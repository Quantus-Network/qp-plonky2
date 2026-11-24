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

theorem publicBatchWrapper89_f11525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10629)) := by
  have c73656 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73657 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73658 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73659 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73660 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73661 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_8 := arithEq_of_rows h (row := 1194) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_8
  simp only [← c73656, ← c73657, ← c73658] at e_1194_8
  have e_1194_9 := arithEq_of_rows h (row := 1194) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_9
  simp only [← c73659, ← c73660, k2, ← c73661] at e_1194_9
  have hr := e_1194_9
  simp only [e_1194_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10630)) := by
  have c73662 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73663 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73664 := (publicBatchWrapper89_copies2302 a h).1
  have c73665 := (publicBatchWrapper89_copies2302 a h).2.1
  have c73666 := (publicBatchWrapper89_copies2302 a h).2.2.1
  have c73667 := (publicBatchWrapper89_copies2302 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_10 := arithEq_of_rows h (row := 1194) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_10
  simp only [← c73662, ← c73663, ← c73664] at e_1194_10
  have e_1194_11 := arithEq_of_rows h (row := 1194) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_11
  simp only [← c73665, ← c73666, k2, ← c73667] at e_1194_11
  have hr := e_1194_11
  simp only [e_1194_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10631)) := by
  have c73668 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.1
  have c73669 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.1
  have c73670 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.1
  have c73671 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.1
  have c73672 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.1
  have c73673 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_12 := arithEq_of_rows h (row := 1194) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_12
  simp only [← c73668, ← c73669, ← c73670] at e_1194_12
  have e_1194_13 := arithEq_of_rows h (row := 1194) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_13
  simp only [← c73671, ← c73672, k2, ← c73673] at e_1194_13
  have hr := e_1194_13
  simp only [e_1194_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10632)) := by
  have c73674 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73675 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73676 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73677 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73678 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73679 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_14 := arithEq_of_rows h (row := 1194) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_14
  simp only [← c73674, ← c73675, ← c73676] at e_1194_14
  have e_1194_15 := arithEq_of_rows h (row := 1194) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_15
  simp only [← c73677, ← c73678, k2, ← c73679] at e_1194_15
  have hr := e_1194_15
  simp only [e_1194_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10633)) := by
  have c73680 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73681 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73682 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73683 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73684 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73685 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_16 := arithEq_of_rows h (row := 1194) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_16
  simp only [← c73680, ← c73681, ← c73682] at e_1194_16
  have e_1194_17 := arithEq_of_rows h (row := 1194) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_17
  simp only [← c73683, ← c73684, k2, ← c73685] at e_1194_17
  have hr := e_1194_17
  simp only [e_1194_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10634)) := by
  have c73686 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73687 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73688 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73689 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73690 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73691 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_18 := arithEq_of_rows h (row := 1194) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_18
  simp only [← c73686, ← c73687, ← c73688] at e_1194_18
  have e_1194_19 := arithEq_of_rows h (row := 1194) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_19
  simp only [← c73689, ← c73690, k2, ← c73691] at e_1194_19
  have hr := e_1194_19
  simp only [e_1194_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10635)) := by
  have c73692 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73693 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73694 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73695 := (publicBatchWrapper89_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73696 := (publicBatchWrapper89_copies2303 a h).1
  have c73697 := (publicBatchWrapper89_copies2303 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_0 := arithEq_of_rows h (row := 1195) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_0
  simp only [← c73692, ← c73693, ← c73694] at e_1195_0
  have e_1195_1 := arithEq_of_rows h (row := 1195) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_1
  simp only [← c73695, ← c73696, k2, ← c73697] at e_1195_1
  have hr := e_1195_1
  simp only [e_1195_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10636)) := by
  have c73698 := (publicBatchWrapper89_copies2303 a h).2.2.1
  have c73699 := (publicBatchWrapper89_copies2303 a h).2.2.2.1
  have c73700 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.1
  have c73701 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.1
  have c73702 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.1
  have c73703 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_2 := arithEq_of_rows h (row := 1195) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_2
  simp only [← c73698, ← c73699, ← c73700] at e_1195_2
  have e_1195_3 := arithEq_of_rows h (row := 1195) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_3
  simp only [← c73701, ← c73702, k2, ← c73703] at e_1195_3
  have hr := e_1195_3
  simp only [e_1195_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10637)) := by
  have c73704 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.1
  have c73705 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.1
  have c73706 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73707 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73708 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73709 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_4 := arithEq_of_rows h (row := 1195) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_4
  simp only [← c73704, ← c73705, ← c73706] at e_1195_4
  have e_1195_5 := arithEq_of_rows h (row := 1195) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_5
  simp only [← c73707, ← c73708, k2, ← c73709] at e_1195_5
  have hr := e_1195_5
  simp only [e_1195_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10638)) := by
  have c73710 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73711 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73712 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73713 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73714 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73715 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_6 := arithEq_of_rows h (row := 1195) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_6
  simp only [← c73710, ← c73711, ← c73712] at e_1195_6
  have e_1195_7 := arithEq_of_rows h (row := 1195) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_7
  simp only [← c73713, ← c73714, k2, ← c73715] at e_1195_7
  have hr := e_1195_7
  simp only [e_1195_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10639)) := by
  have c73716 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73717 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73718 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73719 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73720 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73721 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_8 := arithEq_of_rows h (row := 1195) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_8
  simp only [← c73716, ← c73717, ← c73718] at e_1195_8
  have e_1195_9 := arithEq_of_rows h (row := 1195) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_9
  simp only [← c73719, ← c73720, k2, ← c73721] at e_1195_9
  have hr := e_1195_9
  simp only [e_1195_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10640)) := by
  have c73722 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73723 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73724 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73725 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73726 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73727 := (publicBatchWrapper89_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_10 := arithEq_of_rows h (row := 1195) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_10
  simp only [← c73722, ← c73723, ← c73724] at e_1195_10
  have e_1195_11 := arithEq_of_rows h (row := 1195) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_11
  simp only [← c73725, ← c73726, k2, ← c73727] at e_1195_11
  have hr := e_1195_11
  simp only [e_1195_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10641)) := by
  have c73728 := (publicBatchWrapper89_copies2304 a h).1
  have c73729 := (publicBatchWrapper89_copies2304 a h).2.1
  have c73730 := (publicBatchWrapper89_copies2304 a h).2.2.1
  have c73731 := (publicBatchWrapper89_copies2304 a h).2.2.2.1
  have c73732 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.1
  have c73733 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_12 := arithEq_of_rows h (row := 1195) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_12
  simp only [← c73728, ← c73729, ← c73730] at e_1195_12
  have e_1195_13 := arithEq_of_rows h (row := 1195) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_13
  simp only [← c73731, ← c73732, k2, ← c73733] at e_1195_13
  have hr := e_1195_13
  simp only [e_1195_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10642)) := by
  have c73734 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.1
  have c73735 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.1
  have c73736 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.1
  have c73737 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.1
  have c73738 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73739 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_14 := arithEq_of_rows h (row := 1195) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_14
  simp only [← c73734, ← c73735, ← c73736] at e_1195_14
  have e_1195_15 := arithEq_of_rows h (row := 1195) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_15
  simp only [← c73737, ← c73738, k2, ← c73739] at e_1195_15
  have hr := e_1195_15
  simp only [e_1195_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10643)) := by
  have c73740 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73741 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73742 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73743 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73744 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73745 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_16 := arithEq_of_rows h (row := 1195) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_16
  simp only [← c73740, ← c73741, ← c73742] at e_1195_16
  have e_1195_17 := arithEq_of_rows h (row := 1195) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_17
  simp only [← c73743, ← c73744, k2, ← c73745] at e_1195_17
  have hr := e_1195_17
  simp only [e_1195_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1195 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10644)) := by
  have c73746 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73747 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73748 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73749 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73750 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73751 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1195_18 := arithEq_of_rows h (row := 1195) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_18
  simp only [← c73746, ← c73747, ← c73748] at e_1195_18
  have e_1195_19 := arithEq_of_rows h (row := 1195) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_19
  simp only [← c73749, ← c73750, k2, ← c73751] at e_1195_19
  have hr := e_1195_19
  simp only [e_1195_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10645)) := by
  have c73752 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73753 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73754 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73755 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73756 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73757 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_0 := arithEq_of_rows h (row := 1196) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_0
  simp only [← c73752, ← c73753, ← c73754] at e_1196_0
  have e_1196_1 := arithEq_of_rows h (row := 1196) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_1
  simp only [← c73755, ← c73756, k2, ← c73757] at e_1196_1
  have hr := e_1196_1
  simp only [e_1196_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10646)) := by
  have c73758 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73759 := (publicBatchWrapper89_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73760 := (publicBatchWrapper89_copies2305 a h).1
  have c73761 := (publicBatchWrapper89_copies2305 a h).2.1
  have c73762 := (publicBatchWrapper89_copies2305 a h).2.2.1
  have c73763 := (publicBatchWrapper89_copies2305 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_2 := arithEq_of_rows h (row := 1196) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_2
  simp only [← c73758, ← c73759, ← c73760] at e_1196_2
  have e_1196_3 := arithEq_of_rows h (row := 1196) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_3
  simp only [← c73761, ← c73762, k2, ← c73763] at e_1196_3
  have hr := e_1196_3
  simp only [e_1196_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10647)) := by
  have c73764 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.1
  have c73765 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.1
  have c73766 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.1
  have c73767 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.1
  have c73768 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.1
  have c73769 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_4 := arithEq_of_rows h (row := 1196) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_4
  simp only [← c73764, ← c73765, ← c73766] at e_1196_4
  have e_1196_5 := arithEq_of_rows h (row := 1196) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_5
  simp only [← c73767, ← c73768, k2, ← c73769] at e_1196_5
  have hr := e_1196_5
  simp only [e_1196_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10648)) := by
  have c73770 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73771 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73772 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73773 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73774 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73775 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_6 := arithEq_of_rows h (row := 1196) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_6
  simp only [← c73770, ← c73771, ← c73772] at e_1196_6
  have e_1196_7 := arithEq_of_rows h (row := 1196) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_7
  simp only [← c73773, ← c73774, k2, ← c73775] at e_1196_7
  have hr := e_1196_7
  simp only [e_1196_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10649)) := by
  have c73776 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73777 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73778 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73779 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73780 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73781 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_8 := arithEq_of_rows h (row := 1196) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_8
  simp only [← c73776, ← c73777, ← c73778] at e_1196_8
  have e_1196_9 := arithEq_of_rows h (row := 1196) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_9
  simp only [← c73779, ← c73780, k2, ← c73781] at e_1196_9
  have hr := e_1196_9
  simp only [e_1196_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10650)) := by
  have c73782 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73783 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73784 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73785 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73786 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73787 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_10 := arithEq_of_rows h (row := 1196) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_10
  simp only [← c73782, ← c73783, ← c73784] at e_1196_10
  have e_1196_11 := arithEq_of_rows h (row := 1196) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_11
  simp only [← c73785, ← c73786, k2, ← c73787] at e_1196_11
  have hr := e_1196_11
  simp only [e_1196_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10651)) := by
  have c73788 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73789 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73790 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73791 := (publicBatchWrapper89_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73792 := (publicBatchWrapper89_copies2306 a h).1
  have c73793 := (publicBatchWrapper89_copies2306 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_12 := arithEq_of_rows h (row := 1196) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_12
  simp only [← c73788, ← c73789, ← c73790] at e_1196_12
  have e_1196_13 := arithEq_of_rows h (row := 1196) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_13
  simp only [← c73791, ← c73792, k2, ← c73793] at e_1196_13
  have hr := e_1196_13
  simp only [e_1196_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10652)) := by
  have c73794 := (publicBatchWrapper89_copies2306 a h).2.2.1
  have c73795 := (publicBatchWrapper89_copies2306 a h).2.2.2.1
  have c73796 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.1
  have c73797 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.1
  have c73798 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.1
  have c73799 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_14 := arithEq_of_rows h (row := 1196) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_14
  simp only [← c73794, ← c73795, ← c73796] at e_1196_14
  have e_1196_15 := arithEq_of_rows h (row := 1196) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_15
  simp only [← c73797, ← c73798, k2, ← c73799] at e_1196_15
  have hr := e_1196_15
  simp only [e_1196_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10653)) := by
  have c73800 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.1
  have c73801 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.1
  have c73802 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73803 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73804 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73805 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_16 := arithEq_of_rows h (row := 1196) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_16
  simp only [← c73800, ← c73801, ← c73802] at e_1196_16
  have e_1196_17 := arithEq_of_rows h (row := 1196) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_17
  simp only [← c73803, ← c73804, k2, ← c73805] at e_1196_17
  have hr := e_1196_17
  simp only [e_1196_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1196 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10654)) := by
  have c73806 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73807 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73808 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73809 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73810 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73811 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1196_18 := arithEq_of_rows h (row := 1196) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_18
  simp only [← c73806, ← c73807, ← c73808] at e_1196_18
  have e_1196_19 := arithEq_of_rows h (row := 1196) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_19
  simp only [← c73809, ← c73810, k2, ← c73811] at e_1196_19
  have hr := e_1196_19
  simp only [e_1196_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10655)) := by
  have c73812 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73813 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73814 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73815 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73816 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73817 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_0 := arithEq_of_rows h (row := 1197) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_0
  simp only [← c73812, ← c73813, ← c73814] at e_1197_0
  have e_1197_1 := arithEq_of_rows h (row := 1197) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_1
  simp only [← c73815, ← c73816, k2, ← c73817] at e_1197_1
  have hr := e_1197_1
  simp only [e_1197_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10656)) := by
  have c73818 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73819 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73820 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73821 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73822 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73823 := (publicBatchWrapper89_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_2 := arithEq_of_rows h (row := 1197) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_2
  simp only [← c73818, ← c73819, ← c73820] at e_1197_2
  have e_1197_3 := arithEq_of_rows h (row := 1197) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_3
  simp only [← c73821, ← c73822, k2, ← c73823] at e_1197_3
  have hr := e_1197_3
  simp only [e_1197_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10657)) := by
  have c73824 := (publicBatchWrapper89_copies2307 a h).1
  have c73825 := (publicBatchWrapper89_copies2307 a h).2.1
  have c73826 := (publicBatchWrapper89_copies2307 a h).2.2.1
  have c73827 := (publicBatchWrapper89_copies2307 a h).2.2.2.1
  have c73828 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.1
  have c73829 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_4 := arithEq_of_rows h (row := 1197) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_4
  simp only [← c73824, ← c73825, ← c73826] at e_1197_4
  have e_1197_5 := arithEq_of_rows h (row := 1197) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_5
  simp only [← c73827, ← c73828, k2, ← c73829] at e_1197_5
  have hr := e_1197_5
  simp only [e_1197_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10658)) := by
  have c73830 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.1
  have c73831 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.1
  have c73832 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.1
  have c73833 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.1
  have c73834 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73835 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_6 := arithEq_of_rows h (row := 1197) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_6
  simp only [← c73830, ← c73831, ← c73832] at e_1197_6
  have e_1197_7 := arithEq_of_rows h (row := 1197) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_7
  simp only [← c73833, ← c73834, k2, ← c73835] at e_1197_7
  have hr := e_1197_7
  simp only [e_1197_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10659)) := by
  have c73836 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73837 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73838 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73839 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73840 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73841 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_8 := arithEq_of_rows h (row := 1197) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_8
  simp only [← c73836, ← c73837, ← c73838] at e_1197_8
  have e_1197_9 := arithEq_of_rows h (row := 1197) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_9
  simp only [← c73839, ← c73840, k2, ← c73841] at e_1197_9
  have hr := e_1197_9
  simp only [e_1197_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10660)) := by
  have c73842 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73843 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73844 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73845 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73846 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73847 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_10 := arithEq_of_rows h (row := 1197) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_10
  simp only [← c73842, ← c73843, ← c73844] at e_1197_10
  have e_1197_11 := arithEq_of_rows h (row := 1197) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_11
  simp only [← c73845, ← c73846, k2, ← c73847] at e_1197_11
  have hr := e_1197_11
  simp only [e_1197_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10661)) := by
  have c73848 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73849 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73850 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73851 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73852 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73853 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_12 := arithEq_of_rows h (row := 1197) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_12
  simp only [← c73848, ← c73849, ← c73850] at e_1197_12
  have e_1197_13 := arithEq_of_rows h (row := 1197) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_13
  simp only [← c73851, ← c73852, k2, ← c73853] at e_1197_13
  have hr := e_1197_13
  simp only [e_1197_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10662)) := by
  have c73854 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73855 := (publicBatchWrapper89_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73856 := (publicBatchWrapper89_copies2308 a h).1
  have c73857 := (publicBatchWrapper89_copies2308 a h).2.1
  have c73858 := (publicBatchWrapper89_copies2308 a h).2.2.1
  have c73859 := (publicBatchWrapper89_copies2308 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_14 := arithEq_of_rows h (row := 1197) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_14
  simp only [← c73854, ← c73855, ← c73856] at e_1197_14
  have e_1197_15 := arithEq_of_rows h (row := 1197) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_15
  simp only [← c73857, ← c73858, k2, ← c73859] at e_1197_15
  have hr := e_1197_15
  simp only [e_1197_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10663)) := by
  have c73860 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.1
  have c73861 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.1
  have c73862 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.1
  have c73863 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.1
  have c73864 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.1
  have c73865 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_16 := arithEq_of_rows h (row := 1197) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_16
  simp only [← c73860, ← c73861, ← c73862] at e_1197_16
  have e_1197_17 := arithEq_of_rows h (row := 1197) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_17
  simp only [← c73863, ← c73864, k2, ← c73865] at e_1197_17
  have hr := e_1197_17
  simp only [e_1197_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1197 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10664)) := by
  have c73866 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73867 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73868 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73869 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73870 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73871 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1197_18 := arithEq_of_rows h (row := 1197) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_18
  simp only [← c73866, ← c73867, ← c73868] at e_1197_18
  have e_1197_19 := arithEq_of_rows h (row := 1197) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_19
  simp only [← c73869, ← c73870, k2, ← c73871] at e_1197_19
  have hr := e_1197_19
  simp only [e_1197_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10665)) := by
  have c73872 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73873 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73874 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73875 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73876 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73877 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_0 := arithEq_of_rows h (row := 1198) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_0
  simp only [← c73872, ← c73873, ← c73874] at e_1198_0
  have e_1198_1 := arithEq_of_rows h (row := 1198) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_1
  simp only [← c73875, ← c73876, k2, ← c73877] at e_1198_1
  have hr := e_1198_1
  simp only [e_1198_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10666)) := by
  have c73878 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73879 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73880 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73881 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73882 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73883 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_2 := arithEq_of_rows h (row := 1198) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_2
  simp only [← c73878, ← c73879, ← c73880] at e_1198_2
  have e_1198_3 := arithEq_of_rows h (row := 1198) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_3
  simp only [← c73881, ← c73882, k2, ← c73883] at e_1198_3
  have hr := e_1198_3
  simp only [e_1198_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10667)) := by
  have c73884 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73885 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73886 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73887 := (publicBatchWrapper89_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73888 := (publicBatchWrapper89_copies2309 a h).1
  have c73889 := (publicBatchWrapper89_copies2309 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_4 := arithEq_of_rows h (row := 1198) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_4
  simp only [← c73884, ← c73885, ← c73886] at e_1198_4
  have e_1198_5 := arithEq_of_rows h (row := 1198) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_5
  simp only [← c73887, ← c73888, k2, ← c73889] at e_1198_5
  have hr := e_1198_5
  simp only [e_1198_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10668)) := by
  have c73890 := (publicBatchWrapper89_copies2309 a h).2.2.1
  have c73891 := (publicBatchWrapper89_copies2309 a h).2.2.2.1
  have c73892 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.1
  have c73893 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.1
  have c73894 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.1
  have c73895 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_6 := arithEq_of_rows h (row := 1198) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_6
  simp only [← c73890, ← c73891, ← c73892] at e_1198_6
  have e_1198_7 := arithEq_of_rows h (row := 1198) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_7
  simp only [← c73893, ← c73894, k2, ← c73895] at e_1198_7
  have hr := e_1198_7
  simp only [e_1198_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10669)) := by
  have c73896 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.1
  have c73897 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.1
  have c73898 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73899 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73900 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73901 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_8 := arithEq_of_rows h (row := 1198) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_8
  simp only [← c73896, ← c73897, ← c73898] at e_1198_8
  have e_1198_9 := arithEq_of_rows h (row := 1198) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_9
  simp only [← c73899, ← c73900, k2, ← c73901] at e_1198_9
  have hr := e_1198_9
  simp only [e_1198_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10670)) := by
  have c73902 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73903 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73904 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73905 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73906 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73907 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_10 := arithEq_of_rows h (row := 1198) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_10
  simp only [← c73902, ← c73903, ← c73904] at e_1198_10
  have e_1198_11 := arithEq_of_rows h (row := 1198) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_11
  simp only [← c73905, ← c73906, k2, ← c73907] at e_1198_11
  have hr := e_1198_11
  simp only [e_1198_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10671)) := by
  have c73908 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73909 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73910 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73911 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73912 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73913 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_12 := arithEq_of_rows h (row := 1198) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_12
  simp only [← c73908, ← c73909, ← c73910] at e_1198_12
  have e_1198_13 := arithEq_of_rows h (row := 1198) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_13
  simp only [← c73911, ← c73912, k2, ← c73913] at e_1198_13
  have hr := e_1198_13
  simp only [e_1198_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10672)) := by
  have c73914 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73915 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73916 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73917 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73918 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73919 := (publicBatchWrapper89_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_14 := arithEq_of_rows h (row := 1198) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_14
  simp only [← c73914, ← c73915, ← c73916] at e_1198_14
  have e_1198_15 := arithEq_of_rows h (row := 1198) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_15
  simp only [← c73917, ← c73918, k2, ← c73919] at e_1198_15
  have hr := e_1198_15
  simp only [e_1198_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10673)) := by
  have c73920 := (publicBatchWrapper89_copies2310 a h).1
  have c73921 := (publicBatchWrapper89_copies2310 a h).2.1
  have c73922 := (publicBatchWrapper89_copies2310 a h).2.2.1
  have c73923 := (publicBatchWrapper89_copies2310 a h).2.2.2.1
  have c73924 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.1
  have c73925 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_16 := arithEq_of_rows h (row := 1198) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_16
  simp only [← c73920, ← c73921, ← c73922] at e_1198_16
  have e_1198_17 := arithEq_of_rows h (row := 1198) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_17
  simp only [← c73923, ← c73924, k2, ← c73925] at e_1198_17
  have hr := e_1198_17
  simp only [e_1198_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1198 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10674)) := by
  have c73926 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.1
  have c73927 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.1
  have c73928 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.1
  have c73929 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.1
  have c73930 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73931 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1198_18 := arithEq_of_rows h (row := 1198) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_18
  simp only [← c73926, ← c73927, ← c73928] at e_1198_18
  have e_1198_19 := arithEq_of_rows h (row := 1198) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_19
  simp only [← c73929, ← c73930, k2, ← c73931] at e_1198_19
  have hr := e_1198_19
  simp only [e_1198_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10675)) := by
  have c73932 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73933 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73934 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73935 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73936 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73937 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_0 := arithEq_of_rows h (row := 1199) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_0
  simp only [← c73932, ← c73933, ← c73934] at e_1199_0
  have e_1199_1 := arithEq_of_rows h (row := 1199) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_1
  simp only [← c73935, ← c73936, k2, ← c73937] at e_1199_1
  have hr := e_1199_1
  simp only [e_1199_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10676)) := by
  have c73938 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73939 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73940 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73941 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73942 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73943 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_2 := arithEq_of_rows h (row := 1199) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_2
  simp only [← c73938, ← c73939, ← c73940] at e_1199_2
  have e_1199_3 := arithEq_of_rows h (row := 1199) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_3
  simp only [← c73941, ← c73942, k2, ← c73943] at e_1199_3
  have hr := e_1199_3
  simp only [e_1199_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10677)) := by
  have c73944 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73945 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73946 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73947 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73948 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73949 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_4 := arithEq_of_rows h (row := 1199) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_4
  simp only [← c73944, ← c73945, ← c73946] at e_1199_4
  have e_1199_5 := arithEq_of_rows h (row := 1199) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_5
  simp only [← c73947, ← c73948, k2, ← c73949] at e_1199_5
  have hr := e_1199_5
  simp only [e_1199_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10678)) := by
  have c73950 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73951 := (publicBatchWrapper89_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73952 := (publicBatchWrapper89_copies2311 a h).1
  have c73953 := (publicBatchWrapper89_copies2311 a h).2.1
  have c73954 := (publicBatchWrapper89_copies2311 a h).2.2.1
  have c73955 := (publicBatchWrapper89_copies2311 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_6 := arithEq_of_rows h (row := 1199) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_6
  simp only [← c73950, ← c73951, ← c73952] at e_1199_6
  have e_1199_7 := arithEq_of_rows h (row := 1199) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_7
  simp only [← c73953, ← c73954, k2, ← c73955] at e_1199_7
  have hr := e_1199_7
  simp only [e_1199_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10679)) := by
  have c73956 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.1
  have c73957 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.1
  have c73958 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.1
  have c73959 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.1
  have c73960 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.1
  have c73961 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_8 := arithEq_of_rows h (row := 1199) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_8
  simp only [← c73956, ← c73957, ← c73958] at e_1199_8
  have e_1199_9 := arithEq_of_rows h (row := 1199) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_9
  simp only [← c73959, ← c73960, k2, ← c73961] at e_1199_9
  have hr := e_1199_9
  simp only [e_1199_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10680)) := by
  have c73962 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73963 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73964 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73965 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73966 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73967 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_10 := arithEq_of_rows h (row := 1199) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_10
  simp only [← c73962, ← c73963, ← c73964] at e_1199_10
  have e_1199_11 := arithEq_of_rows h (row := 1199) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_11
  simp only [← c73965, ← c73966, k2, ← c73967] at e_1199_11
  have hr := e_1199_11
  simp only [e_1199_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10681)) := by
  have c73968 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73969 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73970 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73971 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73972 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73973 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_12 := arithEq_of_rows h (row := 1199) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_12
  simp only [← c73968, ← c73969, ← c73970] at e_1199_12
  have e_1199_13 := arithEq_of_rows h (row := 1199) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_13
  simp only [← c73971, ← c73972, k2, ← c73973] at e_1199_13
  have hr := e_1199_13
  simp only [e_1199_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10682)) := by
  have c73974 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73975 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73976 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73977 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73978 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73979 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_14 := arithEq_of_rows h (row := 1199) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_14
  simp only [← c73974, ← c73975, ← c73976] at e_1199_14
  have e_1199_15 := arithEq_of_rows h (row := 1199) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_15
  simp only [← c73977, ← c73978, k2, ← c73979] at e_1199_15
  have hr := e_1199_15
  simp only [e_1199_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10683)) := by
  have c73980 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73981 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73982 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73983 := (publicBatchWrapper89_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73984 := (publicBatchWrapper89_copies2312 a h).1
  have c73985 := (publicBatchWrapper89_copies2312 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_16 := arithEq_of_rows h (row := 1199) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_16
  simp only [← c73980, ← c73981, ← c73982] at e_1199_16
  have e_1199_17 := arithEq_of_rows h (row := 1199) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_17
  simp only [← c73983, ← c73984, k2, ← c73985] at e_1199_17
  have hr := e_1199_17
  simp only [e_1199_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1199 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10684)) := by
  have c73986 := (publicBatchWrapper89_copies2312 a h).2.2.1
  have c73987 := (publicBatchWrapper89_copies2312 a h).2.2.2.1
  have c73988 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.1
  have c73989 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.1
  have c73990 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.1
  have c73991 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1199_18 := arithEq_of_rows h (row := 1199) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_18
  simp only [← c73986, ← c73987, ← c73988] at e_1199_18
  have e_1199_19 := arithEq_of_rows h (row := 1199) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_19
  simp only [← c73989, ← c73990, k2, ← c73991] at e_1199_19
  have hr := e_1199_19
  simp only [e_1199_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10685)) := by
  have c73992 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.1
  have c73993 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.1
  have c73994 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73995 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73996 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73997 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_0 := arithEq_of_rows h (row := 1200) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_0
  simp only [← c73992, ← c73993, ← c73994] at e_1200_0
  have e_1200_1 := arithEq_of_rows h (row := 1200) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_1
  simp only [← c73995, ← c73996, k2, ← c73997] at e_1200_1
  have hr := e_1200_1
  simp only [e_1200_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10686)) := by
  have c73998 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73999 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74000 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74001 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74002 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74003 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_2 := arithEq_of_rows h (row := 1200) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_2
  simp only [← c73998, ← c73999, ← c74000] at e_1200_2
  have e_1200_3 := arithEq_of_rows h (row := 1200) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_3
  simp only [← c74001, ← c74002, k2, ← c74003] at e_1200_3
  have hr := e_1200_3
  simp only [e_1200_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10687)) := by
  have c74004 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74005 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74006 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74007 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74008 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74009 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_4 := arithEq_of_rows h (row := 1200) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_4
  simp only [← c74004, ← c74005, ← c74006] at e_1200_4
  have e_1200_5 := arithEq_of_rows h (row := 1200) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_5
  simp only [← c74007, ← c74008, k2, ← c74009] at e_1200_5
  have hr := e_1200_5
  simp only [e_1200_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10688)) := by
  have c74010 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74011 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74012 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74013 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74014 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74015 := (publicBatchWrapper89_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_6 := arithEq_of_rows h (row := 1200) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_6
  simp only [← c74010, ← c74011, ← c74012] at e_1200_6
  have e_1200_7 := arithEq_of_rows h (row := 1200) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_7
  simp only [← c74013, ← c74014, k2, ← c74015] at e_1200_7
  have hr := e_1200_7
  simp only [e_1200_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10689)) := by
  have c74016 := (publicBatchWrapper89_copies2313 a h).1
  have c74017 := (publicBatchWrapper89_copies2313 a h).2.1
  have c74018 := (publicBatchWrapper89_copies2313 a h).2.2.1
  have c74019 := (publicBatchWrapper89_copies2313 a h).2.2.2.1
  have c74020 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.1
  have c74021 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_8 := arithEq_of_rows h (row := 1200) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_8
  simp only [← c74016, ← c74017, ← c74018] at e_1200_8
  have e_1200_9 := arithEq_of_rows h (row := 1200) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_9
  simp only [← c74019, ← c74020, k2, ← c74021] at e_1200_9
  have hr := e_1200_9
  simp only [e_1200_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10690)) := by
  have c74022 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.1
  have c74023 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.1
  have c74024 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.1
  have c74025 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.1
  have c74026 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74027 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_10 := arithEq_of_rows h (row := 1200) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_10
  simp only [← c74022, ← c74023, ← c74024] at e_1200_10
  have e_1200_11 := arithEq_of_rows h (row := 1200) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_11
  simp only [← c74025, ← c74026, k2, ← c74027] at e_1200_11
  have hr := e_1200_11
  simp only [e_1200_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10691)) := by
  have c74028 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74029 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74030 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74031 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74032 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74033 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_12 := arithEq_of_rows h (row := 1200) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_12
  simp only [← c74028, ← c74029, ← c74030] at e_1200_12
  have e_1200_13 := arithEq_of_rows h (row := 1200) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_13
  simp only [← c74031, ← c74032, k2, ← c74033] at e_1200_13
  have hr := e_1200_13
  simp only [e_1200_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10692)) := by
  have c74034 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74035 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74036 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74037 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74038 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74039 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_14 := arithEq_of_rows h (row := 1200) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_14
  simp only [← c74034, ← c74035, ← c74036] at e_1200_14
  have e_1200_15 := arithEq_of_rows h (row := 1200) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_15
  simp only [← c74037, ← c74038, k2, ← c74039] at e_1200_15
  have hr := e_1200_15
  simp only [e_1200_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10693)) := by
  have c74040 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74041 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74042 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74043 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74044 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74045 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_16 := arithEq_of_rows h (row := 1200) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_16
  simp only [← c74040, ← c74041, ← c74042] at e_1200_16
  have e_1200_17 := arithEq_of_rows h (row := 1200) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_17
  simp only [← c74043, ← c74044, k2, ← c74045] at e_1200_17
  have hr := e_1200_17
  simp only [e_1200_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1200 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10694)) := by
  have c74046 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74047 := (publicBatchWrapper89_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74048 := (publicBatchWrapper89_copies2314 a h).1
  have c74049 := (publicBatchWrapper89_copies2314 a h).2.1
  have c74050 := (publicBatchWrapper89_copies2314 a h).2.2.1
  have c74051 := (publicBatchWrapper89_copies2314 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1200_18 := arithEq_of_rows h (row := 1200) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_18
  simp only [← c74046, ← c74047, ← c74048] at e_1200_18
  have e_1200_19 := arithEq_of_rows h (row := 1200) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_19
  simp only [← c74049, ← c74050, k2, ← c74051] at e_1200_19
  have hr := e_1200_19
  simp only [e_1200_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10695)) := by
  have c74052 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.1
  have c74053 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.1
  have c74054 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.1
  have c74055 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.1
  have c74056 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.1
  have c74057 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_0 := arithEq_of_rows h (row := 1201) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_0
  simp only [← c74052, ← c74053, ← c74054] at e_1201_0
  have e_1201_1 := arithEq_of_rows h (row := 1201) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_1
  simp only [← c74055, ← c74056, k2, ← c74057] at e_1201_1
  have hr := e_1201_1
  simp only [e_1201_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10696)) := by
  have c74058 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74059 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74060 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74061 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74062 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74063 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_2 := arithEq_of_rows h (row := 1201) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_2
  simp only [← c74058, ← c74059, ← c74060] at e_1201_2
  have e_1201_3 := arithEq_of_rows h (row := 1201) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_3
  simp only [← c74061, ← c74062, k2, ← c74063] at e_1201_3
  have hr := e_1201_3
  simp only [e_1201_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10697)) := by
  have c74064 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74065 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74066 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74067 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74068 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74069 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_4 := arithEq_of_rows h (row := 1201) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_4
  simp only [← c74064, ← c74065, ← c74066] at e_1201_4
  have e_1201_5 := arithEq_of_rows h (row := 1201) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_5
  simp only [← c74067, ← c74068, k2, ← c74069] at e_1201_5
  have hr := e_1201_5
  simp only [e_1201_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10698)) := by
  have c74070 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74071 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74072 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74073 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74074 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74075 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_6 := arithEq_of_rows h (row := 1201) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_6
  simp only [← c74070, ← c74071, ← c74072] at e_1201_6
  have e_1201_7 := arithEq_of_rows h (row := 1201) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_7
  simp only [← c74073, ← c74074, k2, ← c74075] at e_1201_7
  have hr := e_1201_7
  simp only [e_1201_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10699)) := by
  have c74076 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74077 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74078 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74079 := (publicBatchWrapper89_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74080 := (publicBatchWrapper89_copies2315 a h).1
  have c74081 := (publicBatchWrapper89_copies2315 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_8 := arithEq_of_rows h (row := 1201) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_8
  simp only [← c74076, ← c74077, ← c74078] at e_1201_8
  have e_1201_9 := arithEq_of_rows h (row := 1201) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_9
  simp only [← c74079, ← c74080, k2, ← c74081] at e_1201_9
  have hr := e_1201_9
  simp only [e_1201_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10700)) := by
  have c74082 := (publicBatchWrapper89_copies2315 a h).2.2.1
  have c74083 := (publicBatchWrapper89_copies2315 a h).2.2.2.1
  have c74084 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.1
  have c74085 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.1
  have c74086 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.1
  have c74087 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_10 := arithEq_of_rows h (row := 1201) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_10
  simp only [← c74082, ← c74083, ← c74084] at e_1201_10
  have e_1201_11 := arithEq_of_rows h (row := 1201) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_11
  simp only [← c74085, ← c74086, k2, ← c74087] at e_1201_11
  have hr := e_1201_11
  simp only [e_1201_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10701)) := by
  have c74088 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.1
  have c74089 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.1
  have c74090 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74091 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74092 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74093 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_12 := arithEq_of_rows h (row := 1201) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_12
  simp only [← c74088, ← c74089, ← c74090] at e_1201_12
  have e_1201_13 := arithEq_of_rows h (row := 1201) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_13
  simp only [← c74091, ← c74092, k2, ← c74093] at e_1201_13
  have hr := e_1201_13
  simp only [e_1201_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10702)) := by
  have c74094 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74095 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74096 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74097 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74098 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74099 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_14 := arithEq_of_rows h (row := 1201) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_14
  simp only [← c74094, ← c74095, ← c74096] at e_1201_14
  have e_1201_15 := arithEq_of_rows h (row := 1201) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_15
  simp only [← c74097, ← c74098, k2, ← c74099] at e_1201_15
  have hr := e_1201_15
  simp only [e_1201_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10703)) := by
  have c74100 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74101 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74102 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74103 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74104 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74105 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_16 := arithEq_of_rows h (row := 1201) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_16
  simp only [← c74100, ← c74101, ← c74102] at e_1201_16
  have e_1201_17 := arithEq_of_rows h (row := 1201) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_17
  simp only [← c74103, ← c74104, k2, ← c74105] at e_1201_17
  have hr := e_1201_17
  simp only [e_1201_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1201 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10704)) := by
  have c74106 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74107 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74108 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74109 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74110 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74111 := (publicBatchWrapper89_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1201_18 := arithEq_of_rows h (row := 1201) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_18
  simp only [← c74106, ← c74107, ← c74108] at e_1201_18
  have e_1201_19 := arithEq_of_rows h (row := 1201) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_19
  simp only [← c74109, ← c74110, k2, ← c74111] at e_1201_19
  have hr := e_1201_19
  simp only [e_1201_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10705)) := by
  have c74112 := (publicBatchWrapper89_copies2316 a h).1
  have c74113 := (publicBatchWrapper89_copies2316 a h).2.1
  have c74114 := (publicBatchWrapper89_copies2316 a h).2.2.1
  have c74115 := (publicBatchWrapper89_copies2316 a h).2.2.2.1
  have c74116 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.1
  have c74117 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_0 := arithEq_of_rows h (row := 1202) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_0
  simp only [← c74112, ← c74113, ← c74114] at e_1202_0
  have e_1202_1 := arithEq_of_rows h (row := 1202) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_1
  simp only [← c74115, ← c74116, k2, ← c74117] at e_1202_1
  have hr := e_1202_1
  simp only [e_1202_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10706)) := by
  have c74118 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.1
  have c74119 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.1
  have c74120 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.1
  have c74121 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.1
  have c74122 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74123 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_2 := arithEq_of_rows h (row := 1202) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_2
  simp only [← c74118, ← c74119, ← c74120] at e_1202_2
  have e_1202_3 := arithEq_of_rows h (row := 1202) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_3
  simp only [← c74121, ← c74122, k2, ← c74123] at e_1202_3
  have hr := e_1202_3
  simp only [e_1202_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10707)) := by
  have c74124 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74125 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74126 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74127 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74128 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74129 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_4 := arithEq_of_rows h (row := 1202) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_4
  simp only [← c74124, ← c74125, ← c74126] at e_1202_4
  have e_1202_5 := arithEq_of_rows h (row := 1202) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_5
  simp only [← c74127, ← c74128, k2, ← c74129] at e_1202_5
  have hr := e_1202_5
  simp only [e_1202_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10708)) := by
  have c74130 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74131 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74132 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74133 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74134 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74135 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_6 := arithEq_of_rows h (row := 1202) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_6
  simp only [← c74130, ← c74131, ← c74132] at e_1202_6
  have e_1202_7 := arithEq_of_rows h (row := 1202) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_7
  simp only [← c74133, ← c74134, k2, ← c74135] at e_1202_7
  have hr := e_1202_7
  simp only [e_1202_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10709)) := by
  have c74136 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74137 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74138 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74139 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74140 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74141 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_8 := arithEq_of_rows h (row := 1202) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_8
  simp only [← c74136, ← c74137, ← c74138] at e_1202_8
  have e_1202_9 := arithEq_of_rows h (row := 1202) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_9
  simp only [← c74139, ← c74140, k2, ← c74141] at e_1202_9
  have hr := e_1202_9
  simp only [e_1202_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10710)) := by
  have c74142 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74143 := (publicBatchWrapper89_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74144 := (publicBatchWrapper89_copies2317 a h).1
  have c74145 := (publicBatchWrapper89_copies2317 a h).2.1
  have c74146 := (publicBatchWrapper89_copies2317 a h).2.2.1
  have c74147 := (publicBatchWrapper89_copies2317 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_10 := arithEq_of_rows h (row := 1202) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_10
  simp only [← c74142, ← c74143, ← c74144] at e_1202_10
  have e_1202_11 := arithEq_of_rows h (row := 1202) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_11
  simp only [← c74145, ← c74146, k2, ← c74147] at e_1202_11
  have hr := e_1202_11
  simp only [e_1202_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10711)) := by
  have c74148 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.1
  have c74149 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.1
  have c74150 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.1
  have c74151 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.1
  have c74152 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.1
  have c74153 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_12 := arithEq_of_rows h (row := 1202) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_12
  simp only [← c74148, ← c74149, ← c74150] at e_1202_12
  have e_1202_13 := arithEq_of_rows h (row := 1202) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_13
  simp only [← c74151, ← c74152, k2, ← c74153] at e_1202_13
  have hr := e_1202_13
  simp only [e_1202_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10712)) := by
  have c74154 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74155 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74156 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74157 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74158 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74159 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_14 := arithEq_of_rows h (row := 1202) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_14
  simp only [← c74154, ← c74155, ← c74156] at e_1202_14
  have e_1202_15 := arithEq_of_rows h (row := 1202) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_15
  simp only [← c74157, ← c74158, k2, ← c74159] at e_1202_15
  have hr := e_1202_15
  simp only [e_1202_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10713)) := by
  have c74160 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74161 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74162 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74163 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74164 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74165 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_16 := arithEq_of_rows h (row := 1202) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_16
  simp only [← c74160, ← c74161, ← c74162] at e_1202_16
  have e_1202_17 := arithEq_of_rows h (row := 1202) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_17
  simp only [← c74163, ← c74164, k2, ← c74165] at e_1202_17
  have hr := e_1202_17
  simp only [e_1202_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1202 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10714)) := by
  have c74166 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74167 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74168 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74169 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74170 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74171 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1202_18 := arithEq_of_rows h (row := 1202) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_18
  simp only [← c74166, ← c74167, ← c74168] at e_1202_18
  have e_1202_19 := arithEq_of_rows h (row := 1202) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_19
  simp only [← c74169, ← c74170, k2, ← c74171] at e_1202_19
  have hr := e_1202_19
  simp only [e_1202_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10715)) := by
  have c74172 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74173 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74174 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74175 := (publicBatchWrapper89_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74176 := (publicBatchWrapper89_copies2318 a h).1
  have c74177 := (publicBatchWrapper89_copies2318 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_0 := arithEq_of_rows h (row := 1203) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_0
  simp only [← c74172, ← c74173, ← c74174] at e_1203_0
  have e_1203_1 := arithEq_of_rows h (row := 1203) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_1
  simp only [← c74175, ← c74176, k2, ← c74177] at e_1203_1
  have hr := e_1203_1
  simp only [e_1203_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10716)) := by
  have c74178 := (publicBatchWrapper89_copies2318 a h).2.2.1
  have c74179 := (publicBatchWrapper89_copies2318 a h).2.2.2.1
  have c74180 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.1
  have c74181 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.1
  have c74182 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.1
  have c74183 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_2 := arithEq_of_rows h (row := 1203) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_2
  simp only [← c74178, ← c74179, ← c74180] at e_1203_2
  have e_1203_3 := arithEq_of_rows h (row := 1203) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_3
  simp only [← c74181, ← c74182, k2, ← c74183] at e_1203_3
  have hr := e_1203_3
  simp only [e_1203_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10717)) := by
  have c74184 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.1
  have c74185 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.1
  have c74186 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74187 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74188 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74189 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_4 := arithEq_of_rows h (row := 1203) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_4
  simp only [← c74184, ← c74185, ← c74186] at e_1203_4
  have e_1203_5 := arithEq_of_rows h (row := 1203) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_5
  simp only [← c74187, ← c74188, k2, ← c74189] at e_1203_5
  have hr := e_1203_5
  simp only [e_1203_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10718)) := by
  have c74190 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74191 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74192 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74193 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74194 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74195 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_6 := arithEq_of_rows h (row := 1203) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_6
  simp only [← c74190, ← c74191, ← c74192] at e_1203_6
  have e_1203_7 := arithEq_of_rows h (row := 1203) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_7
  simp only [← c74193, ← c74194, k2, ← c74195] at e_1203_7
  have hr := e_1203_7
  simp only [e_1203_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10719)) := by
  have c74196 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74197 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74198 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74199 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74200 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74201 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_8 := arithEq_of_rows h (row := 1203) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_8
  simp only [← c74196, ← c74197, ← c74198] at e_1203_8
  have e_1203_9 := arithEq_of_rows h (row := 1203) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_9
  simp only [← c74199, ← c74200, k2, ← c74201] at e_1203_9
  have hr := e_1203_9
  simp only [e_1203_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10720)) := by
  have c74202 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74203 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74204 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74205 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74206 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74207 := (publicBatchWrapper89_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_10 := arithEq_of_rows h (row := 1203) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_10
  simp only [← c74202, ← c74203, ← c74204] at e_1203_10
  have e_1203_11 := arithEq_of_rows h (row := 1203) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_11
  simp only [← c74205, ← c74206, k2, ← c74207] at e_1203_11
  have hr := e_1203_11
  simp only [e_1203_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10721)) := by
  have c74208 := (publicBatchWrapper89_copies2319 a h).1
  have c74209 := (publicBatchWrapper89_copies2319 a h).2.1
  have c74210 := (publicBatchWrapper89_copies2319 a h).2.2.1
  have c74211 := (publicBatchWrapper89_copies2319 a h).2.2.2.1
  have c74212 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.1
  have c74213 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_12 := arithEq_of_rows h (row := 1203) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_12
  simp only [← c74208, ← c74209, ← c74210] at e_1203_12
  have e_1203_13 := arithEq_of_rows h (row := 1203) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_13
  simp only [← c74211, ← c74212, k2, ← c74213] at e_1203_13
  have hr := e_1203_13
  simp only [e_1203_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10722)) := by
  have c74214 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.1
  have c74215 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.1
  have c74216 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.1
  have c74217 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.1
  have c74218 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74219 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_14 := arithEq_of_rows h (row := 1203) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_14
  simp only [← c74214, ← c74215, ← c74216] at e_1203_14
  have e_1203_15 := arithEq_of_rows h (row := 1203) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_15
  simp only [← c74217, ← c74218, k2, ← c74219] at e_1203_15
  have hr := e_1203_15
  simp only [e_1203_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10723)) := by
  have c74220 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74221 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74222 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74223 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74224 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74225 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_16 := arithEq_of_rows h (row := 1203) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_16
  simp only [← c74220, ← c74221, ← c74222] at e_1203_16
  have e_1203_17 := arithEq_of_rows h (row := 1203) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_17
  simp only [← c74223, ← c74224, k2, ← c74225] at e_1203_17
  have hr := e_1203_17
  simp only [e_1203_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1203 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10724)) := by
  have c74226 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74227 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74228 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74229 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74230 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74231 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1203_18 := arithEq_of_rows h (row := 1203) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_18
  simp only [← c74226, ← c74227, ← c74228] at e_1203_18
  have e_1203_19 := arithEq_of_rows h (row := 1203) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_19
  simp only [← c74229, ← c74230, k2, ← c74231] at e_1203_19
  have hr := e_1203_19
  simp only [e_1203_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10725)) := by
  have c74232 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74233 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74234 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74235 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74236 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74237 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_0 := arithEq_of_rows h (row := 1204) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_0
  simp only [← c74232, ← c74233, ← c74234] at e_1204_0
  have e_1204_1 := arithEq_of_rows h (row := 1204) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_1
  simp only [← c74235, ← c74236, k2, ← c74237] at e_1204_1
  have hr := e_1204_1
  simp only [e_1204_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10726)) := by
  have c74238 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74239 := (publicBatchWrapper89_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74240 := (publicBatchWrapper89_copies2320 a h).1
  have c74241 := (publicBatchWrapper89_copies2320 a h).2.1
  have c74242 := (publicBatchWrapper89_copies2320 a h).2.2.1
  have c74243 := (publicBatchWrapper89_copies2320 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_2 := arithEq_of_rows h (row := 1204) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_2
  simp only [← c74238, ← c74239, ← c74240] at e_1204_2
  have e_1204_3 := arithEq_of_rows h (row := 1204) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_3
  simp only [← c74241, ← c74242, k2, ← c74243] at e_1204_3
  have hr := e_1204_3
  simp only [e_1204_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10727)) := by
  have c74244 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.1
  have c74245 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.1
  have c74246 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.1
  have c74247 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.1
  have c74248 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.1
  have c74249 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_4 := arithEq_of_rows h (row := 1204) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_4
  simp only [← c74244, ← c74245, ← c74246] at e_1204_4
  have e_1204_5 := arithEq_of_rows h (row := 1204) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_5
  simp only [← c74247, ← c74248, k2, ← c74249] at e_1204_5
  have hr := e_1204_5
  simp only [e_1204_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10728)) := by
  have c74250 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74251 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74252 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74253 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74254 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74255 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_6 := arithEq_of_rows h (row := 1204) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_6
  simp only [← c74250, ← c74251, ← c74252] at e_1204_6
  have e_1204_7 := arithEq_of_rows h (row := 1204) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_7
  simp only [← c74253, ← c74254, k2, ← c74255] at e_1204_7
  have hr := e_1204_7
  simp only [e_1204_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10729)) := by
  have c74256 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74257 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74258 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74259 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74260 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74261 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_8 := arithEq_of_rows h (row := 1204) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_8
  simp only [← c74256, ← c74257, ← c74258] at e_1204_8
  have e_1204_9 := arithEq_of_rows h (row := 1204) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_9
  simp only [← c74259, ← c74260, k2, ← c74261] at e_1204_9
  have hr := e_1204_9
  simp only [e_1204_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10730)) := by
  have c74262 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74263 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74264 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74265 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74266 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74267 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_10 := arithEq_of_rows h (row := 1204) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_10
  simp only [← c74262, ← c74263, ← c74264] at e_1204_10
  have e_1204_11 := arithEq_of_rows h (row := 1204) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_11
  simp only [← c74265, ← c74266, k2, ← c74267] at e_1204_11
  have hr := e_1204_11
  simp only [e_1204_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10731)) := by
  have c74268 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74269 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74270 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74271 := (publicBatchWrapper89_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74272 := (publicBatchWrapper89_copies2321 a h).1
  have c74273 := (publicBatchWrapper89_copies2321 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_12 := arithEq_of_rows h (row := 1204) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_12
  simp only [← c74268, ← c74269, ← c74270] at e_1204_12
  have e_1204_13 := arithEq_of_rows h (row := 1204) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_13
  simp only [← c74271, ← c74272, k2, ← c74273] at e_1204_13
  have hr := e_1204_13
  simp only [e_1204_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10732)) := by
  have c74274 := (publicBatchWrapper89_copies2321 a h).2.2.1
  have c74275 := (publicBatchWrapper89_copies2321 a h).2.2.2.1
  have c74276 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.1
  have c74277 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.1
  have c74278 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.1
  have c74279 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_14 := arithEq_of_rows h (row := 1204) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_14
  simp only [← c74274, ← c74275, ← c74276] at e_1204_14
  have e_1204_15 := arithEq_of_rows h (row := 1204) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_15
  simp only [← c74277, ← c74278, k2, ← c74279] at e_1204_15
  have hr := e_1204_15
  simp only [e_1204_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10733)) := by
  have c74280 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.1
  have c74281 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.1
  have c74282 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74283 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74284 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74285 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_16 := arithEq_of_rows h (row := 1204) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_16
  simp only [← c74280, ← c74281, ← c74282] at e_1204_16
  have e_1204_17 := arithEq_of_rows h (row := 1204) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_17
  simp only [← c74283, ← c74284, k2, ← c74285] at e_1204_17
  have hr := e_1204_17
  simp only [e_1204_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1204 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10734)) := by
  have c74286 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74287 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74288 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74289 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74290 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74291 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1204_18 := arithEq_of_rows h (row := 1204) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_18
  simp only [← c74286, ← c74287, ← c74288] at e_1204_18
  have e_1204_19 := arithEq_of_rows h (row := 1204) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_19
  simp only [← c74289, ← c74290, k2, ← c74291] at e_1204_19
  have hr := e_1204_19
  simp only [e_1204_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10735)) := by
  have c74292 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74293 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74294 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74295 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74296 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74297 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_0 := arithEq_of_rows h (row := 1205) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_0
  simp only [← c74292, ← c74293, ← c74294] at e_1205_0
  have e_1205_1 := arithEq_of_rows h (row := 1205) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_1
  simp only [← c74295, ← c74296, k2, ← c74297] at e_1205_1
  have hr := e_1205_1
  simp only [e_1205_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10736)) := by
  have c74298 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74299 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74300 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74301 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74302 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74303 := (publicBatchWrapper89_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_2 := arithEq_of_rows h (row := 1205) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_2
  simp only [← c74298, ← c74299, ← c74300] at e_1205_2
  have e_1205_3 := arithEq_of_rows h (row := 1205) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_3
  simp only [← c74301, ← c74302, k2, ← c74303] at e_1205_3
  have hr := e_1205_3
  simp only [e_1205_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10737)) := by
  have c74304 := (publicBatchWrapper89_copies2322 a h).1
  have c74305 := (publicBatchWrapper89_copies2322 a h).2.1
  have c74306 := (publicBatchWrapper89_copies2322 a h).2.2.1
  have c74307 := (publicBatchWrapper89_copies2322 a h).2.2.2.1
  have c74308 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.1
  have c74309 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_4 := arithEq_of_rows h (row := 1205) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_4
  simp only [← c74304, ← c74305, ← c74306] at e_1205_4
  have e_1205_5 := arithEq_of_rows h (row := 1205) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_5
  simp only [← c74307, ← c74308, k2, ← c74309] at e_1205_5
  have hr := e_1205_5
  simp only [e_1205_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10738)) := by
  have c74310 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.1
  have c74311 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.1
  have c74312 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.1
  have c74313 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.1
  have c74314 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74315 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_6 := arithEq_of_rows h (row := 1205) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_6
  simp only [← c74310, ← c74311, ← c74312] at e_1205_6
  have e_1205_7 := arithEq_of_rows h (row := 1205) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_7
  simp only [← c74313, ← c74314, k2, ← c74315] at e_1205_7
  have hr := e_1205_7
  simp only [e_1205_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10739)) := by
  have c74316 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74317 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74318 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74319 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74320 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74321 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_8 := arithEq_of_rows h (row := 1205) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_8
  simp only [← c74316, ← c74317, ← c74318] at e_1205_8
  have e_1205_9 := arithEq_of_rows h (row := 1205) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_9
  simp only [← c74319, ← c74320, k2, ← c74321] at e_1205_9
  have hr := e_1205_9
  simp only [e_1205_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10740)) := by
  have c74322 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74323 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74324 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74325 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74326 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74327 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_10 := arithEq_of_rows h (row := 1205) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_10
  simp only [← c74322, ← c74323, ← c74324] at e_1205_10
  have e_1205_11 := arithEq_of_rows h (row := 1205) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_11
  simp only [← c74325, ← c74326, k2, ← c74327] at e_1205_11
  have hr := e_1205_11
  simp only [e_1205_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10741)) := by
  have c74328 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74329 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74330 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74331 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74332 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74333 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_12 := arithEq_of_rows h (row := 1205) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_12
  simp only [← c74328, ← c74329, ← c74330] at e_1205_12
  have e_1205_13 := arithEq_of_rows h (row := 1205) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_13
  simp only [← c74331, ← c74332, k2, ← c74333] at e_1205_13
  have hr := e_1205_13
  simp only [e_1205_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10742)) := by
  have c74334 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74335 := (publicBatchWrapper89_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74336 := (publicBatchWrapper89_copies2323 a h).1
  have c74337 := (publicBatchWrapper89_copies2323 a h).2.1
  have c74338 := (publicBatchWrapper89_copies2323 a h).2.2.1
  have c74339 := (publicBatchWrapper89_copies2323 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_14 := arithEq_of_rows h (row := 1205) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_14
  simp only [← c74334, ← c74335, ← c74336] at e_1205_14
  have e_1205_15 := arithEq_of_rows h (row := 1205) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_15
  simp only [← c74337, ← c74338, k2, ← c74339] at e_1205_15
  have hr := e_1205_15
  simp only [e_1205_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10743)) := by
  have c74340 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.1
  have c74341 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.1
  have c74342 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.1
  have c74343 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.1
  have c74344 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.1
  have c74345 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_16 := arithEq_of_rows h (row := 1205) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_16
  simp only [← c74340, ← c74341, ← c74342] at e_1205_16
  have e_1205_17 := arithEq_of_rows h (row := 1205) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_17
  simp only [← c74343, ← c74344, k2, ← c74345] at e_1205_17
  have hr := e_1205_17
  simp only [e_1205_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1205 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10744)) := by
  have c74346 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74347 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74348 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74349 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74350 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74351 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1205_18 := arithEq_of_rows h (row := 1205) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_18
  simp only [← c74346, ← c74347, ← c74348] at e_1205_18
  have e_1205_19 := arithEq_of_rows h (row := 1205) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_19
  simp only [← c74349, ← c74350, k2, ← c74351] at e_1205_19
  have hr := e_1205_19
  simp only [e_1205_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10745)) := by
  have c74352 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74353 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74354 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74355 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74356 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74357 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_0 := arithEq_of_rows h (row := 1206) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_0
  simp only [← c74352, ← c74353, ← c74354] at e_1206_0
  have e_1206_1 := arithEq_of_rows h (row := 1206) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_1
  simp only [← c74355, ← c74356, k2, ← c74357] at e_1206_1
  have hr := e_1206_1
  simp only [e_1206_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10746)) := by
  have c74358 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74359 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74360 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74361 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74362 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74363 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_2 := arithEq_of_rows h (row := 1206) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_2
  simp only [← c74358, ← c74359, ← c74360] at e_1206_2
  have e_1206_3 := arithEq_of_rows h (row := 1206) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_3
  simp only [← c74361, ← c74362, k2, ← c74363] at e_1206_3
  have hr := e_1206_3
  simp only [e_1206_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10747)) := by
  have c74364 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74365 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74366 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74367 := (publicBatchWrapper89_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74368 := (publicBatchWrapper89_copies2324 a h).1
  have c74369 := (publicBatchWrapper89_copies2324 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_4 := arithEq_of_rows h (row := 1206) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_4
  simp only [← c74364, ← c74365, ← c74366] at e_1206_4
  have e_1206_5 := arithEq_of_rows h (row := 1206) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_5
  simp only [← c74367, ← c74368, k2, ← c74369] at e_1206_5
  have hr := e_1206_5
  simp only [e_1206_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10748)) := by
  have c74370 := (publicBatchWrapper89_copies2324 a h).2.2.1
  have c74371 := (publicBatchWrapper89_copies2324 a h).2.2.2.1
  have c74372 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.1
  have c74373 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.1
  have c74374 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.1
  have c74375 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_6 := arithEq_of_rows h (row := 1206) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_6
  simp only [← c74370, ← c74371, ← c74372] at e_1206_6
  have e_1206_7 := arithEq_of_rows h (row := 1206) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_7
  simp only [← c74373, ← c74374, k2, ← c74375] at e_1206_7
  have hr := e_1206_7
  simp only [e_1206_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10749)) := by
  have c74376 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.1
  have c74377 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.1
  have c74378 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74379 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74380 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74381 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_8 := arithEq_of_rows h (row := 1206) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_8
  simp only [← c74376, ← c74377, ← c74378] at e_1206_8
  have e_1206_9 := arithEq_of_rows h (row := 1206) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_9
  simp only [← c74379, ← c74380, k2, ← c74381] at e_1206_9
  have hr := e_1206_9
  simp only [e_1206_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10750)) := by
  have c74382 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74383 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74384 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74385 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74386 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74387 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_10 := arithEq_of_rows h (row := 1206) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_10
  simp only [← c74382, ← c74383, ← c74384] at e_1206_10
  have e_1206_11 := arithEq_of_rows h (row := 1206) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_11
  simp only [← c74385, ← c74386, k2, ← c74387] at e_1206_11
  have hr := e_1206_11
  simp only [e_1206_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10751)) := by
  have c74388 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74389 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74390 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74391 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74392 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74393 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_12 := arithEq_of_rows h (row := 1206) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_12
  simp only [← c74388, ← c74389, ← c74390] at e_1206_12
  have e_1206_13 := arithEq_of_rows h (row := 1206) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_13
  simp only [← c74391, ← c74392, k2, ← c74393] at e_1206_13
  have hr := e_1206_13
  simp only [e_1206_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10752)) := by
  have c74394 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74395 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74396 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74397 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74398 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74399 := (publicBatchWrapper89_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_14 := arithEq_of_rows h (row := 1206) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_14
  simp only [← c74394, ← c74395, ← c74396] at e_1206_14
  have e_1206_15 := arithEq_of_rows h (row := 1206) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_15
  simp only [← c74397, ← c74398, k2, ← c74399] at e_1206_15
  have hr := e_1206_15
  simp only [e_1206_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10753)) := by
  have c74400 := (publicBatchWrapper89_copies2325 a h).1
  have c74401 := (publicBatchWrapper89_copies2325 a h).2.1
  have c74402 := (publicBatchWrapper89_copies2325 a h).2.2.1
  have c74403 := (publicBatchWrapper89_copies2325 a h).2.2.2.1
  have c74404 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.1
  have c74405 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_16 := arithEq_of_rows h (row := 1206) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_16
  simp only [← c74400, ← c74401, ← c74402] at e_1206_16
  have e_1206_17 := arithEq_of_rows h (row := 1206) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_17
  simp only [← c74403, ← c74404, k2, ← c74405] at e_1206_17
  have hr := e_1206_17
  simp only [e_1206_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1206 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10754)) := by
  have c74406 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.1
  have c74407 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.1
  have c74408 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.1
  have c74409 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.1
  have c74410 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74411 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1206_18 := arithEq_of_rows h (row := 1206) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_18
  simp only [← c74406, ← c74407, ← c74408] at e_1206_18
  have e_1206_19 := arithEq_of_rows h (row := 1206) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_19
  simp only [← c74409, ← c74410, k2, ← c74411] at e_1206_19
  have hr := e_1206_19
  simp only [e_1206_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10755)) := by
  have c74412 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74413 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74414 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74415 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74416 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74417 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_0 := arithEq_of_rows h (row := 1207) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_0
  simp only [← c74412, ← c74413, ← c74414] at e_1207_0
  have e_1207_1 := arithEq_of_rows h (row := 1207) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_1
  simp only [← c74415, ← c74416, k2, ← c74417] at e_1207_1
  have hr := e_1207_1
  simp only [e_1207_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10756)) := by
  have c74418 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74419 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74420 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74421 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74422 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74423 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_2 := arithEq_of_rows h (row := 1207) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_2
  simp only [← c74418, ← c74419, ← c74420] at e_1207_2
  have e_1207_3 := arithEq_of_rows h (row := 1207) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_3
  simp only [← c74421, ← c74422, k2, ← c74423] at e_1207_3
  have hr := e_1207_3
  simp only [e_1207_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10757)) := by
  have c74424 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74425 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74426 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74427 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74428 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74429 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_4 := arithEq_of_rows h (row := 1207) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_4
  simp only [← c74424, ← c74425, ← c74426] at e_1207_4
  have e_1207_5 := arithEq_of_rows h (row := 1207) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_5
  simp only [← c74427, ← c74428, k2, ← c74429] at e_1207_5
  have hr := e_1207_5
  simp only [e_1207_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10758)) := by
  have c74430 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74431 := (publicBatchWrapper89_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74432 := (publicBatchWrapper89_copies2326 a h).1
  have c74433 := (publicBatchWrapper89_copies2326 a h).2.1
  have c74434 := (publicBatchWrapper89_copies2326 a h).2.2.1
  have c74435 := (publicBatchWrapper89_copies2326 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_6 := arithEq_of_rows h (row := 1207) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_6
  simp only [← c74430, ← c74431, ← c74432] at e_1207_6
  have e_1207_7 := arithEq_of_rows h (row := 1207) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_7
  simp only [← c74433, ← c74434, k2, ← c74435] at e_1207_7
  have hr := e_1207_7
  simp only [e_1207_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10759)) := by
  have c74436 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.1
  have c74437 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.1
  have c74438 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.1
  have c74439 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.1
  have c74440 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.1
  have c74441 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_8 := arithEq_of_rows h (row := 1207) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_8
  simp only [← c74436, ← c74437, ← c74438] at e_1207_8
  have e_1207_9 := arithEq_of_rows h (row := 1207) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_9
  simp only [← c74439, ← c74440, k2, ← c74441] at e_1207_9
  have hr := e_1207_9
  simp only [e_1207_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10760)) := by
  have c74442 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74443 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74444 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74445 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74446 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74447 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_10 := arithEq_of_rows h (row := 1207) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_10
  simp only [← c74442, ← c74443, ← c74444] at e_1207_10
  have e_1207_11 := arithEq_of_rows h (row := 1207) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_11
  simp only [← c74445, ← c74446, k2, ← c74447] at e_1207_11
  have hr := e_1207_11
  simp only [e_1207_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10761)) := by
  have c74448 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74449 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74450 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74451 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74452 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74453 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_12 := arithEq_of_rows h (row := 1207) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_12
  simp only [← c74448, ← c74449, ← c74450] at e_1207_12
  have e_1207_13 := arithEq_of_rows h (row := 1207) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_13
  simp only [← c74451, ← c74452, k2, ← c74453] at e_1207_13
  have hr := e_1207_13
  simp only [e_1207_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10762)) := by
  have c74454 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74455 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74456 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74457 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74458 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74459 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_14 := arithEq_of_rows h (row := 1207) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_14
  simp only [← c74454, ← c74455, ← c74456] at e_1207_14
  have e_1207_15 := arithEq_of_rows h (row := 1207) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_15
  simp only [← c74457, ← c74458, k2, ← c74459] at e_1207_15
  have hr := e_1207_15
  simp only [e_1207_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10763)) := by
  have c74460 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74461 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74462 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74463 := (publicBatchWrapper89_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74464 := (publicBatchWrapper89_copies2327 a h).1
  have c74465 := (publicBatchWrapper89_copies2327 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_16 := arithEq_of_rows h (row := 1207) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_16
  simp only [← c74460, ← c74461, ← c74462] at e_1207_16
  have e_1207_17 := arithEq_of_rows h (row := 1207) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_17
  simp only [← c74463, ← c74464, k2, ← c74465] at e_1207_17
  have hr := e_1207_17
  simp only [e_1207_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1207 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10764)) := by
  have c74466 := (publicBatchWrapper89_copies2327 a h).2.2.1
  have c74467 := (publicBatchWrapper89_copies2327 a h).2.2.2.1
  have c74468 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.1
  have c74469 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.1
  have c74470 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.1
  have c74471 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1207_18 := arithEq_of_rows h (row := 1207) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_18
  simp only [← c74466, ← c74467, ← c74468] at e_1207_18
  have e_1207_19 := arithEq_of_rows h (row := 1207) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_19
  simp only [← c74469, ← c74470, k2, ← c74471] at e_1207_19
  have hr := e_1207_19
  simp only [e_1207_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10765)) := by
  have c74472 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.1
  have c74473 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.1
  have c74474 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74475 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74476 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74477 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_0 := arithEq_of_rows h (row := 1208) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_0
  simp only [← c74472, ← c74473, ← c74474] at e_1208_0
  have e_1208_1 := arithEq_of_rows h (row := 1208) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_1
  simp only [← c74475, ← c74476, k2, ← c74477] at e_1208_1
  have hr := e_1208_1
  simp only [e_1208_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10766)) := by
  have c74478 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74479 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74480 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74481 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74482 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74483 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_2 := arithEq_of_rows h (row := 1208) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_2
  simp only [← c74478, ← c74479, ← c74480] at e_1208_2
  have e_1208_3 := arithEq_of_rows h (row := 1208) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_3
  simp only [← c74481, ← c74482, k2, ← c74483] at e_1208_3
  have hr := e_1208_3
  simp only [e_1208_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10767)) := by
  have c74484 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74485 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74486 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74487 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74488 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74489 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_4 := arithEq_of_rows h (row := 1208) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_4
  simp only [← c74484, ← c74485, ← c74486] at e_1208_4
  have e_1208_5 := arithEq_of_rows h (row := 1208) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_5
  simp only [← c74487, ← c74488, k2, ← c74489] at e_1208_5
  have hr := e_1208_5
  simp only [e_1208_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10768)) := by
  have c74490 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74491 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74492 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74493 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74494 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74495 := (publicBatchWrapper89_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_6 := arithEq_of_rows h (row := 1208) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_6
  simp only [← c74490, ← c74491, ← c74492] at e_1208_6
  have e_1208_7 := arithEq_of_rows h (row := 1208) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_7
  simp only [← c74493, ← c74494, k2, ← c74495] at e_1208_7
  have hr := e_1208_7
  simp only [e_1208_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10769)) := by
  have c74496 := (publicBatchWrapper89_copies2328 a h).1
  have c74497 := (publicBatchWrapper89_copies2328 a h).2.1
  have c74498 := (publicBatchWrapper89_copies2328 a h).2.2.1
  have c74499 := (publicBatchWrapper89_copies2328 a h).2.2.2.1
  have c74500 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.1
  have c74501 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_8 := arithEq_of_rows h (row := 1208) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_8
  simp only [← c74496, ← c74497, ← c74498] at e_1208_8
  have e_1208_9 := arithEq_of_rows h (row := 1208) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_9
  simp only [← c74499, ← c74500, k2, ← c74501] at e_1208_9
  have hr := e_1208_9
  simp only [e_1208_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10810)) := by
  have c74502 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.1
  have c74503 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.1
  have c74504 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.1
  have c74505 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.1
  have c74506 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74507 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_10 := arithEq_of_rows h (row := 1208) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_10
  simp only [← c74502, ← c74503, ← c74504] at e_1208_10
  have e_1208_11 := arithEq_of_rows h (row := 1208) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_11
  simp only [← c74505, ← c74506, k2, ← c74507] at e_1208_11
  have hr := e_1208_11
  simp only [e_1208_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10811)) := by
  have c74508 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74509 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74510 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74511 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74512 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74513 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_12 := arithEq_of_rows h (row := 1208) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_12
  simp only [← c74508, ← c74509, ← c74510] at e_1208_12
  have e_1208_13 := arithEq_of_rows h (row := 1208) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_13
  simp only [← c74511, ← c74512, k2, ← c74513] at e_1208_13
  have hr := e_1208_13
  simp only [e_1208_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10812)) := by
  have c74514 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74515 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74516 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74517 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74518 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74519 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_14 := arithEq_of_rows h (row := 1208) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_14
  simp only [← c74514, ← c74515, ← c74516] at e_1208_14
  have e_1208_15 := arithEq_of_rows h (row := 1208) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_15
  simp only [← c74517, ← c74518, k2, ← c74519] at e_1208_15
  have hr := e_1208_15
  simp only [e_1208_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10813)) := by
  have c74520 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74521 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74522 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74523 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74524 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74525 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_16 := arithEq_of_rows h (row := 1208) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_16
  simp only [← c74520, ← c74521, ← c74522] at e_1208_16
  have e_1208_17 := arithEq_of_rows h (row := 1208) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_17
  simp only [← c74523, ← c74524, k2, ← c74525] at e_1208_17
  have hr := e_1208_17
  simp only [e_1208_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1208 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10814)) := by
  have c74526 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74527 := (publicBatchWrapper89_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74528 := (publicBatchWrapper89_copies2329 a h).1
  have c74529 := (publicBatchWrapper89_copies2329 a h).2.1
  have c74530 := (publicBatchWrapper89_copies2329 a h).2.2.1
  have c74531 := (publicBatchWrapper89_copies2329 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1208_18 := arithEq_of_rows h (row := 1208) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_18
  simp only [← c74526, ← c74527, ← c74528] at e_1208_18
  have e_1208_19 := arithEq_of_rows h (row := 1208) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_19
  simp only [← c74529, ← c74530, k2, ← c74531] at e_1208_19
  have hr := e_1208_19
  simp only [e_1208_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10815)) := by
  have c74532 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.1
  have c74533 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.1
  have c74534 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.1
  have c74535 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.1
  have c74536 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.1
  have c74537 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_0 := arithEq_of_rows h (row := 1209) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_0
  simp only [← c74532, ← c74533, ← c74534] at e_1209_0
  have e_1209_1 := arithEq_of_rows h (row := 1209) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_1
  simp only [← c74535, ← c74536, k2, ← c74537] at e_1209_1
  have hr := e_1209_1
  simp only [e_1209_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10816)) := by
  have c74538 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74539 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74540 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74541 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74542 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74543 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_2 := arithEq_of_rows h (row := 1209) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_2
  simp only [← c74538, ← c74539, ← c74540] at e_1209_2
  have e_1209_3 := arithEq_of_rows h (row := 1209) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_3
  simp only [← c74541, ← c74542, k2, ← c74543] at e_1209_3
  have hr := e_1209_3
  simp only [e_1209_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10817)) := by
  have c74544 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74545 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74546 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74547 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74548 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74549 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_4 := arithEq_of_rows h (row := 1209) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_4
  simp only [← c74544, ← c74545, ← c74546] at e_1209_4
  have e_1209_5 := arithEq_of_rows h (row := 1209) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_5
  simp only [← c74547, ← c74548, k2, ← c74549] at e_1209_5
  have hr := e_1209_5
  simp only [e_1209_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10818)) := by
  have c74550 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74551 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74552 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74553 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74554 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74555 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_6 := arithEq_of_rows h (row := 1209) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_6
  simp only [← c74550, ← c74551, ← c74552] at e_1209_6
  have e_1209_7 := arithEq_of_rows h (row := 1209) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_7
  simp only [← c74553, ← c74554, k2, ← c74555] at e_1209_7
  have hr := e_1209_7
  simp only [e_1209_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10819)) := by
  have c74556 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74557 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74558 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74559 := (publicBatchWrapper89_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74560 := (publicBatchWrapper89_copies2330 a h).1
  have c74561 := (publicBatchWrapper89_copies2330 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_8 := arithEq_of_rows h (row := 1209) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_8
  simp only [← c74556, ← c74557, ← c74558] at e_1209_8
  have e_1209_9 := arithEq_of_rows h (row := 1209) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_9
  simp only [← c74559, ← c74560, k2, ← c74561] at e_1209_9
  have hr := e_1209_9
  simp only [e_1209_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10820)) := by
  have c74562 := (publicBatchWrapper89_copies2330 a h).2.2.1
  have c74563 := (publicBatchWrapper89_copies2330 a h).2.2.2.1
  have c74564 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.1
  have c74565 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.1
  have c74566 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.1
  have c74567 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_10 := arithEq_of_rows h (row := 1209) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_10
  simp only [← c74562, ← c74563, ← c74564] at e_1209_10
  have e_1209_11 := arithEq_of_rows h (row := 1209) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_11
  simp only [← c74565, ← c74566, k2, ← c74567] at e_1209_11
  have hr := e_1209_11
  simp only [e_1209_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10821)) := by
  have c74568 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.1
  have c74569 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.1
  have c74570 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74571 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74572 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74573 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_12 := arithEq_of_rows h (row := 1209) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_12
  simp only [← c74568, ← c74569, ← c74570] at e_1209_12
  have e_1209_13 := arithEq_of_rows h (row := 1209) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_13
  simp only [← c74571, ← c74572, k2, ← c74573] at e_1209_13
  have hr := e_1209_13
  simp only [e_1209_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10822)) := by
  have c74574 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74575 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74576 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74577 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74578 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74579 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_14 := arithEq_of_rows h (row := 1209) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_14
  simp only [← c74574, ← c74575, ← c74576] at e_1209_14
  have e_1209_15 := arithEq_of_rows h (row := 1209) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_15
  simp only [← c74577, ← c74578, k2, ← c74579] at e_1209_15
  have hr := e_1209_15
  simp only [e_1209_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10823)) := by
  have c74580 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74581 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74582 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74583 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74584 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74585 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_16 := arithEq_of_rows h (row := 1209) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_16
  simp only [← c74580, ← c74581, ← c74582] at e_1209_16
  have e_1209_17 := arithEq_of_rows h (row := 1209) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_17
  simp only [← c74583, ← c74584, k2, ← c74585] at e_1209_17
  have hr := e_1209_17
  simp only [e_1209_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1209 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10824)) := by
  have c74586 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74587 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74588 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74589 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74590 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74591 := (publicBatchWrapper89_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1209_18 := arithEq_of_rows h (row := 1209) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_18
  simp only [← c74586, ← c74587, ← c74588] at e_1209_18
  have e_1209_19 := arithEq_of_rows h (row := 1209) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_19
  simp only [← c74589, ← c74590, k2, ← c74591] at e_1209_19
  have hr := e_1209_19
  simp only [e_1209_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10825)) := by
  have c74592 := (publicBatchWrapper89_copies2331 a h).1
  have c74593 := (publicBatchWrapper89_copies2331 a h).2.1
  have c74594 := (publicBatchWrapper89_copies2331 a h).2.2.1
  have c74595 := (publicBatchWrapper89_copies2331 a h).2.2.2.1
  have c74596 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.1
  have c74597 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_0 := arithEq_of_rows h (row := 1210) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_0
  simp only [← c74592, ← c74593, ← c74594] at e_1210_0
  have e_1210_1 := arithEq_of_rows h (row := 1210) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_1
  simp only [← c74595, ← c74596, k2, ← c74597] at e_1210_1
  have hr := e_1210_1
  simp only [e_1210_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10826)) := by
  have c74598 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.1
  have c74599 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.1
  have c74600 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.1
  have c74601 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.1
  have c74602 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74603 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_2 := arithEq_of_rows h (row := 1210) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_2
  simp only [← c74598, ← c74599, ← c74600] at e_1210_2
  have e_1210_3 := arithEq_of_rows h (row := 1210) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_3
  simp only [← c74601, ← c74602, k2, ← c74603] at e_1210_3
  have hr := e_1210_3
  simp only [e_1210_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10827)) := by
  have c74604 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74605 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74606 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74607 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74608 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74609 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_4 := arithEq_of_rows h (row := 1210) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_4
  simp only [← c74604, ← c74605, ← c74606] at e_1210_4
  have e_1210_5 := arithEq_of_rows h (row := 1210) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_5
  simp only [← c74607, ← c74608, k2, ← c74609] at e_1210_5
  have hr := e_1210_5
  simp only [e_1210_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10828)) := by
  have c74610 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74611 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74612 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74613 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74614 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74615 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_6 := arithEq_of_rows h (row := 1210) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_6
  simp only [← c74610, ← c74611, ← c74612] at e_1210_6
  have e_1210_7 := arithEq_of_rows h (row := 1210) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_7
  simp only [← c74613, ← c74614, k2, ← c74615] at e_1210_7
  have hr := e_1210_7
  simp only [e_1210_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10829)) := by
  have c74616 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74617 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74618 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74619 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74620 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74621 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_8 := arithEq_of_rows h (row := 1210) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_8
  simp only [← c74616, ← c74617, ← c74618] at e_1210_8
  have e_1210_9 := arithEq_of_rows h (row := 1210) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_9
  simp only [← c74619, ← c74620, k2, ← c74621] at e_1210_9
  have hr := e_1210_9
  simp only [e_1210_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10830)) := by
  have c74622 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74623 := (publicBatchWrapper89_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74624 := (publicBatchWrapper89_copies2332 a h).1
  have c74625 := (publicBatchWrapper89_copies2332 a h).2.1
  have c74626 := (publicBatchWrapper89_copies2332 a h).2.2.1
  have c74627 := (publicBatchWrapper89_copies2332 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_10 := arithEq_of_rows h (row := 1210) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_10
  simp only [← c74622, ← c74623, ← c74624] at e_1210_10
  have e_1210_11 := arithEq_of_rows h (row := 1210) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_11
  simp only [← c74625, ← c74626, k2, ← c74627] at e_1210_11
  have hr := e_1210_11
  simp only [e_1210_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10831)) := by
  have c74628 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.1
  have c74629 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.1
  have c74630 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.1
  have c74631 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.1
  have c74632 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.1
  have c74633 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_12 := arithEq_of_rows h (row := 1210) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_12
  simp only [← c74628, ← c74629, ← c74630] at e_1210_12
  have e_1210_13 := arithEq_of_rows h (row := 1210) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_13
  simp only [← c74631, ← c74632, k2, ← c74633] at e_1210_13
  have hr := e_1210_13
  simp only [e_1210_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10832)) := by
  have c74634 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74635 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74636 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74637 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74638 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74639 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_14 := arithEq_of_rows h (row := 1210) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_14
  simp only [← c74634, ← c74635, ← c74636] at e_1210_14
  have e_1210_15 := arithEq_of_rows h (row := 1210) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_15
  simp only [← c74637, ← c74638, k2, ← c74639] at e_1210_15
  have hr := e_1210_15
  simp only [e_1210_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10833)) := by
  have c74640 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74641 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74642 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74643 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74644 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74645 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_16 := arithEq_of_rows h (row := 1210) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_16
  simp only [← c74640, ← c74641, ← c74642] at e_1210_16
  have e_1210_17 := arithEq_of_rows h (row := 1210) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_17
  simp only [← c74643, ← c74644, k2, ← c74645] at e_1210_17
  have hr := e_1210_17
  simp only [e_1210_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1210 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10834)) := by
  have c74646 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74647 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74648 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74649 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74650 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74651 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1210_18 := arithEq_of_rows h (row := 1210) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_18
  simp only [← c74646, ← c74647, ← c74648] at e_1210_18
  have e_1210_19 := arithEq_of_rows h (row := 1210) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_19
  simp only [← c74649, ← c74650, k2, ← c74651] at e_1210_19
  have hr := e_1210_19
  simp only [e_1210_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10835)) := by
  have c74652 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74653 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74654 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74655 := (publicBatchWrapper89_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74656 := (publicBatchWrapper89_copies2333 a h).1
  have c74657 := (publicBatchWrapper89_copies2333 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_0 := arithEq_of_rows h (row := 1211) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_0
  simp only [← c74652, ← c74653, ← c74654] at e_1211_0
  have e_1211_1 := arithEq_of_rows h (row := 1211) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_1
  simp only [← c74655, ← c74656, k2, ← c74657] at e_1211_1
  have hr := e_1211_1
  simp only [e_1211_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10836)) := by
  have c74658 := (publicBatchWrapper89_copies2333 a h).2.2.1
  have c74659 := (publicBatchWrapper89_copies2333 a h).2.2.2.1
  have c74660 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.1
  have c74661 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.1
  have c74662 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.1
  have c74663 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_2 := arithEq_of_rows h (row := 1211) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_2
  simp only [← c74658, ← c74659, ← c74660] at e_1211_2
  have e_1211_3 := arithEq_of_rows h (row := 1211) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_3
  simp only [← c74661, ← c74662, k2, ← c74663] at e_1211_3
  have hr := e_1211_3
  simp only [e_1211_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10837)) := by
  have c74664 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.1
  have c74665 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.1
  have c74666 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74667 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74668 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74669 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_4 := arithEq_of_rows h (row := 1211) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_4
  simp only [← c74664, ← c74665, ← c74666] at e_1211_4
  have e_1211_5 := arithEq_of_rows h (row := 1211) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_5
  simp only [← c74667, ← c74668, k2, ← c74669] at e_1211_5
  have hr := e_1211_5
  simp only [e_1211_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10838)) := by
  have c74670 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74671 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74672 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74673 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74674 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74675 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_6 := arithEq_of_rows h (row := 1211) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_6
  simp only [← c74670, ← c74671, ← c74672] at e_1211_6
  have e_1211_7 := arithEq_of_rows h (row := 1211) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_7
  simp only [← c74673, ← c74674, k2, ← c74675] at e_1211_7
  have hr := e_1211_7
  simp only [e_1211_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10839)) := by
  have c74676 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74677 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74678 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74679 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74680 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74681 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_8 := arithEq_of_rows h (row := 1211) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_8
  simp only [← c74676, ← c74677, ← c74678] at e_1211_8
  have e_1211_9 := arithEq_of_rows h (row := 1211) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_9
  simp only [← c74679, ← c74680, k2, ← c74681] at e_1211_9
  have hr := e_1211_9
  simp only [e_1211_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10840)) := by
  have c74682 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74683 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74684 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74685 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74686 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74687 := (publicBatchWrapper89_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_10 := arithEq_of_rows h (row := 1211) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_10
  simp only [← c74682, ← c74683, ← c74684] at e_1211_10
  have e_1211_11 := arithEq_of_rows h (row := 1211) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_11
  simp only [← c74685, ← c74686, k2, ← c74687] at e_1211_11
  have hr := e_1211_11
  simp only [e_1211_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10841)) := by
  have c74688 := (publicBatchWrapper89_copies2334 a h).1
  have c74689 := (publicBatchWrapper89_copies2334 a h).2.1
  have c74690 := (publicBatchWrapper89_copies2334 a h).2.2.1
  have c74691 := (publicBatchWrapper89_copies2334 a h).2.2.2.1
  have c74692 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.1
  have c74693 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_12 := arithEq_of_rows h (row := 1211) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_12
  simp only [← c74688, ← c74689, ← c74690] at e_1211_12
  have e_1211_13 := arithEq_of_rows h (row := 1211) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_13
  simp only [← c74691, ← c74692, k2, ← c74693] at e_1211_13
  have hr := e_1211_13
  simp only [e_1211_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10842)) := by
  have c74694 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.1
  have c74695 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.1
  have c74696 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.1
  have c74697 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.1
  have c74698 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74699 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_14 := arithEq_of_rows h (row := 1211) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_14
  simp only [← c74694, ← c74695, ← c74696] at e_1211_14
  have e_1211_15 := arithEq_of_rows h (row := 1211) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_15
  simp only [← c74697, ← c74698, k2, ← c74699] at e_1211_15
  have hr := e_1211_15
  simp only [e_1211_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10843)) := by
  have c74700 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74701 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74702 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74703 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74704 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74705 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_16 := arithEq_of_rows h (row := 1211) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_16
  simp only [← c74700, ← c74701, ← c74702] at e_1211_16
  have e_1211_17 := arithEq_of_rows h (row := 1211) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_17
  simp only [← c74703, ← c74704, k2, ← c74705] at e_1211_17
  have hr := e_1211_17
  simp only [e_1211_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1211 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10844)) := by
  have c74706 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74707 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74708 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74709 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74710 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74711 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1211_18 := arithEq_of_rows h (row := 1211) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_18
  simp only [← c74706, ← c74707, ← c74708] at e_1211_18
  have e_1211_19 := arithEq_of_rows h (row := 1211) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_19
  simp only [← c74709, ← c74710, k2, ← c74711] at e_1211_19
  have hr := e_1211_19
  simp only [e_1211_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10845)) := by
  have c74712 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74713 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74714 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74715 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74716 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74717 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_0 := arithEq_of_rows h (row := 1212) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_0
  simp only [← c74712, ← c74713, ← c74714] at e_1212_0
  have e_1212_1 := arithEq_of_rows h (row := 1212) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_1
  simp only [← c74715, ← c74716, k2, ← c74717] at e_1212_1
  have hr := e_1212_1
  simp only [e_1212_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10846)) := by
  have c74718 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74719 := (publicBatchWrapper89_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74720 := (publicBatchWrapper89_copies2335 a h).1
  have c74721 := (publicBatchWrapper89_copies2335 a h).2.1
  have c74722 := (publicBatchWrapper89_copies2335 a h).2.2.1
  have c74723 := (publicBatchWrapper89_copies2335 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_2 := arithEq_of_rows h (row := 1212) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_2
  simp only [← c74718, ← c74719, ← c74720] at e_1212_2
  have e_1212_3 := arithEq_of_rows h (row := 1212) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_3
  simp only [← c74721, ← c74722, k2, ← c74723] at e_1212_3
  have hr := e_1212_3
  simp only [e_1212_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10847)) := by
  have c74724 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.1
  have c74725 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.1
  have c74726 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.1
  have c74727 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.1
  have c74728 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.1
  have c74729 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_4 := arithEq_of_rows h (row := 1212) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_4
  simp only [← c74724, ← c74725, ← c74726] at e_1212_4
  have e_1212_5 := arithEq_of_rows h (row := 1212) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_5
  simp only [← c74727, ← c74728, k2, ← c74729] at e_1212_5
  have hr := e_1212_5
  simp only [e_1212_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10848)) := by
  have c74730 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74731 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74732 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74733 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74734 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74735 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_6 := arithEq_of_rows h (row := 1212) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_6
  simp only [← c74730, ← c74731, ← c74732] at e_1212_6
  have e_1212_7 := arithEq_of_rows h (row := 1212) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_7
  simp only [← c74733, ← c74734, k2, ← c74735] at e_1212_7
  have hr := e_1212_7
  simp only [e_1212_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10849)) := by
  have c74736 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74737 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74738 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74739 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74740 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74741 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_8 := arithEq_of_rows h (row := 1212) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_8
  simp only [← c74736, ← c74737, ← c74738] at e_1212_8
  have e_1212_9 := arithEq_of_rows h (row := 1212) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_9
  simp only [← c74739, ← c74740, k2, ← c74741] at e_1212_9
  have hr := e_1212_9
  simp only [e_1212_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10850)) := by
  have c74742 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74743 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74744 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74745 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74746 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74747 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_10 := arithEq_of_rows h (row := 1212) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_10
  simp only [← c74742, ← c74743, ← c74744] at e_1212_10
  have e_1212_11 := arithEq_of_rows h (row := 1212) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_11
  simp only [← c74745, ← c74746, k2, ← c74747] at e_1212_11
  have hr := e_1212_11
  simp only [e_1212_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10851)) := by
  have c74748 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74749 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74750 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74751 := (publicBatchWrapper89_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74752 := (publicBatchWrapper89_copies2336 a h).1
  have c74753 := (publicBatchWrapper89_copies2336 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_12 := arithEq_of_rows h (row := 1212) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_12
  simp only [← c74748, ← c74749, ← c74750] at e_1212_12
  have e_1212_13 := arithEq_of_rows h (row := 1212) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_13
  simp only [← c74751, ← c74752, k2, ← c74753] at e_1212_13
  have hr := e_1212_13
  simp only [e_1212_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10852)) := by
  have c74754 := (publicBatchWrapper89_copies2336 a h).2.2.1
  have c74755 := (publicBatchWrapper89_copies2336 a h).2.2.2.1
  have c74756 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.1
  have c74757 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.1
  have c74758 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.1
  have c74759 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_14 := arithEq_of_rows h (row := 1212) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_14
  simp only [← c74754, ← c74755, ← c74756] at e_1212_14
  have e_1212_15 := arithEq_of_rows h (row := 1212) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_15
  simp only [← c74757, ← c74758, k2, ← c74759] at e_1212_15
  have hr := e_1212_15
  simp only [e_1212_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10853)) := by
  have c74760 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.1
  have c74761 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.1
  have c74762 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74763 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74764 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74765 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_16 := arithEq_of_rows h (row := 1212) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_16
  simp only [← c74760, ← c74761, ← c74762] at e_1212_16
  have e_1212_17 := arithEq_of_rows h (row := 1212) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_17
  simp only [← c74763, ← c74764, k2, ← c74765] at e_1212_17
  have hr := e_1212_17
  simp only [e_1212_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1212 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10854)) := by
  have c74766 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74767 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74768 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74769 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74770 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74771 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1212_18 := arithEq_of_rows h (row := 1212) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_18
  simp only [← c74766, ← c74767, ← c74768] at e_1212_18
  have e_1212_19 := arithEq_of_rows h (row := 1212) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_19
  simp only [← c74769, ← c74770, k2, ← c74771] at e_1212_19
  have hr := e_1212_19
  simp only [e_1212_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10855)) := by
  have c74772 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74773 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74774 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74775 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74776 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74777 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_0 := arithEq_of_rows h (row := 1213) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_0
  simp only [← c74772, ← c74773, ← c74774] at e_1213_0
  have e_1213_1 := arithEq_of_rows h (row := 1213) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_1
  simp only [← c74775, ← c74776, k2, ← c74777] at e_1213_1
  have hr := e_1213_1
  simp only [e_1213_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10856)) := by
  have c74778 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74779 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74780 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74781 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74782 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74783 := (publicBatchWrapper89_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_2 := arithEq_of_rows h (row := 1213) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_2
  simp only [← c74778, ← c74779, ← c74780] at e_1213_2
  have e_1213_3 := arithEq_of_rows h (row := 1213) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_3
  simp only [← c74781, ← c74782, k2, ← c74783] at e_1213_3
  have hr := e_1213_3
  simp only [e_1213_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10857)) := by
  have c74784 := (publicBatchWrapper89_copies2337 a h).1
  have c74785 := (publicBatchWrapper89_copies2337 a h).2.1
  have c74786 := (publicBatchWrapper89_copies2337 a h).2.2.1
  have c74787 := (publicBatchWrapper89_copies2337 a h).2.2.2.1
  have c74788 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.1
  have c74789 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_4 := arithEq_of_rows h (row := 1213) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_4
  simp only [← c74784, ← c74785, ← c74786] at e_1213_4
  have e_1213_5 := arithEq_of_rows h (row := 1213) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_5
  simp only [← c74787, ← c74788, k2, ← c74789] at e_1213_5
  have hr := e_1213_5
  simp only [e_1213_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10858)) := by
  have c74790 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.1
  have c74791 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.1
  have c74792 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.1
  have c74793 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.1
  have c74794 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74795 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_6 := arithEq_of_rows h (row := 1213) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_6
  simp only [← c74790, ← c74791, ← c74792] at e_1213_6
  have e_1213_7 := arithEq_of_rows h (row := 1213) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_7
  simp only [← c74793, ← c74794, k2, ← c74795] at e_1213_7
  have hr := e_1213_7
  simp only [e_1213_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10859)) := by
  have c74796 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74797 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74798 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74799 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74800 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74801 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_8 := arithEq_of_rows h (row := 1213) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_8
  simp only [← c74796, ← c74797, ← c74798] at e_1213_8
  have e_1213_9 := arithEq_of_rows h (row := 1213) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_9
  simp only [← c74799, ← c74800, k2, ← c74801] at e_1213_9
  have hr := e_1213_9
  simp only [e_1213_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10860)) := by
  have c74802 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74803 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74804 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74805 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74806 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74807 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_10 := arithEq_of_rows h (row := 1213) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_10
  simp only [← c74802, ← c74803, ← c74804] at e_1213_10
  have e_1213_11 := arithEq_of_rows h (row := 1213) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_11
  simp only [← c74805, ← c74806, k2, ← c74807] at e_1213_11
  have hr := e_1213_11
  simp only [e_1213_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10861)) := by
  have c74808 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74809 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74810 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74811 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74812 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74813 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_12 := arithEq_of_rows h (row := 1213) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_12
  simp only [← c74808, ← c74809, ← c74810] at e_1213_12
  have e_1213_13 := arithEq_of_rows h (row := 1213) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_13
  simp only [← c74811, ← c74812, k2, ← c74813] at e_1213_13
  have hr := e_1213_13
  simp only [e_1213_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10862)) := by
  have c74814 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74815 := (publicBatchWrapper89_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74816 := (publicBatchWrapper89_copies2338 a h).1
  have c74817 := (publicBatchWrapper89_copies2338 a h).2.1
  have c74818 := (publicBatchWrapper89_copies2338 a h).2.2.1
  have c74819 := (publicBatchWrapper89_copies2338 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_14 := arithEq_of_rows h (row := 1213) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_14
  simp only [← c74814, ← c74815, ← c74816] at e_1213_14
  have e_1213_15 := arithEq_of_rows h (row := 1213) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_15
  simp only [← c74817, ← c74818, k2, ← c74819] at e_1213_15
  have hr := e_1213_15
  simp only [e_1213_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10863)) := by
  have c74820 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.1
  have c74821 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.1
  have c74822 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.1
  have c74823 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.1
  have c74824 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.1
  have c74825 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_16 := arithEq_of_rows h (row := 1213) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_16
  simp only [← c74820, ← c74821, ← c74822] at e_1213_16
  have e_1213_17 := arithEq_of_rows h (row := 1213) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_17
  simp only [← c74823, ← c74824, k2, ← c74825] at e_1213_17
  have hr := e_1213_17
  simp only [e_1213_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1213 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10864)) := by
  have c74826 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74827 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74828 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74829 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74830 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74831 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1213_18 := arithEq_of_rows h (row := 1213) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_18
  simp only [← c74826, ← c74827, ← c74828] at e_1213_18
  have e_1213_19 := arithEq_of_rows h (row := 1213) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_19
  simp only [← c74829, ← c74830, k2, ← c74831] at e_1213_19
  have hr := e_1213_19
  simp only [e_1213_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10865)) := by
  have c74832 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74833 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74834 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74835 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74836 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74837 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_0 := arithEq_of_rows h (row := 1214) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_0
  simp only [← c74832, ← c74833, ← c74834] at e_1214_0
  have e_1214_1 := arithEq_of_rows h (row := 1214) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_1
  simp only [← c74835, ← c74836, k2, ← c74837] at e_1214_1
  have hr := e_1214_1
  simp only [e_1214_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10866)) := by
  have c74838 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74839 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74840 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74841 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74842 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74843 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_2 := arithEq_of_rows h (row := 1214) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_2
  simp only [← c74838, ← c74839, ← c74840] at e_1214_2
  have e_1214_3 := arithEq_of_rows h (row := 1214) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_3
  simp only [← c74841, ← c74842, k2, ← c74843] at e_1214_3
  have hr := e_1214_3
  simp only [e_1214_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10867)) := by
  have c74844 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74845 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74846 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74847 := (publicBatchWrapper89_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74848 := (publicBatchWrapper89_copies2339 a h).1
  have c74849 := (publicBatchWrapper89_copies2339 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_4 := arithEq_of_rows h (row := 1214) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_4
  simp only [← c74844, ← c74845, ← c74846] at e_1214_4
  have e_1214_5 := arithEq_of_rows h (row := 1214) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_5
  simp only [← c74847, ← c74848, k2, ← c74849] at e_1214_5
  have hr := e_1214_5
  simp only [e_1214_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10868)) := by
  have c74850 := (publicBatchWrapper89_copies2339 a h).2.2.1
  have c74851 := (publicBatchWrapper89_copies2339 a h).2.2.2.1
  have c74852 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.1
  have c74853 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.1
  have c74854 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.1
  have c74855 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_6 := arithEq_of_rows h (row := 1214) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_6
  simp only [← c74850, ← c74851, ← c74852] at e_1214_6
  have e_1214_7 := arithEq_of_rows h (row := 1214) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_7
  simp only [← c74853, ← c74854, k2, ← c74855] at e_1214_7
  have hr := e_1214_7
  simp only [e_1214_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10869)) := by
  have c74856 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.1
  have c74857 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.1
  have c74858 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74859 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74860 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74861 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_8 := arithEq_of_rows h (row := 1214) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_8
  simp only [← c74856, ← c74857, ← c74858] at e_1214_8
  have e_1214_9 := arithEq_of_rows h (row := 1214) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_9
  simp only [← c74859, ← c74860, k2, ← c74861] at e_1214_9
  have hr := e_1214_9
  simp only [e_1214_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10870)) := by
  have c74862 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74863 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74864 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74865 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74866 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74867 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_10 := arithEq_of_rows h (row := 1214) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_10
  simp only [← c74862, ← c74863, ← c74864] at e_1214_10
  have e_1214_11 := arithEq_of_rows h (row := 1214) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_11
  simp only [← c74865, ← c74866, k2, ← c74867] at e_1214_11
  have hr := e_1214_11
  simp only [e_1214_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10871)) := by
  have c74868 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74869 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74870 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74871 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74872 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74873 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_12 := arithEq_of_rows h (row := 1214) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_12
  simp only [← c74868, ← c74869, ← c74870] at e_1214_12
  have e_1214_13 := arithEq_of_rows h (row := 1214) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_13
  simp only [← c74871, ← c74872, k2, ← c74873] at e_1214_13
  have hr := e_1214_13
  simp only [e_1214_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10872)) := by
  have c74874 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74875 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74876 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74877 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74878 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74879 := (publicBatchWrapper89_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_14 := arithEq_of_rows h (row := 1214) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_14
  simp only [← c74874, ← c74875, ← c74876] at e_1214_14
  have e_1214_15 := arithEq_of_rows h (row := 1214) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_15
  simp only [← c74877, ← c74878, k2, ← c74879] at e_1214_15
  have hr := e_1214_15
  simp only [e_1214_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10873)) := by
  have c74880 := (publicBatchWrapper89_copies2340 a h).1
  have c74881 := (publicBatchWrapper89_copies2340 a h).2.1
  have c74882 := (publicBatchWrapper89_copies2340 a h).2.2.1
  have c74883 := (publicBatchWrapper89_copies2340 a h).2.2.2.1
  have c74884 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.1
  have c74885 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_16 := arithEq_of_rows h (row := 1214) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_16
  simp only [← c74880, ← c74881, ← c74882] at e_1214_16
  have e_1214_17 := arithEq_of_rows h (row := 1214) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_17
  simp only [← c74883, ← c74884, k2, ← c74885] at e_1214_17
  have hr := e_1214_17
  simp only [e_1214_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1214 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10874)) := by
  have c74886 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.1
  have c74887 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.1
  have c74888 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.1
  have c74889 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.1
  have c74890 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74891 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1214_18 := arithEq_of_rows h (row := 1214) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_18
  simp only [← c74886, ← c74887, ← c74888] at e_1214_18
  have e_1214_19 := arithEq_of_rows h (row := 1214) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_19
  simp only [← c74889, ← c74890, k2, ← c74891] at e_1214_19
  have hr := e_1214_19
  simp only [e_1214_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10875)) := by
  have c74892 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74893 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74894 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74895 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74896 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74897 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_0 := arithEq_of_rows h (row := 1215) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_0
  simp only [← c74892, ← c74893, ← c74894] at e_1215_0
  have e_1215_1 := arithEq_of_rows h (row := 1215) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_1
  simp only [← c74895, ← c74896, k2, ← c74897] at e_1215_1
  have hr := e_1215_1
  simp only [e_1215_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10876)) := by
  have c74898 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74899 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74900 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74901 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74902 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74903 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_2 := arithEq_of_rows h (row := 1215) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_2
  simp only [← c74898, ← c74899, ← c74900] at e_1215_2
  have e_1215_3 := arithEq_of_rows h (row := 1215) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_3
  simp only [← c74901, ← c74902, k2, ← c74903] at e_1215_3
  have hr := e_1215_3
  simp only [e_1215_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10877)) := by
  have c74904 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74905 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74906 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74907 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74908 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74909 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_4 := arithEq_of_rows h (row := 1215) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_4
  simp only [← c74904, ← c74905, ← c74906] at e_1215_4
  have e_1215_5 := arithEq_of_rows h (row := 1215) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_5
  simp only [← c74907, ← c74908, k2, ← c74909] at e_1215_5
  have hr := e_1215_5
  simp only [e_1215_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10878)) := by
  have c74910 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74911 := (publicBatchWrapper89_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74912 := (publicBatchWrapper89_copies2341 a h).1
  have c74913 := (publicBatchWrapper89_copies2341 a h).2.1
  have c74914 := (publicBatchWrapper89_copies2341 a h).2.2.1
  have c74915 := (publicBatchWrapper89_copies2341 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_6 := arithEq_of_rows h (row := 1215) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_6
  simp only [← c74910, ← c74911, ← c74912] at e_1215_6
  have e_1215_7 := arithEq_of_rows h (row := 1215) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_7
  simp only [← c74913, ← c74914, k2, ← c74915] at e_1215_7
  have hr := e_1215_7
  simp only [e_1215_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10879)) := by
  have c74916 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.1
  have c74917 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.1
  have c74918 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.1
  have c74919 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.1
  have c74920 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.1
  have c74921 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_8 := arithEq_of_rows h (row := 1215) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_8
  simp only [← c74916, ← c74917, ← c74918] at e_1215_8
  have e_1215_9 := arithEq_of_rows h (row := 1215) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_9
  simp only [← c74919, ← c74920, k2, ← c74921] at e_1215_9
  have hr := e_1215_9
  simp only [e_1215_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10880)) := by
  have c74922 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74923 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74924 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74925 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74926 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74927 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_10 := arithEq_of_rows h (row := 1215) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_10
  simp only [← c74922, ← c74923, ← c74924] at e_1215_10
  have e_1215_11 := arithEq_of_rows h (row := 1215) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_11
  simp only [← c74925, ← c74926, k2, ← c74927] at e_1215_11
  have hr := e_1215_11
  simp only [e_1215_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10881)) := by
  have c74928 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74929 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74930 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74931 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74932 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74933 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_12 := arithEq_of_rows h (row := 1215) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_12
  simp only [← c74928, ← c74929, ← c74930] at e_1215_12
  have e_1215_13 := arithEq_of_rows h (row := 1215) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_13
  simp only [← c74931, ← c74932, k2, ← c74933] at e_1215_13
  have hr := e_1215_13
  simp only [e_1215_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10882)) := by
  have c74934 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74935 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74936 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74937 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74938 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74939 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_14 := arithEq_of_rows h (row := 1215) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_14
  simp only [← c74934, ← c74935, ← c74936] at e_1215_14
  have e_1215_15 := arithEq_of_rows h (row := 1215) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_15
  simp only [← c74937, ← c74938, k2, ← c74939] at e_1215_15
  have hr := e_1215_15
  simp only [e_1215_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10883)) := by
  have c74940 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74941 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74942 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74943 := (publicBatchWrapper89_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74944 := (publicBatchWrapper89_copies2342 a h).1
  have c74945 := (publicBatchWrapper89_copies2342 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_16 := arithEq_of_rows h (row := 1215) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_16
  simp only [← c74940, ← c74941, ← c74942] at e_1215_16
  have e_1215_17 := arithEq_of_rows h (row := 1215) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_17
  simp only [← c74943, ← c74944, k2, ← c74945] at e_1215_17
  have hr := e_1215_17
  simp only [e_1215_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1215 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10884)) := by
  have c74946 := (publicBatchWrapper89_copies2342 a h).2.2.1
  have c74947 := (publicBatchWrapper89_copies2342 a h).2.2.2.1
  have c74948 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.1
  have c74949 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.1
  have c74950 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.1
  have c74951 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1215_18 := arithEq_of_rows h (row := 1215) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_18
  simp only [← c74946, ← c74947, ← c74948] at e_1215_18
  have e_1215_19 := arithEq_of_rows h (row := 1215) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_19
  simp only [← c74949, ← c74950, k2, ← c74951] at e_1215_19
  have hr := e_1215_19
  simp only [e_1215_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10885)) := by
  have c74952 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.1
  have c74953 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.1
  have c74954 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74955 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74956 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74957 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_0 := arithEq_of_rows h (row := 1216) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_0
  simp only [← c74952, ← c74953, ← c74954] at e_1216_0
  have e_1216_1 := arithEq_of_rows h (row := 1216) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_1
  simp only [← c74955, ← c74956, k2, ← c74957] at e_1216_1
  have hr := e_1216_1
  simp only [e_1216_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10886)) := by
  have c74958 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74959 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74960 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74961 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74962 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74963 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_2 := arithEq_of_rows h (row := 1216) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_2
  simp only [← c74958, ← c74959, ← c74960] at e_1216_2
  have e_1216_3 := arithEq_of_rows h (row := 1216) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_3
  simp only [← c74961, ← c74962, k2, ← c74963] at e_1216_3
  have hr := e_1216_3
  simp only [e_1216_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10887)) := by
  have c74964 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74965 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74966 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74967 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74968 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74969 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_4 := arithEq_of_rows h (row := 1216) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_4
  simp only [← c74964, ← c74965, ← c74966] at e_1216_4
  have e_1216_5 := arithEq_of_rows h (row := 1216) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_5
  simp only [← c74967, ← c74968, k2, ← c74969] at e_1216_5
  have hr := e_1216_5
  simp only [e_1216_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10888)) := by
  have c74970 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74971 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74972 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74973 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74974 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74975 := (publicBatchWrapper89_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_6 := arithEq_of_rows h (row := 1216) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_6
  simp only [← c74970, ← c74971, ← c74972] at e_1216_6
  have e_1216_7 := arithEq_of_rows h (row := 1216) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_7
  simp only [← c74973, ← c74974, k2, ← c74975] at e_1216_7
  have hr := e_1216_7
  simp only [e_1216_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10889)) := by
  have c74976 := (publicBatchWrapper89_copies2343 a h).1
  have c74977 := (publicBatchWrapper89_copies2343 a h).2.1
  have c74978 := (publicBatchWrapper89_copies2343 a h).2.2.1
  have c74979 := (publicBatchWrapper89_copies2343 a h).2.2.2.1
  have c74980 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.1
  have c74981 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_8 := arithEq_of_rows h (row := 1216) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_8
  simp only [← c74976, ← c74977, ← c74978] at e_1216_8
  have e_1216_9 := arithEq_of_rows h (row := 1216) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_9
  simp only [← c74979, ← c74980, k2, ← c74981] at e_1216_9
  have hr := e_1216_9
  simp only [e_1216_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10890)) := by
  have c74982 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.1
  have c74983 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.1
  have c74984 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.1
  have c74985 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.1
  have c74986 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74987 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_10 := arithEq_of_rows h (row := 1216) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_10
  simp only [← c74982, ← c74983, ← c74984] at e_1216_10
  have e_1216_11 := arithEq_of_rows h (row := 1216) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_11
  simp only [← c74985, ← c74986, k2, ← c74987] at e_1216_11
  have hr := e_1216_11
  simp only [e_1216_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10891)) := by
  have c74988 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74989 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74990 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74991 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74992 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74993 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_12 := arithEq_of_rows h (row := 1216) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_12
  simp only [← c74988, ← c74989, ← c74990] at e_1216_12
  have e_1216_13 := arithEq_of_rows h (row := 1216) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_13
  simp only [← c74991, ← c74992, k2, ← c74993] at e_1216_13
  have hr := e_1216_13
  simp only [e_1216_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10892)) := by
  have c74994 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74995 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74996 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74997 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74998 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74999 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_14 := arithEq_of_rows h (row := 1216) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_14
  simp only [← c74994, ← c74995, ← c74996] at e_1216_14
  have e_1216_15 := arithEq_of_rows h (row := 1216) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_15
  simp only [← c74997, ← c74998, k2, ← c74999] at e_1216_15
  have hr := e_1216_15
  simp only [e_1216_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10893)) := by
  have c75000 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75001 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75002 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75003 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75004 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75005 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_16 := arithEq_of_rows h (row := 1216) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_16
  simp only [← c75000, ← c75001, ← c75002] at e_1216_16
  have e_1216_17 := arithEq_of_rows h (row := 1216) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_17
  simp only [← c75003, ← c75004, k2, ← c75005] at e_1216_17
  have hr := e_1216_17
  simp only [e_1216_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1216 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10894)) := by
  have c75006 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75007 := (publicBatchWrapper89_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75008 := (publicBatchWrapper89_copies2344 a h).1
  have c75009 := (publicBatchWrapper89_copies2344 a h).2.1
  have c75010 := (publicBatchWrapper89_copies2344 a h).2.2.1
  have c75011 := (publicBatchWrapper89_copies2344 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1216_18 := arithEq_of_rows h (row := 1216) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_18
  simp only [← c75006, ← c75007, ← c75008] at e_1216_18
  have e_1216_19 := arithEq_of_rows h (row := 1216) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_19
  simp only [← c75009, ← c75010, k2, ← c75011] at e_1216_19
  have hr := e_1216_19
  simp only [e_1216_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10895)) := by
  have c75012 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.1
  have c75013 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.1
  have c75014 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.1
  have c75015 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.1
  have c75016 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.1
  have c75017 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_0 := arithEq_of_rows h (row := 1217) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_0
  simp only [← c75012, ← c75013, ← c75014] at e_1217_0
  have e_1217_1 := arithEq_of_rows h (row := 1217) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_1
  simp only [← c75015, ← c75016, k2, ← c75017] at e_1217_1
  have hr := e_1217_1
  simp only [e_1217_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10896)) := by
  have c75018 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75019 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75020 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75021 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75022 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75023 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_2 := arithEq_of_rows h (row := 1217) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_2
  simp only [← c75018, ← c75019, ← c75020] at e_1217_2
  have e_1217_3 := arithEq_of_rows h (row := 1217) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_3
  simp only [← c75021, ← c75022, k2, ← c75023] at e_1217_3
  have hr := e_1217_3
  simp only [e_1217_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10897)) := by
  have c75024 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75025 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75026 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75027 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75028 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75029 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_4 := arithEq_of_rows h (row := 1217) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_4
  simp only [← c75024, ← c75025, ← c75026] at e_1217_4
  have e_1217_5 := arithEq_of_rows h (row := 1217) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_5
  simp only [← c75027, ← c75028, k2, ← c75029] at e_1217_5
  have hr := e_1217_5
  simp only [e_1217_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10898)) := by
  have c75030 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75031 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75032 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75033 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75034 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75035 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_6 := arithEq_of_rows h (row := 1217) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_6
  simp only [← c75030, ← c75031, ← c75032] at e_1217_6
  have e_1217_7 := arithEq_of_rows h (row := 1217) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_7
  simp only [← c75033, ← c75034, k2, ← c75035] at e_1217_7
  have hr := e_1217_7
  simp only [e_1217_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10899)) := by
  have c75036 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75037 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75038 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75039 := (publicBatchWrapper89_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75040 := (publicBatchWrapper89_copies2345 a h).1
  have c75041 := (publicBatchWrapper89_copies2345 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_8 := arithEq_of_rows h (row := 1217) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_8
  simp only [← c75036, ← c75037, ← c75038] at e_1217_8
  have e_1217_9 := arithEq_of_rows h (row := 1217) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_9
  simp only [← c75039, ← c75040, k2, ← c75041] at e_1217_9
  have hr := e_1217_9
  simp only [e_1217_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10900)) := by
  have c75042 := (publicBatchWrapper89_copies2345 a h).2.2.1
  have c75043 := (publicBatchWrapper89_copies2345 a h).2.2.2.1
  have c75044 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.1
  have c75045 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.1
  have c75046 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.1
  have c75047 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_10 := arithEq_of_rows h (row := 1217) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_10
  simp only [← c75042, ← c75043, ← c75044] at e_1217_10
  have e_1217_11 := arithEq_of_rows h (row := 1217) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_11
  simp only [← c75045, ← c75046, k2, ← c75047] at e_1217_11
  have hr := e_1217_11
  simp only [e_1217_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10901)) := by
  have c75048 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.1
  have c75049 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.1
  have c75050 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75051 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75052 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75053 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_12 := arithEq_of_rows h (row := 1217) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_12
  simp only [← c75048, ← c75049, ← c75050] at e_1217_12
  have e_1217_13 := arithEq_of_rows h (row := 1217) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_13
  simp only [← c75051, ← c75052, k2, ← c75053] at e_1217_13
  have hr := e_1217_13
  simp only [e_1217_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10902)) := by
  have c75054 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75055 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75056 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75057 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75058 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75059 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_14 := arithEq_of_rows h (row := 1217) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_14
  simp only [← c75054, ← c75055, ← c75056] at e_1217_14
  have e_1217_15 := arithEq_of_rows h (row := 1217) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_15
  simp only [← c75057, ← c75058, k2, ← c75059] at e_1217_15
  have hr := e_1217_15
  simp only [e_1217_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10903)) := by
  have c75060 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75061 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75062 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75063 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75064 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75065 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_16 := arithEq_of_rows h (row := 1217) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_16
  simp only [← c75060, ← c75061, ← c75062] at e_1217_16
  have e_1217_17 := arithEq_of_rows h (row := 1217) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_17
  simp only [← c75063, ← c75064, k2, ← c75065] at e_1217_17
  have hr := e_1217_17
  simp only [e_1217_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1217 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10904)) := by
  have c75066 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75067 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75068 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75069 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75070 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75071 := (publicBatchWrapper89_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1217_18 := arithEq_of_rows h (row := 1217) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_18
  simp only [← c75066, ← c75067, ← c75068] at e_1217_18
  have e_1217_19 := arithEq_of_rows h (row := 1217) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_19
  simp only [← c75069, ← c75070, k2, ← c75071] at e_1217_19
  have hr := e_1217_19
  simp only [e_1217_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10905)) := by
  have c75072 := (publicBatchWrapper89_copies2346 a h).1
  have c75073 := (publicBatchWrapper89_copies2346 a h).2.1
  have c75074 := (publicBatchWrapper89_copies2346 a h).2.2.1
  have c75075 := (publicBatchWrapper89_copies2346 a h).2.2.2.1
  have c75076 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.1
  have c75077 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_0 := arithEq_of_rows h (row := 1218) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_0
  simp only [← c75072, ← c75073, ← c75074] at e_1218_0
  have e_1218_1 := arithEq_of_rows h (row := 1218) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_1
  simp only [← c75075, ← c75076, k2, ← c75077] at e_1218_1
  have hr := e_1218_1
  simp only [e_1218_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10906)) := by
  have c75078 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.1
  have c75079 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.1
  have c75080 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.1
  have c75081 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.1
  have c75082 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75083 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_2 := arithEq_of_rows h (row := 1218) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_2
  simp only [← c75078, ← c75079, ← c75080] at e_1218_2
  have e_1218_3 := arithEq_of_rows h (row := 1218) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_3
  simp only [← c75081, ← c75082, k2, ← c75083] at e_1218_3
  have hr := e_1218_3
  simp only [e_1218_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10907)) := by
  have c75084 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75085 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75086 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75087 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75088 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75089 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_4 := arithEq_of_rows h (row := 1218) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_4
  simp only [← c75084, ← c75085, ← c75086] at e_1218_4
  have e_1218_5 := arithEq_of_rows h (row := 1218) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_5
  simp only [← c75087, ← c75088, k2, ← c75089] at e_1218_5
  have hr := e_1218_5
  simp only [e_1218_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10908)) := by
  have c75090 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75091 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75092 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75093 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75094 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75095 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_6 := arithEq_of_rows h (row := 1218) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_6
  simp only [← c75090, ← c75091, ← c75092] at e_1218_6
  have e_1218_7 := arithEq_of_rows h (row := 1218) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_7
  simp only [← c75093, ← c75094, k2, ← c75095] at e_1218_7
  have hr := e_1218_7
  simp only [e_1218_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10909)) := by
  have c75096 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75097 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75098 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75099 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75100 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75101 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_8 := arithEq_of_rows h (row := 1218) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_8
  simp only [← c75096, ← c75097, ← c75098] at e_1218_8
  have e_1218_9 := arithEq_of_rows h (row := 1218) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_9
  simp only [← c75099, ← c75100, k2, ← c75101] at e_1218_9
  have hr := e_1218_9
  simp only [e_1218_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10910)) := by
  have c75102 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75103 := (publicBatchWrapper89_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75104 := (publicBatchWrapper89_copies2347 a h).1
  have c75105 := (publicBatchWrapper89_copies2347 a h).2.1
  have c75106 := (publicBatchWrapper89_copies2347 a h).2.2.1
  have c75107 := (publicBatchWrapper89_copies2347 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_10 := arithEq_of_rows h (row := 1218) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_10
  simp only [← c75102, ← c75103, ← c75104] at e_1218_10
  have e_1218_11 := arithEq_of_rows h (row := 1218) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_11
  simp only [← c75105, ← c75106, k2, ← c75107] at e_1218_11
  have hr := e_1218_11
  simp only [e_1218_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10911)) := by
  have c75108 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.1
  have c75109 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.1
  have c75110 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.1
  have c75111 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.1
  have c75112 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.1
  have c75113 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_12 := arithEq_of_rows h (row := 1218) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_12
  simp only [← c75108, ← c75109, ← c75110] at e_1218_12
  have e_1218_13 := arithEq_of_rows h (row := 1218) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_13
  simp only [← c75111, ← c75112, k2, ← c75113] at e_1218_13
  have hr := e_1218_13
  simp only [e_1218_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10912)) := by
  have c75114 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75115 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75116 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75117 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75118 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75119 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_14 := arithEq_of_rows h (row := 1218) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_14
  simp only [← c75114, ← c75115, ← c75116] at e_1218_14
  have e_1218_15 := arithEq_of_rows h (row := 1218) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_15
  simp only [← c75117, ← c75118, k2, ← c75119] at e_1218_15
  have hr := e_1218_15
  simp only [e_1218_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10913)) := by
  have c75120 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75121 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75122 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75123 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75124 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75125 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_16 := arithEq_of_rows h (row := 1218) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_16
  simp only [← c75120, ← c75121, ← c75122] at e_1218_16
  have e_1218_17 := arithEq_of_rows h (row := 1218) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_17
  simp only [← c75123, ← c75124, k2, ← c75125] at e_1218_17
  have hr := e_1218_17
  simp only [e_1218_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1218 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10914)) := by
  have c75126 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75127 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75128 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75129 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75130 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75131 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1218_18 := arithEq_of_rows h (row := 1218) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_18
  simp only [← c75126, ← c75127, ← c75128] at e_1218_18
  have e_1218_19 := arithEq_of_rows h (row := 1218) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_19
  simp only [← c75129, ← c75130, k2, ← c75131] at e_1218_19
  have hr := e_1218_19
  simp only [e_1218_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10915)) := by
  have c75132 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75133 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75134 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75135 := (publicBatchWrapper89_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75136 := (publicBatchWrapper89_copies2348 a h).1
  have c75137 := (publicBatchWrapper89_copies2348 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_0 := arithEq_of_rows h (row := 1219) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_0
  simp only [← c75132, ← c75133, ← c75134] at e_1219_0
  have e_1219_1 := arithEq_of_rows h (row := 1219) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_1
  simp only [← c75135, ← c75136, k2, ← c75137] at e_1219_1
  have hr := e_1219_1
  simp only [e_1219_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10916)) := by
  have c75138 := (publicBatchWrapper89_copies2348 a h).2.2.1
  have c75139 := (publicBatchWrapper89_copies2348 a h).2.2.2.1
  have c75140 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.1
  have c75141 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.1
  have c75142 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.1
  have c75143 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_2 := arithEq_of_rows h (row := 1219) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_2
  simp only [← c75138, ← c75139, ← c75140] at e_1219_2
  have e_1219_3 := arithEq_of_rows h (row := 1219) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_3
  simp only [← c75141, ← c75142, k2, ← c75143] at e_1219_3
  have hr := e_1219_3
  simp only [e_1219_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10917)) := by
  have c75144 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.1
  have c75145 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.1
  have c75146 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75147 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75148 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75149 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_4 := arithEq_of_rows h (row := 1219) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_4
  simp only [← c75144, ← c75145, ← c75146] at e_1219_4
  have e_1219_5 := arithEq_of_rows h (row := 1219) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_5
  simp only [← c75147, ← c75148, k2, ← c75149] at e_1219_5
  have hr := e_1219_5
  simp only [e_1219_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10918)) := by
  have c75150 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75151 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75152 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75153 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75154 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75155 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_6 := arithEq_of_rows h (row := 1219) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_6
  simp only [← c75150, ← c75151, ← c75152] at e_1219_6
  have e_1219_7 := arithEq_of_rows h (row := 1219) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_7
  simp only [← c75153, ← c75154, k2, ← c75155] at e_1219_7
  have hr := e_1219_7
  simp only [e_1219_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10919)) := by
  have c75156 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75157 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75158 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75159 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75160 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75161 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_8 := arithEq_of_rows h (row := 1219) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_8
  simp only [← c75156, ← c75157, ← c75158] at e_1219_8
  have e_1219_9 := arithEq_of_rows h (row := 1219) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_9
  simp only [← c75159, ← c75160, k2, ← c75161] at e_1219_9
  have hr := e_1219_9
  simp only [e_1219_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10920)) := by
  have c75162 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75163 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75164 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75165 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75166 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75167 := (publicBatchWrapper89_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_10 := arithEq_of_rows h (row := 1219) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_10
  simp only [← c75162, ← c75163, ← c75164] at e_1219_10
  have e_1219_11 := arithEq_of_rows h (row := 1219) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_11
  simp only [← c75165, ← c75166, k2, ← c75167] at e_1219_11
  have hr := e_1219_11
  simp only [e_1219_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10921)) := by
  have c75168 := (publicBatchWrapper89_copies2349 a h).1
  have c75169 := (publicBatchWrapper89_copies2349 a h).2.1
  have c75170 := (publicBatchWrapper89_copies2349 a h).2.2.1
  have c75171 := (publicBatchWrapper89_copies2349 a h).2.2.2.1
  have c75172 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.1
  have c75173 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_12 := arithEq_of_rows h (row := 1219) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_12
  simp only [← c75168, ← c75169, ← c75170] at e_1219_12
  have e_1219_13 := arithEq_of_rows h (row := 1219) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_13
  simp only [← c75171, ← c75172, k2, ← c75173] at e_1219_13
  have hr := e_1219_13
  simp only [e_1219_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10922)) := by
  have c75174 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.1
  have c75175 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.1
  have c75176 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.1
  have c75177 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.1
  have c75178 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75179 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_14 := arithEq_of_rows h (row := 1219) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_14
  simp only [← c75174, ← c75175, ← c75176] at e_1219_14
  have e_1219_15 := arithEq_of_rows h (row := 1219) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_15
  simp only [← c75177, ← c75178, k2, ← c75179] at e_1219_15
  have hr := e_1219_15
  simp only [e_1219_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10923)) := by
  have c75180 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75181 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75182 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75183 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75184 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75185 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_16 := arithEq_of_rows h (row := 1219) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_16
  simp only [← c75180, ← c75181, ← c75182] at e_1219_16
  have e_1219_17 := arithEq_of_rows h (row := 1219) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_17
  simp only [← c75183, ← c75184, k2, ← c75185] at e_1219_17
  have hr := e_1219_17
  simp only [e_1219_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1219 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10924)) := by
  have c75186 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75187 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75188 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75189 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75190 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75191 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1219_18 := arithEq_of_rows h (row := 1219) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_18
  simp only [← c75186, ← c75187, ← c75188] at e_1219_18
  have e_1219_19 := arithEq_of_rows h (row := 1219) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_19
  simp only [← c75189, ← c75190, k2, ← c75191] at e_1219_19
  have hr := e_1219_19
  simp only [e_1219_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10925)) := by
  have c75192 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75193 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75194 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75195 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75196 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75197 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_0 := arithEq_of_rows h (row := 1220) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_0
  simp only [← c75192, ← c75193, ← c75194] at e_1220_0
  have e_1220_1 := arithEq_of_rows h (row := 1220) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_1
  simp only [← c75195, ← c75196, k2, ← c75197] at e_1220_1
  have hr := e_1220_1
  simp only [e_1220_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10926)) := by
  have c75198 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75199 := (publicBatchWrapper89_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75200 := (publicBatchWrapper89_copies2350 a h).1
  have c75201 := (publicBatchWrapper89_copies2350 a h).2.1
  have c75202 := (publicBatchWrapper89_copies2350 a h).2.2.1
  have c75203 := (publicBatchWrapper89_copies2350 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_2 := arithEq_of_rows h (row := 1220) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_2
  simp only [← c75198, ← c75199, ← c75200] at e_1220_2
  have e_1220_3 := arithEq_of_rows h (row := 1220) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_3
  simp only [← c75201, ← c75202, k2, ← c75203] at e_1220_3
  have hr := e_1220_3
  simp only [e_1220_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10927)) := by
  have c75204 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.1
  have c75205 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.1
  have c75206 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.1
  have c75207 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.1
  have c75208 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.1
  have c75209 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_4 := arithEq_of_rows h (row := 1220) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_4
  simp only [← c75204, ← c75205, ← c75206] at e_1220_4
  have e_1220_5 := arithEq_of_rows h (row := 1220) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_5
  simp only [← c75207, ← c75208, k2, ← c75209] at e_1220_5
  have hr := e_1220_5
  simp only [e_1220_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10928)) := by
  have c75210 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75211 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75212 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75213 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75214 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75215 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_6 := arithEq_of_rows h (row := 1220) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_6
  simp only [← c75210, ← c75211, ← c75212] at e_1220_6
  have e_1220_7 := arithEq_of_rows h (row := 1220) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_7
  simp only [← c75213, ← c75214, k2, ← c75215] at e_1220_7
  have hr := e_1220_7
  simp only [e_1220_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10929)) := by
  have c75216 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75217 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75218 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75219 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75220 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75221 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_8 := arithEq_of_rows h (row := 1220) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_8
  simp only [← c75216, ← c75217, ← c75218] at e_1220_8
  have e_1220_9 := arithEq_of_rows h (row := 1220) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_9
  simp only [← c75219, ← c75220, k2, ← c75221] at e_1220_9
  have hr := e_1220_9
  simp only [e_1220_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10930)) := by
  have c75222 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75223 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75224 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75225 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75226 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75227 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_10 := arithEq_of_rows h (row := 1220) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_10
  simp only [← c75222, ← c75223, ← c75224] at e_1220_10
  have e_1220_11 := arithEq_of_rows h (row := 1220) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_11
  simp only [← c75225, ← c75226, k2, ← c75227] at e_1220_11
  have hr := e_1220_11
  simp only [e_1220_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10931)) := by
  have c75228 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75229 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75230 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75231 := (publicBatchWrapper89_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75232 := (publicBatchWrapper89_copies2351 a h).1
  have c75233 := (publicBatchWrapper89_copies2351 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_12 := arithEq_of_rows h (row := 1220) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_12
  simp only [← c75228, ← c75229, ← c75230] at e_1220_12
  have e_1220_13 := arithEq_of_rows h (row := 1220) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_13
  simp only [← c75231, ← c75232, k2, ← c75233] at e_1220_13
  have hr := e_1220_13
  simp only [e_1220_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10932)) := by
  have c75234 := (publicBatchWrapper89_copies2351 a h).2.2.1
  have c75235 := (publicBatchWrapper89_copies2351 a h).2.2.2.1
  have c75236 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.1
  have c75237 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.1
  have c75238 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.1
  have c75239 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_14 := arithEq_of_rows h (row := 1220) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_14
  simp only [← c75234, ← c75235, ← c75236] at e_1220_14
  have e_1220_15 := arithEq_of_rows h (row := 1220) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_15
  simp only [← c75237, ← c75238, k2, ← c75239] at e_1220_15
  have hr := e_1220_15
  simp only [e_1220_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10933)) := by
  have c75240 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.1
  have c75241 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.1
  have c75242 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75243 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75244 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75245 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_16 := arithEq_of_rows h (row := 1220) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_16
  simp only [← c75240, ← c75241, ← c75242] at e_1220_16
  have e_1220_17 := arithEq_of_rows h (row := 1220) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_17
  simp only [← c75243, ← c75244, k2, ← c75245] at e_1220_17
  have hr := e_1220_17
  simp only [e_1220_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1220 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10934)) := by
  have c75246 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75247 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75248 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75249 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75250 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75251 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1220_18 := arithEq_of_rows h (row := 1220) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_18
  simp only [← c75246, ← c75247, ← c75248] at e_1220_18
  have e_1220_19 := arithEq_of_rows h (row := 1220) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_19
  simp only [← c75249, ← c75250, k2, ← c75251] at e_1220_19
  have hr := e_1220_19
  simp only [e_1220_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10935)) := by
  have c75252 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75253 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75254 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75255 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75256 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75257 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_0 := arithEq_of_rows h (row := 1221) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_0
  simp only [← c75252, ← c75253, ← c75254] at e_1221_0
  have e_1221_1 := arithEq_of_rows h (row := 1221) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_1
  simp only [← c75255, ← c75256, k2, ← c75257] at e_1221_1
  have hr := e_1221_1
  simp only [e_1221_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10936)) := by
  have c75258 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75259 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75260 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75261 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75262 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75263 := (publicBatchWrapper89_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_2 := arithEq_of_rows h (row := 1221) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_2
  simp only [← c75258, ← c75259, ← c75260] at e_1221_2
  have e_1221_3 := arithEq_of_rows h (row := 1221) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_3
  simp only [← c75261, ← c75262, k2, ← c75263] at e_1221_3
  have hr := e_1221_3
  simp only [e_1221_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10937)) := by
  have c75264 := (publicBatchWrapper89_copies2352 a h).1
  have c75265 := (publicBatchWrapper89_copies2352 a h).2.1
  have c75266 := (publicBatchWrapper89_copies2352 a h).2.2.1
  have c75267 := (publicBatchWrapper89_copies2352 a h).2.2.2.1
  have c75268 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.1
  have c75269 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_4 := arithEq_of_rows h (row := 1221) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_4
  simp only [← c75264, ← c75265, ← c75266] at e_1221_4
  have e_1221_5 := arithEq_of_rows h (row := 1221) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_5
  simp only [← c75267, ← c75268, k2, ← c75269] at e_1221_5
  have hr := e_1221_5
  simp only [e_1221_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10938)) := by
  have c75270 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.1
  have c75271 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.1
  have c75272 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.1
  have c75273 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.1
  have c75274 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75275 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_6 := arithEq_of_rows h (row := 1221) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_6
  simp only [← c75270, ← c75271, ← c75272] at e_1221_6
  have e_1221_7 := arithEq_of_rows h (row := 1221) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_7
  simp only [← c75273, ← c75274, k2, ← c75275] at e_1221_7
  have hr := e_1221_7
  simp only [e_1221_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10939)) := by
  have c75276 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75277 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75278 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75279 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75280 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75281 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_8 := arithEq_of_rows h (row := 1221) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_8
  simp only [← c75276, ← c75277, ← c75278] at e_1221_8
  have e_1221_9 := arithEq_of_rows h (row := 1221) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_9
  simp only [← c75279, ← c75280, k2, ← c75281] at e_1221_9
  have hr := e_1221_9
  simp only [e_1221_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10940)) := by
  have c75282 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75283 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75284 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75285 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75286 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75287 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_10 := arithEq_of_rows h (row := 1221) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_10
  simp only [← c75282, ← c75283, ← c75284] at e_1221_10
  have e_1221_11 := arithEq_of_rows h (row := 1221) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_11
  simp only [← c75285, ← c75286, k2, ← c75287] at e_1221_11
  have hr := e_1221_11
  simp only [e_1221_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10941)) := by
  have c75288 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75289 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75290 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75291 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75292 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75293 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_12 := arithEq_of_rows h (row := 1221) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_12
  simp only [← c75288, ← c75289, ← c75290] at e_1221_12
  have e_1221_13 := arithEq_of_rows h (row := 1221) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_13
  simp only [← c75291, ← c75292, k2, ← c75293] at e_1221_13
  have hr := e_1221_13
  simp only [e_1221_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10942)) := by
  have c75294 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75295 := (publicBatchWrapper89_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75296 := (publicBatchWrapper89_copies2353 a h).1
  have c75297 := (publicBatchWrapper89_copies2353 a h).2.1
  have c75298 := (publicBatchWrapper89_copies2353 a h).2.2.1
  have c75299 := (publicBatchWrapper89_copies2353 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_14 := arithEq_of_rows h (row := 1221) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_14
  simp only [← c75294, ← c75295, ← c75296] at e_1221_14
  have e_1221_15 := arithEq_of_rows h (row := 1221) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_15
  simp only [← c75297, ← c75298, k2, ← c75299] at e_1221_15
  have hr := e_1221_15
  simp only [e_1221_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10943)) := by
  have c75300 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.1
  have c75301 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.1
  have c75302 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.1
  have c75303 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.1
  have c75304 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.1
  have c75305 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_16 := arithEq_of_rows h (row := 1221) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_16
  simp only [← c75300, ← c75301, ← c75302] at e_1221_16
  have e_1221_17 := arithEq_of_rows h (row := 1221) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_17
  simp only [← c75303, ← c75304, k2, ← c75305] at e_1221_17
  have hr := e_1221_17
  simp only [e_1221_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1221 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10944)) := by
  have c75306 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75307 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75308 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75309 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75310 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75311 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1221_18 := arithEq_of_rows h (row := 1221) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_18
  simp only [← c75306, ← c75307, ← c75308] at e_1221_18
  have e_1221_19 := arithEq_of_rows h (row := 1221) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_19
  simp only [← c75309, ← c75310, k2, ← c75311] at e_1221_19
  have hr := e_1221_19
  simp only [e_1221_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10945)) := by
  have c75312 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75313 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75314 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75315 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75316 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75317 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_0 := arithEq_of_rows h (row := 1222) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_0
  simp only [← c75312, ← c75313, ← c75314] at e_1222_0
  have e_1222_1 := arithEq_of_rows h (row := 1222) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_1
  simp only [← c75315, ← c75316, k2, ← c75317] at e_1222_1
  have hr := e_1222_1
  simp only [e_1222_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10946)) := by
  have c75318 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75319 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75320 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75321 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75322 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75323 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_2 := arithEq_of_rows h (row := 1222) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_2
  simp only [← c75318, ← c75319, ← c75320] at e_1222_2
  have e_1222_3 := arithEq_of_rows h (row := 1222) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_3
  simp only [← c75321, ← c75322, k2, ← c75323] at e_1222_3
  have hr := e_1222_3
  simp only [e_1222_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10947)) := by
  have c75324 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75325 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75326 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75327 := (publicBatchWrapper89_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75328 := (publicBatchWrapper89_copies2354 a h).1
  have c75329 := (publicBatchWrapper89_copies2354 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_4 := arithEq_of_rows h (row := 1222) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_4
  simp only [← c75324, ← c75325, ← c75326] at e_1222_4
  have e_1222_5 := arithEq_of_rows h (row := 1222) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_5
  simp only [← c75327, ← c75328, k2, ← c75329] at e_1222_5
  have hr := e_1222_5
  simp only [e_1222_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10948)) := by
  have c75330 := (publicBatchWrapper89_copies2354 a h).2.2.1
  have c75331 := (publicBatchWrapper89_copies2354 a h).2.2.2.1
  have c75332 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.1
  have c75333 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.1
  have c75334 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.1
  have c75335 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_6 := arithEq_of_rows h (row := 1222) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_6
  simp only [← c75330, ← c75331, ← c75332] at e_1222_6
  have e_1222_7 := arithEq_of_rows h (row := 1222) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_7
  simp only [← c75333, ← c75334, k2, ← c75335] at e_1222_7
  have hr := e_1222_7
  simp only [e_1222_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10949)) := by
  have c75336 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.1
  have c75337 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.1
  have c75338 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75339 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75340 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75341 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_8 := arithEq_of_rows h (row := 1222) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_8
  simp only [← c75336, ← c75337, ← c75338] at e_1222_8
  have e_1222_9 := arithEq_of_rows h (row := 1222) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_9
  simp only [← c75339, ← c75340, k2, ← c75341] at e_1222_9
  have hr := e_1222_9
  simp only [e_1222_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10950)) := by
  have c75342 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75343 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75344 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75345 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75346 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75347 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_10 := arithEq_of_rows h (row := 1222) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_10
  simp only [← c75342, ← c75343, ← c75344] at e_1222_10
  have e_1222_11 := arithEq_of_rows h (row := 1222) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_11
  simp only [← c75345, ← c75346, k2, ← c75347] at e_1222_11
  have hr := e_1222_11
  simp only [e_1222_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10951)) := by
  have c75348 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75349 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75350 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75351 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75352 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75353 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_12 := arithEq_of_rows h (row := 1222) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_12
  simp only [← c75348, ← c75349, ← c75350] at e_1222_12
  have e_1222_13 := arithEq_of_rows h (row := 1222) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_13
  simp only [← c75351, ← c75352, k2, ← c75353] at e_1222_13
  have hr := e_1222_13
  simp only [e_1222_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10952)) := by
  have c75354 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75355 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75356 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75357 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75358 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75359 := (publicBatchWrapper89_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_14 := arithEq_of_rows h (row := 1222) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_14
  simp only [← c75354, ← c75355, ← c75356] at e_1222_14
  have e_1222_15 := arithEq_of_rows h (row := 1222) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_15
  simp only [← c75357, ← c75358, k2, ← c75359] at e_1222_15
  have hr := e_1222_15
  simp only [e_1222_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10953)) := by
  have c75360 := (publicBatchWrapper89_copies2355 a h).1
  have c75361 := (publicBatchWrapper89_copies2355 a h).2.1
  have c75362 := (publicBatchWrapper89_copies2355 a h).2.2.1
  have c75363 := (publicBatchWrapper89_copies2355 a h).2.2.2.1
  have c75364 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.1
  have c75365 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_16 := arithEq_of_rows h (row := 1222) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_16
  simp only [← c75360, ← c75361, ← c75362] at e_1222_16
  have e_1222_17 := arithEq_of_rows h (row := 1222) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_17
  simp only [← c75363, ← c75364, k2, ← c75365] at e_1222_17
  have hr := e_1222_17
  simp only [e_1222_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1222 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10954)) := by
  have c75366 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.1
  have c75367 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.1
  have c75368 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.1
  have c75369 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.1
  have c75370 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75371 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1222_18 := arithEq_of_rows h (row := 1222) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_18
  simp only [← c75366, ← c75367, ← c75368] at e_1222_18
  have e_1222_19 := arithEq_of_rows h (row := 1222) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_19
  simp only [← c75369, ← c75370, k2, ← c75371] at e_1222_19
  have hr := e_1222_19
  simp only [e_1222_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10955)) := by
  have c75372 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75373 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75374 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75375 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75376 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75377 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_0 := arithEq_of_rows h (row := 1223) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_0
  simp only [← c75372, ← c75373, ← c75374] at e_1223_0
  have e_1223_1 := arithEq_of_rows h (row := 1223) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_1
  simp only [← c75375, ← c75376, k2, ← c75377] at e_1223_1
  have hr := e_1223_1
  simp only [e_1223_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10956)) := by
  have c75378 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75379 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75380 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75381 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75382 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75383 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_2 := arithEq_of_rows h (row := 1223) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_2
  simp only [← c75378, ← c75379, ← c75380] at e_1223_2
  have e_1223_3 := arithEq_of_rows h (row := 1223) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_3
  simp only [← c75381, ← c75382, k2, ← c75383] at e_1223_3
  have hr := e_1223_3
  simp only [e_1223_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10957)) := by
  have c75384 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75385 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75386 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75387 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75388 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75389 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_4 := arithEq_of_rows h (row := 1223) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_4
  simp only [← c75384, ← c75385, ← c75386] at e_1223_4
  have e_1223_5 := arithEq_of_rows h (row := 1223) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_5
  simp only [← c75387, ← c75388, k2, ← c75389] at e_1223_5
  have hr := e_1223_5
  simp only [e_1223_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10958)) := by
  have c75390 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75391 := (publicBatchWrapper89_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75392 := (publicBatchWrapper89_copies2356 a h).1
  have c75393 := (publicBatchWrapper89_copies2356 a h).2.1
  have c75394 := (publicBatchWrapper89_copies2356 a h).2.2.1
  have c75395 := (publicBatchWrapper89_copies2356 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_6 := arithEq_of_rows h (row := 1223) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_6
  simp only [← c75390, ← c75391, ← c75392] at e_1223_6
  have e_1223_7 := arithEq_of_rows h (row := 1223) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_7
  simp only [← c75393, ← c75394, k2, ← c75395] at e_1223_7
  have hr := e_1223_7
  simp only [e_1223_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10959)) := by
  have c75396 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.1
  have c75397 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.1
  have c75398 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.1
  have c75399 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.1
  have c75400 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.1
  have c75401 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_8 := arithEq_of_rows h (row := 1223) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_8
  simp only [← c75396, ← c75397, ← c75398] at e_1223_8
  have e_1223_9 := arithEq_of_rows h (row := 1223) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_9
  simp only [← c75399, ← c75400, k2, ← c75401] at e_1223_9
  have hr := e_1223_9
  simp only [e_1223_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10960)) := by
  have c75402 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75403 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75404 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75405 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75406 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75407 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_10 := arithEq_of_rows h (row := 1223) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_10
  simp only [← c75402, ← c75403, ← c75404] at e_1223_10
  have e_1223_11 := arithEq_of_rows h (row := 1223) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_11
  simp only [← c75405, ← c75406, k2, ← c75407] at e_1223_11
  have hr := e_1223_11
  simp only [e_1223_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10961)) := by
  have c75408 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75409 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75410 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75411 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75412 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75413 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_12 := arithEq_of_rows h (row := 1223) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_12
  simp only [← c75408, ← c75409, ← c75410] at e_1223_12
  have e_1223_13 := arithEq_of_rows h (row := 1223) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_13
  simp only [← c75411, ← c75412, k2, ← c75413] at e_1223_13
  have hr := e_1223_13
  simp only [e_1223_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10962)) := by
  have c75414 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75415 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75416 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75417 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75418 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75419 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_14 := arithEq_of_rows h (row := 1223) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_14
  simp only [← c75414, ← c75415, ← c75416] at e_1223_14
  have e_1223_15 := arithEq_of_rows h (row := 1223) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_15
  simp only [← c75417, ← c75418, k2, ← c75419] at e_1223_15
  have hr := e_1223_15
  simp only [e_1223_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10963)) := by
  have c75420 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75421 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75422 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75423 := (publicBatchWrapper89_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75424 := (publicBatchWrapper89_copies2357 a h).1
  have c75425 := (publicBatchWrapper89_copies2357 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_16 := arithEq_of_rows h (row := 1223) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_16
  simp only [← c75420, ← c75421, ← c75422] at e_1223_16
  have e_1223_17 := arithEq_of_rows h (row := 1223) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_17
  simp only [← c75423, ← c75424, k2, ← c75425] at e_1223_17
  have hr := e_1223_17
  simp only [e_1223_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1223 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10964)) := by
  have c75426 := (publicBatchWrapper89_copies2357 a h).2.2.1
  have c75427 := (publicBatchWrapper89_copies2357 a h).2.2.2.1
  have c75428 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.1
  have c75429 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.1
  have c75430 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.1
  have c75431 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1223_18 := arithEq_of_rows h (row := 1223) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_18
  simp only [← c75426, ← c75427, ← c75428] at e_1223_18
  have e_1223_19 := arithEq_of_rows h (row := 1223) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_19
  simp only [← c75429, ← c75430, k2, ← c75431] at e_1223_19
  have hr := e_1223_19
  simp only [e_1223_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10965)) := by
  have c75432 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.1
  have c75433 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.1
  have c75434 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75435 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75436 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75437 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_0 := arithEq_of_rows h (row := 1224) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_0
  simp only [← c75432, ← c75433, ← c75434] at e_1224_0
  have e_1224_1 := arithEq_of_rows h (row := 1224) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_1
  simp only [← c75435, ← c75436, k2, ← c75437] at e_1224_1
  have hr := e_1224_1
  simp only [e_1224_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10966)) := by
  have c75438 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75439 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75440 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75441 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75442 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75443 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_2 := arithEq_of_rows h (row := 1224) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_2
  simp only [← c75438, ← c75439, ← c75440] at e_1224_2
  have e_1224_3 := arithEq_of_rows h (row := 1224) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_3
  simp only [← c75441, ← c75442, k2, ← c75443] at e_1224_3
  have hr := e_1224_3
  simp only [e_1224_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10967)) := by
  have c75444 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75445 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75446 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75447 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75448 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75449 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_4 := arithEq_of_rows h (row := 1224) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_4
  simp only [← c75444, ← c75445, ← c75446] at e_1224_4
  have e_1224_5 := arithEq_of_rows h (row := 1224) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_5
  simp only [← c75447, ← c75448, k2, ← c75449] at e_1224_5
  have hr := e_1224_5
  simp only [e_1224_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10968)) := by
  have c75450 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75451 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75452 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75453 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75454 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75455 := (publicBatchWrapper89_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_6 := arithEq_of_rows h (row := 1224) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_6
  simp only [← c75450, ← c75451, ← c75452] at e_1224_6
  have e_1224_7 := arithEq_of_rows h (row := 1224) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_7
  simp only [← c75453, ← c75454, k2, ← c75455] at e_1224_7
  have hr := e_1224_7
  simp only [e_1224_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10969)) := by
  have c75456 := (publicBatchWrapper89_copies2358 a h).1
  have c75457 := (publicBatchWrapper89_copies2358 a h).2.1
  have c75458 := (publicBatchWrapper89_copies2358 a h).2.2.1
  have c75459 := (publicBatchWrapper89_copies2358 a h).2.2.2.1
  have c75460 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.1
  have c75461 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_8 := arithEq_of_rows h (row := 1224) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_8
  simp only [← c75456, ← c75457, ← c75458] at e_1224_8
  have e_1224_9 := arithEq_of_rows h (row := 1224) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_9
  simp only [← c75459, ← c75460, k2, ← c75461] at e_1224_9
  have hr := e_1224_9
  simp only [e_1224_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11010)) := by
  have c75462 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.1
  have c75463 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.1
  have c75464 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.1
  have c75465 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.1
  have c75466 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75467 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_10 := arithEq_of_rows h (row := 1224) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_10
  simp only [← c75462, ← c75463, ← c75464] at e_1224_10
  have e_1224_11 := arithEq_of_rows h (row := 1224) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_11
  simp only [← c75465, ← c75466, k2, ← c75467] at e_1224_11
  have hr := e_1224_11
  simp only [e_1224_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11011)) := by
  have c75468 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75469 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75470 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75471 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75472 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75473 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_12 := arithEq_of_rows h (row := 1224) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_12
  simp only [← c75468, ← c75469, ← c75470] at e_1224_12
  have e_1224_13 := arithEq_of_rows h (row := 1224) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_13
  simp only [← c75471, ← c75472, k2, ← c75473] at e_1224_13
  have hr := e_1224_13
  simp only [e_1224_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11012)) := by
  have c75474 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75475 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75476 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75477 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75478 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75479 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_14 := arithEq_of_rows h (row := 1224) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_14
  simp only [← c75474, ← c75475, ← c75476] at e_1224_14
  have e_1224_15 := arithEq_of_rows h (row := 1224) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_15
  simp only [← c75477, ← c75478, k2, ← c75479] at e_1224_15
  have hr := e_1224_15
  simp only [e_1224_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11013)) := by
  have c75480 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75481 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75482 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75483 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75484 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75485 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_16 := arithEq_of_rows h (row := 1224) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_16
  simp only [← c75480, ← c75481, ← c75482] at e_1224_16
  have e_1224_17 := arithEq_of_rows h (row := 1224) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_17
  simp only [← c75483, ← c75484, k2, ← c75485] at e_1224_17
  have hr := e_1224_17
  simp only [e_1224_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1224 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11014)) := by
  have c75486 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75487 := (publicBatchWrapper89_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75488 := (publicBatchWrapper89_copies2359 a h).1
  have c75489 := (publicBatchWrapper89_copies2359 a h).2.1
  have c75490 := (publicBatchWrapper89_copies2359 a h).2.2.1
  have c75491 := (publicBatchWrapper89_copies2359 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1224_18 := arithEq_of_rows h (row := 1224) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_18
  simp only [← c75486, ← c75487, ← c75488] at e_1224_18
  have e_1224_19 := arithEq_of_rows h (row := 1224) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_19
  simp only [← c75489, ← c75490, k2, ← c75491] at e_1224_19
  have hr := e_1224_19
  simp only [e_1224_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11015)) := by
  have c75492 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.1
  have c75493 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.1
  have c75494 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.1
  have c75495 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.1
  have c75496 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.1
  have c75497 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_0 := arithEq_of_rows h (row := 1225) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_0
  simp only [← c75492, ← c75493, ← c75494] at e_1225_0
  have e_1225_1 := arithEq_of_rows h (row := 1225) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_1
  simp only [← c75495, ← c75496, k2, ← c75497] at e_1225_1
  have hr := e_1225_1
  simp only [e_1225_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11016)) := by
  have c75498 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75499 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75500 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75501 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75502 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75503 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_2 := arithEq_of_rows h (row := 1225) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_2
  simp only [← c75498, ← c75499, ← c75500] at e_1225_2
  have e_1225_3 := arithEq_of_rows h (row := 1225) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_3
  simp only [← c75501, ← c75502, k2, ← c75503] at e_1225_3
  have hr := e_1225_3
  simp only [e_1225_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11017)) := by
  have c75504 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75505 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75506 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75507 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75508 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75509 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_4 := arithEq_of_rows h (row := 1225) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_4
  simp only [← c75504, ← c75505, ← c75506] at e_1225_4
  have e_1225_5 := arithEq_of_rows h (row := 1225) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_5
  simp only [← c75507, ← c75508, k2, ← c75509] at e_1225_5
  have hr := e_1225_5
  simp only [e_1225_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11018)) := by
  have c75510 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75511 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75512 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75513 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75514 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75515 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_6 := arithEq_of_rows h (row := 1225) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_6
  simp only [← c75510, ← c75511, ← c75512] at e_1225_6
  have e_1225_7 := arithEq_of_rows h (row := 1225) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_7
  simp only [← c75513, ← c75514, k2, ← c75515] at e_1225_7
  have hr := e_1225_7
  simp only [e_1225_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11019)) := by
  have c75516 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75517 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75518 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75519 := (publicBatchWrapper89_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75520 := (publicBatchWrapper89_copies2360 a h).1
  have c75521 := (publicBatchWrapper89_copies2360 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_8 := arithEq_of_rows h (row := 1225) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_8
  simp only [← c75516, ← c75517, ← c75518] at e_1225_8
  have e_1225_9 := arithEq_of_rows h (row := 1225) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_9
  simp only [← c75519, ← c75520, k2, ← c75521] at e_1225_9
  have hr := e_1225_9
  simp only [e_1225_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11020)) := by
  have c75522 := (publicBatchWrapper89_copies2360 a h).2.2.1
  have c75523 := (publicBatchWrapper89_copies2360 a h).2.2.2.1
  have c75524 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.1
  have c75525 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.1
  have c75526 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.1
  have c75527 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_10 := arithEq_of_rows h (row := 1225) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_10
  simp only [← c75522, ← c75523, ← c75524] at e_1225_10
  have e_1225_11 := arithEq_of_rows h (row := 1225) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_11
  simp only [← c75525, ← c75526, k2, ← c75527] at e_1225_11
  have hr := e_1225_11
  simp only [e_1225_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11021)) := by
  have c75528 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.1
  have c75529 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.1
  have c75530 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75531 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75532 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75533 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_12 := arithEq_of_rows h (row := 1225) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_12
  simp only [← c75528, ← c75529, ← c75530] at e_1225_12
  have e_1225_13 := arithEq_of_rows h (row := 1225) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_13
  simp only [← c75531, ← c75532, k2, ← c75533] at e_1225_13
  have hr := e_1225_13
  simp only [e_1225_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11022)) := by
  have c75534 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75535 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75536 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75537 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75538 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75539 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_14 := arithEq_of_rows h (row := 1225) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_14
  simp only [← c75534, ← c75535, ← c75536] at e_1225_14
  have e_1225_15 := arithEq_of_rows h (row := 1225) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_15
  simp only [← c75537, ← c75538, k2, ← c75539] at e_1225_15
  have hr := e_1225_15
  simp only [e_1225_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11023)) := by
  have c75540 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75541 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75542 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75543 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75544 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75545 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_16 := arithEq_of_rows h (row := 1225) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_16
  simp only [← c75540, ← c75541, ← c75542] at e_1225_16
  have e_1225_17 := arithEq_of_rows h (row := 1225) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_17
  simp only [← c75543, ← c75544, k2, ← c75545] at e_1225_17
  have hr := e_1225_17
  simp only [e_1225_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1225 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11024)) := by
  have c75546 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75547 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75548 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75549 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75550 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75551 := (publicBatchWrapper89_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1225_18 := arithEq_of_rows h (row := 1225) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_18
  simp only [← c75546, ← c75547, ← c75548] at e_1225_18
  have e_1225_19 := arithEq_of_rows h (row := 1225) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_19
  simp only [← c75549, ← c75550, k2, ← c75551] at e_1225_19
  have hr := e_1225_19
  simp only [e_1225_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11025)) := by
  have c75552 := (publicBatchWrapper89_copies2361 a h).1
  have c75553 := (publicBatchWrapper89_copies2361 a h).2.1
  have c75554 := (publicBatchWrapper89_copies2361 a h).2.2.1
  have c75555 := (publicBatchWrapper89_copies2361 a h).2.2.2.1
  have c75556 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.1
  have c75557 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_0 := arithEq_of_rows h (row := 1226) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_0
  simp only [← c75552, ← c75553, ← c75554] at e_1226_0
  have e_1226_1 := arithEq_of_rows h (row := 1226) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_1
  simp only [← c75555, ← c75556, k2, ← c75557] at e_1226_1
  have hr := e_1226_1
  simp only [e_1226_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11026)) := by
  have c75558 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.1
  have c75559 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.1
  have c75560 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.1
  have c75561 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.1
  have c75562 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75563 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_2 := arithEq_of_rows h (row := 1226) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_2
  simp only [← c75558, ← c75559, ← c75560] at e_1226_2
  have e_1226_3 := arithEq_of_rows h (row := 1226) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_3
  simp only [← c75561, ← c75562, k2, ← c75563] at e_1226_3
  have hr := e_1226_3
  simp only [e_1226_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11027)) := by
  have c75564 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75565 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75566 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75567 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75568 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75569 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_4 := arithEq_of_rows h (row := 1226) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_4
  simp only [← c75564, ← c75565, ← c75566] at e_1226_4
  have e_1226_5 := arithEq_of_rows h (row := 1226) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_5
  simp only [← c75567, ← c75568, k2, ← c75569] at e_1226_5
  have hr := e_1226_5
  simp only [e_1226_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11028)) := by
  have c75570 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75571 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75572 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75573 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75574 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75575 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_6 := arithEq_of_rows h (row := 1226) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_6
  simp only [← c75570, ← c75571, ← c75572] at e_1226_6
  have e_1226_7 := arithEq_of_rows h (row := 1226) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_7
  simp only [← c75573, ← c75574, k2, ← c75575] at e_1226_7
  have hr := e_1226_7
  simp only [e_1226_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11029)) := by
  have c75576 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75577 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75578 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75579 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75580 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75581 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_8 := arithEq_of_rows h (row := 1226) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_8
  simp only [← c75576, ← c75577, ← c75578] at e_1226_8
  have e_1226_9 := arithEq_of_rows h (row := 1226) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_9
  simp only [← c75579, ← c75580, k2, ← c75581] at e_1226_9
  have hr := e_1226_9
  simp only [e_1226_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11030)) := by
  have c75582 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75583 := (publicBatchWrapper89_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75584 := (publicBatchWrapper89_copies2362 a h).1
  have c75585 := (publicBatchWrapper89_copies2362 a h).2.1
  have c75586 := (publicBatchWrapper89_copies2362 a h).2.2.1
  have c75587 := (publicBatchWrapper89_copies2362 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_10 := arithEq_of_rows h (row := 1226) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_10
  simp only [← c75582, ← c75583, ← c75584] at e_1226_10
  have e_1226_11 := arithEq_of_rows h (row := 1226) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_11
  simp only [← c75585, ← c75586, k2, ← c75587] at e_1226_11
  have hr := e_1226_11
  simp only [e_1226_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11031)) := by
  have c75588 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.1
  have c75589 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.1
  have c75590 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.1
  have c75591 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.1
  have c75592 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.1
  have c75593 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_12 := arithEq_of_rows h (row := 1226) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_12
  simp only [← c75588, ← c75589, ← c75590] at e_1226_12
  have e_1226_13 := arithEq_of_rows h (row := 1226) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_13
  simp only [← c75591, ← c75592, k2, ← c75593] at e_1226_13
  have hr := e_1226_13
  simp only [e_1226_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11032)) := by
  have c75594 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75595 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75596 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75597 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75598 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75599 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_14 := arithEq_of_rows h (row := 1226) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_14
  simp only [← c75594, ← c75595, ← c75596] at e_1226_14
  have e_1226_15 := arithEq_of_rows h (row := 1226) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_15
  simp only [← c75597, ← c75598, k2, ← c75599] at e_1226_15
  have hr := e_1226_15
  simp only [e_1226_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11033)) := by
  have c75600 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75601 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75602 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75603 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75604 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75605 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_16 := arithEq_of_rows h (row := 1226) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_16
  simp only [← c75600, ← c75601, ← c75602] at e_1226_16
  have e_1226_17 := arithEq_of_rows h (row := 1226) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_17
  simp only [← c75603, ← c75604, k2, ← c75605] at e_1226_17
  have hr := e_1226_17
  simp only [e_1226_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1226 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11034)) := by
  have c75606 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75607 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75608 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75609 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75610 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75611 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1226_18 := arithEq_of_rows h (row := 1226) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_18
  simp only [← c75606, ← c75607, ← c75608] at e_1226_18
  have e_1226_19 := arithEq_of_rows h (row := 1226) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_19
  simp only [← c75609, ← c75610, k2, ← c75611] at e_1226_19
  have hr := e_1226_19
  simp only [e_1226_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11035)) := by
  have c75612 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75613 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75614 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75615 := (publicBatchWrapper89_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75616 := (publicBatchWrapper89_copies2363 a h).1
  have c75617 := (publicBatchWrapper89_copies2363 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_0 := arithEq_of_rows h (row := 1227) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_0
  simp only [← c75612, ← c75613, ← c75614] at e_1227_0
  have e_1227_1 := arithEq_of_rows h (row := 1227) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_1
  simp only [← c75615, ← c75616, k2, ← c75617] at e_1227_1
  have hr := e_1227_1
  simp only [e_1227_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11036)) := by
  have c75618 := (publicBatchWrapper89_copies2363 a h).2.2.1
  have c75619 := (publicBatchWrapper89_copies2363 a h).2.2.2.1
  have c75620 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.1
  have c75621 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.1
  have c75622 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.1
  have c75623 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_2 := arithEq_of_rows h (row := 1227) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_2
  simp only [← c75618, ← c75619, ← c75620] at e_1227_2
  have e_1227_3 := arithEq_of_rows h (row := 1227) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_3
  simp only [← c75621, ← c75622, k2, ← c75623] at e_1227_3
  have hr := e_1227_3
  simp only [e_1227_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11037)) := by
  have c75624 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.1
  have c75625 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.1
  have c75626 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75627 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75628 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75629 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_4 := arithEq_of_rows h (row := 1227) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_4
  simp only [← c75624, ← c75625, ← c75626] at e_1227_4
  have e_1227_5 := arithEq_of_rows h (row := 1227) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_5
  simp only [← c75627, ← c75628, k2, ← c75629] at e_1227_5
  have hr := e_1227_5
  simp only [e_1227_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11038)) := by
  have c75630 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75631 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75632 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75633 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75634 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75635 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_6 := arithEq_of_rows h (row := 1227) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_6
  simp only [← c75630, ← c75631, ← c75632] at e_1227_6
  have e_1227_7 := arithEq_of_rows h (row := 1227) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_7
  simp only [← c75633, ← c75634, k2, ← c75635] at e_1227_7
  have hr := e_1227_7
  simp only [e_1227_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11039)) := by
  have c75636 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75637 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75638 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75639 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75640 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75641 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_8 := arithEq_of_rows h (row := 1227) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_8
  simp only [← c75636, ← c75637, ← c75638] at e_1227_8
  have e_1227_9 := arithEq_of_rows h (row := 1227) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_9
  simp only [← c75639, ← c75640, k2, ← c75641] at e_1227_9
  have hr := e_1227_9
  simp only [e_1227_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11040)) := by
  have c75642 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75643 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75644 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75645 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75646 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75647 := (publicBatchWrapper89_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_10 := arithEq_of_rows h (row := 1227) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_10
  simp only [← c75642, ← c75643, ← c75644] at e_1227_10
  have e_1227_11 := arithEq_of_rows h (row := 1227) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_11
  simp only [← c75645, ← c75646, k2, ← c75647] at e_1227_11
  have hr := e_1227_11
  simp only [e_1227_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11041)) := by
  have c75648 := (publicBatchWrapper89_copies2364 a h).1
  have c75649 := (publicBatchWrapper89_copies2364 a h).2.1
  have c75650 := (publicBatchWrapper89_copies2364 a h).2.2.1
  have c75651 := (publicBatchWrapper89_copies2364 a h).2.2.2.1
  have c75652 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.1
  have c75653 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_12 := arithEq_of_rows h (row := 1227) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_12
  simp only [← c75648, ← c75649, ← c75650] at e_1227_12
  have e_1227_13 := arithEq_of_rows h (row := 1227) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_13
  simp only [← c75651, ← c75652, k2, ← c75653] at e_1227_13
  have hr := e_1227_13
  simp only [e_1227_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11042)) := by
  have c75654 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.1
  have c75655 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.1
  have c75656 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.1
  have c75657 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.1
  have c75658 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75659 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_14 := arithEq_of_rows h (row := 1227) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_14
  simp only [← c75654, ← c75655, ← c75656] at e_1227_14
  have e_1227_15 := arithEq_of_rows h (row := 1227) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_15
  simp only [← c75657, ← c75658, k2, ← c75659] at e_1227_15
  have hr := e_1227_15
  simp only [e_1227_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11043)) := by
  have c75660 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75661 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75662 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75663 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75664 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75665 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_16 := arithEq_of_rows h (row := 1227) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_16
  simp only [← c75660, ← c75661, ← c75662] at e_1227_16
  have e_1227_17 := arithEq_of_rows h (row := 1227) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_17
  simp only [← c75663, ← c75664, k2, ← c75665] at e_1227_17
  have hr := e_1227_17
  simp only [e_1227_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1227 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11044)) := by
  have c75666 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75667 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75668 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75669 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75670 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75671 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1227_18 := arithEq_of_rows h (row := 1227) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_18
  simp only [← c75666, ← c75667, ← c75668] at e_1227_18
  have e_1227_19 := arithEq_of_rows h (row := 1227) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_19
  simp only [← c75669, ← c75670, k2, ← c75671] at e_1227_19
  have hr := e_1227_19
  simp only [e_1227_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11045)) := by
  have c75672 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75673 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75674 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75675 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75676 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75677 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_0 := arithEq_of_rows h (row := 1228) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_0
  simp only [← c75672, ← c75673, ← c75674] at e_1228_0
  have e_1228_1 := arithEq_of_rows h (row := 1228) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_1
  simp only [← c75675, ← c75676, k2, ← c75677] at e_1228_1
  have hr := e_1228_1
  simp only [e_1228_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11046)) := by
  have c75678 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75679 := (publicBatchWrapper89_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75680 := (publicBatchWrapper89_copies2365 a h).1
  have c75681 := (publicBatchWrapper89_copies2365 a h).2.1
  have c75682 := (publicBatchWrapper89_copies2365 a h).2.2.1
  have c75683 := (publicBatchWrapper89_copies2365 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_2 := arithEq_of_rows h (row := 1228) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_2
  simp only [← c75678, ← c75679, ← c75680] at e_1228_2
  have e_1228_3 := arithEq_of_rows h (row := 1228) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_3
  simp only [← c75681, ← c75682, k2, ← c75683] at e_1228_3
  have hr := e_1228_3
  simp only [e_1228_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11047)) := by
  have c75684 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.1
  have c75685 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.1
  have c75686 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.1
  have c75687 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.1
  have c75688 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.1
  have c75689 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_4 := arithEq_of_rows h (row := 1228) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_4
  simp only [← c75684, ← c75685, ← c75686] at e_1228_4
  have e_1228_5 := arithEq_of_rows h (row := 1228) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_5
  simp only [← c75687, ← c75688, k2, ← c75689] at e_1228_5
  have hr := e_1228_5
  simp only [e_1228_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11048)) := by
  have c75690 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75691 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75692 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75693 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75694 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75695 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_6 := arithEq_of_rows h (row := 1228) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_6
  simp only [← c75690, ← c75691, ← c75692] at e_1228_6
  have e_1228_7 := arithEq_of_rows h (row := 1228) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_7
  simp only [← c75693, ← c75694, k2, ← c75695] at e_1228_7
  have hr := e_1228_7
  simp only [e_1228_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11049)) := by
  have c75696 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75697 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75698 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75699 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75700 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75701 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_8 := arithEq_of_rows h (row := 1228) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_8
  simp only [← c75696, ← c75697, ← c75698] at e_1228_8
  have e_1228_9 := arithEq_of_rows h (row := 1228) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_9
  simp only [← c75699, ← c75700, k2, ← c75701] at e_1228_9
  have hr := e_1228_9
  simp only [e_1228_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11050)) := by
  have c75702 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75703 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75704 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75705 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75706 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75707 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_10 := arithEq_of_rows h (row := 1228) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_10
  simp only [← c75702, ← c75703, ← c75704] at e_1228_10
  have e_1228_11 := arithEq_of_rows h (row := 1228) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_11
  simp only [← c75705, ← c75706, k2, ← c75707] at e_1228_11
  have hr := e_1228_11
  simp only [e_1228_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11051)) := by
  have c75708 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75709 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75710 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75711 := (publicBatchWrapper89_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75712 := (publicBatchWrapper89_copies2366 a h).1
  have c75713 := (publicBatchWrapper89_copies2366 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_12 := arithEq_of_rows h (row := 1228) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_12
  simp only [← c75708, ← c75709, ← c75710] at e_1228_12
  have e_1228_13 := arithEq_of_rows h (row := 1228) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_13
  simp only [← c75711, ← c75712, k2, ← c75713] at e_1228_13
  have hr := e_1228_13
  simp only [e_1228_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11052)) := by
  have c75714 := (publicBatchWrapper89_copies2366 a h).2.2.1
  have c75715 := (publicBatchWrapper89_copies2366 a h).2.2.2.1
  have c75716 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.1
  have c75717 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.1
  have c75718 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.1
  have c75719 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_14 := arithEq_of_rows h (row := 1228) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_14
  simp only [← c75714, ← c75715, ← c75716] at e_1228_14
  have e_1228_15 := arithEq_of_rows h (row := 1228) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_15
  simp only [← c75717, ← c75718, k2, ← c75719] at e_1228_15
  have hr := e_1228_15
  simp only [e_1228_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11053)) := by
  have c75720 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.1
  have c75721 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.1
  have c75722 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75723 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75724 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75725 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_16 := arithEq_of_rows h (row := 1228) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_16
  simp only [← c75720, ← c75721, ← c75722] at e_1228_16
  have e_1228_17 := arithEq_of_rows h (row := 1228) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_17
  simp only [← c75723, ← c75724, k2, ← c75725] at e_1228_17
  have hr := e_1228_17
  simp only [e_1228_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1228 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11054)) := by
  have c75726 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75727 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75728 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75729 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75730 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75731 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1228_18 := arithEq_of_rows h (row := 1228) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_18
  simp only [← c75726, ← c75727, ← c75728] at e_1228_18
  have e_1228_19 := arithEq_of_rows h (row := 1228) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_19
  simp only [← c75729, ← c75730, k2, ← c75731] at e_1228_19
  have hr := e_1228_19
  simp only [e_1228_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11055)) := by
  have c75732 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75733 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75734 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75735 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75736 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75737 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_0 := arithEq_of_rows h (row := 1229) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_0
  simp only [← c75732, ← c75733, ← c75734] at e_1229_0
  have e_1229_1 := arithEq_of_rows h (row := 1229) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_1
  simp only [← c75735, ← c75736, k2, ← c75737] at e_1229_1
  have hr := e_1229_1
  simp only [e_1229_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11056)) := by
  have c75738 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75739 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75740 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75741 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75742 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75743 := (publicBatchWrapper89_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_2 := arithEq_of_rows h (row := 1229) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_2
  simp only [← c75738, ← c75739, ← c75740] at e_1229_2
  have e_1229_3 := arithEq_of_rows h (row := 1229) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_3
  simp only [← c75741, ← c75742, k2, ← c75743] at e_1229_3
  have hr := e_1229_3
  simp only [e_1229_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11057)) := by
  have c75744 := (publicBatchWrapper89_copies2367 a h).1
  have c75745 := (publicBatchWrapper89_copies2367 a h).2.1
  have c75746 := (publicBatchWrapper89_copies2367 a h).2.2.1
  have c75747 := (publicBatchWrapper89_copies2367 a h).2.2.2.1
  have c75748 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.1
  have c75749 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_4 := arithEq_of_rows h (row := 1229) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_4
  simp only [← c75744, ← c75745, ← c75746] at e_1229_4
  have e_1229_5 := arithEq_of_rows h (row := 1229) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_5
  simp only [← c75747, ← c75748, k2, ← c75749] at e_1229_5
  have hr := e_1229_5
  simp only [e_1229_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11058)) := by
  have c75750 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.1
  have c75751 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.1
  have c75752 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.1
  have c75753 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.1
  have c75754 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75755 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_6 := arithEq_of_rows h (row := 1229) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_6
  simp only [← c75750, ← c75751, ← c75752] at e_1229_6
  have e_1229_7 := arithEq_of_rows h (row := 1229) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_7
  simp only [← c75753, ← c75754, k2, ← c75755] at e_1229_7
  have hr := e_1229_7
  simp only [e_1229_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11059)) := by
  have c75756 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75757 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75758 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75759 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75760 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75761 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_8 := arithEq_of_rows h (row := 1229) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_8
  simp only [← c75756, ← c75757, ← c75758] at e_1229_8
  have e_1229_9 := arithEq_of_rows h (row := 1229) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_9
  simp only [← c75759, ← c75760, k2, ← c75761] at e_1229_9
  have hr := e_1229_9
  simp only [e_1229_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11060)) := by
  have c75762 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75763 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75764 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75765 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75766 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75767 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_10 := arithEq_of_rows h (row := 1229) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_10
  simp only [← c75762, ← c75763, ← c75764] at e_1229_10
  have e_1229_11 := arithEq_of_rows h (row := 1229) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_11
  simp only [← c75765, ← c75766, k2, ← c75767] at e_1229_11
  have hr := e_1229_11
  simp only [e_1229_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11061)) := by
  have c75768 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75769 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75770 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75771 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75772 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75773 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_12 := arithEq_of_rows h (row := 1229) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_12
  simp only [← c75768, ← c75769, ← c75770] at e_1229_12
  have e_1229_13 := arithEq_of_rows h (row := 1229) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_13
  simp only [← c75771, ← c75772, k2, ← c75773] at e_1229_13
  have hr := e_1229_13
  simp only [e_1229_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11062)) := by
  have c75774 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75775 := (publicBatchWrapper89_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75776 := (publicBatchWrapper89_copies2368 a h).1
  have c75777 := (publicBatchWrapper89_copies2368 a h).2.1
  have c75778 := (publicBatchWrapper89_copies2368 a h).2.2.1
  have c75779 := (publicBatchWrapper89_copies2368 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_14 := arithEq_of_rows h (row := 1229) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_14
  simp only [← c75774, ← c75775, ← c75776] at e_1229_14
  have e_1229_15 := arithEq_of_rows h (row := 1229) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_15
  simp only [← c75777, ← c75778, k2, ← c75779] at e_1229_15
  have hr := e_1229_15
  simp only [e_1229_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11063)) := by
  have c75780 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.1
  have c75781 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.1
  have c75782 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.1
  have c75783 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.1
  have c75784 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.1
  have c75785 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_16 := arithEq_of_rows h (row := 1229) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_16
  simp only [← c75780, ← c75781, ← c75782] at e_1229_16
  have e_1229_17 := arithEq_of_rows h (row := 1229) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_17
  simp only [← c75783, ← c75784, k2, ← c75785] at e_1229_17
  have hr := e_1229_17
  simp only [e_1229_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1229 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11064)) := by
  have c75786 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75787 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75788 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75789 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75790 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75791 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1229_18 := arithEq_of_rows h (row := 1229) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_18
  simp only [← c75786, ← c75787, ← c75788] at e_1229_18
  have e_1229_19 := arithEq_of_rows h (row := 1229) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_19
  simp only [← c75789, ← c75790, k2, ← c75791] at e_1229_19
  have hr := e_1229_19
  simp only [e_1229_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11065)) := by
  have c75792 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75793 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75794 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75795 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75796 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75797 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_0 := arithEq_of_rows h (row := 1230) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_0
  simp only [← c75792, ← c75793, ← c75794] at e_1230_0
  have e_1230_1 := arithEq_of_rows h (row := 1230) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_1
  simp only [← c75795, ← c75796, k2, ← c75797] at e_1230_1
  have hr := e_1230_1
  simp only [e_1230_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11066)) := by
  have c75798 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75799 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75800 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75801 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75802 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75803 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_2 := arithEq_of_rows h (row := 1230) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_2
  simp only [← c75798, ← c75799, ← c75800] at e_1230_2
  have e_1230_3 := arithEq_of_rows h (row := 1230) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_3
  simp only [← c75801, ← c75802, k2, ← c75803] at e_1230_3
  have hr := e_1230_3
  simp only [e_1230_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11067)) := by
  have c75804 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75805 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75806 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75807 := (publicBatchWrapper89_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75808 := (publicBatchWrapper89_copies2369 a h).1
  have c75809 := (publicBatchWrapper89_copies2369 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_4 := arithEq_of_rows h (row := 1230) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_4
  simp only [← c75804, ← c75805, ← c75806] at e_1230_4
  have e_1230_5 := arithEq_of_rows h (row := 1230) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_5
  simp only [← c75807, ← c75808, k2, ← c75809] at e_1230_5
  have hr := e_1230_5
  simp only [e_1230_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11068)) := by
  have c75810 := (publicBatchWrapper89_copies2369 a h).2.2.1
  have c75811 := (publicBatchWrapper89_copies2369 a h).2.2.2.1
  have c75812 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.1
  have c75813 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.1
  have c75814 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.1
  have c75815 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_6 := arithEq_of_rows h (row := 1230) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_6
  simp only [← c75810, ← c75811, ← c75812] at e_1230_6
  have e_1230_7 := arithEq_of_rows h (row := 1230) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_7
  simp only [← c75813, ← c75814, k2, ← c75815] at e_1230_7
  have hr := e_1230_7
  simp only [e_1230_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11069)) := by
  have c75816 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.1
  have c75817 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.1
  have c75818 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75819 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75820 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75821 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_8 := arithEq_of_rows h (row := 1230) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_8
  simp only [← c75816, ← c75817, ← c75818] at e_1230_8
  have e_1230_9 := arithEq_of_rows h (row := 1230) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_9
  simp only [← c75819, ← c75820, k2, ← c75821] at e_1230_9
  have hr := e_1230_9
  simp only [e_1230_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11070)) := by
  have c75822 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75823 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75824 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75825 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75826 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75827 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_10 := arithEq_of_rows h (row := 1230) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_10
  simp only [← c75822, ← c75823, ← c75824] at e_1230_10
  have e_1230_11 := arithEq_of_rows h (row := 1230) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_11
  simp only [← c75825, ← c75826, k2, ← c75827] at e_1230_11
  have hr := e_1230_11
  simp only [e_1230_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11071)) := by
  have c75828 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75829 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75830 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75831 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75832 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75833 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_12 := arithEq_of_rows h (row := 1230) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_12
  simp only [← c75828, ← c75829, ← c75830] at e_1230_12
  have e_1230_13 := arithEq_of_rows h (row := 1230) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_13
  simp only [← c75831, ← c75832, k2, ← c75833] at e_1230_13
  have hr := e_1230_13
  simp only [e_1230_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11072)) := by
  have c75834 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75835 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75836 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75837 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75838 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75839 := (publicBatchWrapper89_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_14 := arithEq_of_rows h (row := 1230) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_14
  simp only [← c75834, ← c75835, ← c75836] at e_1230_14
  have e_1230_15 := arithEq_of_rows h (row := 1230) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_15
  simp only [← c75837, ← c75838, k2, ← c75839] at e_1230_15
  have hr := e_1230_15
  simp only [e_1230_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11073)) := by
  have c75840 := (publicBatchWrapper89_copies2370 a h).1
  have c75841 := (publicBatchWrapper89_copies2370 a h).2.1
  have c75842 := (publicBatchWrapper89_copies2370 a h).2.2.1
  have c75843 := (publicBatchWrapper89_copies2370 a h).2.2.2.1
  have c75844 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.1
  have c75845 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_16 := arithEq_of_rows h (row := 1230) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_16
  simp only [← c75840, ← c75841, ← c75842] at e_1230_16
  have e_1230_17 := arithEq_of_rows h (row := 1230) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_17
  simp only [← c75843, ← c75844, k2, ← c75845] at e_1230_17
  have hr := e_1230_17
  simp only [e_1230_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1230 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11074)) := by
  have c75846 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.1
  have c75847 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.1
  have c75848 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.1
  have c75849 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.1
  have c75850 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75851 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1230_18 := arithEq_of_rows h (row := 1230) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_18
  simp only [← c75846, ← c75847, ← c75848] at e_1230_18
  have e_1230_19 := arithEq_of_rows h (row := 1230) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_19
  simp only [← c75849, ← c75850, k2, ← c75851] at e_1230_19
  have hr := e_1230_19
  simp only [e_1230_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11075)) := by
  have c75852 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75853 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75854 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75855 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75856 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75857 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_0 := arithEq_of_rows h (row := 1231) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_0
  simp only [← c75852, ← c75853, ← c75854] at e_1231_0
  have e_1231_1 := arithEq_of_rows h (row := 1231) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_1
  simp only [← c75855, ← c75856, k2, ← c75857] at e_1231_1
  have hr := e_1231_1
  simp only [e_1231_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11076)) := by
  have c75858 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75859 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75860 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75861 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75862 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75863 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_2 := arithEq_of_rows h (row := 1231) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_2
  simp only [← c75858, ← c75859, ← c75860] at e_1231_2
  have e_1231_3 := arithEq_of_rows h (row := 1231) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_3
  simp only [← c75861, ← c75862, k2, ← c75863] at e_1231_3
  have hr := e_1231_3
  simp only [e_1231_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11077)) := by
  have c75864 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75865 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75866 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75867 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75868 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75869 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_4 := arithEq_of_rows h (row := 1231) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_4
  simp only [← c75864, ← c75865, ← c75866] at e_1231_4
  have e_1231_5 := arithEq_of_rows h (row := 1231) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_5
  simp only [← c75867, ← c75868, k2, ← c75869] at e_1231_5
  have hr := e_1231_5
  simp only [e_1231_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11078)) := by
  have c75870 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75871 := (publicBatchWrapper89_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75872 := (publicBatchWrapper89_copies2371 a h).1
  have c75873 := (publicBatchWrapper89_copies2371 a h).2.1
  have c75874 := (publicBatchWrapper89_copies2371 a h).2.2.1
  have c75875 := (publicBatchWrapper89_copies2371 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_6 := arithEq_of_rows h (row := 1231) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_6
  simp only [← c75870, ← c75871, ← c75872] at e_1231_6
  have e_1231_7 := arithEq_of_rows h (row := 1231) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_7
  simp only [← c75873, ← c75874, k2, ← c75875] at e_1231_7
  have hr := e_1231_7
  simp only [e_1231_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11079)) := by
  have c75876 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.1
  have c75877 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.1
  have c75878 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.1
  have c75879 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.1
  have c75880 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.1
  have c75881 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_8 := arithEq_of_rows h (row := 1231) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_8
  simp only [← c75876, ← c75877, ← c75878] at e_1231_8
  have e_1231_9 := arithEq_of_rows h (row := 1231) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_9
  simp only [← c75879, ← c75880, k2, ← c75881] at e_1231_9
  have hr := e_1231_9
  simp only [e_1231_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11080)) := by
  have c75882 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75883 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75884 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75885 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75886 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75887 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_10 := arithEq_of_rows h (row := 1231) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_10
  simp only [← c75882, ← c75883, ← c75884] at e_1231_10
  have e_1231_11 := arithEq_of_rows h (row := 1231) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_11
  simp only [← c75885, ← c75886, k2, ← c75887] at e_1231_11
  have hr := e_1231_11
  simp only [e_1231_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11081)) := by
  have c75888 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75889 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75890 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75891 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75892 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75893 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_12 := arithEq_of_rows h (row := 1231) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_12
  simp only [← c75888, ← c75889, ← c75890] at e_1231_12
  have e_1231_13 := arithEq_of_rows h (row := 1231) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_13
  simp only [← c75891, ← c75892, k2, ← c75893] at e_1231_13
  have hr := e_1231_13
  simp only [e_1231_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11082)) := by
  have c75894 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75895 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75896 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75897 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75898 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75899 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_14 := arithEq_of_rows h (row := 1231) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_14
  simp only [← c75894, ← c75895, ← c75896] at e_1231_14
  have e_1231_15 := arithEq_of_rows h (row := 1231) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_15
  simp only [← c75897, ← c75898, k2, ← c75899] at e_1231_15
  have hr := e_1231_15
  simp only [e_1231_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11083)) := by
  have c75900 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75901 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75902 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75903 := (publicBatchWrapper89_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75904 := (publicBatchWrapper89_copies2372 a h).1
  have c75905 := (publicBatchWrapper89_copies2372 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_16 := arithEq_of_rows h (row := 1231) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_16
  simp only [← c75900, ← c75901, ← c75902] at e_1231_16
  have e_1231_17 := arithEq_of_rows h (row := 1231) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_17
  simp only [← c75903, ← c75904, k2, ← c75905] at e_1231_17
  have hr := e_1231_17
  simp only [e_1231_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1231 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11084)) := by
  have c75906 := (publicBatchWrapper89_copies2372 a h).2.2.1
  have c75907 := (publicBatchWrapper89_copies2372 a h).2.2.2.1
  have c75908 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.1
  have c75909 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.1
  have c75910 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.1
  have c75911 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1231_18 := arithEq_of_rows h (row := 1231) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_18
  simp only [← c75906, ← c75907, ← c75908] at e_1231_18
  have e_1231_19 := arithEq_of_rows h (row := 1231) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_19
  simp only [← c75909, ← c75910, k2, ← c75911] at e_1231_19
  have hr := e_1231_19
  simp only [e_1231_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11085)) := by
  have c75912 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.1
  have c75913 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.1
  have c75914 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75915 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75916 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75917 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_0 := arithEq_of_rows h (row := 1232) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_0
  simp only [← c75912, ← c75913, ← c75914] at e_1232_0
  have e_1232_1 := arithEq_of_rows h (row := 1232) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_1
  simp only [← c75915, ← c75916, k2, ← c75917] at e_1232_1
  have hr := e_1232_1
  simp only [e_1232_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11086)) := by
  have c75918 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75919 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75920 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75921 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75922 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75923 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_2 := arithEq_of_rows h (row := 1232) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_2
  simp only [← c75918, ← c75919, ← c75920] at e_1232_2
  have e_1232_3 := arithEq_of_rows h (row := 1232) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_3
  simp only [← c75921, ← c75922, k2, ← c75923] at e_1232_3
  have hr := e_1232_3
  simp only [e_1232_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11087)) := by
  have c75924 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75925 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75926 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75927 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75928 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75929 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_4 := arithEq_of_rows h (row := 1232) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_4
  simp only [← c75924, ← c75925, ← c75926] at e_1232_4
  have e_1232_5 := arithEq_of_rows h (row := 1232) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_5
  simp only [← c75927, ← c75928, k2, ← c75929] at e_1232_5
  have hr := e_1232_5
  simp only [e_1232_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11088)) := by
  have c75930 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75931 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75932 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75933 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75934 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75935 := (publicBatchWrapper89_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_6 := arithEq_of_rows h (row := 1232) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_6
  simp only [← c75930, ← c75931, ← c75932] at e_1232_6
  have e_1232_7 := arithEq_of_rows h (row := 1232) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_7
  simp only [← c75933, ← c75934, k2, ← c75935] at e_1232_7
  have hr := e_1232_7
  simp only [e_1232_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11089)) := by
  have c75936 := (publicBatchWrapper89_copies2373 a h).1
  have c75937 := (publicBatchWrapper89_copies2373 a h).2.1
  have c75938 := (publicBatchWrapper89_copies2373 a h).2.2.1
  have c75939 := (publicBatchWrapper89_copies2373 a h).2.2.2.1
  have c75940 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.1
  have c75941 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_8 := arithEq_of_rows h (row := 1232) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_8
  simp only [← c75936, ← c75937, ← c75938] at e_1232_8
  have e_1232_9 := arithEq_of_rows h (row := 1232) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_9
  simp only [← c75939, ← c75940, k2, ← c75941] at e_1232_9
  have hr := e_1232_9
  simp only [e_1232_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11090)) := by
  have c75942 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.1
  have c75943 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.1
  have c75944 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.1
  have c75945 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.1
  have c75946 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75947 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_10 := arithEq_of_rows h (row := 1232) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_10
  simp only [← c75942, ← c75943, ← c75944] at e_1232_10
  have e_1232_11 := arithEq_of_rows h (row := 1232) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_11
  simp only [← c75945, ← c75946, k2, ← c75947] at e_1232_11
  have hr := e_1232_11
  simp only [e_1232_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11091)) := by
  have c75948 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75949 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75950 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75951 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75952 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75953 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_12 := arithEq_of_rows h (row := 1232) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_12
  simp only [← c75948, ← c75949, ← c75950] at e_1232_12
  have e_1232_13 := arithEq_of_rows h (row := 1232) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_13
  simp only [← c75951, ← c75952, k2, ← c75953] at e_1232_13
  have hr := e_1232_13
  simp only [e_1232_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11092)) := by
  have c75954 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75955 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75956 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75957 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75958 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75959 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_14 := arithEq_of_rows h (row := 1232) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_14
  simp only [← c75954, ← c75955, ← c75956] at e_1232_14
  have e_1232_15 := arithEq_of_rows h (row := 1232) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_15
  simp only [← c75957, ← c75958, k2, ← c75959] at e_1232_15
  have hr := e_1232_15
  simp only [e_1232_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11093)) := by
  have c75960 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75961 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75962 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75963 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75964 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75965 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_16 := arithEq_of_rows h (row := 1232) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_16
  simp only [← c75960, ← c75961, ← c75962] at e_1232_16
  have e_1232_17 := arithEq_of_rows h (row := 1232) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_17
  simp only [← c75963, ← c75964, k2, ← c75965] at e_1232_17
  have hr := e_1232_17
  simp only [e_1232_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1232 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11094)) := by
  have c75966 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75967 := (publicBatchWrapper89_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75968 := (publicBatchWrapper89_copies2374 a h).1
  have c75969 := (publicBatchWrapper89_copies2374 a h).2.1
  have c75970 := (publicBatchWrapper89_copies2374 a h).2.2.1
  have c75971 := (publicBatchWrapper89_copies2374 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1232_18 := arithEq_of_rows h (row := 1232) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_18
  simp only [← c75966, ← c75967, ← c75968] at e_1232_18
  have e_1232_19 := arithEq_of_rows h (row := 1232) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_19
  simp only [← c75969, ← c75970, k2, ← c75971] at e_1232_19
  have hr := e_1232_19
  simp only [e_1232_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11095)) := by
  have c75972 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.1
  have c75973 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.1
  have c75974 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.1
  have c75975 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.1
  have c75976 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.1
  have c75977 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_0 := arithEq_of_rows h (row := 1233) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_0
  simp only [← c75972, ← c75973, ← c75974] at e_1233_0
  have e_1233_1 := arithEq_of_rows h (row := 1233) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_1
  simp only [← c75975, ← c75976, k2, ← c75977] at e_1233_1
  have hr := e_1233_1
  simp only [e_1233_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11096)) := by
  have c75978 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75979 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75980 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75981 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75982 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75983 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_2 := arithEq_of_rows h (row := 1233) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_2
  simp only [← c75978, ← c75979, ← c75980] at e_1233_2
  have e_1233_3 := arithEq_of_rows h (row := 1233) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_3
  simp only [← c75981, ← c75982, k2, ← c75983] at e_1233_3
  have hr := e_1233_3
  simp only [e_1233_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11097)) := by
  have c75984 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75985 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75986 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75987 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75988 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75989 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_4 := arithEq_of_rows h (row := 1233) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_4
  simp only [← c75984, ← c75985, ← c75986] at e_1233_4
  have e_1233_5 := arithEq_of_rows h (row := 1233) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_5
  simp only [← c75987, ← c75988, k2, ← c75989] at e_1233_5
  have hr := e_1233_5
  simp only [e_1233_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11098)) := by
  have c75990 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75991 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75992 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75993 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75994 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75995 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_6 := arithEq_of_rows h (row := 1233) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_6
  simp only [← c75990, ← c75991, ← c75992] at e_1233_6
  have e_1233_7 := arithEq_of_rows h (row := 1233) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_7
  simp only [← c75993, ← c75994, k2, ← c75995] at e_1233_7
  have hr := e_1233_7
  simp only [e_1233_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11099)) := by
  have c75996 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75997 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75998 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75999 := (publicBatchWrapper89_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76000 := (publicBatchWrapper89_copies2375 a h).1
  have c76001 := (publicBatchWrapper89_copies2375 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_8 := arithEq_of_rows h (row := 1233) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_8
  simp only [← c75996, ← c75997, ← c75998] at e_1233_8
  have e_1233_9 := arithEq_of_rows h (row := 1233) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_9
  simp only [← c75999, ← c76000, k2, ← c76001] at e_1233_9
  have hr := e_1233_9
  simp only [e_1233_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11100)) := by
  have c76002 := (publicBatchWrapper89_copies2375 a h).2.2.1
  have c76003 := (publicBatchWrapper89_copies2375 a h).2.2.2.1
  have c76004 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.1
  have c76005 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.1
  have c76006 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.1
  have c76007 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_10 := arithEq_of_rows h (row := 1233) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_10
  simp only [← c76002, ← c76003, ← c76004] at e_1233_10
  have e_1233_11 := arithEq_of_rows h (row := 1233) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_11
  simp only [← c76005, ← c76006, k2, ← c76007] at e_1233_11
  have hr := e_1233_11
  simp only [e_1233_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11101)) := by
  have c76008 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.1
  have c76009 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.1
  have c76010 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76011 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76012 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76013 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_12 := arithEq_of_rows h (row := 1233) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_12
  simp only [← c76008, ← c76009, ← c76010] at e_1233_12
  have e_1233_13 := arithEq_of_rows h (row := 1233) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_13
  simp only [← c76011, ← c76012, k2, ← c76013] at e_1233_13
  have hr := e_1233_13
  simp only [e_1233_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11102)) := by
  have c76014 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76015 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76016 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76017 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76018 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76019 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_14 := arithEq_of_rows h (row := 1233) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_14
  simp only [← c76014, ← c76015, ← c76016] at e_1233_14
  have e_1233_15 := arithEq_of_rows h (row := 1233) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_15
  simp only [← c76017, ← c76018, k2, ← c76019] at e_1233_15
  have hr := e_1233_15
  simp only [e_1233_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11103)) := by
  have c76020 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76021 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76022 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76023 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76024 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76025 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_16 := arithEq_of_rows h (row := 1233) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_16
  simp only [← c76020, ← c76021, ← c76022] at e_1233_16
  have e_1233_17 := arithEq_of_rows h (row := 1233) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_17
  simp only [← c76023, ← c76024, k2, ← c76025] at e_1233_17
  have hr := e_1233_17
  simp only [e_1233_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1233 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11104)) := by
  have c76026 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76027 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76028 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76029 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76030 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76031 := (publicBatchWrapper89_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1233_18 := arithEq_of_rows h (row := 1233) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_18
  simp only [← c76026, ← c76027, ← c76028] at e_1233_18
  have e_1233_19 := arithEq_of_rows h (row := 1233) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_19
  simp only [← c76029, ← c76030, k2, ← c76031] at e_1233_19
  have hr := e_1233_19
  simp only [e_1233_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11105)) := by
  have c76032 := (publicBatchWrapper89_copies2376 a h).1
  have c76033 := (publicBatchWrapper89_copies2376 a h).2.1
  have c76034 := (publicBatchWrapper89_copies2376 a h).2.2.1
  have c76035 := (publicBatchWrapper89_copies2376 a h).2.2.2.1
  have c76036 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.1
  have c76037 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_0 := arithEq_of_rows h (row := 1234) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_0
  simp only [← c76032, ← c76033, ← c76034] at e_1234_0
  have e_1234_1 := arithEq_of_rows h (row := 1234) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_1
  simp only [← c76035, ← c76036, k2, ← c76037] at e_1234_1
  have hr := e_1234_1
  simp only [e_1234_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11106)) := by
  have c76038 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.1
  have c76039 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.1
  have c76040 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.1
  have c76041 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.1
  have c76042 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76043 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_2 := arithEq_of_rows h (row := 1234) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_2
  simp only [← c76038, ← c76039, ← c76040] at e_1234_2
  have e_1234_3 := arithEq_of_rows h (row := 1234) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_3
  simp only [← c76041, ← c76042, k2, ← c76043] at e_1234_3
  have hr := e_1234_3
  simp only [e_1234_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11107)) := by
  have c76044 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76045 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76046 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76047 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76048 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76049 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_4 := arithEq_of_rows h (row := 1234) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_4
  simp only [← c76044, ← c76045, ← c76046] at e_1234_4
  have e_1234_5 := arithEq_of_rows h (row := 1234) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_5
  simp only [← c76047, ← c76048, k2, ← c76049] at e_1234_5
  have hr := e_1234_5
  simp only [e_1234_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11108)) := by
  have c76050 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76051 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76052 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76053 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76054 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76055 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_6 := arithEq_of_rows h (row := 1234) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_6
  simp only [← c76050, ← c76051, ← c76052] at e_1234_6
  have e_1234_7 := arithEq_of_rows h (row := 1234) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_7
  simp only [← c76053, ← c76054, k2, ← c76055] at e_1234_7
  have hr := e_1234_7
  simp only [e_1234_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11109)) := by
  have c76056 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76057 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76058 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76059 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76060 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76061 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_8 := arithEq_of_rows h (row := 1234) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_8
  simp only [← c76056, ← c76057, ← c76058] at e_1234_8
  have e_1234_9 := arithEq_of_rows h (row := 1234) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_9
  simp only [← c76059, ← c76060, k2, ← c76061] at e_1234_9
  have hr := e_1234_9
  simp only [e_1234_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11110)) := by
  have c76062 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76063 := (publicBatchWrapper89_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76064 := (publicBatchWrapper89_copies2377 a h).1
  have c76065 := (publicBatchWrapper89_copies2377 a h).2.1
  have c76066 := (publicBatchWrapper89_copies2377 a h).2.2.1
  have c76067 := (publicBatchWrapper89_copies2377 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_10 := arithEq_of_rows h (row := 1234) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_10
  simp only [← c76062, ← c76063, ← c76064] at e_1234_10
  have e_1234_11 := arithEq_of_rows h (row := 1234) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_11
  simp only [← c76065, ← c76066, k2, ← c76067] at e_1234_11
  have hr := e_1234_11
  simp only [e_1234_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11111)) := by
  have c76068 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.1
  have c76069 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.1
  have c76070 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.1
  have c76071 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.1
  have c76072 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.1
  have c76073 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_12 := arithEq_of_rows h (row := 1234) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_12
  simp only [← c76068, ← c76069, ← c76070] at e_1234_12
  have e_1234_13 := arithEq_of_rows h (row := 1234) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_13
  simp only [← c76071, ← c76072, k2, ← c76073] at e_1234_13
  have hr := e_1234_13
  simp only [e_1234_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11112)) := by
  have c76074 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76075 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76076 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76077 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76078 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76079 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_14 := arithEq_of_rows h (row := 1234) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_14
  simp only [← c76074, ← c76075, ← c76076] at e_1234_14
  have e_1234_15 := arithEq_of_rows h (row := 1234) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_15
  simp only [← c76077, ← c76078, k2, ← c76079] at e_1234_15
  have hr := e_1234_15
  simp only [e_1234_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11113)) := by
  have c76080 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76081 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76082 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76083 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76084 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76085 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_16 := arithEq_of_rows h (row := 1234) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_16
  simp only [← c76080, ← c76081, ← c76082] at e_1234_16
  have e_1234_17 := arithEq_of_rows h (row := 1234) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_17
  simp only [← c76083, ← c76084, k2, ← c76085] at e_1234_17
  have hr := e_1234_17
  simp only [e_1234_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1234 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11114)) := by
  have c76086 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76087 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76088 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76089 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76090 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76091 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1234_18 := arithEq_of_rows h (row := 1234) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_18
  simp only [← c76086, ← c76087, ← c76088] at e_1234_18
  have e_1234_19 := arithEq_of_rows h (row := 1234) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_19
  simp only [← c76089, ← c76090, k2, ← c76091] at e_1234_19
  have hr := e_1234_19
  simp only [e_1234_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11115)) := by
  have c76092 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76093 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76094 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76095 := (publicBatchWrapper89_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76096 := (publicBatchWrapper89_copies2378 a h).1
  have c76097 := (publicBatchWrapper89_copies2378 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_0 := arithEq_of_rows h (row := 1235) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_0
  simp only [← c76092, ← c76093, ← c76094] at e_1235_0
  have e_1235_1 := arithEq_of_rows h (row := 1235) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_1
  simp only [← c76095, ← c76096, k2, ← c76097] at e_1235_1
  have hr := e_1235_1
  simp only [e_1235_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11116)) := by
  have c76098 := (publicBatchWrapper89_copies2378 a h).2.2.1
  have c76099 := (publicBatchWrapper89_copies2378 a h).2.2.2.1
  have c76100 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.1
  have c76101 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.1
  have c76102 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.1
  have c76103 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_2 := arithEq_of_rows h (row := 1235) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_2
  simp only [← c76098, ← c76099, ← c76100] at e_1235_2
  have e_1235_3 := arithEq_of_rows h (row := 1235) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_3
  simp only [← c76101, ← c76102, k2, ← c76103] at e_1235_3
  have hr := e_1235_3
  simp only [e_1235_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11117)) := by
  have c76104 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.1
  have c76105 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.1
  have c76106 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76107 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76108 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76109 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_4 := arithEq_of_rows h (row := 1235) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_4
  simp only [← c76104, ← c76105, ← c76106] at e_1235_4
  have e_1235_5 := arithEq_of_rows h (row := 1235) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_5
  simp only [← c76107, ← c76108, k2, ← c76109] at e_1235_5
  have hr := e_1235_5
  simp only [e_1235_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11118)) := by
  have c76110 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76111 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76112 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76113 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76114 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76115 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_6 := arithEq_of_rows h (row := 1235) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_6
  simp only [← c76110, ← c76111, ← c76112] at e_1235_6
  have e_1235_7 := arithEq_of_rows h (row := 1235) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_7
  simp only [← c76113, ← c76114, k2, ← c76115] at e_1235_7
  have hr := e_1235_7
  simp only [e_1235_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11119)) := by
  have c76116 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76117 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76118 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76119 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76120 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76121 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_8 := arithEq_of_rows h (row := 1235) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_8
  simp only [← c76116, ← c76117, ← c76118] at e_1235_8
  have e_1235_9 := arithEq_of_rows h (row := 1235) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_9
  simp only [← c76119, ← c76120, k2, ← c76121] at e_1235_9
  have hr := e_1235_9
  simp only [e_1235_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11120)) := by
  have c76122 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76123 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76124 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76125 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76126 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76127 := (publicBatchWrapper89_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_10 := arithEq_of_rows h (row := 1235) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_10
  simp only [← c76122, ← c76123, ← c76124] at e_1235_10
  have e_1235_11 := arithEq_of_rows h (row := 1235) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_11
  simp only [← c76125, ← c76126, k2, ← c76127] at e_1235_11
  have hr := e_1235_11
  simp only [e_1235_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11121)) := by
  have c76128 := (publicBatchWrapper89_copies2379 a h).1
  have c76129 := (publicBatchWrapper89_copies2379 a h).2.1
  have c76130 := (publicBatchWrapper89_copies2379 a h).2.2.1
  have c76131 := (publicBatchWrapper89_copies2379 a h).2.2.2.1
  have c76132 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.1
  have c76133 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_12 := arithEq_of_rows h (row := 1235) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_12
  simp only [← c76128, ← c76129, ← c76130] at e_1235_12
  have e_1235_13 := arithEq_of_rows h (row := 1235) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_13
  simp only [← c76131, ← c76132, k2, ← c76133] at e_1235_13
  have hr := e_1235_13
  simp only [e_1235_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11122)) := by
  have c76134 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.1
  have c76135 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.1
  have c76136 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.1
  have c76137 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.1
  have c76138 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76139 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_14 := arithEq_of_rows h (row := 1235) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_14
  simp only [← c76134, ← c76135, ← c76136] at e_1235_14
  have e_1235_15 := arithEq_of_rows h (row := 1235) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_15
  simp only [← c76137, ← c76138, k2, ← c76139] at e_1235_15
  have hr := e_1235_15
  simp only [e_1235_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11123)) := by
  have c76140 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76141 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76142 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76143 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76144 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76145 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_16 := arithEq_of_rows h (row := 1235) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_16
  simp only [← c76140, ← c76141, ← c76142] at e_1235_16
  have e_1235_17 := arithEq_of_rows h (row := 1235) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_17
  simp only [← c76143, ← c76144, k2, ← c76145] at e_1235_17
  have hr := e_1235_17
  simp only [e_1235_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1235 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11124)) := by
  have c76146 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76147 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76148 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76149 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76150 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76151 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1235_18 := arithEq_of_rows h (row := 1235) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_18
  simp only [← c76146, ← c76147, ← c76148] at e_1235_18
  have e_1235_19 := arithEq_of_rows h (row := 1235) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_19
  simp only [← c76149, ← c76150, k2, ← c76151] at e_1235_19
  have hr := e_1235_19
  simp only [e_1235_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11125)) := by
  have c76152 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76153 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76154 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76155 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76156 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76157 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_0 := arithEq_of_rows h (row := 1236) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_0
  simp only [← c76152, ← c76153, ← c76154] at e_1236_0
  have e_1236_1 := arithEq_of_rows h (row := 1236) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_1
  simp only [← c76155, ← c76156, k2, ← c76157] at e_1236_1
  have hr := e_1236_1
  simp only [e_1236_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11126)) := by
  have c76158 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76159 := (publicBatchWrapper89_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76160 := (publicBatchWrapper89_copies2380 a h).1
  have c76161 := (publicBatchWrapper89_copies2380 a h).2.1
  have c76162 := (publicBatchWrapper89_copies2380 a h).2.2.1
  have c76163 := (publicBatchWrapper89_copies2380 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_2 := arithEq_of_rows h (row := 1236) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_2
  simp only [← c76158, ← c76159, ← c76160] at e_1236_2
  have e_1236_3 := arithEq_of_rows h (row := 1236) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_3
  simp only [← c76161, ← c76162, k2, ← c76163] at e_1236_3
  have hr := e_1236_3
  simp only [e_1236_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11127)) := by
  have c76164 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.1
  have c76165 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.1
  have c76166 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.1
  have c76167 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.1
  have c76168 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.1
  have c76169 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_4 := arithEq_of_rows h (row := 1236) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_4
  simp only [← c76164, ← c76165, ← c76166] at e_1236_4
  have e_1236_5 := arithEq_of_rows h (row := 1236) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_5
  simp only [← c76167, ← c76168, k2, ← c76169] at e_1236_5
  have hr := e_1236_5
  simp only [e_1236_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11128)) := by
  have c76170 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76171 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76172 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76173 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76174 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76175 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_6 := arithEq_of_rows h (row := 1236) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_6
  simp only [← c76170, ← c76171, ← c76172] at e_1236_6
  have e_1236_7 := arithEq_of_rows h (row := 1236) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_7
  simp only [← c76173, ← c76174, k2, ← c76175] at e_1236_7
  have hr := e_1236_7
  simp only [e_1236_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11129)) := by
  have c76176 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76177 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76178 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76179 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76180 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76181 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_8 := arithEq_of_rows h (row := 1236) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_8
  simp only [← c76176, ← c76177, ← c76178] at e_1236_8
  have e_1236_9 := arithEq_of_rows h (row := 1236) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_9
  simp only [← c76179, ← c76180, k2, ← c76181] at e_1236_9
  have hr := e_1236_9
  simp only [e_1236_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11130)) := by
  have c76182 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76183 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76184 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76185 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76186 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76187 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_10 := arithEq_of_rows h (row := 1236) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_10
  simp only [← c76182, ← c76183, ← c76184] at e_1236_10
  have e_1236_11 := arithEq_of_rows h (row := 1236) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_11
  simp only [← c76185, ← c76186, k2, ← c76187] at e_1236_11
  have hr := e_1236_11
  simp only [e_1236_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11131)) := by
  have c76188 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76189 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76190 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76191 := (publicBatchWrapper89_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76192 := (publicBatchWrapper89_copies2381 a h).1
  have c76193 := (publicBatchWrapper89_copies2381 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_12 := arithEq_of_rows h (row := 1236) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_12
  simp only [← c76188, ← c76189, ← c76190] at e_1236_12
  have e_1236_13 := arithEq_of_rows h (row := 1236) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_13
  simp only [← c76191, ← c76192, k2, ← c76193] at e_1236_13
  have hr := e_1236_13
  simp only [e_1236_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11132)) := by
  have c76194 := (publicBatchWrapper89_copies2381 a h).2.2.1
  have c76195 := (publicBatchWrapper89_copies2381 a h).2.2.2.1
  have c76196 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.1
  have c76197 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.1
  have c76198 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.1
  have c76199 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_14 := arithEq_of_rows h (row := 1236) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_14
  simp only [← c76194, ← c76195, ← c76196] at e_1236_14
  have e_1236_15 := arithEq_of_rows h (row := 1236) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_15
  simp only [← c76197, ← c76198, k2, ← c76199] at e_1236_15
  have hr := e_1236_15
  simp only [e_1236_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11133)) := by
  have c76200 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.1
  have c76201 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.1
  have c76202 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76203 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76204 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76205 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_16 := arithEq_of_rows h (row := 1236) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_16
  simp only [← c76200, ← c76201, ← c76202] at e_1236_16
  have e_1236_17 := arithEq_of_rows h (row := 1236) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_17
  simp only [← c76203, ← c76204, k2, ← c76205] at e_1236_17
  have hr := e_1236_17
  simp only [e_1236_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1236 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11134)) := by
  have c76206 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76207 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76208 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76209 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76210 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76211 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1236_18 := arithEq_of_rows h (row := 1236) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_18
  simp only [← c76206, ← c76207, ← c76208] at e_1236_18
  have e_1236_19 := arithEq_of_rows h (row := 1236) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_19
  simp only [← c76209, ← c76210, k2, ← c76211] at e_1236_19
  have hr := e_1236_19
  simp only [e_1236_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11135)) := by
  have c76212 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76213 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76214 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76215 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76216 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76217 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_0 := arithEq_of_rows h (row := 1237) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_0
  simp only [← c76212, ← c76213, ← c76214] at e_1237_0
  have e_1237_1 := arithEq_of_rows h (row := 1237) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_1
  simp only [← c76215, ← c76216, k2, ← c76217] at e_1237_1
  have hr := e_1237_1
  simp only [e_1237_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11136)) := by
  have c76218 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76219 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76220 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76221 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76222 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76223 := (publicBatchWrapper89_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_2 := arithEq_of_rows h (row := 1237) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_2
  simp only [← c76218, ← c76219, ← c76220] at e_1237_2
  have e_1237_3 := arithEq_of_rows h (row := 1237) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_3
  simp only [← c76221, ← c76222, k2, ← c76223] at e_1237_3
  have hr := e_1237_3
  simp only [e_1237_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11137)) := by
  have c76224 := (publicBatchWrapper89_copies2382 a h).1
  have c76225 := (publicBatchWrapper89_copies2382 a h).2.1
  have c76226 := (publicBatchWrapper89_copies2382 a h).2.2.1
  have c76227 := (publicBatchWrapper89_copies2382 a h).2.2.2.1
  have c76228 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.1
  have c76229 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_4 := arithEq_of_rows h (row := 1237) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_4
  simp only [← c76224, ← c76225, ← c76226] at e_1237_4
  have e_1237_5 := arithEq_of_rows h (row := 1237) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_5
  simp only [← c76227, ← c76228, k2, ← c76229] at e_1237_5
  have hr := e_1237_5
  simp only [e_1237_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11138)) := by
  have c76230 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.1
  have c76231 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.1
  have c76232 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.1
  have c76233 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.1
  have c76234 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76235 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_6 := arithEq_of_rows h (row := 1237) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_6
  simp only [← c76230, ← c76231, ← c76232] at e_1237_6
  have e_1237_7 := arithEq_of_rows h (row := 1237) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_7
  simp only [← c76233, ← c76234, k2, ← c76235] at e_1237_7
  have hr := e_1237_7
  simp only [e_1237_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11139)) := by
  have c76236 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76237 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76238 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76239 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76240 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76241 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_8 := arithEq_of_rows h (row := 1237) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_8
  simp only [← c76236, ← c76237, ← c76238] at e_1237_8
  have e_1237_9 := arithEq_of_rows h (row := 1237) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_9
  simp only [← c76239, ← c76240, k2, ← c76241] at e_1237_9
  have hr := e_1237_9
  simp only [e_1237_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11140)) := by
  have c76242 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76243 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76244 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76245 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76246 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76247 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_10 := arithEq_of_rows h (row := 1237) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_10
  simp only [← c76242, ← c76243, ← c76244] at e_1237_10
  have e_1237_11 := arithEq_of_rows h (row := 1237) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_11
  simp only [← c76245, ← c76246, k2, ← c76247] at e_1237_11
  have hr := e_1237_11
  simp only [e_1237_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11141)) := by
  have c76248 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76249 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76250 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76251 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76252 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76253 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_12 := arithEq_of_rows h (row := 1237) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_12
  simp only [← c76248, ← c76249, ← c76250] at e_1237_12
  have e_1237_13 := arithEq_of_rows h (row := 1237) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_13
  simp only [← c76251, ← c76252, k2, ← c76253] at e_1237_13
  have hr := e_1237_13
  simp only [e_1237_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11142)) := by
  have c76254 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76255 := (publicBatchWrapper89_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76256 := (publicBatchWrapper89_copies2383 a h).1
  have c76257 := (publicBatchWrapper89_copies2383 a h).2.1
  have c76258 := (publicBatchWrapper89_copies2383 a h).2.2.1
  have c76259 := (publicBatchWrapper89_copies2383 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_14 := arithEq_of_rows h (row := 1237) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_14
  simp only [← c76254, ← c76255, ← c76256] at e_1237_14
  have e_1237_15 := arithEq_of_rows h (row := 1237) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_15
  simp only [← c76257, ← c76258, k2, ← c76259] at e_1237_15
  have hr := e_1237_15
  simp only [e_1237_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11143)) := by
  have c76260 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.1
  have c76261 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.1
  have c76262 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.1
  have c76263 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.1
  have c76264 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.1
  have c76265 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_16 := arithEq_of_rows h (row := 1237) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_16
  simp only [← c76260, ← c76261, ← c76262] at e_1237_16
  have e_1237_17 := arithEq_of_rows h (row := 1237) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_17
  simp only [← c76263, ← c76264, k2, ← c76265] at e_1237_17
  have hr := e_1237_17
  simp only [e_1237_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1237 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11144)) := by
  have c76266 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76267 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76268 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76269 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76270 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76271 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1237_18 := arithEq_of_rows h (row := 1237) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_18
  simp only [← c76266, ← c76267, ← c76268] at e_1237_18
  have e_1237_19 := arithEq_of_rows h (row := 1237) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_19
  simp only [← c76269, ← c76270, k2, ← c76271] at e_1237_19
  have hr := e_1237_19
  simp only [e_1237_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11145)) := by
  have c76272 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76273 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76274 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76275 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76276 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76277 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_0 := arithEq_of_rows h (row := 1238) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_0
  simp only [← c76272, ← c76273, ← c76274] at e_1238_0
  have e_1238_1 := arithEq_of_rows h (row := 1238) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_1
  simp only [← c76275, ← c76276, k2, ← c76277] at e_1238_1
  have hr := e_1238_1
  simp only [e_1238_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11146)) := by
  have c76278 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76279 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76280 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76281 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76282 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76283 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_2 := arithEq_of_rows h (row := 1238) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_2
  simp only [← c76278, ← c76279, ← c76280] at e_1238_2
  have e_1238_3 := arithEq_of_rows h (row := 1238) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_3
  simp only [← c76281, ← c76282, k2, ← c76283] at e_1238_3
  have hr := e_1238_3
  simp only [e_1238_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11147)) := by
  have c76284 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76285 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76286 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76287 := (publicBatchWrapper89_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76288 := (publicBatchWrapper89_copies2384 a h).1
  have c76289 := (publicBatchWrapper89_copies2384 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_4 := arithEq_of_rows h (row := 1238) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_4
  simp only [← c76284, ← c76285, ← c76286] at e_1238_4
  have e_1238_5 := arithEq_of_rows h (row := 1238) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_5
  simp only [← c76287, ← c76288, k2, ← c76289] at e_1238_5
  have hr := e_1238_5
  simp only [e_1238_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11148)) := by
  have c76290 := (publicBatchWrapper89_copies2384 a h).2.2.1
  have c76291 := (publicBatchWrapper89_copies2384 a h).2.2.2.1
  have c76292 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.1
  have c76293 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.1
  have c76294 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.1
  have c76295 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_6 := arithEq_of_rows h (row := 1238) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_6
  simp only [← c76290, ← c76291, ← c76292] at e_1238_6
  have e_1238_7 := arithEq_of_rows h (row := 1238) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_7
  simp only [← c76293, ← c76294, k2, ← c76295] at e_1238_7
  have hr := e_1238_7
  simp only [e_1238_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11149)) := by
  have c76296 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.1
  have c76297 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.1
  have c76298 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76299 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76300 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76301 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_8 := arithEq_of_rows h (row := 1238) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_8
  simp only [← c76296, ← c76297, ← c76298] at e_1238_8
  have e_1238_9 := arithEq_of_rows h (row := 1238) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_9
  simp only [← c76299, ← c76300, k2, ← c76301] at e_1238_9
  have hr := e_1238_9
  simp only [e_1238_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11150)) := by
  have c76302 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76303 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76304 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76305 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76306 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76307 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_10 := arithEq_of_rows h (row := 1238) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_10
  simp only [← c76302, ← c76303, ← c76304] at e_1238_10
  have e_1238_11 := arithEq_of_rows h (row := 1238) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_11
  simp only [← c76305, ← c76306, k2, ← c76307] at e_1238_11
  have hr := e_1238_11
  simp only [e_1238_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11151)) := by
  have c76308 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76309 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76310 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76311 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76312 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76313 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_12 := arithEq_of_rows h (row := 1238) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_12
  simp only [← c76308, ← c76309, ← c76310] at e_1238_12
  have e_1238_13 := arithEq_of_rows h (row := 1238) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_13
  simp only [← c76311, ← c76312, k2, ← c76313] at e_1238_13
  have hr := e_1238_13
  simp only [e_1238_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11152)) := by
  have c76314 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76315 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76316 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76317 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76318 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76319 := (publicBatchWrapper89_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_14 := arithEq_of_rows h (row := 1238) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_14
  simp only [← c76314, ← c76315, ← c76316] at e_1238_14
  have e_1238_15 := arithEq_of_rows h (row := 1238) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_15
  simp only [← c76317, ← c76318, k2, ← c76319] at e_1238_15
  have hr := e_1238_15
  simp only [e_1238_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11153)) := by
  have c76320 := (publicBatchWrapper89_copies2385 a h).1
  have c76321 := (publicBatchWrapper89_copies2385 a h).2.1
  have c76322 := (publicBatchWrapper89_copies2385 a h).2.2.1
  have c76323 := (publicBatchWrapper89_copies2385 a h).2.2.2.1
  have c76324 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.1
  have c76325 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_16 := arithEq_of_rows h (row := 1238) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_16
  simp only [← c76320, ← c76321, ← c76322] at e_1238_16
  have e_1238_17 := arithEq_of_rows h (row := 1238) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_17
  simp only [← c76323, ← c76324, k2, ← c76325] at e_1238_17
  have hr := e_1238_17
  simp only [e_1238_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1238 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11154)) := by
  have c76326 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.1
  have c76327 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.1
  have c76328 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.1
  have c76329 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.1
  have c76330 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76331 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1238_18 := arithEq_of_rows h (row := 1238) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_18
  simp only [← c76326, ← c76327, ← c76328] at e_1238_18
  have e_1238_19 := arithEq_of_rows h (row := 1238) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_19
  simp only [← c76329, ← c76330, k2, ← c76331] at e_1238_19
  have hr := e_1238_19
  simp only [e_1238_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11155)) := by
  have c76332 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76333 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76334 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76335 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76336 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76337 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_0 := arithEq_of_rows h (row := 1239) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_0
  simp only [← c76332, ← c76333, ← c76334] at e_1239_0
  have e_1239_1 := arithEq_of_rows h (row := 1239) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_1
  simp only [← c76335, ← c76336, k2, ← c76337] at e_1239_1
  have hr := e_1239_1
  simp only [e_1239_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11156)) := by
  have c76338 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76339 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76340 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76341 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76342 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76343 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_2 := arithEq_of_rows h (row := 1239) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_2
  simp only [← c76338, ← c76339, ← c76340] at e_1239_2
  have e_1239_3 := arithEq_of_rows h (row := 1239) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_3
  simp only [← c76341, ← c76342, k2, ← c76343] at e_1239_3
  have hr := e_1239_3
  simp only [e_1239_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11157)) := by
  have c76344 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76345 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76346 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76347 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76348 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76349 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_4 := arithEq_of_rows h (row := 1239) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_4
  simp only [← c76344, ← c76345, ← c76346] at e_1239_4
  have e_1239_5 := arithEq_of_rows h (row := 1239) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_5
  simp only [← c76347, ← c76348, k2, ← c76349] at e_1239_5
  have hr := e_1239_5
  simp only [e_1239_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11158)) := by
  have c76350 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76351 := (publicBatchWrapper89_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76352 := (publicBatchWrapper89_copies2386 a h).1
  have c76353 := (publicBatchWrapper89_copies2386 a h).2.1
  have c76354 := (publicBatchWrapper89_copies2386 a h).2.2.1
  have c76355 := (publicBatchWrapper89_copies2386 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_6 := arithEq_of_rows h (row := 1239) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_6
  simp only [← c76350, ← c76351, ← c76352] at e_1239_6
  have e_1239_7 := arithEq_of_rows h (row := 1239) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_7
  simp only [← c76353, ← c76354, k2, ← c76355] at e_1239_7
  have hr := e_1239_7
  simp only [e_1239_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11159)) := by
  have c76356 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.1
  have c76357 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.1
  have c76358 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.1
  have c76359 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.1
  have c76360 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.1
  have c76361 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_8 := arithEq_of_rows h (row := 1239) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_8
  simp only [← c76356, ← c76357, ← c76358] at e_1239_8
  have e_1239_9 := arithEq_of_rows h (row := 1239) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_9
  simp only [← c76359, ← c76360, k2, ← c76361] at e_1239_9
  have hr := e_1239_9
  simp only [e_1239_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11160)) := by
  have c76362 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76363 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76364 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76365 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76366 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76367 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_10 := arithEq_of_rows h (row := 1239) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_10
  simp only [← c76362, ← c76363, ← c76364] at e_1239_10
  have e_1239_11 := arithEq_of_rows h (row := 1239) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_11
  simp only [← c76365, ← c76366, k2, ← c76367] at e_1239_11
  have hr := e_1239_11
  simp only [e_1239_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11161)) := by
  have c76368 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76369 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76370 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76371 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76372 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76373 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_12 := arithEq_of_rows h (row := 1239) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_12
  simp only [← c76368, ← c76369, ← c76370] at e_1239_12
  have e_1239_13 := arithEq_of_rows h (row := 1239) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_13
  simp only [← c76371, ← c76372, k2, ← c76373] at e_1239_13
  have hr := e_1239_13
  simp only [e_1239_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11162)) := by
  have c76374 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76375 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76376 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76377 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76378 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76379 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_14 := arithEq_of_rows h (row := 1239) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_14
  simp only [← c76374, ← c76375, ← c76376] at e_1239_14
  have e_1239_15 := arithEq_of_rows h (row := 1239) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_15
  simp only [← c76377, ← c76378, k2, ← c76379] at e_1239_15
  have hr := e_1239_15
  simp only [e_1239_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11163)) := by
  have c76380 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76381 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76382 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76383 := (publicBatchWrapper89_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76384 := (publicBatchWrapper89_copies2387 a h).1
  have c76385 := (publicBatchWrapper89_copies2387 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_16 := arithEq_of_rows h (row := 1239) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_16
  simp only [← c76380, ← c76381, ← c76382] at e_1239_16
  have e_1239_17 := arithEq_of_rows h (row := 1239) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_17
  simp only [← c76383, ← c76384, k2, ← c76385] at e_1239_17
  have hr := e_1239_17
  simp only [e_1239_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1239 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11164)) := by
  have c76386 := (publicBatchWrapper89_copies2387 a h).2.2.1
  have c76387 := (publicBatchWrapper89_copies2387 a h).2.2.2.1
  have c76388 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.1
  have c76389 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.1
  have c76390 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.1
  have c76391 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1239_18 := arithEq_of_rows h (row := 1239) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_18
  simp only [← c76386, ← c76387, ← c76388] at e_1239_18
  have e_1239_19 := arithEq_of_rows h (row := 1239) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_19
  simp only [← c76389, ← c76390, k2, ← c76391] at e_1239_19
  have hr := e_1239_19
  simp only [e_1239_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11165)) := by
  have c76392 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.1
  have c76393 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.1
  have c76394 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76395 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76396 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76397 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_0 := arithEq_of_rows h (row := 1240) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_0
  simp only [← c76392, ← c76393, ← c76394] at e_1240_0
  have e_1240_1 := arithEq_of_rows h (row := 1240) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_1
  simp only [← c76395, ← c76396, k2, ← c76397] at e_1240_1
  have hr := e_1240_1
  simp only [e_1240_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11166)) := by
  have c76398 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76399 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76400 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76401 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76402 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76403 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_2 := arithEq_of_rows h (row := 1240) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_2
  simp only [← c76398, ← c76399, ← c76400] at e_1240_2
  have e_1240_3 := arithEq_of_rows h (row := 1240) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_3
  simp only [← c76401, ← c76402, k2, ← c76403] at e_1240_3
  have hr := e_1240_3
  simp only [e_1240_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11167)) := by
  have c76404 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76405 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76406 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76407 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76408 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76409 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_4 := arithEq_of_rows h (row := 1240) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_4
  simp only [← c76404, ← c76405, ← c76406] at e_1240_4
  have e_1240_5 := arithEq_of_rows h (row := 1240) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_5
  simp only [← c76407, ← c76408, k2, ← c76409] at e_1240_5
  have hr := e_1240_5
  simp only [e_1240_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11168)) := by
  have c76410 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76411 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76412 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76413 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76414 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76415 := (publicBatchWrapper89_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_6 := arithEq_of_rows h (row := 1240) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_6
  simp only [← c76410, ← c76411, ← c76412] at e_1240_6
  have e_1240_7 := arithEq_of_rows h (row := 1240) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_7
  simp only [← c76413, ← c76414, k2, ← c76415] at e_1240_7
  have hr := e_1240_7
  simp only [e_1240_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11169)) := by
  have c76416 := (publicBatchWrapper89_copies2388 a h).1
  have c76417 := (publicBatchWrapper89_copies2388 a h).2.1
  have c76418 := (publicBatchWrapper89_copies2388 a h).2.2.1
  have c76419 := (publicBatchWrapper89_copies2388 a h).2.2.2.1
  have c76420 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.1
  have c76421 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_8 := arithEq_of_rows h (row := 1240) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_8
  simp only [← c76416, ← c76417, ← c76418] at e_1240_8
  have e_1240_9 := arithEq_of_rows h (row := 1240) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_9
  simp only [← c76419, ← c76420, k2, ← c76421] at e_1240_9
  have hr := e_1240_9
  simp only [e_1240_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11210)) := by
  have c76422 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.1
  have c76423 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.1
  have c76424 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.1
  have c76425 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.1
  have c76426 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76427 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_10 := arithEq_of_rows h (row := 1240) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_10
  simp only [← c76422, ← c76423, ← c76424] at e_1240_10
  have e_1240_11 := arithEq_of_rows h (row := 1240) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_11
  simp only [← c76425, ← c76426, k2, ← c76427] at e_1240_11
  have hr := e_1240_11
  simp only [e_1240_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11211)) := by
  have c76428 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76429 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76430 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76431 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76432 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76433 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_12 := arithEq_of_rows h (row := 1240) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_12
  simp only [← c76428, ← c76429, ← c76430] at e_1240_12
  have e_1240_13 := arithEq_of_rows h (row := 1240) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_13
  simp only [← c76431, ← c76432, k2, ← c76433] at e_1240_13
  have hr := e_1240_13
  simp only [e_1240_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11212)) := by
  have c76434 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76435 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76436 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76437 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76438 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76439 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_14 := arithEq_of_rows h (row := 1240) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_14
  simp only [← c76434, ← c76435, ← c76436] at e_1240_14
  have e_1240_15 := arithEq_of_rows h (row := 1240) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_15
  simp only [← c76437, ← c76438, k2, ← c76439] at e_1240_15
  have hr := e_1240_15
  simp only [e_1240_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11213)) := by
  have c76440 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76441 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76442 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76443 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76444 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76445 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_16 := arithEq_of_rows h (row := 1240) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_16
  simp only [← c76440, ← c76441, ← c76442] at e_1240_16
  have e_1240_17 := arithEq_of_rows h (row := 1240) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_17
  simp only [← c76443, ← c76444, k2, ← c76445] at e_1240_17
  have hr := e_1240_17
  simp only [e_1240_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1240 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11214)) := by
  have c76446 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76447 := (publicBatchWrapper89_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76448 := (publicBatchWrapper89_copies2389 a h).1
  have c76449 := (publicBatchWrapper89_copies2389 a h).2.1
  have c76450 := (publicBatchWrapper89_copies2389 a h).2.2.1
  have c76451 := (publicBatchWrapper89_copies2389 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1240_18 := arithEq_of_rows h (row := 1240) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_18
  simp only [← c76446, ← c76447, ← c76448] at e_1240_18
  have e_1240_19 := arithEq_of_rows h (row := 1240) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_19
  simp only [← c76449, ← c76450, k2, ← c76451] at e_1240_19
  have hr := e_1240_19
  simp only [e_1240_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11215)) := by
  have c76452 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.1
  have c76453 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.1
  have c76454 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.1
  have c76455 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.1
  have c76456 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.1
  have c76457 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_0 := arithEq_of_rows h (row := 1241) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_0
  simp only [← c76452, ← c76453, ← c76454] at e_1241_0
  have e_1241_1 := arithEq_of_rows h (row := 1241) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_1
  simp only [← c76455, ← c76456, k2, ← c76457] at e_1241_1
  have hr := e_1241_1
  simp only [e_1241_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11216)) := by
  have c76458 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76459 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76460 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76461 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76462 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76463 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_2 := arithEq_of_rows h (row := 1241) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_2
  simp only [← c76458, ← c76459, ← c76460] at e_1241_2
  have e_1241_3 := arithEq_of_rows h (row := 1241) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_3
  simp only [← c76461, ← c76462, k2, ← c76463] at e_1241_3
  have hr := e_1241_3
  simp only [e_1241_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11217)) := by
  have c76464 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76465 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76466 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76467 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76468 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76469 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_4 := arithEq_of_rows h (row := 1241) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_4
  simp only [← c76464, ← c76465, ← c76466] at e_1241_4
  have e_1241_5 := arithEq_of_rows h (row := 1241) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_5
  simp only [← c76467, ← c76468, k2, ← c76469] at e_1241_5
  have hr := e_1241_5
  simp only [e_1241_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11218)) := by
  have c76470 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76471 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76472 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76473 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76474 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76475 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_6 := arithEq_of_rows h (row := 1241) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_6
  simp only [← c76470, ← c76471, ← c76472] at e_1241_6
  have e_1241_7 := arithEq_of_rows h (row := 1241) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_7
  simp only [← c76473, ← c76474, k2, ← c76475] at e_1241_7
  have hr := e_1241_7
  simp only [e_1241_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11219)) := by
  have c76476 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76477 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76478 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76479 := (publicBatchWrapper89_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76480 := (publicBatchWrapper89_copies2390 a h).1
  have c76481 := (publicBatchWrapper89_copies2390 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_8 := arithEq_of_rows h (row := 1241) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_8
  simp only [← c76476, ← c76477, ← c76478] at e_1241_8
  have e_1241_9 := arithEq_of_rows h (row := 1241) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_9
  simp only [← c76479, ← c76480, k2, ← c76481] at e_1241_9
  have hr := e_1241_9
  simp only [e_1241_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11220)) := by
  have c76482 := (publicBatchWrapper89_copies2390 a h).2.2.1
  have c76483 := (publicBatchWrapper89_copies2390 a h).2.2.2.1
  have c76484 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.1
  have c76485 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.1
  have c76486 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.1
  have c76487 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_10 := arithEq_of_rows h (row := 1241) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_10
  simp only [← c76482, ← c76483, ← c76484] at e_1241_10
  have e_1241_11 := arithEq_of_rows h (row := 1241) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_11
  simp only [← c76485, ← c76486, k2, ← c76487] at e_1241_11
  have hr := e_1241_11
  simp only [e_1241_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11221)) := by
  have c76488 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.1
  have c76489 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.1
  have c76490 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76491 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76492 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76493 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_12 := arithEq_of_rows h (row := 1241) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_12
  simp only [← c76488, ← c76489, ← c76490] at e_1241_12
  have e_1241_13 := arithEq_of_rows h (row := 1241) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_13
  simp only [← c76491, ← c76492, k2, ← c76493] at e_1241_13
  have hr := e_1241_13
  simp only [e_1241_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11222)) := by
  have c76494 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76495 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76496 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76497 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76498 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76499 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_14 := arithEq_of_rows h (row := 1241) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_14
  simp only [← c76494, ← c76495, ← c76496] at e_1241_14
  have e_1241_15 := arithEq_of_rows h (row := 1241) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_15
  simp only [← c76497, ← c76498, k2, ← c76499] at e_1241_15
  have hr := e_1241_15
  simp only [e_1241_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11223)) := by
  have c76500 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76501 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76502 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76503 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76504 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76505 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_16 := arithEq_of_rows h (row := 1241) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_16
  simp only [← c76500, ← c76501, ← c76502] at e_1241_16
  have e_1241_17 := arithEq_of_rows h (row := 1241) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_17
  simp only [← c76503, ← c76504, k2, ← c76505] at e_1241_17
  have hr := e_1241_17
  simp only [e_1241_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1241 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11224)) := by
  have c76506 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76507 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76508 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76509 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76510 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76511 := (publicBatchWrapper89_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1241_18 := arithEq_of_rows h (row := 1241) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_18
  simp only [← c76506, ← c76507, ← c76508] at e_1241_18
  have e_1241_19 := arithEq_of_rows h (row := 1241) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_19
  simp only [← c76509, ← c76510, k2, ← c76511] at e_1241_19
  have hr := e_1241_19
  simp only [e_1241_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11225)) := by
  have c76512 := (publicBatchWrapper89_copies2391 a h).1
  have c76513 := (publicBatchWrapper89_copies2391 a h).2.1
  have c76514 := (publicBatchWrapper89_copies2391 a h).2.2.1
  have c76515 := (publicBatchWrapper89_copies2391 a h).2.2.2.1
  have c76516 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.1
  have c76517 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_0 := arithEq_of_rows h (row := 1242) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_0
  simp only [← c76512, ← c76513, ← c76514] at e_1242_0
  have e_1242_1 := arithEq_of_rows h (row := 1242) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_1
  simp only [← c76515, ← c76516, k2, ← c76517] at e_1242_1
  have hr := e_1242_1
  simp only [e_1242_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11226)) := by
  have c76518 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.1
  have c76519 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.1
  have c76520 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.1
  have c76521 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.1
  have c76522 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76523 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_2 := arithEq_of_rows h (row := 1242) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_2
  simp only [← c76518, ← c76519, ← c76520] at e_1242_2
  have e_1242_3 := arithEq_of_rows h (row := 1242) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_3
  simp only [← c76521, ← c76522, k2, ← c76523] at e_1242_3
  have hr := e_1242_3
  simp only [e_1242_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11227)) := by
  have c76524 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76525 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76526 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76527 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76528 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76529 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_4 := arithEq_of_rows h (row := 1242) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_4
  simp only [← c76524, ← c76525, ← c76526] at e_1242_4
  have e_1242_5 := arithEq_of_rows h (row := 1242) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_5
  simp only [← c76527, ← c76528, k2, ← c76529] at e_1242_5
  have hr := e_1242_5
  simp only [e_1242_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11228)) := by
  have c76530 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76531 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76532 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76533 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76534 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76535 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_6 := arithEq_of_rows h (row := 1242) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_6
  simp only [← c76530, ← c76531, ← c76532] at e_1242_6
  have e_1242_7 := arithEq_of_rows h (row := 1242) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_7
  simp only [← c76533, ← c76534, k2, ← c76535] at e_1242_7
  have hr := e_1242_7
  simp only [e_1242_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11229)) := by
  have c76536 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76537 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76538 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76539 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76540 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76541 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_8 := arithEq_of_rows h (row := 1242) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_8
  simp only [← c76536, ← c76537, ← c76538] at e_1242_8
  have e_1242_9 := arithEq_of_rows h (row := 1242) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_9
  simp only [← c76539, ← c76540, k2, ← c76541] at e_1242_9
  have hr := e_1242_9
  simp only [e_1242_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11230)) := by
  have c76542 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76543 := (publicBatchWrapper89_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76544 := (publicBatchWrapper89_copies2392 a h).1
  have c76545 := (publicBatchWrapper89_copies2392 a h).2.1
  have c76546 := (publicBatchWrapper89_copies2392 a h).2.2.1
  have c76547 := (publicBatchWrapper89_copies2392 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_10 := arithEq_of_rows h (row := 1242) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_10
  simp only [← c76542, ← c76543, ← c76544] at e_1242_10
  have e_1242_11 := arithEq_of_rows h (row := 1242) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_11
  simp only [← c76545, ← c76546, k2, ← c76547] at e_1242_11
  have hr := e_1242_11
  simp only [e_1242_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11231)) := by
  have c76548 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.1
  have c76549 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.1
  have c76550 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.1
  have c76551 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.1
  have c76552 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.1
  have c76553 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_12 := arithEq_of_rows h (row := 1242) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_12
  simp only [← c76548, ← c76549, ← c76550] at e_1242_12
  have e_1242_13 := arithEq_of_rows h (row := 1242) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_13
  simp only [← c76551, ← c76552, k2, ← c76553] at e_1242_13
  have hr := e_1242_13
  simp only [e_1242_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11232)) := by
  have c76554 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76555 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76556 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76557 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76558 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76559 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_14 := arithEq_of_rows h (row := 1242) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_14
  simp only [← c76554, ← c76555, ← c76556] at e_1242_14
  have e_1242_15 := arithEq_of_rows h (row := 1242) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_15
  simp only [← c76557, ← c76558, k2, ← c76559] at e_1242_15
  have hr := e_1242_15
  simp only [e_1242_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11233)) := by
  have c76560 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76561 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76562 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76563 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76564 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76565 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_16 := arithEq_of_rows h (row := 1242) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_16
  simp only [← c76560, ← c76561, ← c76562] at e_1242_16
  have e_1242_17 := arithEq_of_rows h (row := 1242) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_17
  simp only [← c76563, ← c76564, k2, ← c76565] at e_1242_17
  have hr := e_1242_17
  simp only [e_1242_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1242 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11234)) := by
  have c76566 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76567 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76568 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76569 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76570 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76571 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1242_18 := arithEq_of_rows h (row := 1242) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_18
  simp only [← c76566, ← c76567, ← c76568] at e_1242_18
  have e_1242_19 := arithEq_of_rows h (row := 1242) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_19
  simp only [← c76569, ← c76570, k2, ← c76571] at e_1242_19
  have hr := e_1242_19
  simp only [e_1242_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11235)) := by
  have c76572 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76573 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76574 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76575 := (publicBatchWrapper89_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76576 := (publicBatchWrapper89_copies2393 a h).1
  have c76577 := (publicBatchWrapper89_copies2393 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_0 := arithEq_of_rows h (row := 1243) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_0
  simp only [← c76572, ← c76573, ← c76574] at e_1243_0
  have e_1243_1 := arithEq_of_rows h (row := 1243) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_1
  simp only [← c76575, ← c76576, k2, ← c76577] at e_1243_1
  have hr := e_1243_1
  simp only [e_1243_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11236)) := by
  have c76578 := (publicBatchWrapper89_copies2393 a h).2.2.1
  have c76579 := (publicBatchWrapper89_copies2393 a h).2.2.2.1
  have c76580 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.1
  have c76581 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.1
  have c76582 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.1
  have c76583 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_2 := arithEq_of_rows h (row := 1243) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_2
  simp only [← c76578, ← c76579, ← c76580] at e_1243_2
  have e_1243_3 := arithEq_of_rows h (row := 1243) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_3
  simp only [← c76581, ← c76582, k2, ← c76583] at e_1243_3
  have hr := e_1243_3
  simp only [e_1243_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11237)) := by
  have c76584 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.1
  have c76585 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.1
  have c76586 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76587 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76588 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76589 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_4 := arithEq_of_rows h (row := 1243) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_4
  simp only [← c76584, ← c76585, ← c76586] at e_1243_4
  have e_1243_5 := arithEq_of_rows h (row := 1243) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_5
  simp only [← c76587, ← c76588, k2, ← c76589] at e_1243_5
  have hr := e_1243_5
  simp only [e_1243_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11238)) := by
  have c76590 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76591 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76592 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76593 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76594 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76595 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_6 := arithEq_of_rows h (row := 1243) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_6
  simp only [← c76590, ← c76591, ← c76592] at e_1243_6
  have e_1243_7 := arithEq_of_rows h (row := 1243) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_7
  simp only [← c76593, ← c76594, k2, ← c76595] at e_1243_7
  have hr := e_1243_7
  simp only [e_1243_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11239)) := by
  have c76596 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76597 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76598 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76599 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76600 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76601 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_8 := arithEq_of_rows h (row := 1243) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_8
  simp only [← c76596, ← c76597, ← c76598] at e_1243_8
  have e_1243_9 := arithEq_of_rows h (row := 1243) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_9
  simp only [← c76599, ← c76600, k2, ← c76601] at e_1243_9
  have hr := e_1243_9
  simp only [e_1243_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11240)) := by
  have c76602 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76603 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76604 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76605 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76606 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76607 := (publicBatchWrapper89_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_10 := arithEq_of_rows h (row := 1243) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_10
  simp only [← c76602, ← c76603, ← c76604] at e_1243_10
  have e_1243_11 := arithEq_of_rows h (row := 1243) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_11
  simp only [← c76605, ← c76606, k2, ← c76607] at e_1243_11
  have hr := e_1243_11
  simp only [e_1243_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11241)) := by
  have c76608 := (publicBatchWrapper89_copies2394 a h).1
  have c76609 := (publicBatchWrapper89_copies2394 a h).2.1
  have c76610 := (publicBatchWrapper89_copies2394 a h).2.2.1
  have c76611 := (publicBatchWrapper89_copies2394 a h).2.2.2.1
  have c76612 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.1
  have c76613 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_12 := arithEq_of_rows h (row := 1243) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_12
  simp only [← c76608, ← c76609, ← c76610] at e_1243_12
  have e_1243_13 := arithEq_of_rows h (row := 1243) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_13
  simp only [← c76611, ← c76612, k2, ← c76613] at e_1243_13
  have hr := e_1243_13
  simp only [e_1243_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11242)) := by
  have c76614 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.1
  have c76615 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.1
  have c76616 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.1
  have c76617 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.1
  have c76618 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76619 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_14 := arithEq_of_rows h (row := 1243) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_14
  simp only [← c76614, ← c76615, ← c76616] at e_1243_14
  have e_1243_15 := arithEq_of_rows h (row := 1243) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_15
  simp only [← c76617, ← c76618, k2, ← c76619] at e_1243_15
  have hr := e_1243_15
  simp only [e_1243_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11243)) := by
  have c76620 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76621 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76622 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76623 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76624 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76625 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_16 := arithEq_of_rows h (row := 1243) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_16
  simp only [← c76620, ← c76621, ← c76622] at e_1243_16
  have e_1243_17 := arithEq_of_rows h (row := 1243) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_17
  simp only [← c76623, ← c76624, k2, ← c76625] at e_1243_17
  have hr := e_1243_17
  simp only [e_1243_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1243 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11244)) := by
  have c76626 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76627 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76628 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76629 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76630 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76631 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1243_18 := arithEq_of_rows h (row := 1243) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_18
  simp only [← c76626, ← c76627, ← c76628] at e_1243_18
  have e_1243_19 := arithEq_of_rows h (row := 1243) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_19
  simp only [← c76629, ← c76630, k2, ← c76631] at e_1243_19
  have hr := e_1243_19
  simp only [e_1243_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11245)) := by
  have c76632 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76633 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76634 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76635 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76636 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76637 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_0 := arithEq_of_rows h (row := 1244) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_0
  simp only [← c76632, ← c76633, ← c76634] at e_1244_0
  have e_1244_1 := arithEq_of_rows h (row := 1244) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_1
  simp only [← c76635, ← c76636, k2, ← c76637] at e_1244_1
  have hr := e_1244_1
  simp only [e_1244_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11246)) := by
  have c76638 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76639 := (publicBatchWrapper89_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76640 := (publicBatchWrapper89_copies2395 a h).1
  have c76641 := (publicBatchWrapper89_copies2395 a h).2.1
  have c76642 := (publicBatchWrapper89_copies2395 a h).2.2.1
  have c76643 := (publicBatchWrapper89_copies2395 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_2 := arithEq_of_rows h (row := 1244) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_2
  simp only [← c76638, ← c76639, ← c76640] at e_1244_2
  have e_1244_3 := arithEq_of_rows h (row := 1244) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_3
  simp only [← c76641, ← c76642, k2, ← c76643] at e_1244_3
  have hr := e_1244_3
  simp only [e_1244_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11247)) := by
  have c76644 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.1
  have c76645 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.1
  have c76646 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.1
  have c76647 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.1
  have c76648 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.1
  have c76649 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_4 := arithEq_of_rows h (row := 1244) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_4
  simp only [← c76644, ← c76645, ← c76646] at e_1244_4
  have e_1244_5 := arithEq_of_rows h (row := 1244) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_5
  simp only [← c76647, ← c76648, k2, ← c76649] at e_1244_5
  have hr := e_1244_5
  simp only [e_1244_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11248)) := by
  have c76650 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76651 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76652 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76653 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76654 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76655 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_6 := arithEq_of_rows h (row := 1244) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_6
  simp only [← c76650, ← c76651, ← c76652] at e_1244_6
  have e_1244_7 := arithEq_of_rows h (row := 1244) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_7
  simp only [← c76653, ← c76654, k2, ← c76655] at e_1244_7
  have hr := e_1244_7
  simp only [e_1244_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11249)) := by
  have c76656 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76657 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76658 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76659 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76660 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76661 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_8 := arithEq_of_rows h (row := 1244) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_8
  simp only [← c76656, ← c76657, ← c76658] at e_1244_8
  have e_1244_9 := arithEq_of_rows h (row := 1244) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_9
  simp only [← c76659, ← c76660, k2, ← c76661] at e_1244_9
  have hr := e_1244_9
  simp only [e_1244_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11250)) := by
  have c76662 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76663 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76664 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76665 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76666 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76667 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_10 := arithEq_of_rows h (row := 1244) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_10
  simp only [← c76662, ← c76663, ← c76664] at e_1244_10
  have e_1244_11 := arithEq_of_rows h (row := 1244) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_11
  simp only [← c76665, ← c76666, k2, ← c76667] at e_1244_11
  have hr := e_1244_11
  simp only [e_1244_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11251)) := by
  have c76668 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76669 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76670 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76671 := (publicBatchWrapper89_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76672 := (publicBatchWrapper89_copies2396 a h).1
  have c76673 := (publicBatchWrapper89_copies2396 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_12 := arithEq_of_rows h (row := 1244) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_12
  simp only [← c76668, ← c76669, ← c76670] at e_1244_12
  have e_1244_13 := arithEq_of_rows h (row := 1244) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_13
  simp only [← c76671, ← c76672, k2, ← c76673] at e_1244_13
  have hr := e_1244_13
  simp only [e_1244_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11252)) := by
  have c76674 := (publicBatchWrapper89_copies2396 a h).2.2.1
  have c76675 := (publicBatchWrapper89_copies2396 a h).2.2.2.1
  have c76676 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.1
  have c76677 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.1
  have c76678 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.1
  have c76679 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_14 := arithEq_of_rows h (row := 1244) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_14
  simp only [← c76674, ← c76675, ← c76676] at e_1244_14
  have e_1244_15 := arithEq_of_rows h (row := 1244) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_15
  simp only [← c76677, ← c76678, k2, ← c76679] at e_1244_15
  have hr := e_1244_15
  simp only [e_1244_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11253)) := by
  have c76680 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.1
  have c76681 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.1
  have c76682 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76683 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76684 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76685 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_16 := arithEq_of_rows h (row := 1244) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_16
  simp only [← c76680, ← c76681, ← c76682] at e_1244_16
  have e_1244_17 := arithEq_of_rows h (row := 1244) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_17
  simp only [← c76683, ← c76684, k2, ← c76685] at e_1244_17
  have hr := e_1244_17
  simp only [e_1244_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1244 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11254)) := by
  have c76686 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76687 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76688 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76689 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76690 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76691 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1244_18 := arithEq_of_rows h (row := 1244) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_18
  simp only [← c76686, ← c76687, ← c76688] at e_1244_18
  have e_1244_19 := arithEq_of_rows h (row := 1244) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_19
  simp only [← c76689, ← c76690, k2, ← c76691] at e_1244_19
  have hr := e_1244_19
  simp only [e_1244_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11255)) := by
  have c76692 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76693 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76694 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76695 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76696 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76697 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_0 := arithEq_of_rows h (row := 1245) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_0
  simp only [← c76692, ← c76693, ← c76694] at e_1245_0
  have e_1245_1 := arithEq_of_rows h (row := 1245) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_1
  simp only [← c76695, ← c76696, k2, ← c76697] at e_1245_1
  have hr := e_1245_1
  simp only [e_1245_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11256)) := by
  have c76698 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76699 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76700 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76701 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76702 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76703 := (publicBatchWrapper89_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_2 := arithEq_of_rows h (row := 1245) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_2
  simp only [← c76698, ← c76699, ← c76700] at e_1245_2
  have e_1245_3 := arithEq_of_rows h (row := 1245) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_3
  simp only [← c76701, ← c76702, k2, ← c76703] at e_1245_3
  have hr := e_1245_3
  simp only [e_1245_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11257)) := by
  have c76704 := (publicBatchWrapper89_copies2397 a h).1
  have c76705 := (publicBatchWrapper89_copies2397 a h).2.1
  have c76706 := (publicBatchWrapper89_copies2397 a h).2.2.1
  have c76707 := (publicBatchWrapper89_copies2397 a h).2.2.2.1
  have c76708 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.1
  have c76709 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_4 := arithEq_of_rows h (row := 1245) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_4
  simp only [← c76704, ← c76705, ← c76706] at e_1245_4
  have e_1245_5 := arithEq_of_rows h (row := 1245) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_5
  simp only [← c76707, ← c76708, k2, ← c76709] at e_1245_5
  have hr := e_1245_5
  simp only [e_1245_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11258)) := by
  have c76710 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.1
  have c76711 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.1
  have c76712 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.1
  have c76713 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.1
  have c76714 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76715 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_6 := arithEq_of_rows h (row := 1245) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_6
  simp only [← c76710, ← c76711, ← c76712] at e_1245_6
  have e_1245_7 := arithEq_of_rows h (row := 1245) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_7
  simp only [← c76713, ← c76714, k2, ← c76715] at e_1245_7
  have hr := e_1245_7
  simp only [e_1245_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11259)) := by
  have c76716 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76717 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76718 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76719 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76720 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76721 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_8 := arithEq_of_rows h (row := 1245) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_8
  simp only [← c76716, ← c76717, ← c76718] at e_1245_8
  have e_1245_9 := arithEq_of_rows h (row := 1245) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_9
  simp only [← c76719, ← c76720, k2, ← c76721] at e_1245_9
  have hr := e_1245_9
  simp only [e_1245_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11260)) := by
  have c76722 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76723 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76724 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76725 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76726 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76727 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_10 := arithEq_of_rows h (row := 1245) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_10
  simp only [← c76722, ← c76723, ← c76724] at e_1245_10
  have e_1245_11 := arithEq_of_rows h (row := 1245) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_11
  simp only [← c76725, ← c76726, k2, ← c76727] at e_1245_11
  have hr := e_1245_11
  simp only [e_1245_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11261)) := by
  have c76728 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76729 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76730 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76731 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76732 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76733 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_12 := arithEq_of_rows h (row := 1245) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_12
  simp only [← c76728, ← c76729, ← c76730] at e_1245_12
  have e_1245_13 := arithEq_of_rows h (row := 1245) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_13
  simp only [← c76731, ← c76732, k2, ← c76733] at e_1245_13
  have hr := e_1245_13
  simp only [e_1245_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11262)) := by
  have c76734 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76735 := (publicBatchWrapper89_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76736 := (publicBatchWrapper89_copies2398 a h).1
  have c76737 := (publicBatchWrapper89_copies2398 a h).2.1
  have c76738 := (publicBatchWrapper89_copies2398 a h).2.2.1
  have c76739 := (publicBatchWrapper89_copies2398 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_14 := arithEq_of_rows h (row := 1245) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_14
  simp only [← c76734, ← c76735, ← c76736] at e_1245_14
  have e_1245_15 := arithEq_of_rows h (row := 1245) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_15
  simp only [← c76737, ← c76738, k2, ← c76739] at e_1245_15
  have hr := e_1245_15
  simp only [e_1245_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11263)) := by
  have c76740 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.1
  have c76741 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.1
  have c76742 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.1
  have c76743 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.1
  have c76744 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.1
  have c76745 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_16 := arithEq_of_rows h (row := 1245) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_16
  simp only [← c76740, ← c76741, ← c76742] at e_1245_16
  have e_1245_17 := arithEq_of_rows h (row := 1245) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_17
  simp only [← c76743, ← c76744, k2, ← c76745] at e_1245_17
  have hr := e_1245_17
  simp only [e_1245_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1245 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11264)) := by
  have c76746 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76747 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76748 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76749 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76750 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76751 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1245_18 := arithEq_of_rows h (row := 1245) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_18
  simp only [← c76746, ← c76747, ← c76748] at e_1245_18
  have e_1245_19 := arithEq_of_rows h (row := 1245) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_19
  simp only [← c76749, ← c76750, k2, ← c76751] at e_1245_19
  have hr := e_1245_19
  simp only [e_1245_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11265)) := by
  have c76752 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76753 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76754 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76755 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76756 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76757 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_0 := arithEq_of_rows h (row := 1246) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_0
  simp only [← c76752, ← c76753, ← c76754] at e_1246_0
  have e_1246_1 := arithEq_of_rows h (row := 1246) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_1
  simp only [← c76755, ← c76756, k2, ← c76757] at e_1246_1
  have hr := e_1246_1
  simp only [e_1246_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11266)) := by
  have c76758 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76759 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76760 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76761 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76762 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76763 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_2 := arithEq_of_rows h (row := 1246) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_2
  simp only [← c76758, ← c76759, ← c76760] at e_1246_2
  have e_1246_3 := arithEq_of_rows h (row := 1246) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_3
  simp only [← c76761, ← c76762, k2, ← c76763] at e_1246_3
  have hr := e_1246_3
  simp only [e_1246_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11267)) := by
  have c76764 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76765 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76766 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76767 := (publicBatchWrapper89_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76768 := (publicBatchWrapper89_copies2399 a h).1
  have c76769 := (publicBatchWrapper89_copies2399 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_4 := arithEq_of_rows h (row := 1246) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_4
  simp only [← c76764, ← c76765, ← c76766] at e_1246_4
  have e_1246_5 := arithEq_of_rows h (row := 1246) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_5
  simp only [← c76767, ← c76768, k2, ← c76769] at e_1246_5
  have hr := e_1246_5
  simp only [e_1246_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11268)) := by
  have c76770 := (publicBatchWrapper89_copies2399 a h).2.2.1
  have c76771 := (publicBatchWrapper89_copies2399 a h).2.2.2.1
  have c76772 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.1
  have c76773 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.1
  have c76774 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.1
  have c76775 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_6 := arithEq_of_rows h (row := 1246) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_6
  simp only [← c76770, ← c76771, ← c76772] at e_1246_6
  have e_1246_7 := arithEq_of_rows h (row := 1246) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_7
  simp only [← c76773, ← c76774, k2, ← c76775] at e_1246_7
  have hr := e_1246_7
  simp only [e_1246_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11269)) := by
  have c76776 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.1
  have c76777 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.1
  have c76778 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76779 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76780 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76781 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_8 := arithEq_of_rows h (row := 1246) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_8
  simp only [← c76776, ← c76777, ← c76778] at e_1246_8
  have e_1246_9 := arithEq_of_rows h (row := 1246) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_9
  simp only [← c76779, ← c76780, k2, ← c76781] at e_1246_9
  have hr := e_1246_9
  simp only [e_1246_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11270)) := by
  have c76782 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76783 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76784 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76785 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76786 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76787 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_10 := arithEq_of_rows h (row := 1246) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_10
  simp only [← c76782, ← c76783, ← c76784] at e_1246_10
  have e_1246_11 := arithEq_of_rows h (row := 1246) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_11
  simp only [← c76785, ← c76786, k2, ← c76787] at e_1246_11
  have hr := e_1246_11
  simp only [e_1246_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11271)) := by
  have c76788 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76789 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76790 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76791 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76792 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76793 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_12 := arithEq_of_rows h (row := 1246) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_12
  simp only [← c76788, ← c76789, ← c76790] at e_1246_12
  have e_1246_13 := arithEq_of_rows h (row := 1246) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_13
  simp only [← c76791, ← c76792, k2, ← c76793] at e_1246_13
  have hr := e_1246_13
  simp only [e_1246_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11272)) := by
  have c76794 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76795 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76796 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76797 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76798 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76799 := (publicBatchWrapper89_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_14 := arithEq_of_rows h (row := 1246) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_14
  simp only [← c76794, ← c76795, ← c76796] at e_1246_14
  have e_1246_15 := arithEq_of_rows h (row := 1246) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_15
  simp only [← c76797, ← c76798, k2, ← c76799] at e_1246_15
  have hr := e_1246_15
  simp only [e_1246_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11273)) := by
  have c76800 := (publicBatchWrapper89_copies2400 a h).1
  have c76801 := (publicBatchWrapper89_copies2400 a h).2.1
  have c76802 := (publicBatchWrapper89_copies2400 a h).2.2.1
  have c76803 := (publicBatchWrapper89_copies2400 a h).2.2.2.1
  have c76804 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.1
  have c76805 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_16 := arithEq_of_rows h (row := 1246) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_16
  simp only [← c76800, ← c76801, ← c76802] at e_1246_16
  have e_1246_17 := arithEq_of_rows h (row := 1246) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_17
  simp only [← c76803, ← c76804, k2, ← c76805] at e_1246_17
  have hr := e_1246_17
  simp only [e_1246_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1246 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11274)) := by
  have c76806 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.1
  have c76807 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.1
  have c76808 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.1
  have c76809 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.1
  have c76810 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76811 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1246_18 := arithEq_of_rows h (row := 1246) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_18
  simp only [← c76806, ← c76807, ← c76808] at e_1246_18
  have e_1246_19 := arithEq_of_rows h (row := 1246) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_19
  simp only [← c76809, ← c76810, k2, ← c76811] at e_1246_19
  have hr := e_1246_19
  simp only [e_1246_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11275)) := by
  have c76812 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76813 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76814 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76815 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76816 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76817 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_0 := arithEq_of_rows h (row := 1247) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_0
  simp only [← c76812, ← c76813, ← c76814] at e_1247_0
  have e_1247_1 := arithEq_of_rows h (row := 1247) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_1
  simp only [← c76815, ← c76816, k2, ← c76817] at e_1247_1
  have hr := e_1247_1
  simp only [e_1247_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11276)) := by
  have c76818 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76819 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76820 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76821 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76822 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76823 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_2 := arithEq_of_rows h (row := 1247) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_2
  simp only [← c76818, ← c76819, ← c76820] at e_1247_2
  have e_1247_3 := arithEq_of_rows h (row := 1247) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_3
  simp only [← c76821, ← c76822, k2, ← c76823] at e_1247_3
  have hr := e_1247_3
  simp only [e_1247_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11277)) := by
  have c76824 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76825 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76826 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76827 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76828 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76829 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_4 := arithEq_of_rows h (row := 1247) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_4
  simp only [← c76824, ← c76825, ← c76826] at e_1247_4
  have e_1247_5 := arithEq_of_rows h (row := 1247) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_5
  simp only [← c76827, ← c76828, k2, ← c76829] at e_1247_5
  have hr := e_1247_5
  simp only [e_1247_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11278)) := by
  have c76830 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76831 := (publicBatchWrapper89_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76832 := (publicBatchWrapper89_copies2401 a h).1
  have c76833 := (publicBatchWrapper89_copies2401 a h).2.1
  have c76834 := (publicBatchWrapper89_copies2401 a h).2.2.1
  have c76835 := (publicBatchWrapper89_copies2401 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_6 := arithEq_of_rows h (row := 1247) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_6
  simp only [← c76830, ← c76831, ← c76832] at e_1247_6
  have e_1247_7 := arithEq_of_rows h (row := 1247) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_7
  simp only [← c76833, ← c76834, k2, ← c76835] at e_1247_7
  have hr := e_1247_7
  simp only [e_1247_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11279)) := by
  have c76836 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.1
  have c76837 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.1
  have c76838 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.1
  have c76839 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.1
  have c76840 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.1
  have c76841 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_8 := arithEq_of_rows h (row := 1247) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_8
  simp only [← c76836, ← c76837, ← c76838] at e_1247_8
  have e_1247_9 := arithEq_of_rows h (row := 1247) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_9
  simp only [← c76839, ← c76840, k2, ← c76841] at e_1247_9
  have hr := e_1247_9
  simp only [e_1247_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11280)) := by
  have c76842 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76843 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76844 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76845 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76846 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76847 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_10 := arithEq_of_rows h (row := 1247) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_10
  simp only [← c76842, ← c76843, ← c76844] at e_1247_10
  have e_1247_11 := arithEq_of_rows h (row := 1247) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_11
  simp only [← c76845, ← c76846, k2, ← c76847] at e_1247_11
  have hr := e_1247_11
  simp only [e_1247_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11281)) := by
  have c76848 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76849 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76850 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76851 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76852 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76853 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_12 := arithEq_of_rows h (row := 1247) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_12
  simp only [← c76848, ← c76849, ← c76850] at e_1247_12
  have e_1247_13 := arithEq_of_rows h (row := 1247) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_13
  simp only [← c76851, ← c76852, k2, ← c76853] at e_1247_13
  have hr := e_1247_13
  simp only [e_1247_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11282)) := by
  have c76854 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76855 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76856 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76857 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76858 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76859 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_14 := arithEq_of_rows h (row := 1247) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_14
  simp only [← c76854, ← c76855, ← c76856] at e_1247_14
  have e_1247_15 := arithEq_of_rows h (row := 1247) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_15
  simp only [← c76857, ← c76858, k2, ← c76859] at e_1247_15
  have hr := e_1247_15
  simp only [e_1247_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11283)) := by
  have c76860 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76861 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76862 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76863 := (publicBatchWrapper89_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76864 := (publicBatchWrapper89_copies2402 a h).1
  have c76865 := (publicBatchWrapper89_copies2402 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_16 := arithEq_of_rows h (row := 1247) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_16
  simp only [← c76860, ← c76861, ← c76862] at e_1247_16
  have e_1247_17 := arithEq_of_rows h (row := 1247) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_17
  simp only [← c76863, ← c76864, k2, ← c76865] at e_1247_17
  have hr := e_1247_17
  simp only [e_1247_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1247 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11284)) := by
  have c76866 := (publicBatchWrapper89_copies2402 a h).2.2.1
  have c76867 := (publicBatchWrapper89_copies2402 a h).2.2.2.1
  have c76868 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.1
  have c76869 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.1
  have c76870 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.1
  have c76871 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1247_18 := arithEq_of_rows h (row := 1247) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_18
  simp only [← c76866, ← c76867, ← c76868] at e_1247_18
  have e_1247_19 := arithEq_of_rows h (row := 1247) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_19
  simp only [← c76869, ← c76870, k2, ← c76871] at e_1247_19
  have hr := e_1247_19
  simp only [e_1247_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11285)) := by
  have c76872 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.1
  have c76873 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.1
  have c76874 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76875 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76876 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76877 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_0 := arithEq_of_rows h (row := 1248) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_0
  simp only [← c76872, ← c76873, ← c76874] at e_1248_0
  have e_1248_1 := arithEq_of_rows h (row := 1248) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_1
  simp only [← c76875, ← c76876, k2, ← c76877] at e_1248_1
  have hr := e_1248_1
  simp only [e_1248_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11286)) := by
  have c76878 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76879 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76880 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76881 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76882 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76883 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_2 := arithEq_of_rows h (row := 1248) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_2
  simp only [← c76878, ← c76879, ← c76880] at e_1248_2
  have e_1248_3 := arithEq_of_rows h (row := 1248) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_3
  simp only [← c76881, ← c76882, k2, ← c76883] at e_1248_3
  have hr := e_1248_3
  simp only [e_1248_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11287)) := by
  have c76884 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76885 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76886 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76887 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76888 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76889 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_4 := arithEq_of_rows h (row := 1248) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_4
  simp only [← c76884, ← c76885, ← c76886] at e_1248_4
  have e_1248_5 := arithEq_of_rows h (row := 1248) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_5
  simp only [← c76887, ← c76888, k2, ← c76889] at e_1248_5
  have hr := e_1248_5
  simp only [e_1248_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11288)) := by
  have c76890 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76891 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76892 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76893 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76894 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76895 := (publicBatchWrapper89_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_6 := arithEq_of_rows h (row := 1248) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_6
  simp only [← c76890, ← c76891, ← c76892] at e_1248_6
  have e_1248_7 := arithEq_of_rows h (row := 1248) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_7
  simp only [← c76893, ← c76894, k2, ← c76895] at e_1248_7
  have hr := e_1248_7
  simp only [e_1248_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11289)) := by
  have c76896 := (publicBatchWrapper89_copies2403 a h).1
  have c76897 := (publicBatchWrapper89_copies2403 a h).2.1
  have c76898 := (publicBatchWrapper89_copies2403 a h).2.2.1
  have c76899 := (publicBatchWrapper89_copies2403 a h).2.2.2.1
  have c76900 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.1
  have c76901 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_8 := arithEq_of_rows h (row := 1248) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_8
  simp only [← c76896, ← c76897, ← c76898] at e_1248_8
  have e_1248_9 := arithEq_of_rows h (row := 1248) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_9
  simp only [← c76899, ← c76900, k2, ← c76901] at e_1248_9
  have hr := e_1248_9
  simp only [e_1248_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11290)) := by
  have c76902 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.1
  have c76903 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.1
  have c76904 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.1
  have c76905 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.1
  have c76906 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76907 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_10 := arithEq_of_rows h (row := 1248) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_10
  simp only [← c76902, ← c76903, ← c76904] at e_1248_10
  have e_1248_11 := arithEq_of_rows h (row := 1248) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_11
  simp only [← c76905, ← c76906, k2, ← c76907] at e_1248_11
  have hr := e_1248_11
  simp only [e_1248_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11291)) := by
  have c76908 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76909 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76910 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76911 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76912 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76913 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_12 := arithEq_of_rows h (row := 1248) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_12
  simp only [← c76908, ← c76909, ← c76910] at e_1248_12
  have e_1248_13 := arithEq_of_rows h (row := 1248) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_13
  simp only [← c76911, ← c76912, k2, ← c76913] at e_1248_13
  have hr := e_1248_13
  simp only [e_1248_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11292)) := by
  have c76914 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76915 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76916 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76917 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76918 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76919 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_14 := arithEq_of_rows h (row := 1248) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_14
  simp only [← c76914, ← c76915, ← c76916] at e_1248_14
  have e_1248_15 := arithEq_of_rows h (row := 1248) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_15
  simp only [← c76917, ← c76918, k2, ← c76919] at e_1248_15
  have hr := e_1248_15
  simp only [e_1248_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11293)) := by
  have c76920 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76921 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76922 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76923 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76924 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76925 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_16 := arithEq_of_rows h (row := 1248) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_16
  simp only [← c76920, ← c76921, ← c76922] at e_1248_16
  have e_1248_17 := arithEq_of_rows h (row := 1248) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_17
  simp only [← c76923, ← c76924, k2, ← c76925] at e_1248_17
  have hr := e_1248_17
  simp only [e_1248_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1248 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11294)) := by
  have c76926 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76927 := (publicBatchWrapper89_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76928 := (publicBatchWrapper89_copies2404 a h).1
  have c76929 := (publicBatchWrapper89_copies2404 a h).2.1
  have c76930 := (publicBatchWrapper89_copies2404 a h).2.2.1
  have c76931 := (publicBatchWrapper89_copies2404 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1248_18 := arithEq_of_rows h (row := 1248) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_18
  simp only [← c76926, ← c76927, ← c76928] at e_1248_18
  have e_1248_19 := arithEq_of_rows h (row := 1248) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_19
  simp only [← c76929, ← c76930, k2, ← c76931] at e_1248_19
  have hr := e_1248_19
  simp only [e_1248_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11295)) := by
  have c76932 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.1
  have c76933 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.1
  have c76934 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.1
  have c76935 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.1
  have c76936 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.1
  have c76937 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_0 := arithEq_of_rows h (row := 1249) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_0
  simp only [← c76932, ← c76933, ← c76934] at e_1249_0
  have e_1249_1 := arithEq_of_rows h (row := 1249) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_1
  simp only [← c76935, ← c76936, k2, ← c76937] at e_1249_1
  have hr := e_1249_1
  simp only [e_1249_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11296)) := by
  have c76938 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76939 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76940 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76941 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76942 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76943 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_2 := arithEq_of_rows h (row := 1249) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_2
  simp only [← c76938, ← c76939, ← c76940] at e_1249_2
  have e_1249_3 := arithEq_of_rows h (row := 1249) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_3
  simp only [← c76941, ← c76942, k2, ← c76943] at e_1249_3
  have hr := e_1249_3
  simp only [e_1249_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11297)) := by
  have c76944 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76945 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76946 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76947 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76948 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76949 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_4 := arithEq_of_rows h (row := 1249) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_4
  simp only [← c76944, ← c76945, ← c76946] at e_1249_4
  have e_1249_5 := arithEq_of_rows h (row := 1249) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_5
  simp only [← c76947, ← c76948, k2, ← c76949] at e_1249_5
  have hr := e_1249_5
  simp only [e_1249_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11298)) := by
  have c76950 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76951 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76952 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76953 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76954 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76955 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_6 := arithEq_of_rows h (row := 1249) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_6
  simp only [← c76950, ← c76951, ← c76952] at e_1249_6
  have e_1249_7 := arithEq_of_rows h (row := 1249) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_7
  simp only [← c76953, ← c76954, k2, ← c76955] at e_1249_7
  have hr := e_1249_7
  simp only [e_1249_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11299)) := by
  have c76956 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76957 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76958 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76959 := (publicBatchWrapper89_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76960 := (publicBatchWrapper89_copies2405 a h).1
  have c76961 := (publicBatchWrapper89_copies2405 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_8 := arithEq_of_rows h (row := 1249) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_8
  simp only [← c76956, ← c76957, ← c76958] at e_1249_8
  have e_1249_9 := arithEq_of_rows h (row := 1249) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_9
  simp only [← c76959, ← c76960, k2, ← c76961] at e_1249_9
  have hr := e_1249_9
  simp only [e_1249_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11300)) := by
  have c76962 := (publicBatchWrapper89_copies2405 a h).2.2.1
  have c76963 := (publicBatchWrapper89_copies2405 a h).2.2.2.1
  have c76964 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.1
  have c76965 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.1
  have c76966 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.1
  have c76967 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_10 := arithEq_of_rows h (row := 1249) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_10
  simp only [← c76962, ← c76963, ← c76964] at e_1249_10
  have e_1249_11 := arithEq_of_rows h (row := 1249) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_11
  simp only [← c76965, ← c76966, k2, ← c76967] at e_1249_11
  have hr := e_1249_11
  simp only [e_1249_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11301)) := by
  have c76968 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.1
  have c76969 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.1
  have c76970 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76971 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76972 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76973 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_12 := arithEq_of_rows h (row := 1249) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_12
  simp only [← c76968, ← c76969, ← c76970] at e_1249_12
  have e_1249_13 := arithEq_of_rows h (row := 1249) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_13
  simp only [← c76971, ← c76972, k2, ← c76973] at e_1249_13
  have hr := e_1249_13
  simp only [e_1249_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11302)) := by
  have c76974 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76975 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76976 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76977 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76978 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76979 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_14 := arithEq_of_rows h (row := 1249) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_14
  simp only [← c76974, ← c76975, ← c76976] at e_1249_14
  have e_1249_15 := arithEq_of_rows h (row := 1249) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_15
  simp only [← c76977, ← c76978, k2, ← c76979] at e_1249_15
  have hr := e_1249_15
  simp only [e_1249_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11303)) := by
  have c76980 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76981 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76982 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76983 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76984 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76985 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_16 := arithEq_of_rows h (row := 1249) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_16
  simp only [← c76980, ← c76981, ← c76982] at e_1249_16
  have e_1249_17 := arithEq_of_rows h (row := 1249) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_17
  simp only [← c76983, ← c76984, k2, ← c76985] at e_1249_17
  have hr := e_1249_17
  simp only [e_1249_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1249 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11304)) := by
  have c76986 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76987 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76988 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76989 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76990 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76991 := (publicBatchWrapper89_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1249_18 := arithEq_of_rows h (row := 1249) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_18
  simp only [← c76986, ← c76987, ← c76988] at e_1249_18
  have e_1249_19 := arithEq_of_rows h (row := 1249) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_19
  simp only [← c76989, ← c76990, k2, ← c76991] at e_1249_19
  have hr := e_1249_19
  simp only [e_1249_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11305)) := by
  have c76992 := (publicBatchWrapper89_copies2406 a h).1
  have c76993 := (publicBatchWrapper89_copies2406 a h).2.1
  have c76994 := (publicBatchWrapper89_copies2406 a h).2.2.1
  have c76995 := (publicBatchWrapper89_copies2406 a h).2.2.2.1
  have c76996 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.1
  have c76997 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_0 := arithEq_of_rows h (row := 1250) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_0
  simp only [← c76992, ← c76993, ← c76994] at e_1250_0
  have e_1250_1 := arithEq_of_rows h (row := 1250) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_1
  simp only [← c76995, ← c76996, k2, ← c76997] at e_1250_1
  have hr := e_1250_1
  simp only [e_1250_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11306)) := by
  have c76998 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.1
  have c76999 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.1
  have c77000 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.1
  have c77001 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.1
  have c77002 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77003 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_2 := arithEq_of_rows h (row := 1250) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_2
  simp only [← c76998, ← c76999, ← c77000] at e_1250_2
  have e_1250_3 := arithEq_of_rows h (row := 1250) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_3
  simp only [← c77001, ← c77002, k2, ← c77003] at e_1250_3
  have hr := e_1250_3
  simp only [e_1250_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11307)) := by
  have c77004 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77005 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77006 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77007 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77008 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77009 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_4 := arithEq_of_rows h (row := 1250) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_4
  simp only [← c77004, ← c77005, ← c77006] at e_1250_4
  have e_1250_5 := arithEq_of_rows h (row := 1250) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_5
  simp only [← c77007, ← c77008, k2, ← c77009] at e_1250_5
  have hr := e_1250_5
  simp only [e_1250_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11308)) := by
  have c77010 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77011 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77012 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77013 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77014 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77015 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_6 := arithEq_of_rows h (row := 1250) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_6
  simp only [← c77010, ← c77011, ← c77012] at e_1250_6
  have e_1250_7 := arithEq_of_rows h (row := 1250) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_7
  simp only [← c77013, ← c77014, k2, ← c77015] at e_1250_7
  have hr := e_1250_7
  simp only [e_1250_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11309)) := by
  have c77016 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77017 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77018 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77019 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77020 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77021 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_8 := arithEq_of_rows h (row := 1250) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_8
  simp only [← c77016, ← c77017, ← c77018] at e_1250_8
  have e_1250_9 := arithEq_of_rows h (row := 1250) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_9
  simp only [← c77019, ← c77020, k2, ← c77021] at e_1250_9
  have hr := e_1250_9
  simp only [e_1250_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11310)) := by
  have c77022 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77023 := (publicBatchWrapper89_copies2406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77024 := (publicBatchWrapper89_copies2407 a h).1
  have c77025 := (publicBatchWrapper89_copies2407 a h).2.1
  have c77026 := (publicBatchWrapper89_copies2407 a h).2.2.1
  have c77027 := (publicBatchWrapper89_copies2407 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_10 := arithEq_of_rows h (row := 1250) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_10
  simp only [← c77022, ← c77023, ← c77024] at e_1250_10
  have e_1250_11 := arithEq_of_rows h (row := 1250) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_11
  simp only [← c77025, ← c77026, k2, ← c77027] at e_1250_11
  have hr := e_1250_11
  simp only [e_1250_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11311)) := by
  have c77028 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.1
  have c77029 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.1
  have c77030 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.1
  have c77031 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.1
  have c77032 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.1
  have c77033 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_12 := arithEq_of_rows h (row := 1250) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_12
  simp only [← c77028, ← c77029, ← c77030] at e_1250_12
  have e_1250_13 := arithEq_of_rows h (row := 1250) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_13
  simp only [← c77031, ← c77032, k2, ← c77033] at e_1250_13
  have hr := e_1250_13
  simp only [e_1250_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11312)) := by
  have c77034 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77035 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77036 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77037 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77038 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77039 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_14 := arithEq_of_rows h (row := 1250) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_14
  simp only [← c77034, ← c77035, ← c77036] at e_1250_14
  have e_1250_15 := arithEq_of_rows h (row := 1250) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_15
  simp only [← c77037, ← c77038, k2, ← c77039] at e_1250_15
  have hr := e_1250_15
  simp only [e_1250_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11313)) := by
  have c77040 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77041 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77042 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77043 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77044 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77045 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_16 := arithEq_of_rows h (row := 1250) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_16
  simp only [← c77040, ← c77041, ← c77042] at e_1250_16
  have e_1250_17 := arithEq_of_rows h (row := 1250) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_17
  simp only [← c77043, ← c77044, k2, ← c77045] at e_1250_17
  have hr := e_1250_17
  simp only [e_1250_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1250 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11314)) := by
  have c77046 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77047 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77048 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77049 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77050 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77051 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1250_18 := arithEq_of_rows h (row := 1250) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_18
  simp only [← c77046, ← c77047, ← c77048] at e_1250_18
  have e_1250_19 := arithEq_of_rows h (row := 1250) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_19
  simp only [← c77049, ← c77050, k2, ← c77051] at e_1250_19
  have hr := e_1250_19
  simp only [e_1250_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11315)) := by
  have c77052 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77053 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77054 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77055 := (publicBatchWrapper89_copies2407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77056 := (publicBatchWrapper89_copies2408 a h).1
  have c77057 := (publicBatchWrapper89_copies2408 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_0 := arithEq_of_rows h (row := 1251) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_0
  simp only [← c77052, ← c77053, ← c77054] at e_1251_0
  have e_1251_1 := arithEq_of_rows h (row := 1251) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_1
  simp only [← c77055, ← c77056, k2, ← c77057] at e_1251_1
  have hr := e_1251_1
  simp only [e_1251_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11316)) := by
  have c77058 := (publicBatchWrapper89_copies2408 a h).2.2.1
  have c77059 := (publicBatchWrapper89_copies2408 a h).2.2.2.1
  have c77060 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.1
  have c77061 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.1
  have c77062 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.1
  have c77063 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_2 := arithEq_of_rows h (row := 1251) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_2
  simp only [← c77058, ← c77059, ← c77060] at e_1251_2
  have e_1251_3 := arithEq_of_rows h (row := 1251) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_3
  simp only [← c77061, ← c77062, k2, ← c77063] at e_1251_3
  have hr := e_1251_3
  simp only [e_1251_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11317)) := by
  have c77064 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.1
  have c77065 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.1
  have c77066 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77067 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77068 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77069 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_4 := arithEq_of_rows h (row := 1251) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_4
  simp only [← c77064, ← c77065, ← c77066] at e_1251_4
  have e_1251_5 := arithEq_of_rows h (row := 1251) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_5
  simp only [← c77067, ← c77068, k2, ← c77069] at e_1251_5
  have hr := e_1251_5
  simp only [e_1251_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11318)) := by
  have c77070 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77071 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77072 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77073 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77074 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77075 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_6 := arithEq_of_rows h (row := 1251) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_6
  simp only [← c77070, ← c77071, ← c77072] at e_1251_6
  have e_1251_7 := arithEq_of_rows h (row := 1251) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_7
  simp only [← c77073, ← c77074, k2, ← c77075] at e_1251_7
  have hr := e_1251_7
  simp only [e_1251_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11319)) := by
  have c77076 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77077 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77078 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77079 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77080 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77081 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_8 := arithEq_of_rows h (row := 1251) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_8
  simp only [← c77076, ← c77077, ← c77078] at e_1251_8
  have e_1251_9 := arithEq_of_rows h (row := 1251) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_9
  simp only [← c77079, ← c77080, k2, ← c77081] at e_1251_9
  have hr := e_1251_9
  simp only [e_1251_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11320)) := by
  have c77082 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77083 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77084 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77085 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77086 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77087 := (publicBatchWrapper89_copies2408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_10 := arithEq_of_rows h (row := 1251) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_10
  simp only [← c77082, ← c77083, ← c77084] at e_1251_10
  have e_1251_11 := arithEq_of_rows h (row := 1251) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_11
  simp only [← c77085, ← c77086, k2, ← c77087] at e_1251_11
  have hr := e_1251_11
  simp only [e_1251_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11321)) := by
  have c77088 := (publicBatchWrapper89_copies2409 a h).1
  have c77089 := (publicBatchWrapper89_copies2409 a h).2.1
  have c77090 := (publicBatchWrapper89_copies2409 a h).2.2.1
  have c77091 := (publicBatchWrapper89_copies2409 a h).2.2.2.1
  have c77092 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.1
  have c77093 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_12 := arithEq_of_rows h (row := 1251) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_12
  simp only [← c77088, ← c77089, ← c77090] at e_1251_12
  have e_1251_13 := arithEq_of_rows h (row := 1251) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_13
  simp only [← c77091, ← c77092, k2, ← c77093] at e_1251_13
  have hr := e_1251_13
  simp only [e_1251_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11322)) := by
  have c77094 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.1
  have c77095 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.1
  have c77096 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.1
  have c77097 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.1
  have c77098 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77099 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_14 := arithEq_of_rows h (row := 1251) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_14
  simp only [← c77094, ← c77095, ← c77096] at e_1251_14
  have e_1251_15 := arithEq_of_rows h (row := 1251) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_15
  simp only [← c77097, ← c77098, k2, ← c77099] at e_1251_15
  have hr := e_1251_15
  simp only [e_1251_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11323)) := by
  have c77100 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77101 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77102 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77103 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77104 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77105 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_16 := arithEq_of_rows h (row := 1251) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_16
  simp only [← c77100, ← c77101, ← c77102] at e_1251_16
  have e_1251_17 := arithEq_of_rows h (row := 1251) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_17
  simp only [← c77103, ← c77104, k2, ← c77105] at e_1251_17
  have hr := e_1251_17
  simp only [e_1251_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1251 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11324)) := by
  have c77106 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77107 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77108 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77109 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77110 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77111 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1251_18 := arithEq_of_rows h (row := 1251) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_18
  simp only [← c77106, ← c77107, ← c77108] at e_1251_18
  have e_1251_19 := arithEq_of_rows h (row := 1251) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_19
  simp only [← c77109, ← c77110, k2, ← c77111] at e_1251_19
  have hr := e_1251_19
  simp only [e_1251_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11325)) := by
  have c77112 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77113 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77114 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77115 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77116 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77117 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_0 := arithEq_of_rows h (row := 1252) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_0
  simp only [← c77112, ← c77113, ← c77114] at e_1252_0
  have e_1252_1 := arithEq_of_rows h (row := 1252) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_1
  simp only [← c77115, ← c77116, k2, ← c77117] at e_1252_1
  have hr := e_1252_1
  simp only [e_1252_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11326)) := by
  have c77118 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77119 := (publicBatchWrapper89_copies2409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77120 := (publicBatchWrapper89_copies2410 a h).1
  have c77121 := (publicBatchWrapper89_copies2410 a h).2.1
  have c77122 := (publicBatchWrapper89_copies2410 a h).2.2.1
  have c77123 := (publicBatchWrapper89_copies2410 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_2 := arithEq_of_rows h (row := 1252) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_2
  simp only [← c77118, ← c77119, ← c77120] at e_1252_2
  have e_1252_3 := arithEq_of_rows h (row := 1252) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_3
  simp only [← c77121, ← c77122, k2, ← c77123] at e_1252_3
  have hr := e_1252_3
  simp only [e_1252_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11327)) := by
  have c77124 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.1
  have c77125 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.1
  have c77126 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.1
  have c77127 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.1
  have c77128 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.1
  have c77129 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_4 := arithEq_of_rows h (row := 1252) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_4
  simp only [← c77124, ← c77125, ← c77126] at e_1252_4
  have e_1252_5 := arithEq_of_rows h (row := 1252) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_5
  simp only [← c77127, ← c77128, k2, ← c77129] at e_1252_5
  have hr := e_1252_5
  simp only [e_1252_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11328)) := by
  have c77130 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77131 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77132 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77133 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77134 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77135 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_6 := arithEq_of_rows h (row := 1252) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_6
  simp only [← c77130, ← c77131, ← c77132] at e_1252_6
  have e_1252_7 := arithEq_of_rows h (row := 1252) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_7
  simp only [← c77133, ← c77134, k2, ← c77135] at e_1252_7
  have hr := e_1252_7
  simp only [e_1252_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11329)) := by
  have c77136 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77137 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77138 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77139 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77140 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77141 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_8 := arithEq_of_rows h (row := 1252) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_8
  simp only [← c77136, ← c77137, ← c77138] at e_1252_8
  have e_1252_9 := arithEq_of_rows h (row := 1252) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_9
  simp only [← c77139, ← c77140, k2, ← c77141] at e_1252_9
  have hr := e_1252_9
  simp only [e_1252_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11330)) := by
  have c77142 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77143 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77144 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77145 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77146 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77147 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_10 := arithEq_of_rows h (row := 1252) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_10
  simp only [← c77142, ← c77143, ← c77144] at e_1252_10
  have e_1252_11 := arithEq_of_rows h (row := 1252) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_11
  simp only [← c77145, ← c77146, k2, ← c77147] at e_1252_11
  have hr := e_1252_11
  simp only [e_1252_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11331)) := by
  have c77148 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77149 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77150 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77151 := (publicBatchWrapper89_copies2410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77152 := (publicBatchWrapper89_copies2411 a h).1
  have c77153 := (publicBatchWrapper89_copies2411 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_12 := arithEq_of_rows h (row := 1252) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_12
  simp only [← c77148, ← c77149, ← c77150] at e_1252_12
  have e_1252_13 := arithEq_of_rows h (row := 1252) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_13
  simp only [← c77151, ← c77152, k2, ← c77153] at e_1252_13
  have hr := e_1252_13
  simp only [e_1252_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11332)) := by
  have c77154 := (publicBatchWrapper89_copies2411 a h).2.2.1
  have c77155 := (publicBatchWrapper89_copies2411 a h).2.2.2.1
  have c77156 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.1
  have c77157 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.1
  have c77158 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.1
  have c77159 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_14 := arithEq_of_rows h (row := 1252) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_14
  simp only [← c77154, ← c77155, ← c77156] at e_1252_14
  have e_1252_15 := arithEq_of_rows h (row := 1252) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_15
  simp only [← c77157, ← c77158, k2, ← c77159] at e_1252_15
  have hr := e_1252_15
  simp only [e_1252_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11333)) := by
  have c77160 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.1
  have c77161 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.1
  have c77162 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77163 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77164 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77165 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_16 := arithEq_of_rows h (row := 1252) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_16
  simp only [← c77160, ← c77161, ← c77162] at e_1252_16
  have e_1252_17 := arithEq_of_rows h (row := 1252) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_17
  simp only [← c77163, ← c77164, k2, ← c77165] at e_1252_17
  have hr := e_1252_17
  simp only [e_1252_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1252 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11334)) := by
  have c77166 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77167 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77168 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77169 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77170 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77171 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1252_18 := arithEq_of_rows h (row := 1252) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_18
  simp only [← c77166, ← c77167, ← c77168] at e_1252_18
  have e_1252_19 := arithEq_of_rows h (row := 1252) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1252_19
  simp only [← c77169, ← c77170, k2, ← c77171] at e_1252_19
  have hr := e_1252_19
  simp only [e_1252_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11335)) := by
  have c77172 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77173 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77174 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77175 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77176 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77177 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_0 := arithEq_of_rows h (row := 1253) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_0
  simp only [← c77172, ← c77173, ← c77174] at e_1253_0
  have e_1253_1 := arithEq_of_rows h (row := 1253) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_1
  simp only [← c77175, ← c77176, k2, ← c77177] at e_1253_1
  have hr := e_1253_1
  simp only [e_1253_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11336)) := by
  have c77178 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77179 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77180 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77181 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77182 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77183 := (publicBatchWrapper89_copies2411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_2 := arithEq_of_rows h (row := 1253) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_2
  simp only [← c77178, ← c77179, ← c77180] at e_1253_2
  have e_1253_3 := arithEq_of_rows h (row := 1253) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_3
  simp only [← c77181, ← c77182, k2, ← c77183] at e_1253_3
  have hr := e_1253_3
  simp only [e_1253_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11337)) := by
  have c77184 := (publicBatchWrapper89_copies2412 a h).1
  have c77185 := (publicBatchWrapper89_copies2412 a h).2.1
  have c77186 := (publicBatchWrapper89_copies2412 a h).2.2.1
  have c77187 := (publicBatchWrapper89_copies2412 a h).2.2.2.1
  have c77188 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.1
  have c77189 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_4 := arithEq_of_rows h (row := 1253) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_4
  simp only [← c77184, ← c77185, ← c77186] at e_1253_4
  have e_1253_5 := arithEq_of_rows h (row := 1253) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_5
  simp only [← c77187, ← c77188, k2, ← c77189] at e_1253_5
  have hr := e_1253_5
  simp only [e_1253_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11338)) := by
  have c77190 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.1
  have c77191 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.1
  have c77192 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.1
  have c77193 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.1
  have c77194 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77195 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_6 := arithEq_of_rows h (row := 1253) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_6
  simp only [← c77190, ← c77191, ← c77192] at e_1253_6
  have e_1253_7 := arithEq_of_rows h (row := 1253) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_7
  simp only [← c77193, ← c77194, k2, ← c77195] at e_1253_7
  have hr := e_1253_7
  simp only [e_1253_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11339)) := by
  have c77196 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77197 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77198 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77199 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77200 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77201 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_8 := arithEq_of_rows h (row := 1253) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_8
  simp only [← c77196, ← c77197, ← c77198] at e_1253_8
  have e_1253_9 := arithEq_of_rows h (row := 1253) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_9
  simp only [← c77199, ← c77200, k2, ← c77201] at e_1253_9
  have hr := e_1253_9
  simp only [e_1253_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11340)) := by
  have c77202 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77203 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77204 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77205 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77206 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77207 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_10 := arithEq_of_rows h (row := 1253) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_10
  simp only [← c77202, ← c77203, ← c77204] at e_1253_10
  have e_1253_11 := arithEq_of_rows h (row := 1253) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_11
  simp only [← c77205, ← c77206, k2, ← c77207] at e_1253_11
  have hr := e_1253_11
  simp only [e_1253_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11341)) := by
  have c77208 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77209 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77210 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77211 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77212 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77213 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_12 := arithEq_of_rows h (row := 1253) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_12
  simp only [← c77208, ← c77209, ← c77210] at e_1253_12
  have e_1253_13 := arithEq_of_rows h (row := 1253) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_13
  simp only [← c77211, ← c77212, k2, ← c77213] at e_1253_13
  have hr := e_1253_13
  simp only [e_1253_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11342)) := by
  have c77214 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77215 := (publicBatchWrapper89_copies2412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77216 := (publicBatchWrapper89_copies2413 a h).1
  have c77217 := (publicBatchWrapper89_copies2413 a h).2.1
  have c77218 := (publicBatchWrapper89_copies2413 a h).2.2.1
  have c77219 := (publicBatchWrapper89_copies2413 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_14 := arithEq_of_rows h (row := 1253) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_14
  simp only [← c77214, ← c77215, ← c77216] at e_1253_14
  have e_1253_15 := arithEq_of_rows h (row := 1253) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_15
  simp only [← c77217, ← c77218, k2, ← c77219] at e_1253_15
  have hr := e_1253_15
  simp only [e_1253_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11343)) := by
  have c77220 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.1
  have c77221 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.1
  have c77222 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.1
  have c77223 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.1
  have c77224 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.1
  have c77225 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_16 := arithEq_of_rows h (row := 1253) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_16
  simp only [← c77220, ← c77221, ← c77222] at e_1253_16
  have e_1253_17 := arithEq_of_rows h (row := 1253) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_17
  simp only [← c77223, ← c77224, k2, ← c77225] at e_1253_17
  have hr := e_1253_17
  simp only [e_1253_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1253 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11344)) := by
  have c77226 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77227 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77228 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77229 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77230 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77231 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1253_18 := arithEq_of_rows h (row := 1253) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_18
  simp only [← c77226, ← c77227, ← c77228] at e_1253_18
  have e_1253_19 := arithEq_of_rows h (row := 1253) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1253_19
  simp only [← c77229, ← c77230, k2, ← c77231] at e_1253_19
  have hr := e_1253_19
  simp only [e_1253_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11345)) := by
  have c77232 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77233 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77234 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77235 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77236 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77237 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_0 := arithEq_of_rows h (row := 1254) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_0
  simp only [← c77232, ← c77233, ← c77234] at e_1254_0
  have e_1254_1 := arithEq_of_rows h (row := 1254) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_1
  simp only [← c77235, ← c77236, k2, ← c77237] at e_1254_1
  have hr := e_1254_1
  simp only [e_1254_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11346)) := by
  have c77238 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77239 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77240 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77241 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77242 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77243 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_2 := arithEq_of_rows h (row := 1254) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_2
  simp only [← c77238, ← c77239, ← c77240] at e_1254_2
  have e_1254_3 := arithEq_of_rows h (row := 1254) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_3
  simp only [← c77241, ← c77242, k2, ← c77243] at e_1254_3
  have hr := e_1254_3
  simp only [e_1254_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11347)) := by
  have c77244 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77245 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77246 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77247 := (publicBatchWrapper89_copies2413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77248 := (publicBatchWrapper89_copies2414 a h).1
  have c77249 := (publicBatchWrapper89_copies2414 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_4 := arithEq_of_rows h (row := 1254) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_4
  simp only [← c77244, ← c77245, ← c77246] at e_1254_4
  have e_1254_5 := arithEq_of_rows h (row := 1254) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_5
  simp only [← c77247, ← c77248, k2, ← c77249] at e_1254_5
  have hr := e_1254_5
  simp only [e_1254_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11348)) := by
  have c77250 := (publicBatchWrapper89_copies2414 a h).2.2.1
  have c77251 := (publicBatchWrapper89_copies2414 a h).2.2.2.1
  have c77252 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.1
  have c77253 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.1
  have c77254 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.1
  have c77255 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_6 := arithEq_of_rows h (row := 1254) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_6
  simp only [← c77250, ← c77251, ← c77252] at e_1254_6
  have e_1254_7 := arithEq_of_rows h (row := 1254) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_7
  simp only [← c77253, ← c77254, k2, ← c77255] at e_1254_7
  have hr := e_1254_7
  simp only [e_1254_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11349)) := by
  have c77256 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.1
  have c77257 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.1
  have c77258 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77259 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77260 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77261 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_8 := arithEq_of_rows h (row := 1254) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_8
  simp only [← c77256, ← c77257, ← c77258] at e_1254_8
  have e_1254_9 := arithEq_of_rows h (row := 1254) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_9
  simp only [← c77259, ← c77260, k2, ← c77261] at e_1254_9
  have hr := e_1254_9
  simp only [e_1254_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11350)) := by
  have c77262 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77263 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77264 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77265 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77266 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77267 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_10 := arithEq_of_rows h (row := 1254) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_10
  simp only [← c77262, ← c77263, ← c77264] at e_1254_10
  have e_1254_11 := arithEq_of_rows h (row := 1254) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_11
  simp only [← c77265, ← c77266, k2, ← c77267] at e_1254_11
  have hr := e_1254_11
  simp only [e_1254_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11351)) := by
  have c77268 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77269 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77270 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77271 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77272 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77273 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_12 := arithEq_of_rows h (row := 1254) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_12
  simp only [← c77268, ← c77269, ← c77270] at e_1254_12
  have e_1254_13 := arithEq_of_rows h (row := 1254) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_13
  simp only [← c77271, ← c77272, k2, ← c77273] at e_1254_13
  have hr := e_1254_13
  simp only [e_1254_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11352)) := by
  have c77274 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77275 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77276 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77277 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77278 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77279 := (publicBatchWrapper89_copies2414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_14 := arithEq_of_rows h (row := 1254) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_14
  simp only [← c77274, ← c77275, ← c77276] at e_1254_14
  have e_1254_15 := arithEq_of_rows h (row := 1254) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_15
  simp only [← c77277, ← c77278, k2, ← c77279] at e_1254_15
  have hr := e_1254_15
  simp only [e_1254_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11353)) := by
  have c77280 := (publicBatchWrapper89_copies2415 a h).1
  have c77281 := (publicBatchWrapper89_copies2415 a h).2.1
  have c77282 := (publicBatchWrapper89_copies2415 a h).2.2.1
  have c77283 := (publicBatchWrapper89_copies2415 a h).2.2.2.1
  have c77284 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.1
  have c77285 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_16 := arithEq_of_rows h (row := 1254) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_16
  simp only [← c77280, ← c77281, ← c77282] at e_1254_16
  have e_1254_17 := arithEq_of_rows h (row := 1254) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_17
  simp only [← c77283, ← c77284, k2, ← c77285] at e_1254_17
  have hr := e_1254_17
  simp only [e_1254_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1254 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11354)) := by
  have c77286 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.1
  have c77287 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.1
  have c77288 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.1
  have c77289 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.1
  have c77290 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77291 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1254_18 := arithEq_of_rows h (row := 1254) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_18
  simp only [← c77286, ← c77287, ← c77288] at e_1254_18
  have e_1254_19 := arithEq_of_rows h (row := 1254) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1254_19
  simp only [← c77289, ← c77290, k2, ← c77291] at e_1254_19
  have hr := e_1254_19
  simp only [e_1254_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11355)) := by
  have c77292 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77293 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77294 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77295 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77296 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77297 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_0 := arithEq_of_rows h (row := 1255) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_0
  simp only [← c77292, ← c77293, ← c77294] at e_1255_0
  have e_1255_1 := arithEq_of_rows h (row := 1255) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_1
  simp only [← c77295, ← c77296, k2, ← c77297] at e_1255_1
  have hr := e_1255_1
  simp only [e_1255_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11356)) := by
  have c77298 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77299 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77300 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77301 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77302 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77303 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_2 := arithEq_of_rows h (row := 1255) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_2
  simp only [← c77298, ← c77299, ← c77300] at e_1255_2
  have e_1255_3 := arithEq_of_rows h (row := 1255) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_3
  simp only [← c77301, ← c77302, k2, ← c77303] at e_1255_3
  have hr := e_1255_3
  simp only [e_1255_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11357)) := by
  have c77304 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77305 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77306 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77307 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77308 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77309 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_4 := arithEq_of_rows h (row := 1255) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_4
  simp only [← c77304, ← c77305, ← c77306] at e_1255_4
  have e_1255_5 := arithEq_of_rows h (row := 1255) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_5
  simp only [← c77307, ← c77308, k2, ← c77309] at e_1255_5
  have hr := e_1255_5
  simp only [e_1255_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11358)) := by
  have c77310 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77311 := (publicBatchWrapper89_copies2415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77312 := (publicBatchWrapper89_copies2416 a h).1
  have c77313 := (publicBatchWrapper89_copies2416 a h).2.1
  have c77314 := (publicBatchWrapper89_copies2416 a h).2.2.1
  have c77315 := (publicBatchWrapper89_copies2416 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_6 := arithEq_of_rows h (row := 1255) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_6
  simp only [← c77310, ← c77311, ← c77312] at e_1255_6
  have e_1255_7 := arithEq_of_rows h (row := 1255) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_7
  simp only [← c77313, ← c77314, k2, ← c77315] at e_1255_7
  have hr := e_1255_7
  simp only [e_1255_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11359)) := by
  have c77316 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.1
  have c77317 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.1
  have c77318 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.1
  have c77319 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.1
  have c77320 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.1
  have c77321 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_8 := arithEq_of_rows h (row := 1255) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_8
  simp only [← c77316, ← c77317, ← c77318] at e_1255_8
  have e_1255_9 := arithEq_of_rows h (row := 1255) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_9
  simp only [← c77319, ← c77320, k2, ← c77321] at e_1255_9
  have hr := e_1255_9
  simp only [e_1255_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11360)) := by
  have c77322 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77323 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77324 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77325 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77326 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77327 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_10 := arithEq_of_rows h (row := 1255) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_10
  simp only [← c77322, ← c77323, ← c77324] at e_1255_10
  have e_1255_11 := arithEq_of_rows h (row := 1255) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_11
  simp only [← c77325, ← c77326, k2, ← c77327] at e_1255_11
  have hr := e_1255_11
  simp only [e_1255_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11361)) := by
  have c77328 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77329 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77330 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77331 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77332 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77333 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_12 := arithEq_of_rows h (row := 1255) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_12
  simp only [← c77328, ← c77329, ← c77330] at e_1255_12
  have e_1255_13 := arithEq_of_rows h (row := 1255) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_13
  simp only [← c77331, ← c77332, k2, ← c77333] at e_1255_13
  have hr := e_1255_13
  simp only [e_1255_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11362)) := by
  have c77334 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77335 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77336 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77337 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77338 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77339 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_14 := arithEq_of_rows h (row := 1255) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_14
  simp only [← c77334, ← c77335, ← c77336] at e_1255_14
  have e_1255_15 := arithEq_of_rows h (row := 1255) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_15
  simp only [← c77337, ← c77338, k2, ← c77339] at e_1255_15
  have hr := e_1255_15
  simp only [e_1255_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11363)) := by
  have c77340 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77341 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77342 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77343 := (publicBatchWrapper89_copies2416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77344 := (publicBatchWrapper89_copies2417 a h).1
  have c77345 := (publicBatchWrapper89_copies2417 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_16 := arithEq_of_rows h (row := 1255) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_16
  simp only [← c77340, ← c77341, ← c77342] at e_1255_16
  have e_1255_17 := arithEq_of_rows h (row := 1255) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_17
  simp only [← c77343, ← c77344, k2, ← c77345] at e_1255_17
  have hr := e_1255_17
  simp only [e_1255_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1255 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11364)) := by
  have c77346 := (publicBatchWrapper89_copies2417 a h).2.2.1
  have c77347 := (publicBatchWrapper89_copies2417 a h).2.2.2.1
  have c77348 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.1
  have c77349 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.1
  have c77350 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.1
  have c77351 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1255_18 := arithEq_of_rows h (row := 1255) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_18
  simp only [← c77346, ← c77347, ← c77348] at e_1255_18
  have e_1255_19 := arithEq_of_rows h (row := 1255) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1255_19
  simp only [← c77349, ← c77350, k2, ← c77351] at e_1255_19
  have hr := e_1255_19
  simp only [e_1255_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11365)) := by
  have c77352 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.1
  have c77353 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.1
  have c77354 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77355 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77356 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77357 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_0 := arithEq_of_rows h (row := 1256) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_0
  simp only [← c77352, ← c77353, ← c77354] at e_1256_0
  have e_1256_1 := arithEq_of_rows h (row := 1256) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_1
  simp only [← c77355, ← c77356, k2, ← c77357] at e_1256_1
  have hr := e_1256_1
  simp only [e_1256_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11366)) := by
  have c77358 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77359 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77360 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77361 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77362 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77363 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_2 := arithEq_of_rows h (row := 1256) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_2
  simp only [← c77358, ← c77359, ← c77360] at e_1256_2
  have e_1256_3 := arithEq_of_rows h (row := 1256) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_3
  simp only [← c77361, ← c77362, k2, ← c77363] at e_1256_3
  have hr := e_1256_3
  simp only [e_1256_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11367)) := by
  have c77364 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77365 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77366 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77367 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77368 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77369 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_4 := arithEq_of_rows h (row := 1256) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_4
  simp only [← c77364, ← c77365, ← c77366] at e_1256_4
  have e_1256_5 := arithEq_of_rows h (row := 1256) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_5
  simp only [← c77367, ← c77368, k2, ← c77369] at e_1256_5
  have hr := e_1256_5
  simp only [e_1256_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11368)) := by
  have c77370 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77371 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77372 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77373 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77374 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77375 := (publicBatchWrapper89_copies2417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_6 := arithEq_of_rows h (row := 1256) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_6
  simp only [← c77370, ← c77371, ← c77372] at e_1256_6
  have e_1256_7 := arithEq_of_rows h (row := 1256) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_7
  simp only [← c77373, ← c77374, k2, ← c77375] at e_1256_7
  have hr := e_1256_7
  simp only [e_1256_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11369)) := by
  have c77376 := (publicBatchWrapper89_copies2418 a h).1
  have c77377 := (publicBatchWrapper89_copies2418 a h).2.1
  have c77378 := (publicBatchWrapper89_copies2418 a h).2.2.1
  have c77379 := (publicBatchWrapper89_copies2418 a h).2.2.2.1
  have c77380 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.1
  have c77381 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_8 := arithEq_of_rows h (row := 1256) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_8
  simp only [← c77376, ← c77377, ← c77378] at e_1256_8
  have e_1256_9 := arithEq_of_rows h (row := 1256) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_9
  simp only [← c77379, ← c77380, k2, ← c77381] at e_1256_9
  have hr := e_1256_9
  simp only [e_1256_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11410)) := by
  have c77382 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.1
  have c77383 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.1
  have c77384 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.1
  have c77385 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.1
  have c77386 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77387 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_10 := arithEq_of_rows h (row := 1256) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_10
  simp only [← c77382, ← c77383, ← c77384] at e_1256_10
  have e_1256_11 := arithEq_of_rows h (row := 1256) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_11
  simp only [← c77385, ← c77386, k2, ← c77387] at e_1256_11
  have hr := e_1256_11
  simp only [e_1256_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11411)) := by
  have c77388 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77389 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77390 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77391 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77392 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77393 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_12 := arithEq_of_rows h (row := 1256) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_12
  simp only [← c77388, ← c77389, ← c77390] at e_1256_12
  have e_1256_13 := arithEq_of_rows h (row := 1256) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_13
  simp only [← c77391, ← c77392, k2, ← c77393] at e_1256_13
  have hr := e_1256_13
  simp only [e_1256_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11412)) := by
  have c77394 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77395 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77396 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77397 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77398 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77399 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_14 := arithEq_of_rows h (row := 1256) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_14
  simp only [← c77394, ← c77395, ← c77396] at e_1256_14
  have e_1256_15 := arithEq_of_rows h (row := 1256) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_15
  simp only [← c77397, ← c77398, k2, ← c77399] at e_1256_15
  have hr := e_1256_15
  simp only [e_1256_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11413)) := by
  have c77400 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77401 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77402 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77403 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77404 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77405 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_16 := arithEq_of_rows h (row := 1256) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_16
  simp only [← c77400, ← c77401, ← c77402] at e_1256_16
  have e_1256_17 := arithEq_of_rows h (row := 1256) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_17
  simp only [← c77403, ← c77404, k2, ← c77405] at e_1256_17
  have hr := e_1256_17
  simp only [e_1256_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1256 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11414)) := by
  have c77406 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77407 := (publicBatchWrapper89_copies2418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77408 := (publicBatchWrapper89_copies2419 a h).1
  have c77409 := (publicBatchWrapper89_copies2419 a h).2.1
  have c77410 := (publicBatchWrapper89_copies2419 a h).2.2.1
  have c77411 := (publicBatchWrapper89_copies2419 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1256_18 := arithEq_of_rows h (row := 1256) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_18
  simp only [← c77406, ← c77407, ← c77408] at e_1256_18
  have e_1256_19 := arithEq_of_rows h (row := 1256) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1256_19
  simp only [← c77409, ← c77410, k2, ← c77411] at e_1256_19
  have hr := e_1256_19
  simp only [e_1256_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11415)) := by
  have c77412 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.1
  have c77413 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.1
  have c77414 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.1
  have c77415 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.1
  have c77416 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.1
  have c77417 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_0 := arithEq_of_rows h (row := 1257) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_0
  simp only [← c77412, ← c77413, ← c77414] at e_1257_0
  have e_1257_1 := arithEq_of_rows h (row := 1257) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_1
  simp only [← c77415, ← c77416, k2, ← c77417] at e_1257_1
  have hr := e_1257_1
  simp only [e_1257_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11416)) := by
  have c77418 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77419 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77420 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77421 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77422 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77423 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_2 := arithEq_of_rows h (row := 1257) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_2
  simp only [← c77418, ← c77419, ← c77420] at e_1257_2
  have e_1257_3 := arithEq_of_rows h (row := 1257) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_3
  simp only [← c77421, ← c77422, k2, ← c77423] at e_1257_3
  have hr := e_1257_3
  simp only [e_1257_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11417)) := by
  have c77424 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77425 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77426 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77427 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77428 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77429 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_4 := arithEq_of_rows h (row := 1257) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_4
  simp only [← c77424, ← c77425, ← c77426] at e_1257_4
  have e_1257_5 := arithEq_of_rows h (row := 1257) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_5
  simp only [← c77427, ← c77428, k2, ← c77429] at e_1257_5
  have hr := e_1257_5
  simp only [e_1257_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11418)) := by
  have c77430 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77431 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77432 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77433 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77434 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77435 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_6 := arithEq_of_rows h (row := 1257) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_6
  simp only [← c77430, ← c77431, ← c77432] at e_1257_6
  have e_1257_7 := arithEq_of_rows h (row := 1257) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_7
  simp only [← c77433, ← c77434, k2, ← c77435] at e_1257_7
  have hr := e_1257_7
  simp only [e_1257_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11419)) := by
  have c77436 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77437 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77438 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77439 := (publicBatchWrapper89_copies2419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77440 := (publicBatchWrapper89_copies2420 a h).1
  have c77441 := (publicBatchWrapper89_copies2420 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_8 := arithEq_of_rows h (row := 1257) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_8
  simp only [← c77436, ← c77437, ← c77438] at e_1257_8
  have e_1257_9 := arithEq_of_rows h (row := 1257) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_9
  simp only [← c77439, ← c77440, k2, ← c77441] at e_1257_9
  have hr := e_1257_9
  simp only [e_1257_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11420)) := by
  have c77442 := (publicBatchWrapper89_copies2420 a h).2.2.1
  have c77443 := (publicBatchWrapper89_copies2420 a h).2.2.2.1
  have c77444 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.1
  have c77445 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.1
  have c77446 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.1
  have c77447 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_10 := arithEq_of_rows h (row := 1257) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_10
  simp only [← c77442, ← c77443, ← c77444] at e_1257_10
  have e_1257_11 := arithEq_of_rows h (row := 1257) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_11
  simp only [← c77445, ← c77446, k2, ← c77447] at e_1257_11
  have hr := e_1257_11
  simp only [e_1257_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11421)) := by
  have c77448 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.1
  have c77449 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.1
  have c77450 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77451 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77452 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77453 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_12 := arithEq_of_rows h (row := 1257) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_12
  simp only [← c77448, ← c77449, ← c77450] at e_1257_12
  have e_1257_13 := arithEq_of_rows h (row := 1257) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_13
  simp only [← c77451, ← c77452, k2, ← c77453] at e_1257_13
  have hr := e_1257_13
  simp only [e_1257_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11422)) := by
  have c77454 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77455 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77456 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77457 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77458 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77459 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_14 := arithEq_of_rows h (row := 1257) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_14
  simp only [← c77454, ← c77455, ← c77456] at e_1257_14
  have e_1257_15 := arithEq_of_rows h (row := 1257) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_15
  simp only [← c77457, ← c77458, k2, ← c77459] at e_1257_15
  have hr := e_1257_15
  simp only [e_1257_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11423)) := by
  have c77460 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77461 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77462 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77463 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77464 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77465 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_16 := arithEq_of_rows h (row := 1257) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_16
  simp only [← c77460, ← c77461, ← c77462] at e_1257_16
  have e_1257_17 := arithEq_of_rows h (row := 1257) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_17
  simp only [← c77463, ← c77464, k2, ← c77465] at e_1257_17
  have hr := e_1257_17
  simp only [e_1257_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1257 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11424)) := by
  have c77466 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77467 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77468 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77469 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77470 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77471 := (publicBatchWrapper89_copies2420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1257_18 := arithEq_of_rows h (row := 1257) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_18
  simp only [← c77466, ← c77467, ← c77468] at e_1257_18
  have e_1257_19 := arithEq_of_rows h (row := 1257) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1257_19
  simp only [← c77469, ← c77470, k2, ← c77471] at e_1257_19
  have hr := e_1257_19
  simp only [e_1257_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11425)) := by
  have c77472 := (publicBatchWrapper89_copies2421 a h).1
  have c77473 := (publicBatchWrapper89_copies2421 a h).2.1
  have c77474 := (publicBatchWrapper89_copies2421 a h).2.2.1
  have c77475 := (publicBatchWrapper89_copies2421 a h).2.2.2.1
  have c77476 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.1
  have c77477 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_0 := arithEq_of_rows h (row := 1258) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_0
  simp only [← c77472, ← c77473, ← c77474] at e_1258_0
  have e_1258_1 := arithEq_of_rows h (row := 1258) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_1
  simp only [← c77475, ← c77476, k2, ← c77477] at e_1258_1
  have hr := e_1258_1
  simp only [e_1258_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11426)) := by
  have c77478 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.1
  have c77479 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.1
  have c77480 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.1
  have c77481 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.1
  have c77482 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77483 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_2 := arithEq_of_rows h (row := 1258) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_2
  simp only [← c77478, ← c77479, ← c77480] at e_1258_2
  have e_1258_3 := arithEq_of_rows h (row := 1258) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_3
  simp only [← c77481, ← c77482, k2, ← c77483] at e_1258_3
  have hr := e_1258_3
  simp only [e_1258_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11427)) := by
  have c77484 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77485 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77486 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77487 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77488 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77489 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_4 := arithEq_of_rows h (row := 1258) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_4
  simp only [← c77484, ← c77485, ← c77486] at e_1258_4
  have e_1258_5 := arithEq_of_rows h (row := 1258) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_5
  simp only [← c77487, ← c77488, k2, ← c77489] at e_1258_5
  have hr := e_1258_5
  simp only [e_1258_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11428)) := by
  have c77490 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77491 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77492 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77493 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77494 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77495 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_6 := arithEq_of_rows h (row := 1258) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_6
  simp only [← c77490, ← c77491, ← c77492] at e_1258_6
  have e_1258_7 := arithEq_of_rows h (row := 1258) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_7
  simp only [← c77493, ← c77494, k2, ← c77495] at e_1258_7
  have hr := e_1258_7
  simp only [e_1258_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11429)) := by
  have c77496 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77497 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77498 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77499 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77500 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77501 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_8 := arithEq_of_rows h (row := 1258) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_8
  simp only [← c77496, ← c77497, ← c77498] at e_1258_8
  have e_1258_9 := arithEq_of_rows h (row := 1258) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_9
  simp only [← c77499, ← c77500, k2, ← c77501] at e_1258_9
  have hr := e_1258_9
  simp only [e_1258_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11430)) := by
  have c77502 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77503 := (publicBatchWrapper89_copies2421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77504 := (publicBatchWrapper89_copies2422 a h).1
  have c77505 := (publicBatchWrapper89_copies2422 a h).2.1
  have c77506 := (publicBatchWrapper89_copies2422 a h).2.2.1
  have c77507 := (publicBatchWrapper89_copies2422 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_10 := arithEq_of_rows h (row := 1258) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_10
  simp only [← c77502, ← c77503, ← c77504] at e_1258_10
  have e_1258_11 := arithEq_of_rows h (row := 1258) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_11
  simp only [← c77505, ← c77506, k2, ← c77507] at e_1258_11
  have hr := e_1258_11
  simp only [e_1258_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11431)) := by
  have c77508 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.1
  have c77509 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.1
  have c77510 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.1
  have c77511 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.1
  have c77512 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.1
  have c77513 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_12 := arithEq_of_rows h (row := 1258) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_12
  simp only [← c77508, ← c77509, ← c77510] at e_1258_12
  have e_1258_13 := arithEq_of_rows h (row := 1258) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_13
  simp only [← c77511, ← c77512, k2, ← c77513] at e_1258_13
  have hr := e_1258_13
  simp only [e_1258_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11432)) := by
  have c77514 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77515 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77516 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77517 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77518 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77519 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_14 := arithEq_of_rows h (row := 1258) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_14
  simp only [← c77514, ← c77515, ← c77516] at e_1258_14
  have e_1258_15 := arithEq_of_rows h (row := 1258) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_15
  simp only [← c77517, ← c77518, k2, ← c77519] at e_1258_15
  have hr := e_1258_15
  simp only [e_1258_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11433)) := by
  have c77520 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77521 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77522 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77523 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77524 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77525 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_16 := arithEq_of_rows h (row := 1258) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_16
  simp only [← c77520, ← c77521, ← c77522] at e_1258_16
  have e_1258_17 := arithEq_of_rows h (row := 1258) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_17
  simp only [← c77523, ← c77524, k2, ← c77525] at e_1258_17
  have hr := e_1258_17
  simp only [e_1258_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1258 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11434)) := by
  have c77526 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77527 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77528 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77529 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77530 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77531 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1258_18 := arithEq_of_rows h (row := 1258) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_18
  simp only [← c77526, ← c77527, ← c77528] at e_1258_18
  have e_1258_19 := arithEq_of_rows h (row := 1258) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1258_19
  simp only [← c77529, ← c77530, k2, ← c77531] at e_1258_19
  have hr := e_1258_19
  simp only [e_1258_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11435)) := by
  have c77532 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77533 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77534 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77535 := (publicBatchWrapper89_copies2422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77536 := (publicBatchWrapper89_copies2423 a h).1
  have c77537 := (publicBatchWrapper89_copies2423 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_0 := arithEq_of_rows h (row := 1259) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_0
  simp only [← c77532, ← c77533, ← c77534] at e_1259_0
  have e_1259_1 := arithEq_of_rows h (row := 1259) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_1
  simp only [← c77535, ← c77536, k2, ← c77537] at e_1259_1
  have hr := e_1259_1
  simp only [e_1259_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11436)) := by
  have c77538 := (publicBatchWrapper89_copies2423 a h).2.2.1
  have c77539 := (publicBatchWrapper89_copies2423 a h).2.2.2.1
  have c77540 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.1
  have c77541 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.1
  have c77542 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.1
  have c77543 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_2 := arithEq_of_rows h (row := 1259) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_2
  simp only [← c77538, ← c77539, ← c77540] at e_1259_2
  have e_1259_3 := arithEq_of_rows h (row := 1259) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_3
  simp only [← c77541, ← c77542, k2, ← c77543] at e_1259_3
  have hr := e_1259_3
  simp only [e_1259_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11437)) := by
  have c77544 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.1
  have c77545 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.1
  have c77546 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77547 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77548 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77549 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_4 := arithEq_of_rows h (row := 1259) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_4
  simp only [← c77544, ← c77545, ← c77546] at e_1259_4
  have e_1259_5 := arithEq_of_rows h (row := 1259) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_5
  simp only [← c77547, ← c77548, k2, ← c77549] at e_1259_5
  have hr := e_1259_5
  simp only [e_1259_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11438)) := by
  have c77550 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77551 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77552 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77553 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77554 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77555 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_6 := arithEq_of_rows h (row := 1259) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_6
  simp only [← c77550, ← c77551, ← c77552] at e_1259_6
  have e_1259_7 := arithEq_of_rows h (row := 1259) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_7
  simp only [← c77553, ← c77554, k2, ← c77555] at e_1259_7
  have hr := e_1259_7
  simp only [e_1259_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11439)) := by
  have c77556 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77557 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77558 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77559 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77560 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77561 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_8 := arithEq_of_rows h (row := 1259) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_8
  simp only [← c77556, ← c77557, ← c77558] at e_1259_8
  have e_1259_9 := arithEq_of_rows h (row := 1259) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_9
  simp only [← c77559, ← c77560, k2, ← c77561] at e_1259_9
  have hr := e_1259_9
  simp only [e_1259_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11440)) := by
  have c77562 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77563 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77564 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77565 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77566 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77567 := (publicBatchWrapper89_copies2423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_10 := arithEq_of_rows h (row := 1259) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_10
  simp only [← c77562, ← c77563, ← c77564] at e_1259_10
  have e_1259_11 := arithEq_of_rows h (row := 1259) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_11
  simp only [← c77565, ← c77566, k2, ← c77567] at e_1259_11
  have hr := e_1259_11
  simp only [e_1259_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11441)) := by
  have c77568 := (publicBatchWrapper89_copies2424 a h).1
  have c77569 := (publicBatchWrapper89_copies2424 a h).2.1
  have c77570 := (publicBatchWrapper89_copies2424 a h).2.2.1
  have c77571 := (publicBatchWrapper89_copies2424 a h).2.2.2.1
  have c77572 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.1
  have c77573 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_12 := arithEq_of_rows h (row := 1259) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_12
  simp only [← c77568, ← c77569, ← c77570] at e_1259_12
  have e_1259_13 := arithEq_of_rows h (row := 1259) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_13
  simp only [← c77571, ← c77572, k2, ← c77573] at e_1259_13
  have hr := e_1259_13
  simp only [e_1259_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11442)) := by
  have c77574 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.1
  have c77575 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.1
  have c77576 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.1
  have c77577 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.1
  have c77578 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77579 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_14 := arithEq_of_rows h (row := 1259) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_14
  simp only [← c77574, ← c77575, ← c77576] at e_1259_14
  have e_1259_15 := arithEq_of_rows h (row := 1259) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_15
  simp only [← c77577, ← c77578, k2, ← c77579] at e_1259_15
  have hr := e_1259_15
  simp only [e_1259_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11443)) := by
  have c77580 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77581 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77582 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77583 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77584 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77585 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_16 := arithEq_of_rows h (row := 1259) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_16
  simp only [← c77580, ← c77581, ← c77582] at e_1259_16
  have e_1259_17 := arithEq_of_rows h (row := 1259) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_17
  simp only [← c77583, ← c77584, k2, ← c77585] at e_1259_17
  have hr := e_1259_17
  simp only [e_1259_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1259 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11444)) := by
  have c77586 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77587 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77588 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77589 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77590 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77591 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1259_18 := arithEq_of_rows h (row := 1259) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_18
  simp only [← c77586, ← c77587, ← c77588] at e_1259_18
  have e_1259_19 := arithEq_of_rows h (row := 1259) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1259_19
  simp only [← c77589, ← c77590, k2, ← c77591] at e_1259_19
  have hr := e_1259_19
  simp only [e_1259_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11445)) := by
  have c77592 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77593 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77594 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77595 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77596 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77597 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_0 := arithEq_of_rows h (row := 1260) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_0
  simp only [← c77592, ← c77593, ← c77594] at e_1260_0
  have e_1260_1 := arithEq_of_rows h (row := 1260) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_1
  simp only [← c77595, ← c77596, k2, ← c77597] at e_1260_1
  have hr := e_1260_1
  simp only [e_1260_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11446)) := by
  have c77598 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77599 := (publicBatchWrapper89_copies2424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77600 := (publicBatchWrapper89_copies2425 a h).1
  have c77601 := (publicBatchWrapper89_copies2425 a h).2.1
  have c77602 := (publicBatchWrapper89_copies2425 a h).2.2.1
  have c77603 := (publicBatchWrapper89_copies2425 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_2 := arithEq_of_rows h (row := 1260) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_2
  simp only [← c77598, ← c77599, ← c77600] at e_1260_2
  have e_1260_3 := arithEq_of_rows h (row := 1260) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_3
  simp only [← c77601, ← c77602, k2, ← c77603] at e_1260_3
  have hr := e_1260_3
  simp only [e_1260_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11447)) := by
  have c77604 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.1
  have c77605 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.1
  have c77606 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.1
  have c77607 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.1
  have c77608 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.1
  have c77609 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_4 := arithEq_of_rows h (row := 1260) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_4
  simp only [← c77604, ← c77605, ← c77606] at e_1260_4
  have e_1260_5 := arithEq_of_rows h (row := 1260) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_5
  simp only [← c77607, ← c77608, k2, ← c77609] at e_1260_5
  have hr := e_1260_5
  simp only [e_1260_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11448)) := by
  have c77610 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77611 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77612 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77613 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77614 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77615 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_6 := arithEq_of_rows h (row := 1260) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_6
  simp only [← c77610, ← c77611, ← c77612] at e_1260_6
  have e_1260_7 := arithEq_of_rows h (row := 1260) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_7
  simp only [← c77613, ← c77614, k2, ← c77615] at e_1260_7
  have hr := e_1260_7
  simp only [e_1260_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11449)) := by
  have c77616 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77617 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77618 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77619 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77620 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77621 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_8 := arithEq_of_rows h (row := 1260) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_8
  simp only [← c77616, ← c77617, ← c77618] at e_1260_8
  have e_1260_9 := arithEq_of_rows h (row := 1260) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_9
  simp only [← c77619, ← c77620, k2, ← c77621] at e_1260_9
  have hr := e_1260_9
  simp only [e_1260_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11450)) := by
  have c77622 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77623 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77624 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77625 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77626 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77627 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_10 := arithEq_of_rows h (row := 1260) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_10
  simp only [← c77622, ← c77623, ← c77624] at e_1260_10
  have e_1260_11 := arithEq_of_rows h (row := 1260) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_11
  simp only [← c77625, ← c77626, k2, ← c77627] at e_1260_11
  have hr := e_1260_11
  simp only [e_1260_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11451)) := by
  have c77628 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77629 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77630 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77631 := (publicBatchWrapper89_copies2425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77632 := (publicBatchWrapper89_copies2426 a h).1
  have c77633 := (publicBatchWrapper89_copies2426 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_12 := arithEq_of_rows h (row := 1260) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_12
  simp only [← c77628, ← c77629, ← c77630] at e_1260_12
  have e_1260_13 := arithEq_of_rows h (row := 1260) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_13
  simp only [← c77631, ← c77632, k2, ← c77633] at e_1260_13
  have hr := e_1260_13
  simp only [e_1260_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11452)) := by
  have c77634 := (publicBatchWrapper89_copies2426 a h).2.2.1
  have c77635 := (publicBatchWrapper89_copies2426 a h).2.2.2.1
  have c77636 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.1
  have c77637 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.1
  have c77638 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.1
  have c77639 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_14 := arithEq_of_rows h (row := 1260) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_14
  simp only [← c77634, ← c77635, ← c77636] at e_1260_14
  have e_1260_15 := arithEq_of_rows h (row := 1260) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_15
  simp only [← c77637, ← c77638, k2, ← c77639] at e_1260_15
  have hr := e_1260_15
  simp only [e_1260_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11453)) := by
  have c77640 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.1
  have c77641 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.1
  have c77642 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77643 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77644 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77645 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_16 := arithEq_of_rows h (row := 1260) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_16
  simp only [← c77640, ← c77641, ← c77642] at e_1260_16
  have e_1260_17 := arithEq_of_rows h (row := 1260) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_17
  simp only [← c77643, ← c77644, k2, ← c77645] at e_1260_17
  have hr := e_1260_17
  simp only [e_1260_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1260 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11454)) := by
  have c77646 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77647 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77648 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77649 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77650 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77651 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1260_18 := arithEq_of_rows h (row := 1260) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_18
  simp only [← c77646, ← c77647, ← c77648] at e_1260_18
  have e_1260_19 := arithEq_of_rows h (row := 1260) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1260_19
  simp only [← c77649, ← c77650, k2, ← c77651] at e_1260_19
  have hr := e_1260_19
  simp only [e_1260_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11455)) := by
  have c77652 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77653 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77654 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77655 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77656 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77657 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_0 := arithEq_of_rows h (row := 1261) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_0
  simp only [← c77652, ← c77653, ← c77654] at e_1261_0
  have e_1261_1 := arithEq_of_rows h (row := 1261) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_1
  simp only [← c77655, ← c77656, k2, ← c77657] at e_1261_1
  have hr := e_1261_1
  simp only [e_1261_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11456)) := by
  have c77658 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77659 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77660 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77661 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77662 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77663 := (publicBatchWrapper89_copies2426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_2 := arithEq_of_rows h (row := 1261) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_2
  simp only [← c77658, ← c77659, ← c77660] at e_1261_2
  have e_1261_3 := arithEq_of_rows h (row := 1261) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_3
  simp only [← c77661, ← c77662, k2, ← c77663] at e_1261_3
  have hr := e_1261_3
  simp only [e_1261_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11457)) := by
  have c77664 := (publicBatchWrapper89_copies2427 a h).1
  have c77665 := (publicBatchWrapper89_copies2427 a h).2.1
  have c77666 := (publicBatchWrapper89_copies2427 a h).2.2.1
  have c77667 := (publicBatchWrapper89_copies2427 a h).2.2.2.1
  have c77668 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.1
  have c77669 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_4 := arithEq_of_rows h (row := 1261) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_4
  simp only [← c77664, ← c77665, ← c77666] at e_1261_4
  have e_1261_5 := arithEq_of_rows h (row := 1261) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_5
  simp only [← c77667, ← c77668, k2, ← c77669] at e_1261_5
  have hr := e_1261_5
  simp only [e_1261_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11458)) := by
  have c77670 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.1
  have c77671 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.1
  have c77672 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.1
  have c77673 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.1
  have c77674 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77675 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_6 := arithEq_of_rows h (row := 1261) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_6
  simp only [← c77670, ← c77671, ← c77672] at e_1261_6
  have e_1261_7 := arithEq_of_rows h (row := 1261) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_7
  simp only [← c77673, ← c77674, k2, ← c77675] at e_1261_7
  have hr := e_1261_7
  simp only [e_1261_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11459)) := by
  have c77676 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77677 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77678 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77679 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77680 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77681 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_8 := arithEq_of_rows h (row := 1261) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_8
  simp only [← c77676, ← c77677, ← c77678] at e_1261_8
  have e_1261_9 := arithEq_of_rows h (row := 1261) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_9
  simp only [← c77679, ← c77680, k2, ← c77681] at e_1261_9
  have hr := e_1261_9
  simp only [e_1261_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11460)) := by
  have c77682 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77683 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77684 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77685 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77686 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77687 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_10 := arithEq_of_rows h (row := 1261) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_10
  simp only [← c77682, ← c77683, ← c77684] at e_1261_10
  have e_1261_11 := arithEq_of_rows h (row := 1261) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_11
  simp only [← c77685, ← c77686, k2, ← c77687] at e_1261_11
  have hr := e_1261_11
  simp only [e_1261_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11461)) := by
  have c77688 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77689 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77690 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77691 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77692 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77693 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_12 := arithEq_of_rows h (row := 1261) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_12
  simp only [← c77688, ← c77689, ← c77690] at e_1261_12
  have e_1261_13 := arithEq_of_rows h (row := 1261) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_13
  simp only [← c77691, ← c77692, k2, ← c77693] at e_1261_13
  have hr := e_1261_13
  simp only [e_1261_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11462)) := by
  have c77694 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77695 := (publicBatchWrapper89_copies2427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77696 := (publicBatchWrapper89_copies2428 a h).1
  have c77697 := (publicBatchWrapper89_copies2428 a h).2.1
  have c77698 := (publicBatchWrapper89_copies2428 a h).2.2.1
  have c77699 := (publicBatchWrapper89_copies2428 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_14 := arithEq_of_rows h (row := 1261) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_14
  simp only [← c77694, ← c77695, ← c77696] at e_1261_14
  have e_1261_15 := arithEq_of_rows h (row := 1261) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_15
  simp only [← c77697, ← c77698, k2, ← c77699] at e_1261_15
  have hr := e_1261_15
  simp only [e_1261_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11463)) := by
  have c77700 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.1
  have c77701 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.1
  have c77702 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.1
  have c77703 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.1
  have c77704 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.1
  have c77705 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_16 := arithEq_of_rows h (row := 1261) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_16
  simp only [← c77700, ← c77701, ← c77702] at e_1261_16
  have e_1261_17 := arithEq_of_rows h (row := 1261) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_17
  simp only [← c77703, ← c77704, k2, ← c77705] at e_1261_17
  have hr := e_1261_17
  simp only [e_1261_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1261 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11464)) := by
  have c77706 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77707 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77708 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77709 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77710 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77711 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1261_18 := arithEq_of_rows h (row := 1261) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_18
  simp only [← c77706, ← c77707, ← c77708] at e_1261_18
  have e_1261_19 := arithEq_of_rows h (row := 1261) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1261_19
  simp only [← c77709, ← c77710, k2, ← c77711] at e_1261_19
  have hr := e_1261_19
  simp only [e_1261_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11465)) := by
  have c77712 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77713 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77714 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77715 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77716 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77717 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_0 := arithEq_of_rows h (row := 1262) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_0
  simp only [← c77712, ← c77713, ← c77714] at e_1262_0
  have e_1262_1 := arithEq_of_rows h (row := 1262) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_1
  simp only [← c77715, ← c77716, k2, ← c77717] at e_1262_1
  have hr := e_1262_1
  simp only [e_1262_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11466)) := by
  have c77718 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77719 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77720 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77721 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77722 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77723 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_2 := arithEq_of_rows h (row := 1262) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_2
  simp only [← c77718, ← c77719, ← c77720] at e_1262_2
  have e_1262_3 := arithEq_of_rows h (row := 1262) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_3
  simp only [← c77721, ← c77722, k2, ← c77723] at e_1262_3
  have hr := e_1262_3
  simp only [e_1262_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11467)) := by
  have c77724 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77725 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77726 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77727 := (publicBatchWrapper89_copies2428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77728 := (publicBatchWrapper89_copies2429 a h).1
  have c77729 := (publicBatchWrapper89_copies2429 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_4 := arithEq_of_rows h (row := 1262) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_4
  simp only [← c77724, ← c77725, ← c77726] at e_1262_4
  have e_1262_5 := arithEq_of_rows h (row := 1262) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_5
  simp only [← c77727, ← c77728, k2, ← c77729] at e_1262_5
  have hr := e_1262_5
  simp only [e_1262_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11468)) := by
  have c77730 := (publicBatchWrapper89_copies2429 a h).2.2.1
  have c77731 := (publicBatchWrapper89_copies2429 a h).2.2.2.1
  have c77732 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.1
  have c77733 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.1
  have c77734 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.1
  have c77735 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_6 := arithEq_of_rows h (row := 1262) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_6
  simp only [← c77730, ← c77731, ← c77732] at e_1262_6
  have e_1262_7 := arithEq_of_rows h (row := 1262) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_7
  simp only [← c77733, ← c77734, k2, ← c77735] at e_1262_7
  have hr := e_1262_7
  simp only [e_1262_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11469)) := by
  have c77736 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.1
  have c77737 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.1
  have c77738 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77739 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77740 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77741 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_8 := arithEq_of_rows h (row := 1262) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_8
  simp only [← c77736, ← c77737, ← c77738] at e_1262_8
  have e_1262_9 := arithEq_of_rows h (row := 1262) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_9
  simp only [← c77739, ← c77740, k2, ← c77741] at e_1262_9
  have hr := e_1262_9
  simp only [e_1262_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11470)) := by
  have c77742 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77743 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77744 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77745 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77746 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77747 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_10 := arithEq_of_rows h (row := 1262) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_10
  simp only [← c77742, ← c77743, ← c77744] at e_1262_10
  have e_1262_11 := arithEq_of_rows h (row := 1262) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_11
  simp only [← c77745, ← c77746, k2, ← c77747] at e_1262_11
  have hr := e_1262_11
  simp only [e_1262_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11471)) := by
  have c77748 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77749 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77750 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77751 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77752 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77753 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_12 := arithEq_of_rows h (row := 1262) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_12
  simp only [← c77748, ← c77749, ← c77750] at e_1262_12
  have e_1262_13 := arithEq_of_rows h (row := 1262) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_13
  simp only [← c77751, ← c77752, k2, ← c77753] at e_1262_13
  have hr := e_1262_13
  simp only [e_1262_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11472)) := by
  have c77754 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77755 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77756 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77757 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77758 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77759 := (publicBatchWrapper89_copies2429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_14 := arithEq_of_rows h (row := 1262) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_14
  simp only [← c77754, ← c77755, ← c77756] at e_1262_14
  have e_1262_15 := arithEq_of_rows h (row := 1262) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_15
  simp only [← c77757, ← c77758, k2, ← c77759] at e_1262_15
  have hr := e_1262_15
  simp only [e_1262_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11473)) := by
  have c77760 := (publicBatchWrapper89_copies2430 a h).1
  have c77761 := (publicBatchWrapper89_copies2430 a h).2.1
  have c77762 := (publicBatchWrapper89_copies2430 a h).2.2.1
  have c77763 := (publicBatchWrapper89_copies2430 a h).2.2.2.1
  have c77764 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.1
  have c77765 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_16 := arithEq_of_rows h (row := 1262) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_16
  simp only [← c77760, ← c77761, ← c77762] at e_1262_16
  have e_1262_17 := arithEq_of_rows h (row := 1262) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_17
  simp only [← c77763, ← c77764, k2, ← c77765] at e_1262_17
  have hr := e_1262_17
  simp only [e_1262_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1262 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11474)) := by
  have c77766 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.1
  have c77767 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.1
  have c77768 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.1
  have c77769 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.1
  have c77770 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77771 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1262_18 := arithEq_of_rows h (row := 1262) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_18
  simp only [← c77766, ← c77767, ← c77768] at e_1262_18
  have e_1262_19 := arithEq_of_rows h (row := 1262) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1262_19
  simp only [← c77769, ← c77770, k2, ← c77771] at e_1262_19
  have hr := e_1262_19
  simp only [e_1262_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11475)) := by
  have c77772 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77773 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77774 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77775 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77776 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77777 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_0 := arithEq_of_rows h (row := 1263) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_0
  simp only [← c77772, ← c77773, ← c77774] at e_1263_0
  have e_1263_1 := arithEq_of_rows h (row := 1263) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_1
  simp only [← c77775, ← c77776, k2, ← c77777] at e_1263_1
  have hr := e_1263_1
  simp only [e_1263_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11476)) := by
  have c77778 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77779 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77780 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77781 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77782 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77783 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_2 := arithEq_of_rows h (row := 1263) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_2
  simp only [← c77778, ← c77779, ← c77780] at e_1263_2
  have e_1263_3 := arithEq_of_rows h (row := 1263) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_3
  simp only [← c77781, ← c77782, k2, ← c77783] at e_1263_3
  have hr := e_1263_3
  simp only [e_1263_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11477)) := by
  have c77784 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77785 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77786 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77787 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77788 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77789 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_4 := arithEq_of_rows h (row := 1263) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_4
  simp only [← c77784, ← c77785, ← c77786] at e_1263_4
  have e_1263_5 := arithEq_of_rows h (row := 1263) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_5
  simp only [← c77787, ← c77788, k2, ← c77789] at e_1263_5
  have hr := e_1263_5
  simp only [e_1263_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11478)) := by
  have c77790 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77791 := (publicBatchWrapper89_copies2430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77792 := (publicBatchWrapper89_copies2431 a h).1
  have c77793 := (publicBatchWrapper89_copies2431 a h).2.1
  have c77794 := (publicBatchWrapper89_copies2431 a h).2.2.1
  have c77795 := (publicBatchWrapper89_copies2431 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_6 := arithEq_of_rows h (row := 1263) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_6
  simp only [← c77790, ← c77791, ← c77792] at e_1263_6
  have e_1263_7 := arithEq_of_rows h (row := 1263) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_7
  simp only [← c77793, ← c77794, k2, ← c77795] at e_1263_7
  have hr := e_1263_7
  simp only [e_1263_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11479)) := by
  have c77796 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.1
  have c77797 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.1
  have c77798 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.1
  have c77799 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.1
  have c77800 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.1
  have c77801 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_8 := arithEq_of_rows h (row := 1263) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_8
  simp only [← c77796, ← c77797, ← c77798] at e_1263_8
  have e_1263_9 := arithEq_of_rows h (row := 1263) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_9
  simp only [← c77799, ← c77800, k2, ← c77801] at e_1263_9
  have hr := e_1263_9
  simp only [e_1263_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11480)) := by
  have c77802 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77803 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77804 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77805 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77806 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77807 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_10 := arithEq_of_rows h (row := 1263) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_10
  simp only [← c77802, ← c77803, ← c77804] at e_1263_10
  have e_1263_11 := arithEq_of_rows h (row := 1263) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_11
  simp only [← c77805, ← c77806, k2, ← c77807] at e_1263_11
  have hr := e_1263_11
  simp only [e_1263_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11481)) := by
  have c77808 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77809 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77810 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77811 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77812 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77813 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_12 := arithEq_of_rows h (row := 1263) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_12
  simp only [← c77808, ← c77809, ← c77810] at e_1263_12
  have e_1263_13 := arithEq_of_rows h (row := 1263) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_13
  simp only [← c77811, ← c77812, k2, ← c77813] at e_1263_13
  have hr := e_1263_13
  simp only [e_1263_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11482)) := by
  have c77814 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77815 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77816 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77817 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77818 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77819 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_14 := arithEq_of_rows h (row := 1263) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_14
  simp only [← c77814, ← c77815, ← c77816] at e_1263_14
  have e_1263_15 := arithEq_of_rows h (row := 1263) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_15
  simp only [← c77817, ← c77818, k2, ← c77819] at e_1263_15
  have hr := e_1263_15
  simp only [e_1263_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11483)) := by
  have c77820 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77821 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77822 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77823 := (publicBatchWrapper89_copies2431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77824 := (publicBatchWrapper89_copies2432 a h).1
  have c77825 := (publicBatchWrapper89_copies2432 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_16 := arithEq_of_rows h (row := 1263) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_16
  simp only [← c77820, ← c77821, ← c77822] at e_1263_16
  have e_1263_17 := arithEq_of_rows h (row := 1263) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_17
  simp only [← c77823, ← c77824, k2, ← c77825] at e_1263_17
  have hr := e_1263_17
  simp only [e_1263_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1263 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11484)) := by
  have c77826 := (publicBatchWrapper89_copies2432 a h).2.2.1
  have c77827 := (publicBatchWrapper89_copies2432 a h).2.2.2.1
  have c77828 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.1
  have c77829 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.1
  have c77830 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.1
  have c77831 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1263_18 := arithEq_of_rows h (row := 1263) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_18
  simp only [← c77826, ← c77827, ← c77828] at e_1263_18
  have e_1263_19 := arithEq_of_rows h (row := 1263) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1263_19
  simp only [← c77829, ← c77830, k2, ← c77831] at e_1263_19
  have hr := e_1263_19
  simp only [e_1263_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11485)) := by
  have c77832 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.1
  have c77833 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.1
  have c77834 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77835 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77836 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77837 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_0 := arithEq_of_rows h (row := 1264) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_0
  simp only [← c77832, ← c77833, ← c77834] at e_1264_0
  have e_1264_1 := arithEq_of_rows h (row := 1264) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_1
  simp only [← c77835, ← c77836, k2, ← c77837] at e_1264_1
  have hr := e_1264_1
  simp only [e_1264_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11486)) := by
  have c77838 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77839 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77840 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77841 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77842 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77843 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_2 := arithEq_of_rows h (row := 1264) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_2
  simp only [← c77838, ← c77839, ← c77840] at e_1264_2
  have e_1264_3 := arithEq_of_rows h (row := 1264) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_3
  simp only [← c77841, ← c77842, k2, ← c77843] at e_1264_3
  have hr := e_1264_3
  simp only [e_1264_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11487)) := by
  have c77844 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77845 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77846 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77847 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77848 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77849 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_4 := arithEq_of_rows h (row := 1264) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_4
  simp only [← c77844, ← c77845, ← c77846] at e_1264_4
  have e_1264_5 := arithEq_of_rows h (row := 1264) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_5
  simp only [← c77847, ← c77848, k2, ← c77849] at e_1264_5
  have hr := e_1264_5
  simp only [e_1264_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11488)) := by
  have c77850 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77851 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77852 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77853 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77854 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77855 := (publicBatchWrapper89_copies2432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_6 := arithEq_of_rows h (row := 1264) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_6
  simp only [← c77850, ← c77851, ← c77852] at e_1264_6
  have e_1264_7 := arithEq_of_rows h (row := 1264) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_7
  simp only [← c77853, ← c77854, k2, ← c77855] at e_1264_7
  have hr := e_1264_7
  simp only [e_1264_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11489)) := by
  have c77856 := (publicBatchWrapper89_copies2433 a h).1
  have c77857 := (publicBatchWrapper89_copies2433 a h).2.1
  have c77858 := (publicBatchWrapper89_copies2433 a h).2.2.1
  have c77859 := (publicBatchWrapper89_copies2433 a h).2.2.2.1
  have c77860 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.1
  have c77861 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_8 := arithEq_of_rows h (row := 1264) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_8
  simp only [← c77856, ← c77857, ← c77858] at e_1264_8
  have e_1264_9 := arithEq_of_rows h (row := 1264) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_9
  simp only [← c77859, ← c77860, k2, ← c77861] at e_1264_9
  have hr := e_1264_9
  simp only [e_1264_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11490)) := by
  have c77862 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.1
  have c77863 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.1
  have c77864 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.1
  have c77865 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.1
  have c77866 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77867 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_10 := arithEq_of_rows h (row := 1264) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_10
  simp only [← c77862, ← c77863, ← c77864] at e_1264_10
  have e_1264_11 := arithEq_of_rows h (row := 1264) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_11
  simp only [← c77865, ← c77866, k2, ← c77867] at e_1264_11
  have hr := e_1264_11
  simp only [e_1264_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11491)) := by
  have c77868 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77869 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77870 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77871 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77872 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77873 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_12 := arithEq_of_rows h (row := 1264) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_12
  simp only [← c77868, ← c77869, ← c77870] at e_1264_12
  have e_1264_13 := arithEq_of_rows h (row := 1264) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_13
  simp only [← c77871, ← c77872, k2, ← c77873] at e_1264_13
  have hr := e_1264_13
  simp only [e_1264_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11492)) := by
  have c77874 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77875 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77876 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77877 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77878 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77879 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_14 := arithEq_of_rows h (row := 1264) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_14
  simp only [← c77874, ← c77875, ← c77876] at e_1264_14
  have e_1264_15 := arithEq_of_rows h (row := 1264) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_15
  simp only [← c77877, ← c77878, k2, ← c77879] at e_1264_15
  have hr := e_1264_15
  simp only [e_1264_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11493)) := by
  have c77880 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77881 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77882 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77883 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77884 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77885 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_16 := arithEq_of_rows h (row := 1264) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_16
  simp only [← c77880, ← c77881, ← c77882] at e_1264_16
  have e_1264_17 := arithEq_of_rows h (row := 1264) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_17
  simp only [← c77883, ← c77884, k2, ← c77885] at e_1264_17
  have hr := e_1264_17
  simp only [e_1264_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1264 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11494)) := by
  have c77886 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77887 := (publicBatchWrapper89_copies2433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77888 := (publicBatchWrapper89_copies2434 a h).1
  have c77889 := (publicBatchWrapper89_copies2434 a h).2.1
  have c77890 := (publicBatchWrapper89_copies2434 a h).2.2.1
  have c77891 := (publicBatchWrapper89_copies2434 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1264_18 := arithEq_of_rows h (row := 1264) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_18
  simp only [← c77886, ← c77887, ← c77888] at e_1264_18
  have e_1264_19 := arithEq_of_rows h (row := 1264) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1264_19
  simp only [← c77889, ← c77890, k2, ← c77891] at e_1264_19
  have hr := e_1264_19
  simp only [e_1264_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11495)) := by
  have c77892 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.1
  have c77893 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.1
  have c77894 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.1
  have c77895 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.1
  have c77896 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.1
  have c77897 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_0 := arithEq_of_rows h (row := 1265) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_0
  simp only [← c77892, ← c77893, ← c77894] at e_1265_0
  have e_1265_1 := arithEq_of_rows h (row := 1265) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_1
  simp only [← c77895, ← c77896, k2, ← c77897] at e_1265_1
  have hr := e_1265_1
  simp only [e_1265_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11496)) := by
  have c77898 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77899 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77900 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77901 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77902 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77903 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_2 := arithEq_of_rows h (row := 1265) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_2
  simp only [← c77898, ← c77899, ← c77900] at e_1265_2
  have e_1265_3 := arithEq_of_rows h (row := 1265) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_3
  simp only [← c77901, ← c77902, k2, ← c77903] at e_1265_3
  have hr := e_1265_3
  simp only [e_1265_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11497)) := by
  have c77904 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77905 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77906 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77907 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77908 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77909 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_4 := arithEq_of_rows h (row := 1265) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_4
  simp only [← c77904, ← c77905, ← c77906] at e_1265_4
  have e_1265_5 := arithEq_of_rows h (row := 1265) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_5
  simp only [← c77907, ← c77908, k2, ← c77909] at e_1265_5
  have hr := e_1265_5
  simp only [e_1265_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11498)) := by
  have c77910 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77911 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77912 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77913 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77914 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77915 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_6 := arithEq_of_rows h (row := 1265) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_6
  simp only [← c77910, ← c77911, ← c77912] at e_1265_6
  have e_1265_7 := arithEq_of_rows h (row := 1265) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_7
  simp only [← c77913, ← c77914, k2, ← c77915] at e_1265_7
  have hr := e_1265_7
  simp only [e_1265_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11499)) := by
  have c77916 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77917 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77918 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77919 := (publicBatchWrapper89_copies2434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77920 := (publicBatchWrapper89_copies2435 a h).1
  have c77921 := (publicBatchWrapper89_copies2435 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_8 := arithEq_of_rows h (row := 1265) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_8
  simp only [← c77916, ← c77917, ← c77918] at e_1265_8
  have e_1265_9 := arithEq_of_rows h (row := 1265) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_9
  simp only [← c77919, ← c77920, k2, ← c77921] at e_1265_9
  have hr := e_1265_9
  simp only [e_1265_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11500)) := by
  have c77922 := (publicBatchWrapper89_copies2435 a h).2.2.1
  have c77923 := (publicBatchWrapper89_copies2435 a h).2.2.2.1
  have c77924 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.1
  have c77925 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.1
  have c77926 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.1
  have c77927 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_10 := arithEq_of_rows h (row := 1265) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_10
  simp only [← c77922, ← c77923, ← c77924] at e_1265_10
  have e_1265_11 := arithEq_of_rows h (row := 1265) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_11
  simp only [← c77925, ← c77926, k2, ← c77927] at e_1265_11
  have hr := e_1265_11
  simp only [e_1265_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11501)) := by
  have c77928 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.1
  have c77929 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.1
  have c77930 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77931 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77932 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77933 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_12 := arithEq_of_rows h (row := 1265) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_12
  simp only [← c77928, ← c77929, ← c77930] at e_1265_12
  have e_1265_13 := arithEq_of_rows h (row := 1265) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_13
  simp only [← c77931, ← c77932, k2, ← c77933] at e_1265_13
  have hr := e_1265_13
  simp only [e_1265_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11502)) := by
  have c77934 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77935 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77936 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77937 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77938 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77939 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_14 := arithEq_of_rows h (row := 1265) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_14
  simp only [← c77934, ← c77935, ← c77936] at e_1265_14
  have e_1265_15 := arithEq_of_rows h (row := 1265) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_15
  simp only [← c77937, ← c77938, k2, ← c77939] at e_1265_15
  have hr := e_1265_15
  simp only [e_1265_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11503)) := by
  have c77940 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77941 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77942 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77943 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77944 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77945 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_16 := arithEq_of_rows h (row := 1265) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_16
  simp only [← c77940, ← c77941, ← c77942] at e_1265_16
  have e_1265_17 := arithEq_of_rows h (row := 1265) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_17
  simp only [← c77943, ← c77944, k2, ← c77945] at e_1265_17
  have hr := e_1265_17
  simp only [e_1265_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1265 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11504)) := by
  have c77946 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77947 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77948 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77949 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77950 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77951 := (publicBatchWrapper89_copies2435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1265_18 := arithEq_of_rows h (row := 1265) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_18
  simp only [← c77946, ← c77947, ← c77948] at e_1265_18
  have e_1265_19 := arithEq_of_rows h (row := 1265) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1265_19
  simp only [← c77949, ← c77950, k2, ← c77951] at e_1265_19
  have hr := e_1265_19
  simp only [e_1265_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11505)) := by
  have c77952 := (publicBatchWrapper89_copies2436 a h).1
  have c77953 := (publicBatchWrapper89_copies2436 a h).2.1
  have c77954 := (publicBatchWrapper89_copies2436 a h).2.2.1
  have c77955 := (publicBatchWrapper89_copies2436 a h).2.2.2.1
  have c77956 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.1
  have c77957 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_0 := arithEq_of_rows h (row := 1266) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_0
  simp only [← c77952, ← c77953, ← c77954] at e_1266_0
  have e_1266_1 := arithEq_of_rows h (row := 1266) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_1
  simp only [← c77955, ← c77956, k2, ← c77957] at e_1266_1
  have hr := e_1266_1
  simp only [e_1266_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11506)) := by
  have c77958 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.1
  have c77959 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.1
  have c77960 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.1
  have c77961 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.1
  have c77962 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77963 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_2 := arithEq_of_rows h (row := 1266) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_2
  simp only [← c77958, ← c77959, ← c77960] at e_1266_2
  have e_1266_3 := arithEq_of_rows h (row := 1266) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_3
  simp only [← c77961, ← c77962, k2, ← c77963] at e_1266_3
  have hr := e_1266_3
  simp only [e_1266_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11507)) := by
  have c77964 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77965 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77966 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77967 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77968 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77969 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_4 := arithEq_of_rows h (row := 1266) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_4
  simp only [← c77964, ← c77965, ← c77966] at e_1266_4
  have e_1266_5 := arithEq_of_rows h (row := 1266) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_5
  simp only [← c77967, ← c77968, k2, ← c77969] at e_1266_5
  have hr := e_1266_5
  simp only [e_1266_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11508)) := by
  have c77970 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77971 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77972 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77973 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77974 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77975 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_6 := arithEq_of_rows h (row := 1266) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_6
  simp only [← c77970, ← c77971, ← c77972] at e_1266_6
  have e_1266_7 := arithEq_of_rows h (row := 1266) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_7
  simp only [← c77973, ← c77974, k2, ← c77975] at e_1266_7
  have hr := e_1266_7
  simp only [e_1266_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11509)) := by
  have c77976 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77977 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77978 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77979 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77980 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77981 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_8 := arithEq_of_rows h (row := 1266) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_8
  simp only [← c77976, ← c77977, ← c77978] at e_1266_8
  have e_1266_9 := arithEq_of_rows h (row := 1266) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_9
  simp only [← c77979, ← c77980, k2, ← c77981] at e_1266_9
  have hr := e_1266_9
  simp only [e_1266_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11510)) := by
  have c77982 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77983 := (publicBatchWrapper89_copies2436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c77984 := (publicBatchWrapper89_copies2437 a h).1
  have c77985 := (publicBatchWrapper89_copies2437 a h).2.1
  have c77986 := (publicBatchWrapper89_copies2437 a h).2.2.1
  have c77987 := (publicBatchWrapper89_copies2437 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_10 := arithEq_of_rows h (row := 1266) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_10
  simp only [← c77982, ← c77983, ← c77984] at e_1266_10
  have e_1266_11 := arithEq_of_rows h (row := 1266) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_11
  simp only [← c77985, ← c77986, k2, ← c77987] at e_1266_11
  have hr := e_1266_11
  simp only [e_1266_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11511)) := by
  have c77988 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.1
  have c77989 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.1
  have c77990 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.1
  have c77991 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.1
  have c77992 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.1
  have c77993 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_12 := arithEq_of_rows h (row := 1266) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_12
  simp only [← c77988, ← c77989, ← c77990] at e_1266_12
  have e_1266_13 := arithEq_of_rows h (row := 1266) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_13
  simp only [← c77991, ← c77992, k2, ← c77993] at e_1266_13
  have hr := e_1266_13
  simp only [e_1266_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11512)) := by
  have c77994 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c77995 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c77996 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77997 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77998 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77999 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_14 := arithEq_of_rows h (row := 1266) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_14
  simp only [← c77994, ← c77995, ← c77996] at e_1266_14
  have e_1266_15 := arithEq_of_rows h (row := 1266) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_15
  simp only [← c77997, ← c77998, k2, ← c77999] at e_1266_15
  have hr := e_1266_15
  simp only [e_1266_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11513)) := by
  have c78000 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78001 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78002 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78003 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78004 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78005 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_16 := arithEq_of_rows h (row := 1266) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_16
  simp only [← c78000, ← c78001, ← c78002] at e_1266_16
  have e_1266_17 := arithEq_of_rows h (row := 1266) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_17
  simp only [← c78003, ← c78004, k2, ← c78005] at e_1266_17
  have hr := e_1266_17
  simp only [e_1266_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1266 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11514)) := by
  have c78006 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78007 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78008 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78009 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78010 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78011 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1266_18 := arithEq_of_rows h (row := 1266) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_18
  simp only [← c78006, ← c78007, ← c78008] at e_1266_18
  have e_1266_19 := arithEq_of_rows h (row := 1266) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1266_19
  simp only [← c78009, ← c78010, k2, ← c78011] at e_1266_19
  have hr := e_1266_19
  simp only [e_1266_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11515)) := by
  have c78012 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78013 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78014 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78015 := (publicBatchWrapper89_copies2437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78016 := (publicBatchWrapper89_copies2438 a h).1
  have c78017 := (publicBatchWrapper89_copies2438 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_0 := arithEq_of_rows h (row := 1267) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_0
  simp only [← c78012, ← c78013, ← c78014] at e_1267_0
  have e_1267_1 := arithEq_of_rows h (row := 1267) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_1
  simp only [← c78015, ← c78016, k2, ← c78017] at e_1267_1
  have hr := e_1267_1
  simp only [e_1267_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11516)) := by
  have c78018 := (publicBatchWrapper89_copies2438 a h).2.2.1
  have c78019 := (publicBatchWrapper89_copies2438 a h).2.2.2.1
  have c78020 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.1
  have c78021 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.1
  have c78022 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.1
  have c78023 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_2 := arithEq_of_rows h (row := 1267) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_2
  simp only [← c78018, ← c78019, ← c78020] at e_1267_2
  have e_1267_3 := arithEq_of_rows h (row := 1267) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_3
  simp only [← c78021, ← c78022, k2, ← c78023] at e_1267_3
  have hr := e_1267_3
  simp only [e_1267_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11517)) := by
  have c78024 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.1
  have c78025 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.1
  have c78026 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78027 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78028 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78029 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_4 := arithEq_of_rows h (row := 1267) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_4
  simp only [← c78024, ← c78025, ← c78026] at e_1267_4
  have e_1267_5 := arithEq_of_rows h (row := 1267) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_5
  simp only [← c78027, ← c78028, k2, ← c78029] at e_1267_5
  have hr := e_1267_5
  simp only [e_1267_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11518)) := by
  have c78030 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78031 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78032 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78033 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78034 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78035 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_6 := arithEq_of_rows h (row := 1267) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_6
  simp only [← c78030, ← c78031, ← c78032] at e_1267_6
  have e_1267_7 := arithEq_of_rows h (row := 1267) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_7
  simp only [← c78033, ← c78034, k2, ← c78035] at e_1267_7
  have hr := e_1267_7
  simp only [e_1267_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11519)) := by
  have c78036 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78037 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78038 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78039 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78040 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78041 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_8 := arithEq_of_rows h (row := 1267) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_8
  simp only [← c78036, ← c78037, ← c78038] at e_1267_8
  have e_1267_9 := arithEq_of_rows h (row := 1267) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_9
  simp only [← c78039, ← c78040, k2, ← c78041] at e_1267_9
  have hr := e_1267_9
  simp only [e_1267_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11520)) := by
  have c78042 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78043 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78044 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78045 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78046 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78047 := (publicBatchWrapper89_copies2438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_10 := arithEq_of_rows h (row := 1267) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_10
  simp only [← c78042, ← c78043, ← c78044] at e_1267_10
  have e_1267_11 := arithEq_of_rows h (row := 1267) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_11
  simp only [← c78045, ← c78046, k2, ← c78047] at e_1267_11
  have hr := e_1267_11
  simp only [e_1267_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11521)) := by
  have c78048 := (publicBatchWrapper89_copies2439 a h).1
  have c78049 := (publicBatchWrapper89_copies2439 a h).2.1
  have c78050 := (publicBatchWrapper89_copies2439 a h).2.2.1
  have c78051 := (publicBatchWrapper89_copies2439 a h).2.2.2.1
  have c78052 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.1
  have c78053 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_12 := arithEq_of_rows h (row := 1267) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_12
  simp only [← c78048, ← c78049, ← c78050] at e_1267_12
  have e_1267_13 := arithEq_of_rows h (row := 1267) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_13
  simp only [← c78051, ← c78052, k2, ← c78053] at e_1267_13
  have hr := e_1267_13
  simp only [e_1267_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11522)) := by
  have c78054 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.1
  have c78055 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.1
  have c78056 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.1
  have c78057 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.1
  have c78058 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78059 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_14 := arithEq_of_rows h (row := 1267) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_14
  simp only [← c78054, ← c78055, ← c78056] at e_1267_14
  have e_1267_15 := arithEq_of_rows h (row := 1267) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_15
  simp only [← c78057, ← c78058, k2, ← c78059] at e_1267_15
  have hr := e_1267_15
  simp only [e_1267_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11523)) := by
  have c78060 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78061 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78062 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78063 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78064 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78065 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_16 := arithEq_of_rows h (row := 1267) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_16
  simp only [← c78060, ← c78061, ← c78062] at e_1267_16
  have e_1267_17 := arithEq_of_rows h (row := 1267) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_17
  simp only [← c78063, ← c78064, k2, ← c78065] at e_1267_17
  have hr := e_1267_17
  simp only [e_1267_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1267 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11524)) := by
  have c78066 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78067 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78068 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78069 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78070 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78071 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1267_18 := arithEq_of_rows h (row := 1267) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_18
  simp only [← c78066, ← c78067, ← c78068] at e_1267_18
  have e_1267_19 := arithEq_of_rows h (row := 1267) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1267_19
  simp only [← c78069, ← c78070, k2, ← c78071] at e_1267_19
  have hr := e_1267_19
  simp only [e_1267_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11525)) := by
  have c78072 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78073 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78074 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78075 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78076 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78077 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_0 := arithEq_of_rows h (row := 1268) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_0
  simp only [← c78072, ← c78073, ← c78074] at e_1268_0
  have e_1268_1 := arithEq_of_rows h (row := 1268) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_1
  simp only [← c78075, ← c78076, k2, ← c78077] at e_1268_1
  have hr := e_1268_1
  simp only [e_1268_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11526)) := by
  have c78078 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78079 := (publicBatchWrapper89_copies2439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78080 := (publicBatchWrapper89_copies2440 a h).1
  have c78081 := (publicBatchWrapper89_copies2440 a h).2.1
  have c78082 := (publicBatchWrapper89_copies2440 a h).2.2.1
  have c78083 := (publicBatchWrapper89_copies2440 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_2 := arithEq_of_rows h (row := 1268) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_2
  simp only [← c78078, ← c78079, ← c78080] at e_1268_2
  have e_1268_3 := arithEq_of_rows h (row := 1268) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_3
  simp only [← c78081, ← c78082, k2, ← c78083] at e_1268_3
  have hr := e_1268_3
  simp only [e_1268_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11527)) := by
  have c78084 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.1
  have c78085 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.1
  have c78086 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.1
  have c78087 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.1
  have c78088 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.1
  have c78089 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_4 := arithEq_of_rows h (row := 1268) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_4
  simp only [← c78084, ← c78085, ← c78086] at e_1268_4
  have e_1268_5 := arithEq_of_rows h (row := 1268) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_5
  simp only [← c78087, ← c78088, k2, ← c78089] at e_1268_5
  have hr := e_1268_5
  simp only [e_1268_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11528)) := by
  have c78090 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78091 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78092 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78093 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78094 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78095 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_6 := arithEq_of_rows h (row := 1268) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_6
  simp only [← c78090, ← c78091, ← c78092] at e_1268_6
  have e_1268_7 := arithEq_of_rows h (row := 1268) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_7
  simp only [← c78093, ← c78094, k2, ← c78095] at e_1268_7
  have hr := e_1268_7
  simp only [e_1268_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11529)) := by
  have c78096 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78097 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78098 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78099 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78100 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78101 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_8 := arithEq_of_rows h (row := 1268) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_8
  simp only [← c78096, ← c78097, ← c78098] at e_1268_8
  have e_1268_9 := arithEq_of_rows h (row := 1268) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_9
  simp only [← c78099, ← c78100, k2, ← c78101] at e_1268_9
  have hr := e_1268_9
  simp only [e_1268_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11530)) := by
  have c78102 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78103 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78104 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78105 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78106 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78107 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_10 := arithEq_of_rows h (row := 1268) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_10
  simp only [← c78102, ← c78103, ← c78104] at e_1268_10
  have e_1268_11 := arithEq_of_rows h (row := 1268) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_11
  simp only [← c78105, ← c78106, k2, ← c78107] at e_1268_11
  have hr := e_1268_11
  simp only [e_1268_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11531)) := by
  have c78108 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78109 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78110 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78111 := (publicBatchWrapper89_copies2440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78112 := (publicBatchWrapper89_copies2441 a h).1
  have c78113 := (publicBatchWrapper89_copies2441 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_12 := arithEq_of_rows h (row := 1268) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_12
  simp only [← c78108, ← c78109, ← c78110] at e_1268_12
  have e_1268_13 := arithEq_of_rows h (row := 1268) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_13
  simp only [← c78111, ← c78112, k2, ← c78113] at e_1268_13
  have hr := e_1268_13
  simp only [e_1268_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11532)) := by
  have c78114 := (publicBatchWrapper89_copies2441 a h).2.2.1
  have c78115 := (publicBatchWrapper89_copies2441 a h).2.2.2.1
  have c78116 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.1
  have c78117 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.1
  have c78118 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.1
  have c78119 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_14 := arithEq_of_rows h (row := 1268) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_14
  simp only [← c78114, ← c78115, ← c78116] at e_1268_14
  have e_1268_15 := arithEq_of_rows h (row := 1268) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_15
  simp only [← c78117, ← c78118, k2, ← c78119] at e_1268_15
  have hr := e_1268_15
  simp only [e_1268_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11533)) := by
  have c78120 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.1
  have c78121 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.1
  have c78122 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78123 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78124 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78125 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_16 := arithEq_of_rows h (row := 1268) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_16
  simp only [← c78120, ← c78121, ← c78122] at e_1268_16
  have e_1268_17 := arithEq_of_rows h (row := 1268) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_17
  simp only [← c78123, ← c78124, k2, ← c78125] at e_1268_17
  have hr := e_1268_17
  simp only [e_1268_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1268 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11534)) := by
  have c78126 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78127 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78128 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78129 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78130 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78131 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1268_18 := arithEq_of_rows h (row := 1268) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_18
  simp only [← c78126, ← c78127, ← c78128] at e_1268_18
  have e_1268_19 := arithEq_of_rows h (row := 1268) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1268_19
  simp only [← c78129, ← c78130, k2, ← c78131] at e_1268_19
  have hr := e_1268_19
  simp only [e_1268_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11535)) := by
  have c78132 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78133 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78134 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78135 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78136 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78137 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_0 := arithEq_of_rows h (row := 1269) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_0
  simp only [← c78132, ← c78133, ← c78134] at e_1269_0
  have e_1269_1 := arithEq_of_rows h (row := 1269) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_1
  simp only [← c78135, ← c78136, k2, ← c78137] at e_1269_1
  have hr := e_1269_1
  simp only [e_1269_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11536)) := by
  have c78138 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78139 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78140 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78141 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78142 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78143 := (publicBatchWrapper89_copies2441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_2 := arithEq_of_rows h (row := 1269) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_2
  simp only [← c78138, ← c78139, ← c78140] at e_1269_2
  have e_1269_3 := arithEq_of_rows h (row := 1269) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_3
  simp only [← c78141, ← c78142, k2, ← c78143] at e_1269_3
  have hr := e_1269_3
  simp only [e_1269_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11537)) := by
  have c78144 := (publicBatchWrapper89_copies2442 a h).1
  have c78145 := (publicBatchWrapper89_copies2442 a h).2.1
  have c78146 := (publicBatchWrapper89_copies2442 a h).2.2.1
  have c78147 := (publicBatchWrapper89_copies2442 a h).2.2.2.1
  have c78148 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.1
  have c78149 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_4 := arithEq_of_rows h (row := 1269) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_4
  simp only [← c78144, ← c78145, ← c78146] at e_1269_4
  have e_1269_5 := arithEq_of_rows h (row := 1269) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_5
  simp only [← c78147, ← c78148, k2, ← c78149] at e_1269_5
  have hr := e_1269_5
  simp only [e_1269_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11538)) := by
  have c78150 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.1
  have c78151 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.1
  have c78152 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.1
  have c78153 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.1
  have c78154 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78155 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_6 := arithEq_of_rows h (row := 1269) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_6
  simp only [← c78150, ← c78151, ← c78152] at e_1269_6
  have e_1269_7 := arithEq_of_rows h (row := 1269) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_7
  simp only [← c78153, ← c78154, k2, ← c78155] at e_1269_7
  have hr := e_1269_7
  simp only [e_1269_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11539)) := by
  have c78156 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78157 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78158 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78159 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78160 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78161 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_8 := arithEq_of_rows h (row := 1269) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_8
  simp only [← c78156, ← c78157, ← c78158] at e_1269_8
  have e_1269_9 := arithEq_of_rows h (row := 1269) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_9
  simp only [← c78159, ← c78160, k2, ← c78161] at e_1269_9
  have hr := e_1269_9
  simp only [e_1269_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11540)) := by
  have c78162 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78163 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78164 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78165 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78166 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78167 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_10 := arithEq_of_rows h (row := 1269) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_10
  simp only [← c78162, ← c78163, ← c78164] at e_1269_10
  have e_1269_11 := arithEq_of_rows h (row := 1269) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_11
  simp only [← c78165, ← c78166, k2, ← c78167] at e_1269_11
  have hr := e_1269_11
  simp only [e_1269_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11541)) := by
  have c78168 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78169 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78170 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78171 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78172 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78173 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_12 := arithEq_of_rows h (row := 1269) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_12
  simp only [← c78168, ← c78169, ← c78170] at e_1269_12
  have e_1269_13 := arithEq_of_rows h (row := 1269) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_13
  simp only [← c78171, ← c78172, k2, ← c78173] at e_1269_13
  have hr := e_1269_13
  simp only [e_1269_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11542)) := by
  have c78174 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78175 := (publicBatchWrapper89_copies2442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78176 := (publicBatchWrapper89_copies2443 a h).1
  have c78177 := (publicBatchWrapper89_copies2443 a h).2.1
  have c78178 := (publicBatchWrapper89_copies2443 a h).2.2.1
  have c78179 := (publicBatchWrapper89_copies2443 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_14 := arithEq_of_rows h (row := 1269) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_14
  simp only [← c78174, ← c78175, ← c78176] at e_1269_14
  have e_1269_15 := arithEq_of_rows h (row := 1269) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_15
  simp only [← c78177, ← c78178, k2, ← c78179] at e_1269_15
  have hr := e_1269_15
  simp only [e_1269_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11543)) := by
  have c78180 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.1
  have c78181 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.1
  have c78182 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.1
  have c78183 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.1
  have c78184 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.1
  have c78185 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_16 := arithEq_of_rows h (row := 1269) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_16
  simp only [← c78180, ← c78181, ← c78182] at e_1269_16
  have e_1269_17 := arithEq_of_rows h (row := 1269) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_17
  simp only [← c78183, ← c78184, k2, ← c78185] at e_1269_17
  have hr := e_1269_17
  simp only [e_1269_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1269 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11544)) := by
  have c78186 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78187 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78188 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78189 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78190 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78191 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1269_18 := arithEq_of_rows h (row := 1269) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_18
  simp only [← c78186, ← c78187, ← c78188] at e_1269_18
  have e_1269_19 := arithEq_of_rows h (row := 1269) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1269_19
  simp only [← c78189, ← c78190, k2, ← c78191] at e_1269_19
  have hr := e_1269_19
  simp only [e_1269_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11545)) := by
  have c78192 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78193 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78194 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78195 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78196 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78197 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_0 := arithEq_of_rows h (row := 1270) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_0
  simp only [← c78192, ← c78193, ← c78194] at e_1270_0
  have e_1270_1 := arithEq_of_rows h (row := 1270) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_1
  simp only [← c78195, ← c78196, k2, ← c78197] at e_1270_1
  have hr := e_1270_1
  simp only [e_1270_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11546)) := by
  have c78198 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78199 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78200 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78201 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78202 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78203 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_2 := arithEq_of_rows h (row := 1270) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_2
  simp only [← c78198, ← c78199, ← c78200] at e_1270_2
  have e_1270_3 := arithEq_of_rows h (row := 1270) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_3
  simp only [← c78201, ← c78202, k2, ← c78203] at e_1270_3
  have hr := e_1270_3
  simp only [e_1270_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11547)) := by
  have c78204 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78205 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78206 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78207 := (publicBatchWrapper89_copies2443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78208 := (publicBatchWrapper89_copies2444 a h).1
  have c78209 := (publicBatchWrapper89_copies2444 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_4 := arithEq_of_rows h (row := 1270) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_4
  simp only [← c78204, ← c78205, ← c78206] at e_1270_4
  have e_1270_5 := arithEq_of_rows h (row := 1270) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_5
  simp only [← c78207, ← c78208, k2, ← c78209] at e_1270_5
  have hr := e_1270_5
  simp only [e_1270_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11548)) := by
  have c78210 := (publicBatchWrapper89_copies2444 a h).2.2.1
  have c78211 := (publicBatchWrapper89_copies2444 a h).2.2.2.1
  have c78212 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.1
  have c78213 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.1
  have c78214 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.1
  have c78215 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_6 := arithEq_of_rows h (row := 1270) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_6
  simp only [← c78210, ← c78211, ← c78212] at e_1270_6
  have e_1270_7 := arithEq_of_rows h (row := 1270) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_7
  simp only [← c78213, ← c78214, k2, ← c78215] at e_1270_7
  have hr := e_1270_7
  simp only [e_1270_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11549)) := by
  have c78216 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.1
  have c78217 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.1
  have c78218 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78219 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78220 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78221 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_8 := arithEq_of_rows h (row := 1270) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_8
  simp only [← c78216, ← c78217, ← c78218] at e_1270_8
  have e_1270_9 := arithEq_of_rows h (row := 1270) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_9
  simp only [← c78219, ← c78220, k2, ← c78221] at e_1270_9
  have hr := e_1270_9
  simp only [e_1270_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11550)) := by
  have c78222 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78223 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78224 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78225 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78226 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78227 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_10 := arithEq_of_rows h (row := 1270) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_10
  simp only [← c78222, ← c78223, ← c78224] at e_1270_10
  have e_1270_11 := arithEq_of_rows h (row := 1270) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_11
  simp only [← c78225, ← c78226, k2, ← c78227] at e_1270_11
  have hr := e_1270_11
  simp only [e_1270_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11551)) := by
  have c78228 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78229 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78230 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78231 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78232 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78233 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_12 := arithEq_of_rows h (row := 1270) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_12
  simp only [← c78228, ← c78229, ← c78230] at e_1270_12
  have e_1270_13 := arithEq_of_rows h (row := 1270) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_13
  simp only [← c78231, ← c78232, k2, ← c78233] at e_1270_13
  have hr := e_1270_13
  simp only [e_1270_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11552)) := by
  have c78234 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78235 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78236 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78237 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78238 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78239 := (publicBatchWrapper89_copies2444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_14 := arithEq_of_rows h (row := 1270) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_14
  simp only [← c78234, ← c78235, ← c78236] at e_1270_14
  have e_1270_15 := arithEq_of_rows h (row := 1270) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_15
  simp only [← c78237, ← c78238, k2, ← c78239] at e_1270_15
  have hr := e_1270_15
  simp only [e_1270_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11553)) := by
  have c78240 := (publicBatchWrapper89_copies2445 a h).1
  have c78241 := (publicBatchWrapper89_copies2445 a h).2.1
  have c78242 := (publicBatchWrapper89_copies2445 a h).2.2.1
  have c78243 := (publicBatchWrapper89_copies2445 a h).2.2.2.1
  have c78244 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.1
  have c78245 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_16 := arithEq_of_rows h (row := 1270) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_16
  simp only [← c78240, ← c78241, ← c78242] at e_1270_16
  have e_1270_17 := arithEq_of_rows h (row := 1270) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_17
  simp only [← c78243, ← c78244, k2, ← c78245] at e_1270_17
  have hr := e_1270_17
  simp only [e_1270_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1270 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11554)) := by
  have c78246 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.1
  have c78247 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.1
  have c78248 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.1
  have c78249 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.1
  have c78250 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78251 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1270_18 := arithEq_of_rows h (row := 1270) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_18
  simp only [← c78246, ← c78247, ← c78248] at e_1270_18
  have e_1270_19 := arithEq_of_rows h (row := 1270) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1270_19
  simp only [← c78249, ← c78250, k2, ← c78251] at e_1270_19
  have hr := e_1270_19
  simp only [e_1270_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11555)) := by
  have c78252 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78253 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78254 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78255 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78256 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78257 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_0 := arithEq_of_rows h (row := 1271) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_0
  simp only [← c78252, ← c78253, ← c78254] at e_1271_0
  have e_1271_1 := arithEq_of_rows h (row := 1271) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_1
  simp only [← c78255, ← c78256, k2, ← c78257] at e_1271_1
  have hr := e_1271_1
  simp only [e_1271_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11556)) := by
  have c78258 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78259 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78260 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78261 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78262 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78263 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_2 := arithEq_of_rows h (row := 1271) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_2
  simp only [← c78258, ← c78259, ← c78260] at e_1271_2
  have e_1271_3 := arithEq_of_rows h (row := 1271) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_3
  simp only [← c78261, ← c78262, k2, ← c78263] at e_1271_3
  have hr := e_1271_3
  simp only [e_1271_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11557)) := by
  have c78264 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78265 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78266 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78267 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78268 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78269 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_4 := arithEq_of_rows h (row := 1271) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_4
  simp only [← c78264, ← c78265, ← c78266] at e_1271_4
  have e_1271_5 := arithEq_of_rows h (row := 1271) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_5
  simp only [← c78267, ← c78268, k2, ← c78269] at e_1271_5
  have hr := e_1271_5
  simp only [e_1271_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11558)) := by
  have c78270 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78271 := (publicBatchWrapper89_copies2445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78272 := (publicBatchWrapper89_copies2446 a h).1
  have c78273 := (publicBatchWrapper89_copies2446 a h).2.1
  have c78274 := (publicBatchWrapper89_copies2446 a h).2.2.1
  have c78275 := (publicBatchWrapper89_copies2446 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_6 := arithEq_of_rows h (row := 1271) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_6
  simp only [← c78270, ← c78271, ← c78272] at e_1271_6
  have e_1271_7 := arithEq_of_rows h (row := 1271) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_7
  simp only [← c78273, ← c78274, k2, ← c78275] at e_1271_7
  have hr := e_1271_7
  simp only [e_1271_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11559)) := by
  have c78276 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.1
  have c78277 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.1
  have c78278 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.1
  have c78279 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.1
  have c78280 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.1
  have c78281 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_8 := arithEq_of_rows h (row := 1271) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_8
  simp only [← c78276, ← c78277, ← c78278] at e_1271_8
  have e_1271_9 := arithEq_of_rows h (row := 1271) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_9
  simp only [← c78279, ← c78280, k2, ← c78281] at e_1271_9
  have hr := e_1271_9
  simp only [e_1271_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11560)) := by
  have c78282 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78283 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78284 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78285 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78286 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78287 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_10 := arithEq_of_rows h (row := 1271) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_10
  simp only [← c78282, ← c78283, ← c78284] at e_1271_10
  have e_1271_11 := arithEq_of_rows h (row := 1271) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_11
  simp only [← c78285, ← c78286, k2, ← c78287] at e_1271_11
  have hr := e_1271_11
  simp only [e_1271_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11561)) := by
  have c78288 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78289 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78290 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78291 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78292 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78293 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_12 := arithEq_of_rows h (row := 1271) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_12
  simp only [← c78288, ← c78289, ← c78290] at e_1271_12
  have e_1271_13 := arithEq_of_rows h (row := 1271) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_13
  simp only [← c78291, ← c78292, k2, ← c78293] at e_1271_13
  have hr := e_1271_13
  simp only [e_1271_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11562)) := by
  have c78294 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78295 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78296 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78297 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78298 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78299 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_14 := arithEq_of_rows h (row := 1271) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_14
  simp only [← c78294, ← c78295, ← c78296] at e_1271_14
  have e_1271_15 := arithEq_of_rows h (row := 1271) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_15
  simp only [← c78297, ← c78298, k2, ← c78299] at e_1271_15
  have hr := e_1271_15
  simp only [e_1271_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11563)) := by
  have c78300 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78301 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78302 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78303 := (publicBatchWrapper89_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78304 := (publicBatchWrapper89_copies2447 a h).1
  have c78305 := (publicBatchWrapper89_copies2447 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_16 := arithEq_of_rows h (row := 1271) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_16
  simp only [← c78300, ← c78301, ← c78302] at e_1271_16
  have e_1271_17 := arithEq_of_rows h (row := 1271) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_17
  simp only [← c78303, ← c78304, k2, ← c78305] at e_1271_17
  have hr := e_1271_17
  simp only [e_1271_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1271 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11564)) := by
  have c78306 := (publicBatchWrapper89_copies2447 a h).2.2.1
  have c78307 := (publicBatchWrapper89_copies2447 a h).2.2.2.1
  have c78308 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.1
  have c78309 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.1
  have c78310 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.1
  have c78311 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1271_18 := arithEq_of_rows h (row := 1271) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_18
  simp only [← c78306, ← c78307, ← c78308] at e_1271_18
  have e_1271_19 := arithEq_of_rows h (row := 1271) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1271_19
  simp only [← c78309, ← c78310, k2, ← c78311] at e_1271_19
  have hr := e_1271_19
  simp only [e_1271_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11565)) := by
  have c78312 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.1
  have c78313 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.1
  have c78314 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78315 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78316 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78317 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_0 := arithEq_of_rows h (row := 1272) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_0
  simp only [← c78312, ← c78313, ← c78314] at e_1272_0
  have e_1272_1 := arithEq_of_rows h (row := 1272) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_1
  simp only [← c78315, ← c78316, k2, ← c78317] at e_1272_1
  have hr := e_1272_1
  simp only [e_1272_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11566)) := by
  have c78318 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78319 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78320 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78321 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78322 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78323 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_2 := arithEq_of_rows h (row := 1272) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_2
  simp only [← c78318, ← c78319, ← c78320] at e_1272_2
  have e_1272_3 := arithEq_of_rows h (row := 1272) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_3
  simp only [← c78321, ← c78322, k2, ← c78323] at e_1272_3
  have hr := e_1272_3
  simp only [e_1272_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11567)) := by
  have c78324 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78325 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78326 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78327 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78328 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78329 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_4 := arithEq_of_rows h (row := 1272) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_4
  simp only [← c78324, ← c78325, ← c78326] at e_1272_4
  have e_1272_5 := arithEq_of_rows h (row := 1272) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_5
  simp only [← c78327, ← c78328, k2, ← c78329] at e_1272_5
  have hr := e_1272_5
  simp only [e_1272_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11568)) := by
  have c78330 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78331 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78332 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78333 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78334 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78335 := (publicBatchWrapper89_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_6 := arithEq_of_rows h (row := 1272) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_6
  simp only [← c78330, ← c78331, ← c78332] at e_1272_6
  have e_1272_7 := arithEq_of_rows h (row := 1272) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_7
  simp only [← c78333, ← c78334, k2, ← c78335] at e_1272_7
  have hr := e_1272_7
  simp only [e_1272_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11569)) := by
  have c78336 := (publicBatchWrapper89_copies2448 a h).1
  have c78337 := (publicBatchWrapper89_copies2448 a h).2.1
  have c78338 := (publicBatchWrapper89_copies2448 a h).2.2.1
  have c78339 := (publicBatchWrapper89_copies2448 a h).2.2.2.1
  have c78340 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.1
  have c78341 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_8 := arithEq_of_rows h (row := 1272) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_8
  simp only [← c78336, ← c78337, ← c78338] at e_1272_8
  have e_1272_9 := arithEq_of_rows h (row := 1272) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_9
  simp only [← c78339, ← c78340, k2, ← c78341] at e_1272_9
  have hr := e_1272_9
  simp only [e_1272_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11610)) := by
  have c78342 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.1
  have c78343 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.1
  have c78344 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.1
  have c78345 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.1
  have c78346 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78347 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_10 := arithEq_of_rows h (row := 1272) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_10
  simp only [← c78342, ← c78343, ← c78344] at e_1272_10
  have e_1272_11 := arithEq_of_rows h (row := 1272) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_11
  simp only [← c78345, ← c78346, k2, ← c78347] at e_1272_11
  have hr := e_1272_11
  simp only [e_1272_10] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
