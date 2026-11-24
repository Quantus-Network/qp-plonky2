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

theorem publicBatchWrapper89_f7593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5697)) := by
  have c50064 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50065 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50066 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50067 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50068 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50069 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_4 := arithEq_of_rows h (row := 801) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_4
  simp only [← c50064, ← c50065, ← c50066] at e_801_4
  have e_801_5 := arithEq_of_rows h (row := 801) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_5
  simp only [← c50067, ← c50068, k2, ← c50069] at e_801_5
  have hr := e_801_5
  simp only [e_801_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5698)) := by
  have c50070 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50071 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50072 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50073 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50074 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50075 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_6 := arithEq_of_rows h (row := 801) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_6
  simp only [← c50070, ← c50071, ← c50072] at e_801_6
  have e_801_7 := arithEq_of_rows h (row := 801) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_7
  simp only [← c50073, ← c50074, k2, ← c50075] at e_801_7
  have hr := e_801_7
  simp only [e_801_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5699)) := by
  have c50076 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50077 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50078 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50079 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50080 := (publicBatchWrapper89_copies1565 a h).1
  have c50081 := (publicBatchWrapper89_copies1565 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_8 := arithEq_of_rows h (row := 801) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_8
  simp only [← c50076, ← c50077, ← c50078] at e_801_8
  have e_801_9 := arithEq_of_rows h (row := 801) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_9
  simp only [← c50079, ← c50080, k2, ← c50081] at e_801_9
  have hr := e_801_9
  simp only [e_801_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5700)) := by
  have c50082 := (publicBatchWrapper89_copies1565 a h).2.2.1
  have c50083 := (publicBatchWrapper89_copies1565 a h).2.2.2.1
  have c50084 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.1
  have c50085 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.1
  have c50086 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.1
  have c50087 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_10 := arithEq_of_rows h (row := 801) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_10
  simp only [← c50082, ← c50083, ← c50084] at e_801_10
  have e_801_11 := arithEq_of_rows h (row := 801) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_11
  simp only [← c50085, ← c50086, k2, ← c50087] at e_801_11
  have hr := e_801_11
  simp only [e_801_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5701)) := by
  have c50088 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.1
  have c50089 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.1
  have c50090 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50091 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50092 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50093 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_12 := arithEq_of_rows h (row := 801) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_12
  simp only [← c50088, ← c50089, ← c50090] at e_801_12
  have e_801_13 := arithEq_of_rows h (row := 801) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_13
  simp only [← c50091, ← c50092, k2, ← c50093] at e_801_13
  have hr := e_801_13
  simp only [e_801_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5702)) := by
  have c50094 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50095 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50096 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50097 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50098 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50099 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_14 := arithEq_of_rows h (row := 801) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_14
  simp only [← c50094, ← c50095, ← c50096] at e_801_14
  have e_801_15 := arithEq_of_rows h (row := 801) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_15
  simp only [← c50097, ← c50098, k2, ← c50099] at e_801_15
  have hr := e_801_15
  simp only [e_801_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5703)) := by
  have c50100 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50101 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50102 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50103 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50104 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50105 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_16 := arithEq_of_rows h (row := 801) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_16
  simp only [← c50100, ← c50101, ← c50102] at e_801_16
  have e_801_17 := arithEq_of_rows h (row := 801) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_17
  simp only [← c50103, ← c50104, k2, ← c50105] at e_801_17
  have hr := e_801_17
  simp only [e_801_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5704)) := by
  have c50106 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50107 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50108 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50109 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50110 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50111 := (publicBatchWrapper89_copies1565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_18 := arithEq_of_rows h (row := 801) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_18
  simp only [← c50106, ← c50107, ← c50108] at e_801_18
  have e_801_19 := arithEq_of_rows h (row := 801) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_19
  simp only [← c50109, ← c50110, k2, ← c50111] at e_801_19
  have hr := e_801_19
  simp only [e_801_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5705)) := by
  have c50112 := (publicBatchWrapper89_copies1566 a h).1
  have c50113 := (publicBatchWrapper89_copies1566 a h).2.1
  have c50114 := (publicBatchWrapper89_copies1566 a h).2.2.1
  have c50115 := (publicBatchWrapper89_copies1566 a h).2.2.2.1
  have c50116 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.1
  have c50117 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_0 := arithEq_of_rows h (row := 802) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_0
  simp only [← c50112, ← c50113, ← c50114] at e_802_0
  have e_802_1 := arithEq_of_rows h (row := 802) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_1
  simp only [← c50115, ← c50116, k2, ← c50117] at e_802_1
  have hr := e_802_1
  simp only [e_802_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5706)) := by
  have c50118 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.1
  have c50119 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.1
  have c50120 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.1
  have c50121 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.1
  have c50122 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50123 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_2 := arithEq_of_rows h (row := 802) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_2
  simp only [← c50118, ← c50119, ← c50120] at e_802_2
  have e_802_3 := arithEq_of_rows h (row := 802) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_3
  simp only [← c50121, ← c50122, k2, ← c50123] at e_802_3
  have hr := e_802_3
  simp only [e_802_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5707)) := by
  have c50124 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50125 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50126 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50127 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50128 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50129 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_4 := arithEq_of_rows h (row := 802) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_4
  simp only [← c50124, ← c50125, ← c50126] at e_802_4
  have e_802_5 := arithEq_of_rows h (row := 802) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_5
  simp only [← c50127, ← c50128, k2, ← c50129] at e_802_5
  have hr := e_802_5
  simp only [e_802_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5708)) := by
  have c50130 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50131 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50132 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50133 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50134 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50135 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_6 := arithEq_of_rows h (row := 802) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_6
  simp only [← c50130, ← c50131, ← c50132] at e_802_6
  have e_802_7 := arithEq_of_rows h (row := 802) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_7
  simp only [← c50133, ← c50134, k2, ← c50135] at e_802_7
  have hr := e_802_7
  simp only [e_802_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5709)) := by
  have c50136 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50137 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50138 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50139 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50140 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50141 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_8 := arithEq_of_rows h (row := 802) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_8
  simp only [← c50136, ← c50137, ← c50138] at e_802_8
  have e_802_9 := arithEq_of_rows h (row := 802) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_9
  simp only [← c50139, ← c50140, k2, ← c50141] at e_802_9
  have hr := e_802_9
  simp only [e_802_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5710)) := by
  have c50142 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50143 := (publicBatchWrapper89_copies1566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50144 := (publicBatchWrapper89_copies1567 a h).1
  have c50145 := (publicBatchWrapper89_copies1567 a h).2.1
  have c50146 := (publicBatchWrapper89_copies1567 a h).2.2.1
  have c50147 := (publicBatchWrapper89_copies1567 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_10 := arithEq_of_rows h (row := 802) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_10
  simp only [← c50142, ← c50143, ← c50144] at e_802_10
  have e_802_11 := arithEq_of_rows h (row := 802) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_11
  simp only [← c50145, ← c50146, k2, ← c50147] at e_802_11
  have hr := e_802_11
  simp only [e_802_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5711)) := by
  have c50148 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.1
  have c50149 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.1
  have c50150 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.1
  have c50151 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.1
  have c50152 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.1
  have c50153 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_12 := arithEq_of_rows h (row := 802) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_12
  simp only [← c50148, ← c50149, ← c50150] at e_802_12
  have e_802_13 := arithEq_of_rows h (row := 802) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_13
  simp only [← c50151, ← c50152, k2, ← c50153] at e_802_13
  have hr := e_802_13
  simp only [e_802_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5712)) := by
  have c50154 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50155 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50156 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50157 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50158 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50159 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_14 := arithEq_of_rows h (row := 802) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_14
  simp only [← c50154, ← c50155, ← c50156] at e_802_14
  have e_802_15 := arithEq_of_rows h (row := 802) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_15
  simp only [← c50157, ← c50158, k2, ← c50159] at e_802_15
  have hr := e_802_15
  simp only [e_802_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5713)) := by
  have c50160 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50161 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50162 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50163 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50164 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50165 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_16 := arithEq_of_rows h (row := 802) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_16
  simp only [← c50160, ← c50161, ← c50162] at e_802_16
  have e_802_17 := arithEq_of_rows h (row := 802) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_17
  simp only [← c50163, ← c50164, k2, ← c50165] at e_802_17
  have hr := e_802_17
  simp only [e_802_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 802 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5714)) := by
  have c50166 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50167 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50168 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50169 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50170 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50171 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_802_18 := arithEq_of_rows h (row := 802) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_18
  simp only [← c50166, ← c50167, ← c50168] at e_802_18
  have e_802_19 := arithEq_of_rows h (row := 802) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_802_19
  simp only [← c50169, ← c50170, k2, ← c50171] at e_802_19
  have hr := e_802_19
  simp only [e_802_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5715)) := by
  have c50172 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50173 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50174 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50175 := (publicBatchWrapper89_copies1567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50176 := (publicBatchWrapper89_copies1568 a h).1
  have c50177 := (publicBatchWrapper89_copies1568 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_0 := arithEq_of_rows h (row := 803) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_0
  simp only [← c50172, ← c50173, ← c50174] at e_803_0
  have e_803_1 := arithEq_of_rows h (row := 803) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_1
  simp only [← c50175, ← c50176, k2, ← c50177] at e_803_1
  have hr := e_803_1
  simp only [e_803_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5716)) := by
  have c50178 := (publicBatchWrapper89_copies1568 a h).2.2.1
  have c50179 := (publicBatchWrapper89_copies1568 a h).2.2.2.1
  have c50180 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.1
  have c50181 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.1
  have c50182 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.1
  have c50183 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_2 := arithEq_of_rows h (row := 803) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_2
  simp only [← c50178, ← c50179, ← c50180] at e_803_2
  have e_803_3 := arithEq_of_rows h (row := 803) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_3
  simp only [← c50181, ← c50182, k2, ← c50183] at e_803_3
  have hr := e_803_3
  simp only [e_803_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5717)) := by
  have c50184 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.1
  have c50185 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.1
  have c50186 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50187 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50188 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50189 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_4 := arithEq_of_rows h (row := 803) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_4
  simp only [← c50184, ← c50185, ← c50186] at e_803_4
  have e_803_5 := arithEq_of_rows h (row := 803) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_5
  simp only [← c50187, ← c50188, k2, ← c50189] at e_803_5
  have hr := e_803_5
  simp only [e_803_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5718)) := by
  have c50190 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50191 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50192 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50193 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50194 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50195 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_6 := arithEq_of_rows h (row := 803) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_6
  simp only [← c50190, ← c50191, ← c50192] at e_803_6
  have e_803_7 := arithEq_of_rows h (row := 803) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_7
  simp only [← c50193, ← c50194, k2, ← c50195] at e_803_7
  have hr := e_803_7
  simp only [e_803_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5719)) := by
  have c50196 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50197 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50198 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50199 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50200 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50201 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_8 := arithEq_of_rows h (row := 803) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_8
  simp only [← c50196, ← c50197, ← c50198] at e_803_8
  have e_803_9 := arithEq_of_rows h (row := 803) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_9
  simp only [← c50199, ← c50200, k2, ← c50201] at e_803_9
  have hr := e_803_9
  simp only [e_803_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5720)) := by
  have c50202 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50203 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50204 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50205 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50206 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50207 := (publicBatchWrapper89_copies1568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_10 := arithEq_of_rows h (row := 803) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_10
  simp only [← c50202, ← c50203, ← c50204] at e_803_10
  have e_803_11 := arithEq_of_rows h (row := 803) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_11
  simp only [← c50205, ← c50206, k2, ← c50207] at e_803_11
  have hr := e_803_11
  simp only [e_803_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5721)) := by
  have c50208 := (publicBatchWrapper89_copies1569 a h).1
  have c50209 := (publicBatchWrapper89_copies1569 a h).2.1
  have c50210 := (publicBatchWrapper89_copies1569 a h).2.2.1
  have c50211 := (publicBatchWrapper89_copies1569 a h).2.2.2.1
  have c50212 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.1
  have c50213 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_12 := arithEq_of_rows h (row := 803) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_12
  simp only [← c50208, ← c50209, ← c50210] at e_803_12
  have e_803_13 := arithEq_of_rows h (row := 803) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_13
  simp only [← c50211, ← c50212, k2, ← c50213] at e_803_13
  have hr := e_803_13
  simp only [e_803_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5722)) := by
  have c50214 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.1
  have c50215 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.1
  have c50216 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.1
  have c50217 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.1
  have c50218 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50219 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_14 := arithEq_of_rows h (row := 803) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_14
  simp only [← c50214, ← c50215, ← c50216] at e_803_14
  have e_803_15 := arithEq_of_rows h (row := 803) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_15
  simp only [← c50217, ← c50218, k2, ← c50219] at e_803_15
  have hr := e_803_15
  simp only [e_803_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5723)) := by
  have c50220 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50221 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50222 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50223 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50224 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50225 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_16 := arithEq_of_rows h (row := 803) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_16
  simp only [← c50220, ← c50221, ← c50222] at e_803_16
  have e_803_17 := arithEq_of_rows h (row := 803) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_17
  simp only [← c50223, ← c50224, k2, ← c50225] at e_803_17
  have hr := e_803_17
  simp only [e_803_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 803 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5724)) := by
  have c50226 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50227 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50228 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50229 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50230 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50231 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_803_18 := arithEq_of_rows h (row := 803) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_18
  simp only [← c50226, ← c50227, ← c50228] at e_803_18
  have e_803_19 := arithEq_of_rows h (row := 803) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_803_19
  simp only [← c50229, ← c50230, k2, ← c50231] at e_803_19
  have hr := e_803_19
  simp only [e_803_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5725)) := by
  have c50232 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50233 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50234 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50235 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50236 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50237 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_0 := arithEq_of_rows h (row := 804) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_0
  simp only [← c50232, ← c50233, ← c50234] at e_804_0
  have e_804_1 := arithEq_of_rows h (row := 804) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_1
  simp only [← c50235, ← c50236, k2, ← c50237] at e_804_1
  have hr := e_804_1
  simp only [e_804_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5726)) := by
  have c50238 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50239 := (publicBatchWrapper89_copies1569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50240 := (publicBatchWrapper89_copies1570 a h).1
  have c50241 := (publicBatchWrapper89_copies1570 a h).2.1
  have c50242 := (publicBatchWrapper89_copies1570 a h).2.2.1
  have c50243 := (publicBatchWrapper89_copies1570 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_2 := arithEq_of_rows h (row := 804) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_2
  simp only [← c50238, ← c50239, ← c50240] at e_804_2
  have e_804_3 := arithEq_of_rows h (row := 804) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_3
  simp only [← c50241, ← c50242, k2, ← c50243] at e_804_3
  have hr := e_804_3
  simp only [e_804_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5727)) := by
  have c50244 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.1
  have c50245 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.1
  have c50246 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.1
  have c50247 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.1
  have c50248 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.1
  have c50249 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_4 := arithEq_of_rows h (row := 804) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_4
  simp only [← c50244, ← c50245, ← c50246] at e_804_4
  have e_804_5 := arithEq_of_rows h (row := 804) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_5
  simp only [← c50247, ← c50248, k2, ← c50249] at e_804_5
  have hr := e_804_5
  simp only [e_804_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5728)) := by
  have c50250 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50251 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50252 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50253 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50254 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50255 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_6 := arithEq_of_rows h (row := 804) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_6
  simp only [← c50250, ← c50251, ← c50252] at e_804_6
  have e_804_7 := arithEq_of_rows h (row := 804) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_7
  simp only [← c50253, ← c50254, k2, ← c50255] at e_804_7
  have hr := e_804_7
  simp only [e_804_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5729)) := by
  have c50256 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50257 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50258 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50259 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50260 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50261 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_8 := arithEq_of_rows h (row := 804) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_8
  simp only [← c50256, ← c50257, ← c50258] at e_804_8
  have e_804_9 := arithEq_of_rows h (row := 804) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_9
  simp only [← c50259, ← c50260, k2, ← c50261] at e_804_9
  have hr := e_804_9
  simp only [e_804_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5730)) := by
  have c50262 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50263 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50264 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50265 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50266 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50267 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_10 := arithEq_of_rows h (row := 804) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_10
  simp only [← c50262, ← c50263, ← c50264] at e_804_10
  have e_804_11 := arithEq_of_rows h (row := 804) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_11
  simp only [← c50265, ← c50266, k2, ← c50267] at e_804_11
  have hr := e_804_11
  simp only [e_804_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5731)) := by
  have c50268 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50269 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50270 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50271 := (publicBatchWrapper89_copies1570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50272 := (publicBatchWrapper89_copies1571 a h).1
  have c50273 := (publicBatchWrapper89_copies1571 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_12 := arithEq_of_rows h (row := 804) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_12
  simp only [← c50268, ← c50269, ← c50270] at e_804_12
  have e_804_13 := arithEq_of_rows h (row := 804) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_13
  simp only [← c50271, ← c50272, k2, ← c50273] at e_804_13
  have hr := e_804_13
  simp only [e_804_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5732)) := by
  have c50274 := (publicBatchWrapper89_copies1571 a h).2.2.1
  have c50275 := (publicBatchWrapper89_copies1571 a h).2.2.2.1
  have c50276 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.1
  have c50277 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.1
  have c50278 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.1
  have c50279 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_14 := arithEq_of_rows h (row := 804) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_14
  simp only [← c50274, ← c50275, ← c50276] at e_804_14
  have e_804_15 := arithEq_of_rows h (row := 804) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_15
  simp only [← c50277, ← c50278, k2, ← c50279] at e_804_15
  have hr := e_804_15
  simp only [e_804_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5733)) := by
  have c50280 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.1
  have c50281 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.1
  have c50282 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50283 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50284 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50285 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_16 := arithEq_of_rows h (row := 804) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_16
  simp only [← c50280, ← c50281, ← c50282] at e_804_16
  have e_804_17 := arithEq_of_rows h (row := 804) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_17
  simp only [← c50283, ← c50284, k2, ← c50285] at e_804_17
  have hr := e_804_17
  simp only [e_804_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 804 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5734)) := by
  have c50286 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50287 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50288 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50289 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50290 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50291 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_804_18 := arithEq_of_rows h (row := 804) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_18
  simp only [← c50286, ← c50287, ← c50288] at e_804_18
  have e_804_19 := arithEq_of_rows h (row := 804) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_804_19
  simp only [← c50289, ← c50290, k2, ← c50291] at e_804_19
  have hr := e_804_19
  simp only [e_804_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5735)) := by
  have c50292 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50293 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50294 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50295 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50296 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50297 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_0 := arithEq_of_rows h (row := 805) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_0
  simp only [← c50292, ← c50293, ← c50294] at e_805_0
  have e_805_1 := arithEq_of_rows h (row := 805) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_1
  simp only [← c50295, ← c50296, k2, ← c50297] at e_805_1
  have hr := e_805_1
  simp only [e_805_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5736)) := by
  have c50298 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50299 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50300 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50301 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50302 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50303 := (publicBatchWrapper89_copies1571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_2 := arithEq_of_rows h (row := 805) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_2
  simp only [← c50298, ← c50299, ← c50300] at e_805_2
  have e_805_3 := arithEq_of_rows h (row := 805) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_3
  simp only [← c50301, ← c50302, k2, ← c50303] at e_805_3
  have hr := e_805_3
  simp only [e_805_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5737)) := by
  have c50304 := (publicBatchWrapper89_copies1572 a h).1
  have c50305 := (publicBatchWrapper89_copies1572 a h).2.1
  have c50306 := (publicBatchWrapper89_copies1572 a h).2.2.1
  have c50307 := (publicBatchWrapper89_copies1572 a h).2.2.2.1
  have c50308 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.1
  have c50309 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_4 := arithEq_of_rows h (row := 805) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_4
  simp only [← c50304, ← c50305, ← c50306] at e_805_4
  have e_805_5 := arithEq_of_rows h (row := 805) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_5
  simp only [← c50307, ← c50308, k2, ← c50309] at e_805_5
  have hr := e_805_5
  simp only [e_805_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5738)) := by
  have c50310 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.1
  have c50311 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.1
  have c50312 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.1
  have c50313 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.1
  have c50314 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50315 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_6 := arithEq_of_rows h (row := 805) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_6
  simp only [← c50310, ← c50311, ← c50312] at e_805_6
  have e_805_7 := arithEq_of_rows h (row := 805) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_7
  simp only [← c50313, ← c50314, k2, ← c50315] at e_805_7
  have hr := e_805_7
  simp only [e_805_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5739)) := by
  have c50316 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50317 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50318 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50319 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50320 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50321 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_8 := arithEq_of_rows h (row := 805) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_8
  simp only [← c50316, ← c50317, ← c50318] at e_805_8
  have e_805_9 := arithEq_of_rows h (row := 805) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_9
  simp only [← c50319, ← c50320, k2, ← c50321] at e_805_9
  have hr := e_805_9
  simp only [e_805_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5740)) := by
  have c50322 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50323 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50324 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50325 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50326 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50327 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_10 := arithEq_of_rows h (row := 805) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_10
  simp only [← c50322, ← c50323, ← c50324] at e_805_10
  have e_805_11 := arithEq_of_rows h (row := 805) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_11
  simp only [← c50325, ← c50326, k2, ← c50327] at e_805_11
  have hr := e_805_11
  simp only [e_805_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5741)) := by
  have c50328 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50329 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50330 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50331 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50332 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50333 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_12 := arithEq_of_rows h (row := 805) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_12
  simp only [← c50328, ← c50329, ← c50330] at e_805_12
  have e_805_13 := arithEq_of_rows h (row := 805) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_13
  simp only [← c50331, ← c50332, k2, ← c50333] at e_805_13
  have hr := e_805_13
  simp only [e_805_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5742)) := by
  have c50334 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50335 := (publicBatchWrapper89_copies1572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50336 := (publicBatchWrapper89_copies1573 a h).1
  have c50337 := (publicBatchWrapper89_copies1573 a h).2.1
  have c50338 := (publicBatchWrapper89_copies1573 a h).2.2.1
  have c50339 := (publicBatchWrapper89_copies1573 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_14 := arithEq_of_rows h (row := 805) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_14
  simp only [← c50334, ← c50335, ← c50336] at e_805_14
  have e_805_15 := arithEq_of_rows h (row := 805) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_15
  simp only [← c50337, ← c50338, k2, ← c50339] at e_805_15
  have hr := e_805_15
  simp only [e_805_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5743)) := by
  have c50340 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.1
  have c50341 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.1
  have c50342 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.1
  have c50343 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.1
  have c50344 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.1
  have c50345 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_16 := arithEq_of_rows h (row := 805) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_16
  simp only [← c50340, ← c50341, ← c50342] at e_805_16
  have e_805_17 := arithEq_of_rows h (row := 805) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_17
  simp only [← c50343, ← c50344, k2, ← c50345] at e_805_17
  have hr := e_805_17
  simp only [e_805_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 805 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5744)) := by
  have c50346 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50347 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50348 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50349 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50350 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50351 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_805_18 := arithEq_of_rows h (row := 805) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_18
  simp only [← c50346, ← c50347, ← c50348] at e_805_18
  have e_805_19 := arithEq_of_rows h (row := 805) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_805_19
  simp only [← c50349, ← c50350, k2, ← c50351] at e_805_19
  have hr := e_805_19
  simp only [e_805_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5745)) := by
  have c50352 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50353 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50354 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50355 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50356 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50357 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_0 := arithEq_of_rows h (row := 806) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_0
  simp only [← c50352, ← c50353, ← c50354] at e_806_0
  have e_806_1 := arithEq_of_rows h (row := 806) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_1
  simp only [← c50355, ← c50356, k2, ← c50357] at e_806_1
  have hr := e_806_1
  simp only [e_806_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5746)) := by
  have c50358 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50359 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50360 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50361 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50362 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50363 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_2 := arithEq_of_rows h (row := 806) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_2
  simp only [← c50358, ← c50359, ← c50360] at e_806_2
  have e_806_3 := arithEq_of_rows h (row := 806) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_3
  simp only [← c50361, ← c50362, k2, ← c50363] at e_806_3
  have hr := e_806_3
  simp only [e_806_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5747)) := by
  have c50364 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50365 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50366 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50367 := (publicBatchWrapper89_copies1573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50368 := (publicBatchWrapper89_copies1574 a h).1
  have c50369 := (publicBatchWrapper89_copies1574 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_4 := arithEq_of_rows h (row := 806) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_4
  simp only [← c50364, ← c50365, ← c50366] at e_806_4
  have e_806_5 := arithEq_of_rows h (row := 806) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_5
  simp only [← c50367, ← c50368, k2, ← c50369] at e_806_5
  have hr := e_806_5
  simp only [e_806_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5748)) := by
  have c50370 := (publicBatchWrapper89_copies1574 a h).2.2.1
  have c50371 := (publicBatchWrapper89_copies1574 a h).2.2.2.1
  have c50372 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.1
  have c50373 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.1
  have c50374 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.1
  have c50375 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_6 := arithEq_of_rows h (row := 806) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_6
  simp only [← c50370, ← c50371, ← c50372] at e_806_6
  have e_806_7 := arithEq_of_rows h (row := 806) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_7
  simp only [← c50373, ← c50374, k2, ← c50375] at e_806_7
  have hr := e_806_7
  simp only [e_806_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5749)) := by
  have c50376 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.1
  have c50377 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.1
  have c50378 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50379 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50380 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50381 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_8 := arithEq_of_rows h (row := 806) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_8
  simp only [← c50376, ← c50377, ← c50378] at e_806_8
  have e_806_9 := arithEq_of_rows h (row := 806) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_9
  simp only [← c50379, ← c50380, k2, ← c50381] at e_806_9
  have hr := e_806_9
  simp only [e_806_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5750)) := by
  have c50382 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50383 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50384 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50385 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50386 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50387 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_10 := arithEq_of_rows h (row := 806) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_10
  simp only [← c50382, ← c50383, ← c50384] at e_806_10
  have e_806_11 := arithEq_of_rows h (row := 806) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_11
  simp only [← c50385, ← c50386, k2, ← c50387] at e_806_11
  have hr := e_806_11
  simp only [e_806_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5751)) := by
  have c50388 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50389 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50390 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50391 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50392 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50393 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_12 := arithEq_of_rows h (row := 806) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_12
  simp only [← c50388, ← c50389, ← c50390] at e_806_12
  have e_806_13 := arithEq_of_rows h (row := 806) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_13
  simp only [← c50391, ← c50392, k2, ← c50393] at e_806_13
  have hr := e_806_13
  simp only [e_806_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5752)) := by
  have c50394 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50395 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50396 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50397 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50398 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50399 := (publicBatchWrapper89_copies1574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_14 := arithEq_of_rows h (row := 806) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_14
  simp only [← c50394, ← c50395, ← c50396] at e_806_14
  have e_806_15 := arithEq_of_rows h (row := 806) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_15
  simp only [← c50397, ← c50398, k2, ← c50399] at e_806_15
  have hr := e_806_15
  simp only [e_806_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5753)) := by
  have c50400 := (publicBatchWrapper89_copies1575 a h).1
  have c50401 := (publicBatchWrapper89_copies1575 a h).2.1
  have c50402 := (publicBatchWrapper89_copies1575 a h).2.2.1
  have c50403 := (publicBatchWrapper89_copies1575 a h).2.2.2.1
  have c50404 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.1
  have c50405 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_16 := arithEq_of_rows h (row := 806) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_16
  simp only [← c50400, ← c50401, ← c50402] at e_806_16
  have e_806_17 := arithEq_of_rows h (row := 806) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_17
  simp only [← c50403, ← c50404, k2, ← c50405] at e_806_17
  have hr := e_806_17
  simp only [e_806_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 806 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5754)) := by
  have c50406 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.1
  have c50407 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.1
  have c50408 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.1
  have c50409 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.1
  have c50410 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50411 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_806_18 := arithEq_of_rows h (row := 806) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_18
  simp only [← c50406, ← c50407, ← c50408] at e_806_18
  have e_806_19 := arithEq_of_rows h (row := 806) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_806_19
  simp only [← c50409, ← c50410, k2, ← c50411] at e_806_19
  have hr := e_806_19
  simp only [e_806_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5755)) := by
  have c50412 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50413 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50414 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50415 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50416 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50417 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_0 := arithEq_of_rows h (row := 807) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_0
  simp only [← c50412, ← c50413, ← c50414] at e_807_0
  have e_807_1 := arithEq_of_rows h (row := 807) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_1
  simp only [← c50415, ← c50416, k2, ← c50417] at e_807_1
  have hr := e_807_1
  simp only [e_807_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5756)) := by
  have c50418 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50419 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50420 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50421 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50422 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50423 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_2 := arithEq_of_rows h (row := 807) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_2
  simp only [← c50418, ← c50419, ← c50420] at e_807_2
  have e_807_3 := arithEq_of_rows h (row := 807) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_3
  simp only [← c50421, ← c50422, k2, ← c50423] at e_807_3
  have hr := e_807_3
  simp only [e_807_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5757)) := by
  have c50424 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50425 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50426 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50427 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50428 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50429 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_4 := arithEq_of_rows h (row := 807) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_4
  simp only [← c50424, ← c50425, ← c50426] at e_807_4
  have e_807_5 := arithEq_of_rows h (row := 807) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_5
  simp only [← c50427, ← c50428, k2, ← c50429] at e_807_5
  have hr := e_807_5
  simp only [e_807_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5758)) := by
  have c50430 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50431 := (publicBatchWrapper89_copies1575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50432 := (publicBatchWrapper89_copies1576 a h).1
  have c50433 := (publicBatchWrapper89_copies1576 a h).2.1
  have c50434 := (publicBatchWrapper89_copies1576 a h).2.2.1
  have c50435 := (publicBatchWrapper89_copies1576 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_6 := arithEq_of_rows h (row := 807) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_6
  simp only [← c50430, ← c50431, ← c50432] at e_807_6
  have e_807_7 := arithEq_of_rows h (row := 807) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_7
  simp only [← c50433, ← c50434, k2, ← c50435] at e_807_7
  have hr := e_807_7
  simp only [e_807_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5759)) := by
  have c50436 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.1
  have c50437 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.1
  have c50438 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.1
  have c50439 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.1
  have c50440 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.1
  have c50441 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_8 := arithEq_of_rows h (row := 807) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_8
  simp only [← c50436, ← c50437, ← c50438] at e_807_8
  have e_807_9 := arithEq_of_rows h (row := 807) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_9
  simp only [← c50439, ← c50440, k2, ← c50441] at e_807_9
  have hr := e_807_9
  simp only [e_807_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5760)) := by
  have c50442 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50443 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50444 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50445 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50446 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50447 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_10 := arithEq_of_rows h (row := 807) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_10
  simp only [← c50442, ← c50443, ← c50444] at e_807_10
  have e_807_11 := arithEq_of_rows h (row := 807) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_11
  simp only [← c50445, ← c50446, k2, ← c50447] at e_807_11
  have hr := e_807_11
  simp only [e_807_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5761)) := by
  have c50448 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50449 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50450 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50451 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50452 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50453 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_12 := arithEq_of_rows h (row := 807) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_12
  simp only [← c50448, ← c50449, ← c50450] at e_807_12
  have e_807_13 := arithEq_of_rows h (row := 807) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_13
  simp only [← c50451, ← c50452, k2, ← c50453] at e_807_13
  have hr := e_807_13
  simp only [e_807_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5762)) := by
  have c50454 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50455 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50456 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50457 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50458 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50459 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_14 := arithEq_of_rows h (row := 807) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_14
  simp only [← c50454, ← c50455, ← c50456] at e_807_14
  have e_807_15 := arithEq_of_rows h (row := 807) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_15
  simp only [← c50457, ← c50458, k2, ← c50459] at e_807_15
  have hr := e_807_15
  simp only [e_807_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5763)) := by
  have c50460 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50461 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50462 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50463 := (publicBatchWrapper89_copies1576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50464 := (publicBatchWrapper89_copies1577 a h).1
  have c50465 := (publicBatchWrapper89_copies1577 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_16 := arithEq_of_rows h (row := 807) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_16
  simp only [← c50460, ← c50461, ← c50462] at e_807_16
  have e_807_17 := arithEq_of_rows h (row := 807) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_17
  simp only [← c50463, ← c50464, k2, ← c50465] at e_807_17
  have hr := e_807_17
  simp only [e_807_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 807 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5764)) := by
  have c50466 := (publicBatchWrapper89_copies1577 a h).2.2.1
  have c50467 := (publicBatchWrapper89_copies1577 a h).2.2.2.1
  have c50468 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.1
  have c50469 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.1
  have c50470 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.1
  have c50471 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_807_18 := arithEq_of_rows h (row := 807) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_18
  simp only [← c50466, ← c50467, ← c50468] at e_807_18
  have e_807_19 := arithEq_of_rows h (row := 807) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_807_19
  simp only [← c50469, ← c50470, k2, ← c50471] at e_807_19
  have hr := e_807_19
  simp only [e_807_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5765)) := by
  have c50472 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.1
  have c50473 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.1
  have c50474 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50475 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50476 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50477 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_0 := arithEq_of_rows h (row := 808) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_0
  simp only [← c50472, ← c50473, ← c50474] at e_808_0
  have e_808_1 := arithEq_of_rows h (row := 808) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_1
  simp only [← c50475, ← c50476, k2, ← c50477] at e_808_1
  have hr := e_808_1
  simp only [e_808_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5766)) := by
  have c50478 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50479 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50480 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50481 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50482 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50483 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_2 := arithEq_of_rows h (row := 808) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_2
  simp only [← c50478, ← c50479, ← c50480] at e_808_2
  have e_808_3 := arithEq_of_rows h (row := 808) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_3
  simp only [← c50481, ← c50482, k2, ← c50483] at e_808_3
  have hr := e_808_3
  simp only [e_808_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5767)) := by
  have c50484 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50485 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50486 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50487 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50488 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50489 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_4 := arithEq_of_rows h (row := 808) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_4
  simp only [← c50484, ← c50485, ← c50486] at e_808_4
  have e_808_5 := arithEq_of_rows h (row := 808) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_5
  simp only [← c50487, ← c50488, k2, ← c50489] at e_808_5
  have hr := e_808_5
  simp only [e_808_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5768)) := by
  have c50490 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50491 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50492 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50493 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50494 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50495 := (publicBatchWrapper89_copies1577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_6 := arithEq_of_rows h (row := 808) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_6
  simp only [← c50490, ← c50491, ← c50492] at e_808_6
  have e_808_7 := arithEq_of_rows h (row := 808) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_7
  simp only [← c50493, ← c50494, k2, ← c50495] at e_808_7
  have hr := e_808_7
  simp only [e_808_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5769)) := by
  have c50496 := (publicBatchWrapper89_copies1578 a h).1
  have c50497 := (publicBatchWrapper89_copies1578 a h).2.1
  have c50498 := (publicBatchWrapper89_copies1578 a h).2.2.1
  have c50499 := (publicBatchWrapper89_copies1578 a h).2.2.2.1
  have c50500 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.1
  have c50501 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_8 := arithEq_of_rows h (row := 808) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_8
  simp only [← c50496, ← c50497, ← c50498] at e_808_8
  have e_808_9 := arithEq_of_rows h (row := 808) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_9
  simp only [← c50499, ← c50500, k2, ← c50501] at e_808_9
  have hr := e_808_9
  simp only [e_808_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5810)) := by
  have c50502 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.1
  have c50503 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.1
  have c50504 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.1
  have c50505 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.1
  have c50506 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50507 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_10 := arithEq_of_rows h (row := 808) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_10
  simp only [← c50502, ← c50503, ← c50504] at e_808_10
  have e_808_11 := arithEq_of_rows h (row := 808) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_11
  simp only [← c50505, ← c50506, k2, ← c50507] at e_808_11
  have hr := e_808_11
  simp only [e_808_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5811)) := by
  have c50508 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50509 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50510 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50511 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50512 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50513 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_12 := arithEq_of_rows h (row := 808) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_12
  simp only [← c50508, ← c50509, ← c50510] at e_808_12
  have e_808_13 := arithEq_of_rows h (row := 808) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_13
  simp only [← c50511, ← c50512, k2, ← c50513] at e_808_13
  have hr := e_808_13
  simp only [e_808_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5812)) := by
  have c50514 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50515 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50516 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50517 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50518 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50519 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_14 := arithEq_of_rows h (row := 808) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_14
  simp only [← c50514, ← c50515, ← c50516] at e_808_14
  have e_808_15 := arithEq_of_rows h (row := 808) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_15
  simp only [← c50517, ← c50518, k2, ← c50519] at e_808_15
  have hr := e_808_15
  simp only [e_808_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5813)) := by
  have c50520 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50521 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50522 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50523 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50524 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50525 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_16 := arithEq_of_rows h (row := 808) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_16
  simp only [← c50520, ← c50521, ← c50522] at e_808_16
  have e_808_17 := arithEq_of_rows h (row := 808) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_17
  simp only [← c50523, ← c50524, k2, ← c50525] at e_808_17
  have hr := e_808_17
  simp only [e_808_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 808 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5814)) := by
  have c50526 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50527 := (publicBatchWrapper89_copies1578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50528 := (publicBatchWrapper89_copies1579 a h).1
  have c50529 := (publicBatchWrapper89_copies1579 a h).2.1
  have c50530 := (publicBatchWrapper89_copies1579 a h).2.2.1
  have c50531 := (publicBatchWrapper89_copies1579 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_808_18 := arithEq_of_rows h (row := 808) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_18
  simp only [← c50526, ← c50527, ← c50528] at e_808_18
  have e_808_19 := arithEq_of_rows h (row := 808) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_808_19
  simp only [← c50529, ← c50530, k2, ← c50531] at e_808_19
  have hr := e_808_19
  simp only [e_808_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5815)) := by
  have c50532 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.1
  have c50533 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.1
  have c50534 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.1
  have c50535 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.1
  have c50536 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.1
  have c50537 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_0 := arithEq_of_rows h (row := 809) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_0
  simp only [← c50532, ← c50533, ← c50534] at e_809_0
  have e_809_1 := arithEq_of_rows h (row := 809) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_1
  simp only [← c50535, ← c50536, k2, ← c50537] at e_809_1
  have hr := e_809_1
  simp only [e_809_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5816)) := by
  have c50538 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50539 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50540 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50541 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50542 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50543 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_2 := arithEq_of_rows h (row := 809) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_2
  simp only [← c50538, ← c50539, ← c50540] at e_809_2
  have e_809_3 := arithEq_of_rows h (row := 809) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_3
  simp only [← c50541, ← c50542, k2, ← c50543] at e_809_3
  have hr := e_809_3
  simp only [e_809_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5817)) := by
  have c50544 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50545 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50546 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50547 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50548 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50549 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_4 := arithEq_of_rows h (row := 809) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_4
  simp only [← c50544, ← c50545, ← c50546] at e_809_4
  have e_809_5 := arithEq_of_rows h (row := 809) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_5
  simp only [← c50547, ← c50548, k2, ← c50549] at e_809_5
  have hr := e_809_5
  simp only [e_809_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5818)) := by
  have c50550 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50551 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50552 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50553 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50554 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50555 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_6 := arithEq_of_rows h (row := 809) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_6
  simp only [← c50550, ← c50551, ← c50552] at e_809_6
  have e_809_7 := arithEq_of_rows h (row := 809) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_7
  simp only [← c50553, ← c50554, k2, ← c50555] at e_809_7
  have hr := e_809_7
  simp only [e_809_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5819)) := by
  have c50556 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50557 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50558 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50559 := (publicBatchWrapper89_copies1579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50560 := (publicBatchWrapper89_copies1580 a h).1
  have c50561 := (publicBatchWrapper89_copies1580 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_8 := arithEq_of_rows h (row := 809) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_8
  simp only [← c50556, ← c50557, ← c50558] at e_809_8
  have e_809_9 := arithEq_of_rows h (row := 809) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_9
  simp only [← c50559, ← c50560, k2, ← c50561] at e_809_9
  have hr := e_809_9
  simp only [e_809_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5820)) := by
  have c50562 := (publicBatchWrapper89_copies1580 a h).2.2.1
  have c50563 := (publicBatchWrapper89_copies1580 a h).2.2.2.1
  have c50564 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.1
  have c50565 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.1
  have c50566 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.1
  have c50567 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_10 := arithEq_of_rows h (row := 809) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_10
  simp only [← c50562, ← c50563, ← c50564] at e_809_10
  have e_809_11 := arithEq_of_rows h (row := 809) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_11
  simp only [← c50565, ← c50566, k2, ← c50567] at e_809_11
  have hr := e_809_11
  simp only [e_809_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5821)) := by
  have c50568 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.1
  have c50569 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.1
  have c50570 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50571 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50572 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50573 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_12 := arithEq_of_rows h (row := 809) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_12
  simp only [← c50568, ← c50569, ← c50570] at e_809_12
  have e_809_13 := arithEq_of_rows h (row := 809) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_13
  simp only [← c50571, ← c50572, k2, ← c50573] at e_809_13
  have hr := e_809_13
  simp only [e_809_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5822)) := by
  have c50574 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50575 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50576 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50577 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50578 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50579 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_14 := arithEq_of_rows h (row := 809) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_14
  simp only [← c50574, ← c50575, ← c50576] at e_809_14
  have e_809_15 := arithEq_of_rows h (row := 809) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_15
  simp only [← c50577, ← c50578, k2, ← c50579] at e_809_15
  have hr := e_809_15
  simp only [e_809_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5823)) := by
  have c50580 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50581 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50582 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50583 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50584 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50585 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_16 := arithEq_of_rows h (row := 809) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_16
  simp only [← c50580, ← c50581, ← c50582] at e_809_16
  have e_809_17 := arithEq_of_rows h (row := 809) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_17
  simp only [← c50583, ← c50584, k2, ← c50585] at e_809_17
  have hr := e_809_17
  simp only [e_809_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 809 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5824)) := by
  have c50586 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50587 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50588 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50589 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50590 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50591 := (publicBatchWrapper89_copies1580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_809_18 := arithEq_of_rows h (row := 809) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_18
  simp only [← c50586, ← c50587, ← c50588] at e_809_18
  have e_809_19 := arithEq_of_rows h (row := 809) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_809_19
  simp only [← c50589, ← c50590, k2, ← c50591] at e_809_19
  have hr := e_809_19
  simp only [e_809_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5825)) := by
  have c50592 := (publicBatchWrapper89_copies1581 a h).1
  have c50593 := (publicBatchWrapper89_copies1581 a h).2.1
  have c50594 := (publicBatchWrapper89_copies1581 a h).2.2.1
  have c50595 := (publicBatchWrapper89_copies1581 a h).2.2.2.1
  have c50596 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.1
  have c50597 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_0 := arithEq_of_rows h (row := 810) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_0
  simp only [← c50592, ← c50593, ← c50594] at e_810_0
  have e_810_1 := arithEq_of_rows h (row := 810) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_1
  simp only [← c50595, ← c50596, k2, ← c50597] at e_810_1
  have hr := e_810_1
  simp only [e_810_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5826)) := by
  have c50598 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.1
  have c50599 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.1
  have c50600 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.1
  have c50601 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.1
  have c50602 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50603 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_2 := arithEq_of_rows h (row := 810) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_2
  simp only [← c50598, ← c50599, ← c50600] at e_810_2
  have e_810_3 := arithEq_of_rows h (row := 810) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_3
  simp only [← c50601, ← c50602, k2, ← c50603] at e_810_3
  have hr := e_810_3
  simp only [e_810_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5827)) := by
  have c50604 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50605 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50606 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50607 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50608 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50609 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_4 := arithEq_of_rows h (row := 810) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_4
  simp only [← c50604, ← c50605, ← c50606] at e_810_4
  have e_810_5 := arithEq_of_rows h (row := 810) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_5
  simp only [← c50607, ← c50608, k2, ← c50609] at e_810_5
  have hr := e_810_5
  simp only [e_810_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5828)) := by
  have c50610 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50611 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50612 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50613 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50614 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50615 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_6 := arithEq_of_rows h (row := 810) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_6
  simp only [← c50610, ← c50611, ← c50612] at e_810_6
  have e_810_7 := arithEq_of_rows h (row := 810) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_7
  simp only [← c50613, ← c50614, k2, ← c50615] at e_810_7
  have hr := e_810_7
  simp only [e_810_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5829)) := by
  have c50616 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50617 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50618 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50619 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50620 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50621 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_8 := arithEq_of_rows h (row := 810) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_8
  simp only [← c50616, ← c50617, ← c50618] at e_810_8
  have e_810_9 := arithEq_of_rows h (row := 810) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_9
  simp only [← c50619, ← c50620, k2, ← c50621] at e_810_9
  have hr := e_810_9
  simp only [e_810_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5830)) := by
  have c50622 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50623 := (publicBatchWrapper89_copies1581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50624 := (publicBatchWrapper89_copies1582 a h).1
  have c50625 := (publicBatchWrapper89_copies1582 a h).2.1
  have c50626 := (publicBatchWrapper89_copies1582 a h).2.2.1
  have c50627 := (publicBatchWrapper89_copies1582 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_10 := arithEq_of_rows h (row := 810) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_10
  simp only [← c50622, ← c50623, ← c50624] at e_810_10
  have e_810_11 := arithEq_of_rows h (row := 810) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_11
  simp only [← c50625, ← c50626, k2, ← c50627] at e_810_11
  have hr := e_810_11
  simp only [e_810_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5831)) := by
  have c50628 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.1
  have c50629 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.1
  have c50630 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.1
  have c50631 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.1
  have c50632 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.1
  have c50633 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_12 := arithEq_of_rows h (row := 810) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_12
  simp only [← c50628, ← c50629, ← c50630] at e_810_12
  have e_810_13 := arithEq_of_rows h (row := 810) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_13
  simp only [← c50631, ← c50632, k2, ← c50633] at e_810_13
  have hr := e_810_13
  simp only [e_810_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5832)) := by
  have c50634 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50635 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50636 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50637 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50638 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50639 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_14 := arithEq_of_rows h (row := 810) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_14
  simp only [← c50634, ← c50635, ← c50636] at e_810_14
  have e_810_15 := arithEq_of_rows h (row := 810) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_15
  simp only [← c50637, ← c50638, k2, ← c50639] at e_810_15
  have hr := e_810_15
  simp only [e_810_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5833)) := by
  have c50640 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50641 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50642 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50643 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50644 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50645 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_16 := arithEq_of_rows h (row := 810) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_16
  simp only [← c50640, ← c50641, ← c50642] at e_810_16
  have e_810_17 := arithEq_of_rows h (row := 810) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_17
  simp only [← c50643, ← c50644, k2, ← c50645] at e_810_17
  have hr := e_810_17
  simp only [e_810_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 810 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5834)) := by
  have c50646 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50647 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50648 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50649 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50650 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50651 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_810_18 := arithEq_of_rows h (row := 810) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_18
  simp only [← c50646, ← c50647, ← c50648] at e_810_18
  have e_810_19 := arithEq_of_rows h (row := 810) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_810_19
  simp only [← c50649, ← c50650, k2, ← c50651] at e_810_19
  have hr := e_810_19
  simp only [e_810_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5835)) := by
  have c50652 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50653 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50654 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50655 := (publicBatchWrapper89_copies1582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50656 := (publicBatchWrapper89_copies1583 a h).1
  have c50657 := (publicBatchWrapper89_copies1583 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_0 := arithEq_of_rows h (row := 811) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_0
  simp only [← c50652, ← c50653, ← c50654] at e_811_0
  have e_811_1 := arithEq_of_rows h (row := 811) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_1
  simp only [← c50655, ← c50656, k2, ← c50657] at e_811_1
  have hr := e_811_1
  simp only [e_811_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5836)) := by
  have c50658 := (publicBatchWrapper89_copies1583 a h).2.2.1
  have c50659 := (publicBatchWrapper89_copies1583 a h).2.2.2.1
  have c50660 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.1
  have c50661 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.1
  have c50662 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.1
  have c50663 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_2 := arithEq_of_rows h (row := 811) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_2
  simp only [← c50658, ← c50659, ← c50660] at e_811_2
  have e_811_3 := arithEq_of_rows h (row := 811) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_3
  simp only [← c50661, ← c50662, k2, ← c50663] at e_811_3
  have hr := e_811_3
  simp only [e_811_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5837)) := by
  have c50664 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.1
  have c50665 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.1
  have c50666 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50667 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50668 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50669 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_4 := arithEq_of_rows h (row := 811) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_4
  simp only [← c50664, ← c50665, ← c50666] at e_811_4
  have e_811_5 := arithEq_of_rows h (row := 811) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_5
  simp only [← c50667, ← c50668, k2, ← c50669] at e_811_5
  have hr := e_811_5
  simp only [e_811_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5838)) := by
  have c50670 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50671 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50672 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50673 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50674 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50675 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_6 := arithEq_of_rows h (row := 811) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_6
  simp only [← c50670, ← c50671, ← c50672] at e_811_6
  have e_811_7 := arithEq_of_rows h (row := 811) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_7
  simp only [← c50673, ← c50674, k2, ← c50675] at e_811_7
  have hr := e_811_7
  simp only [e_811_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5839)) := by
  have c50676 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50677 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50678 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50679 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50680 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50681 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_8 := arithEq_of_rows h (row := 811) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_8
  simp only [← c50676, ← c50677, ← c50678] at e_811_8
  have e_811_9 := arithEq_of_rows h (row := 811) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_9
  simp only [← c50679, ← c50680, k2, ← c50681] at e_811_9
  have hr := e_811_9
  simp only [e_811_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5840)) := by
  have c50682 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50683 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50684 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50685 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50686 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50687 := (publicBatchWrapper89_copies1583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_10 := arithEq_of_rows h (row := 811) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_10
  simp only [← c50682, ← c50683, ← c50684] at e_811_10
  have e_811_11 := arithEq_of_rows h (row := 811) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_11
  simp only [← c50685, ← c50686, k2, ← c50687] at e_811_11
  have hr := e_811_11
  simp only [e_811_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5841)) := by
  have c50688 := (publicBatchWrapper89_copies1584 a h).1
  have c50689 := (publicBatchWrapper89_copies1584 a h).2.1
  have c50690 := (publicBatchWrapper89_copies1584 a h).2.2.1
  have c50691 := (publicBatchWrapper89_copies1584 a h).2.2.2.1
  have c50692 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.1
  have c50693 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_12 := arithEq_of_rows h (row := 811) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_12
  simp only [← c50688, ← c50689, ← c50690] at e_811_12
  have e_811_13 := arithEq_of_rows h (row := 811) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_13
  simp only [← c50691, ← c50692, k2, ← c50693] at e_811_13
  have hr := e_811_13
  simp only [e_811_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5842)) := by
  have c50694 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.1
  have c50695 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.1
  have c50696 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.1
  have c50697 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.1
  have c50698 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50699 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_14 := arithEq_of_rows h (row := 811) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_14
  simp only [← c50694, ← c50695, ← c50696] at e_811_14
  have e_811_15 := arithEq_of_rows h (row := 811) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_15
  simp only [← c50697, ← c50698, k2, ← c50699] at e_811_15
  have hr := e_811_15
  simp only [e_811_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5843)) := by
  have c50700 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50701 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50702 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50703 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50704 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50705 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_16 := arithEq_of_rows h (row := 811) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_16
  simp only [← c50700, ← c50701, ← c50702] at e_811_16
  have e_811_17 := arithEq_of_rows h (row := 811) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_17
  simp only [← c50703, ← c50704, k2, ← c50705] at e_811_17
  have hr := e_811_17
  simp only [e_811_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 811 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5844)) := by
  have c50706 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50707 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50708 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50709 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50710 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50711 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_811_18 := arithEq_of_rows h (row := 811) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_18
  simp only [← c50706, ← c50707, ← c50708] at e_811_18
  have e_811_19 := arithEq_of_rows h (row := 811) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_811_19
  simp only [← c50709, ← c50710, k2, ← c50711] at e_811_19
  have hr := e_811_19
  simp only [e_811_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5845)) := by
  have c50712 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50713 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50714 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50715 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50716 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50717 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_0 := arithEq_of_rows h (row := 812) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_0
  simp only [← c50712, ← c50713, ← c50714] at e_812_0
  have e_812_1 := arithEq_of_rows h (row := 812) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_1
  simp only [← c50715, ← c50716, k2, ← c50717] at e_812_1
  have hr := e_812_1
  simp only [e_812_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5846)) := by
  have c50718 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50719 := (publicBatchWrapper89_copies1584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50720 := (publicBatchWrapper89_copies1585 a h).1
  have c50721 := (publicBatchWrapper89_copies1585 a h).2.1
  have c50722 := (publicBatchWrapper89_copies1585 a h).2.2.1
  have c50723 := (publicBatchWrapper89_copies1585 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_2 := arithEq_of_rows h (row := 812) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_2
  simp only [← c50718, ← c50719, ← c50720] at e_812_2
  have e_812_3 := arithEq_of_rows h (row := 812) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_3
  simp only [← c50721, ← c50722, k2, ← c50723] at e_812_3
  have hr := e_812_3
  simp only [e_812_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5847)) := by
  have c50724 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.1
  have c50725 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.1
  have c50726 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.1
  have c50727 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.1
  have c50728 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.1
  have c50729 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_4 := arithEq_of_rows h (row := 812) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_4
  simp only [← c50724, ← c50725, ← c50726] at e_812_4
  have e_812_5 := arithEq_of_rows h (row := 812) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_5
  simp only [← c50727, ← c50728, k2, ← c50729] at e_812_5
  have hr := e_812_5
  simp only [e_812_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5848)) := by
  have c50730 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50731 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50732 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50733 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50734 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50735 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_6 := arithEq_of_rows h (row := 812) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_6
  simp only [← c50730, ← c50731, ← c50732] at e_812_6
  have e_812_7 := arithEq_of_rows h (row := 812) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_7
  simp only [← c50733, ← c50734, k2, ← c50735] at e_812_7
  have hr := e_812_7
  simp only [e_812_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5849)) := by
  have c50736 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50737 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50738 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50739 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50740 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50741 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_8 := arithEq_of_rows h (row := 812) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_8
  simp only [← c50736, ← c50737, ← c50738] at e_812_8
  have e_812_9 := arithEq_of_rows h (row := 812) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_9
  simp only [← c50739, ← c50740, k2, ← c50741] at e_812_9
  have hr := e_812_9
  simp only [e_812_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5850)) := by
  have c50742 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50743 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50744 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50745 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50746 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50747 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_10 := arithEq_of_rows h (row := 812) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_10
  simp only [← c50742, ← c50743, ← c50744] at e_812_10
  have e_812_11 := arithEq_of_rows h (row := 812) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_11
  simp only [← c50745, ← c50746, k2, ← c50747] at e_812_11
  have hr := e_812_11
  simp only [e_812_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5851)) := by
  have c50748 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50749 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50750 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50751 := (publicBatchWrapper89_copies1585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50752 := (publicBatchWrapper89_copies1586 a h).1
  have c50753 := (publicBatchWrapper89_copies1586 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_12 := arithEq_of_rows h (row := 812) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_12
  simp only [← c50748, ← c50749, ← c50750] at e_812_12
  have e_812_13 := arithEq_of_rows h (row := 812) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_13
  simp only [← c50751, ← c50752, k2, ← c50753] at e_812_13
  have hr := e_812_13
  simp only [e_812_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5852)) := by
  have c50754 := (publicBatchWrapper89_copies1586 a h).2.2.1
  have c50755 := (publicBatchWrapper89_copies1586 a h).2.2.2.1
  have c50756 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.1
  have c50757 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.1
  have c50758 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.1
  have c50759 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_14 := arithEq_of_rows h (row := 812) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_14
  simp only [← c50754, ← c50755, ← c50756] at e_812_14
  have e_812_15 := arithEq_of_rows h (row := 812) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_15
  simp only [← c50757, ← c50758, k2, ← c50759] at e_812_15
  have hr := e_812_15
  simp only [e_812_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5853)) := by
  have c50760 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.1
  have c50761 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.1
  have c50762 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50763 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50764 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50765 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_16 := arithEq_of_rows h (row := 812) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_16
  simp only [← c50760, ← c50761, ← c50762] at e_812_16
  have e_812_17 := arithEq_of_rows h (row := 812) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_17
  simp only [← c50763, ← c50764, k2, ← c50765] at e_812_17
  have hr := e_812_17
  simp only [e_812_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 812 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5854)) := by
  have c50766 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50767 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50768 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50769 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50770 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50771 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_812_18 := arithEq_of_rows h (row := 812) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_18
  simp only [← c50766, ← c50767, ← c50768] at e_812_18
  have e_812_19 := arithEq_of_rows h (row := 812) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_812_19
  simp only [← c50769, ← c50770, k2, ← c50771] at e_812_19
  have hr := e_812_19
  simp only [e_812_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5855)) := by
  have c50772 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50773 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50774 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50775 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50776 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50777 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_0 := arithEq_of_rows h (row := 813) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_0
  simp only [← c50772, ← c50773, ← c50774] at e_813_0
  have e_813_1 := arithEq_of_rows h (row := 813) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_1
  simp only [← c50775, ← c50776, k2, ← c50777] at e_813_1
  have hr := e_813_1
  simp only [e_813_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5856)) := by
  have c50778 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50779 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50780 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50781 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50782 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50783 := (publicBatchWrapper89_copies1586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_2 := arithEq_of_rows h (row := 813) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_2
  simp only [← c50778, ← c50779, ← c50780] at e_813_2
  have e_813_3 := arithEq_of_rows h (row := 813) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_3
  simp only [← c50781, ← c50782, k2, ← c50783] at e_813_3
  have hr := e_813_3
  simp only [e_813_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5857)) := by
  have c50784 := (publicBatchWrapper89_copies1587 a h).1
  have c50785 := (publicBatchWrapper89_copies1587 a h).2.1
  have c50786 := (publicBatchWrapper89_copies1587 a h).2.2.1
  have c50787 := (publicBatchWrapper89_copies1587 a h).2.2.2.1
  have c50788 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.1
  have c50789 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_4 := arithEq_of_rows h (row := 813) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_4
  simp only [← c50784, ← c50785, ← c50786] at e_813_4
  have e_813_5 := arithEq_of_rows h (row := 813) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_5
  simp only [← c50787, ← c50788, k2, ← c50789] at e_813_5
  have hr := e_813_5
  simp only [e_813_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5858)) := by
  have c50790 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.1
  have c50791 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.1
  have c50792 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.1
  have c50793 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.1
  have c50794 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50795 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_6 := arithEq_of_rows h (row := 813) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_6
  simp only [← c50790, ← c50791, ← c50792] at e_813_6
  have e_813_7 := arithEq_of_rows h (row := 813) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_7
  simp only [← c50793, ← c50794, k2, ← c50795] at e_813_7
  have hr := e_813_7
  simp only [e_813_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5859)) := by
  have c50796 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50797 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50798 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50799 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50800 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50801 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_8 := arithEq_of_rows h (row := 813) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_8
  simp only [← c50796, ← c50797, ← c50798] at e_813_8
  have e_813_9 := arithEq_of_rows h (row := 813) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_9
  simp only [← c50799, ← c50800, k2, ← c50801] at e_813_9
  have hr := e_813_9
  simp only [e_813_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5860)) := by
  have c50802 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50803 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50804 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50805 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50806 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50807 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_10 := arithEq_of_rows h (row := 813) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_10
  simp only [← c50802, ← c50803, ← c50804] at e_813_10
  have e_813_11 := arithEq_of_rows h (row := 813) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_11
  simp only [← c50805, ← c50806, k2, ← c50807] at e_813_11
  have hr := e_813_11
  simp only [e_813_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5861)) := by
  have c50808 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50809 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50810 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50811 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50812 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50813 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_12 := arithEq_of_rows h (row := 813) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_12
  simp only [← c50808, ← c50809, ← c50810] at e_813_12
  have e_813_13 := arithEq_of_rows h (row := 813) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_13
  simp only [← c50811, ← c50812, k2, ← c50813] at e_813_13
  have hr := e_813_13
  simp only [e_813_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5862)) := by
  have c50814 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50815 := (publicBatchWrapper89_copies1587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50816 := (publicBatchWrapper89_copies1588 a h).1
  have c50817 := (publicBatchWrapper89_copies1588 a h).2.1
  have c50818 := (publicBatchWrapper89_copies1588 a h).2.2.1
  have c50819 := (publicBatchWrapper89_copies1588 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_14 := arithEq_of_rows h (row := 813) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_14
  simp only [← c50814, ← c50815, ← c50816] at e_813_14
  have e_813_15 := arithEq_of_rows h (row := 813) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_15
  simp only [← c50817, ← c50818, k2, ← c50819] at e_813_15
  have hr := e_813_15
  simp only [e_813_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5863)) := by
  have c50820 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.1
  have c50821 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.1
  have c50822 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.1
  have c50823 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.1
  have c50824 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.1
  have c50825 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_16 := arithEq_of_rows h (row := 813) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_16
  simp only [← c50820, ← c50821, ← c50822] at e_813_16
  have e_813_17 := arithEq_of_rows h (row := 813) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_17
  simp only [← c50823, ← c50824, k2, ← c50825] at e_813_17
  have hr := e_813_17
  simp only [e_813_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 813 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5864)) := by
  have c50826 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50827 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50828 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50829 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50830 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50831 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_813_18 := arithEq_of_rows h (row := 813) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_18
  simp only [← c50826, ← c50827, ← c50828] at e_813_18
  have e_813_19 := arithEq_of_rows h (row := 813) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_813_19
  simp only [← c50829, ← c50830, k2, ← c50831] at e_813_19
  have hr := e_813_19
  simp only [e_813_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5865)) := by
  have c50832 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50833 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50834 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50835 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50836 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50837 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_0 := arithEq_of_rows h (row := 814) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_0
  simp only [← c50832, ← c50833, ← c50834] at e_814_0
  have e_814_1 := arithEq_of_rows h (row := 814) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_1
  simp only [← c50835, ← c50836, k2, ← c50837] at e_814_1
  have hr := e_814_1
  simp only [e_814_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5866)) := by
  have c50838 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50839 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50840 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50841 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50842 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50843 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_2 := arithEq_of_rows h (row := 814) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_2
  simp only [← c50838, ← c50839, ← c50840] at e_814_2
  have e_814_3 := arithEq_of_rows h (row := 814) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_3
  simp only [← c50841, ← c50842, k2, ← c50843] at e_814_3
  have hr := e_814_3
  simp only [e_814_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5867)) := by
  have c50844 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50845 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50846 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50847 := (publicBatchWrapper89_copies1588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50848 := (publicBatchWrapper89_copies1589 a h).1
  have c50849 := (publicBatchWrapper89_copies1589 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_4 := arithEq_of_rows h (row := 814) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_4
  simp only [← c50844, ← c50845, ← c50846] at e_814_4
  have e_814_5 := arithEq_of_rows h (row := 814) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_5
  simp only [← c50847, ← c50848, k2, ← c50849] at e_814_5
  have hr := e_814_5
  simp only [e_814_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5868)) := by
  have c50850 := (publicBatchWrapper89_copies1589 a h).2.2.1
  have c50851 := (publicBatchWrapper89_copies1589 a h).2.2.2.1
  have c50852 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.1
  have c50853 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.1
  have c50854 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.1
  have c50855 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_6 := arithEq_of_rows h (row := 814) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_6
  simp only [← c50850, ← c50851, ← c50852] at e_814_6
  have e_814_7 := arithEq_of_rows h (row := 814) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_7
  simp only [← c50853, ← c50854, k2, ← c50855] at e_814_7
  have hr := e_814_7
  simp only [e_814_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5869)) := by
  have c50856 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.1
  have c50857 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.1
  have c50858 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50859 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50860 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50861 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_8 := arithEq_of_rows h (row := 814) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_8
  simp only [← c50856, ← c50857, ← c50858] at e_814_8
  have e_814_9 := arithEq_of_rows h (row := 814) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_9
  simp only [← c50859, ← c50860, k2, ← c50861] at e_814_9
  have hr := e_814_9
  simp only [e_814_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5870)) := by
  have c50862 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50863 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50864 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50865 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50866 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50867 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_10 := arithEq_of_rows h (row := 814) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_10
  simp only [← c50862, ← c50863, ← c50864] at e_814_10
  have e_814_11 := arithEq_of_rows h (row := 814) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_11
  simp only [← c50865, ← c50866, k2, ← c50867] at e_814_11
  have hr := e_814_11
  simp only [e_814_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5871)) := by
  have c50868 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50869 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50870 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50871 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50872 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50873 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_12 := arithEq_of_rows h (row := 814) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_12
  simp only [← c50868, ← c50869, ← c50870] at e_814_12
  have e_814_13 := arithEq_of_rows h (row := 814) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_13
  simp only [← c50871, ← c50872, k2, ← c50873] at e_814_13
  have hr := e_814_13
  simp only [e_814_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5872)) := by
  have c50874 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50875 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50876 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50877 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50878 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50879 := (publicBatchWrapper89_copies1589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_14 := arithEq_of_rows h (row := 814) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_14
  simp only [← c50874, ← c50875, ← c50876] at e_814_14
  have e_814_15 := arithEq_of_rows h (row := 814) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_15
  simp only [← c50877, ← c50878, k2, ← c50879] at e_814_15
  have hr := e_814_15
  simp only [e_814_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5873)) := by
  have c50880 := (publicBatchWrapper89_copies1590 a h).1
  have c50881 := (publicBatchWrapper89_copies1590 a h).2.1
  have c50882 := (publicBatchWrapper89_copies1590 a h).2.2.1
  have c50883 := (publicBatchWrapper89_copies1590 a h).2.2.2.1
  have c50884 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.1
  have c50885 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_16 := arithEq_of_rows h (row := 814) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_16
  simp only [← c50880, ← c50881, ← c50882] at e_814_16
  have e_814_17 := arithEq_of_rows h (row := 814) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_17
  simp only [← c50883, ← c50884, k2, ← c50885] at e_814_17
  have hr := e_814_17
  simp only [e_814_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 814 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5874)) := by
  have c50886 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.1
  have c50887 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.1
  have c50888 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.1
  have c50889 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.1
  have c50890 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50891 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_814_18 := arithEq_of_rows h (row := 814) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_18
  simp only [← c50886, ← c50887, ← c50888] at e_814_18
  have e_814_19 := arithEq_of_rows h (row := 814) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_814_19
  simp only [← c50889, ← c50890, k2, ← c50891] at e_814_19
  have hr := e_814_19
  simp only [e_814_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5875)) := by
  have c50892 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50893 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50894 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50895 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50896 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50897 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_0 := arithEq_of_rows h (row := 815) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_0
  simp only [← c50892, ← c50893, ← c50894] at e_815_0
  have e_815_1 := arithEq_of_rows h (row := 815) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_1
  simp only [← c50895, ← c50896, k2, ← c50897] at e_815_1
  have hr := e_815_1
  simp only [e_815_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5876)) := by
  have c50898 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50899 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50900 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50901 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50902 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50903 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_2 := arithEq_of_rows h (row := 815) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_2
  simp only [← c50898, ← c50899, ← c50900] at e_815_2
  have e_815_3 := arithEq_of_rows h (row := 815) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_3
  simp only [← c50901, ← c50902, k2, ← c50903] at e_815_3
  have hr := e_815_3
  simp only [e_815_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5877)) := by
  have c50904 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50905 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50906 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50907 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50908 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50909 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_4 := arithEq_of_rows h (row := 815) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_4
  simp only [← c50904, ← c50905, ← c50906] at e_815_4
  have e_815_5 := arithEq_of_rows h (row := 815) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_5
  simp only [← c50907, ← c50908, k2, ← c50909] at e_815_5
  have hr := e_815_5
  simp only [e_815_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5878)) := by
  have c50910 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50911 := (publicBatchWrapper89_copies1590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50912 := (publicBatchWrapper89_copies1591 a h).1
  have c50913 := (publicBatchWrapper89_copies1591 a h).2.1
  have c50914 := (publicBatchWrapper89_copies1591 a h).2.2.1
  have c50915 := (publicBatchWrapper89_copies1591 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_6 := arithEq_of_rows h (row := 815) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_6
  simp only [← c50910, ← c50911, ← c50912] at e_815_6
  have e_815_7 := arithEq_of_rows h (row := 815) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_7
  simp only [← c50913, ← c50914, k2, ← c50915] at e_815_7
  have hr := e_815_7
  simp only [e_815_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5879)) := by
  have c50916 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.1
  have c50917 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.1
  have c50918 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.1
  have c50919 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.1
  have c50920 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.1
  have c50921 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_8 := arithEq_of_rows h (row := 815) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_8
  simp only [← c50916, ← c50917, ← c50918] at e_815_8
  have e_815_9 := arithEq_of_rows h (row := 815) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_9
  simp only [← c50919, ← c50920, k2, ← c50921] at e_815_9
  have hr := e_815_9
  simp only [e_815_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5880)) := by
  have c50922 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50923 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50924 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50925 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50926 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50927 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_10 := arithEq_of_rows h (row := 815) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_10
  simp only [← c50922, ← c50923, ← c50924] at e_815_10
  have e_815_11 := arithEq_of_rows h (row := 815) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_11
  simp only [← c50925, ← c50926, k2, ← c50927] at e_815_11
  have hr := e_815_11
  simp only [e_815_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5881)) := by
  have c50928 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50929 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50930 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50931 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50932 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50933 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_12 := arithEq_of_rows h (row := 815) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_12
  simp only [← c50928, ← c50929, ← c50930] at e_815_12
  have e_815_13 := arithEq_of_rows h (row := 815) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_13
  simp only [← c50931, ← c50932, k2, ← c50933] at e_815_13
  have hr := e_815_13
  simp only [e_815_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5882)) := by
  have c50934 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50935 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50936 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50937 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50938 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50939 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_14 := arithEq_of_rows h (row := 815) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_14
  simp only [← c50934, ← c50935, ← c50936] at e_815_14
  have e_815_15 := arithEq_of_rows h (row := 815) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_15
  simp only [← c50937, ← c50938, k2, ← c50939] at e_815_15
  have hr := e_815_15
  simp only [e_815_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5883)) := by
  have c50940 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50941 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50942 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50943 := (publicBatchWrapper89_copies1591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50944 := (publicBatchWrapper89_copies1592 a h).1
  have c50945 := (publicBatchWrapper89_copies1592 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_16 := arithEq_of_rows h (row := 815) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_16
  simp only [← c50940, ← c50941, ← c50942] at e_815_16
  have e_815_17 := arithEq_of_rows h (row := 815) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_17
  simp only [← c50943, ← c50944, k2, ← c50945] at e_815_17
  have hr := e_815_17
  simp only [e_815_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 815 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5884)) := by
  have c50946 := (publicBatchWrapper89_copies1592 a h).2.2.1
  have c50947 := (publicBatchWrapper89_copies1592 a h).2.2.2.1
  have c50948 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.1
  have c50949 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.1
  have c50950 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.1
  have c50951 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_815_18 := arithEq_of_rows h (row := 815) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_18
  simp only [← c50946, ← c50947, ← c50948] at e_815_18
  have e_815_19 := arithEq_of_rows h (row := 815) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_815_19
  simp only [← c50949, ← c50950, k2, ← c50951] at e_815_19
  have hr := e_815_19
  simp only [e_815_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5885)) := by
  have c50952 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.1
  have c50953 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.1
  have c50954 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50955 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50956 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50957 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_0 := arithEq_of_rows h (row := 816) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_0
  simp only [← c50952, ← c50953, ← c50954] at e_816_0
  have e_816_1 := arithEq_of_rows h (row := 816) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_1
  simp only [← c50955, ← c50956, k2, ← c50957] at e_816_1
  have hr := e_816_1
  simp only [e_816_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5886)) := by
  have c50958 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50959 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50960 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50961 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50962 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50963 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_2 := arithEq_of_rows h (row := 816) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_2
  simp only [← c50958, ← c50959, ← c50960] at e_816_2
  have e_816_3 := arithEq_of_rows h (row := 816) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_3
  simp only [← c50961, ← c50962, k2, ← c50963] at e_816_3
  have hr := e_816_3
  simp only [e_816_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5887)) := by
  have c50964 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50965 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50966 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50967 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50968 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50969 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_4 := arithEq_of_rows h (row := 816) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_4
  simp only [← c50964, ← c50965, ← c50966] at e_816_4
  have e_816_5 := arithEq_of_rows h (row := 816) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_5
  simp only [← c50967, ← c50968, k2, ← c50969] at e_816_5
  have hr := e_816_5
  simp only [e_816_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5888)) := by
  have c50970 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50971 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50972 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50973 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50974 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50975 := (publicBatchWrapper89_copies1592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_6 := arithEq_of_rows h (row := 816) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_6
  simp only [← c50970, ← c50971, ← c50972] at e_816_6
  have e_816_7 := arithEq_of_rows h (row := 816) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_7
  simp only [← c50973, ← c50974, k2, ← c50975] at e_816_7
  have hr := e_816_7
  simp only [e_816_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5889)) := by
  have c50976 := (publicBatchWrapper89_copies1593 a h).1
  have c50977 := (publicBatchWrapper89_copies1593 a h).2.1
  have c50978 := (publicBatchWrapper89_copies1593 a h).2.2.1
  have c50979 := (publicBatchWrapper89_copies1593 a h).2.2.2.1
  have c50980 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.1
  have c50981 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_8 := arithEq_of_rows h (row := 816) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_8
  simp only [← c50976, ← c50977, ← c50978] at e_816_8
  have e_816_9 := arithEq_of_rows h (row := 816) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_9
  simp only [← c50979, ← c50980, k2, ← c50981] at e_816_9
  have hr := e_816_9
  simp only [e_816_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5890)) := by
  have c50982 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.1
  have c50983 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.1
  have c50984 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.1
  have c50985 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.1
  have c50986 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50987 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_10 := arithEq_of_rows h (row := 816) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_10
  simp only [← c50982, ← c50983, ← c50984] at e_816_10
  have e_816_11 := arithEq_of_rows h (row := 816) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_11
  simp only [← c50985, ← c50986, k2, ← c50987] at e_816_11
  have hr := e_816_11
  simp only [e_816_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5891)) := by
  have c50988 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50989 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50990 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50991 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50992 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50993 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_12 := arithEq_of_rows h (row := 816) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_12
  simp only [← c50988, ← c50989, ← c50990] at e_816_12
  have e_816_13 := arithEq_of_rows h (row := 816) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_13
  simp only [← c50991, ← c50992, k2, ← c50993] at e_816_13
  have hr := e_816_13
  simp only [e_816_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5892)) := by
  have c50994 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50995 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50996 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50997 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50998 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50999 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_14 := arithEq_of_rows h (row := 816) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_14
  simp only [← c50994, ← c50995, ← c50996] at e_816_14
  have e_816_15 := arithEq_of_rows h (row := 816) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_15
  simp only [← c50997, ← c50998, k2, ← c50999] at e_816_15
  have hr := e_816_15
  simp only [e_816_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5893)) := by
  have c51000 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51001 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51002 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51003 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51004 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51005 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_16 := arithEq_of_rows h (row := 816) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_16
  simp only [← c51000, ← c51001, ← c51002] at e_816_16
  have e_816_17 := arithEq_of_rows h (row := 816) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_17
  simp only [← c51003, ← c51004, k2, ← c51005] at e_816_17
  have hr := e_816_17
  simp only [e_816_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 816 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5894)) := by
  have c51006 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51007 := (publicBatchWrapper89_copies1593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51008 := (publicBatchWrapper89_copies1594 a h).1
  have c51009 := (publicBatchWrapper89_copies1594 a h).2.1
  have c51010 := (publicBatchWrapper89_copies1594 a h).2.2.1
  have c51011 := (publicBatchWrapper89_copies1594 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_816_18 := arithEq_of_rows h (row := 816) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_18
  simp only [← c51006, ← c51007, ← c51008] at e_816_18
  have e_816_19 := arithEq_of_rows h (row := 816) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_816_19
  simp only [← c51009, ← c51010, k2, ← c51011] at e_816_19
  have hr := e_816_19
  simp only [e_816_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5895)) := by
  have c51012 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.1
  have c51013 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.1
  have c51014 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.1
  have c51015 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.1
  have c51016 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.1
  have c51017 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_0 := arithEq_of_rows h (row := 817) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_0
  simp only [← c51012, ← c51013, ← c51014] at e_817_0
  have e_817_1 := arithEq_of_rows h (row := 817) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_1
  simp only [← c51015, ← c51016, k2, ← c51017] at e_817_1
  have hr := e_817_1
  simp only [e_817_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5896)) := by
  have c51018 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51019 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51020 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51021 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51022 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51023 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_2 := arithEq_of_rows h (row := 817) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_2
  simp only [← c51018, ← c51019, ← c51020] at e_817_2
  have e_817_3 := arithEq_of_rows h (row := 817) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_3
  simp only [← c51021, ← c51022, k2, ← c51023] at e_817_3
  have hr := e_817_3
  simp only [e_817_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5897)) := by
  have c51024 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51025 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51026 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51027 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51028 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51029 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_4 := arithEq_of_rows h (row := 817) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_4
  simp only [← c51024, ← c51025, ← c51026] at e_817_4
  have e_817_5 := arithEq_of_rows h (row := 817) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_5
  simp only [← c51027, ← c51028, k2, ← c51029] at e_817_5
  have hr := e_817_5
  simp only [e_817_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5898)) := by
  have c51030 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51031 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51032 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51033 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51034 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51035 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_6 := arithEq_of_rows h (row := 817) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_6
  simp only [← c51030, ← c51031, ← c51032] at e_817_6
  have e_817_7 := arithEq_of_rows h (row := 817) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_7
  simp only [← c51033, ← c51034, k2, ← c51035] at e_817_7
  have hr := e_817_7
  simp only [e_817_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5899)) := by
  have c51036 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51037 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51038 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51039 := (publicBatchWrapper89_copies1594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51040 := (publicBatchWrapper89_copies1595 a h).1
  have c51041 := (publicBatchWrapper89_copies1595 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_8 := arithEq_of_rows h (row := 817) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_8
  simp only [← c51036, ← c51037, ← c51038] at e_817_8
  have e_817_9 := arithEq_of_rows h (row := 817) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_9
  simp only [← c51039, ← c51040, k2, ← c51041] at e_817_9
  have hr := e_817_9
  simp only [e_817_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5900)) := by
  have c51042 := (publicBatchWrapper89_copies1595 a h).2.2.1
  have c51043 := (publicBatchWrapper89_copies1595 a h).2.2.2.1
  have c51044 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.1
  have c51045 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.1
  have c51046 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.1
  have c51047 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_10 := arithEq_of_rows h (row := 817) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_10
  simp only [← c51042, ← c51043, ← c51044] at e_817_10
  have e_817_11 := arithEq_of_rows h (row := 817) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_11
  simp only [← c51045, ← c51046, k2, ← c51047] at e_817_11
  have hr := e_817_11
  simp only [e_817_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5901)) := by
  have c51048 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.1
  have c51049 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.1
  have c51050 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51051 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51052 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51053 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_12 := arithEq_of_rows h (row := 817) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_12
  simp only [← c51048, ← c51049, ← c51050] at e_817_12
  have e_817_13 := arithEq_of_rows h (row := 817) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_13
  simp only [← c51051, ← c51052, k2, ← c51053] at e_817_13
  have hr := e_817_13
  simp only [e_817_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5902)) := by
  have c51054 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51055 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51056 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51057 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51058 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51059 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_14 := arithEq_of_rows h (row := 817) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_14
  simp only [← c51054, ← c51055, ← c51056] at e_817_14
  have e_817_15 := arithEq_of_rows h (row := 817) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_15
  simp only [← c51057, ← c51058, k2, ← c51059] at e_817_15
  have hr := e_817_15
  simp only [e_817_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5903)) := by
  have c51060 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51061 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51062 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51063 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51064 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51065 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_16 := arithEq_of_rows h (row := 817) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_16
  simp only [← c51060, ← c51061, ← c51062] at e_817_16
  have e_817_17 := arithEq_of_rows h (row := 817) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_17
  simp only [← c51063, ← c51064, k2, ← c51065] at e_817_17
  have hr := e_817_17
  simp only [e_817_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 817 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5904)) := by
  have c51066 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51067 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51068 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51069 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51070 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51071 := (publicBatchWrapper89_copies1595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_817_18 := arithEq_of_rows h (row := 817) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_18
  simp only [← c51066, ← c51067, ← c51068] at e_817_18
  have e_817_19 := arithEq_of_rows h (row := 817) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_817_19
  simp only [← c51069, ← c51070, k2, ← c51071] at e_817_19
  have hr := e_817_19
  simp only [e_817_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5905)) := by
  have c51072 := (publicBatchWrapper89_copies1596 a h).1
  have c51073 := (publicBatchWrapper89_copies1596 a h).2.1
  have c51074 := (publicBatchWrapper89_copies1596 a h).2.2.1
  have c51075 := (publicBatchWrapper89_copies1596 a h).2.2.2.1
  have c51076 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.1
  have c51077 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_0 := arithEq_of_rows h (row := 818) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_0
  simp only [← c51072, ← c51073, ← c51074] at e_818_0
  have e_818_1 := arithEq_of_rows h (row := 818) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_1
  simp only [← c51075, ← c51076, k2, ← c51077] at e_818_1
  have hr := e_818_1
  simp only [e_818_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5906)) := by
  have c51078 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.1
  have c51079 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.1
  have c51080 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.1
  have c51081 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.1
  have c51082 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51083 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_2 := arithEq_of_rows h (row := 818) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_2
  simp only [← c51078, ← c51079, ← c51080] at e_818_2
  have e_818_3 := arithEq_of_rows h (row := 818) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_3
  simp only [← c51081, ← c51082, k2, ← c51083] at e_818_3
  have hr := e_818_3
  simp only [e_818_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5907)) := by
  have c51084 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51085 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51086 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51087 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51088 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51089 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_4 := arithEq_of_rows h (row := 818) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_4
  simp only [← c51084, ← c51085, ← c51086] at e_818_4
  have e_818_5 := arithEq_of_rows h (row := 818) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_5
  simp only [← c51087, ← c51088, k2, ← c51089] at e_818_5
  have hr := e_818_5
  simp only [e_818_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5908)) := by
  have c51090 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51091 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51092 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51093 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51094 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51095 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_6 := arithEq_of_rows h (row := 818) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_6
  simp only [← c51090, ← c51091, ← c51092] at e_818_6
  have e_818_7 := arithEq_of_rows h (row := 818) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_7
  simp only [← c51093, ← c51094, k2, ← c51095] at e_818_7
  have hr := e_818_7
  simp only [e_818_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5909)) := by
  have c51096 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51097 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51098 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51099 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51100 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51101 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_8 := arithEq_of_rows h (row := 818) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_8
  simp only [← c51096, ← c51097, ← c51098] at e_818_8
  have e_818_9 := arithEq_of_rows h (row := 818) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_9
  simp only [← c51099, ← c51100, k2, ← c51101] at e_818_9
  have hr := e_818_9
  simp only [e_818_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5910)) := by
  have c51102 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51103 := (publicBatchWrapper89_copies1596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51104 := (publicBatchWrapper89_copies1597 a h).1
  have c51105 := (publicBatchWrapper89_copies1597 a h).2.1
  have c51106 := (publicBatchWrapper89_copies1597 a h).2.2.1
  have c51107 := (publicBatchWrapper89_copies1597 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_10 := arithEq_of_rows h (row := 818) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_10
  simp only [← c51102, ← c51103, ← c51104] at e_818_10
  have e_818_11 := arithEq_of_rows h (row := 818) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_11
  simp only [← c51105, ← c51106, k2, ← c51107] at e_818_11
  have hr := e_818_11
  simp only [e_818_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5911)) := by
  have c51108 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.1
  have c51109 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.1
  have c51110 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.1
  have c51111 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.1
  have c51112 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.1
  have c51113 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_12 := arithEq_of_rows h (row := 818) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_12
  simp only [← c51108, ← c51109, ← c51110] at e_818_12
  have e_818_13 := arithEq_of_rows h (row := 818) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_13
  simp only [← c51111, ← c51112, k2, ← c51113] at e_818_13
  have hr := e_818_13
  simp only [e_818_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5912)) := by
  have c51114 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51115 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51116 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51117 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51118 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51119 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_14 := arithEq_of_rows h (row := 818) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_14
  simp only [← c51114, ← c51115, ← c51116] at e_818_14
  have e_818_15 := arithEq_of_rows h (row := 818) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_15
  simp only [← c51117, ← c51118, k2, ← c51119] at e_818_15
  have hr := e_818_15
  simp only [e_818_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5913)) := by
  have c51120 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51121 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51122 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51123 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51124 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51125 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_16 := arithEq_of_rows h (row := 818) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_16
  simp only [← c51120, ← c51121, ← c51122] at e_818_16
  have e_818_17 := arithEq_of_rows h (row := 818) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_17
  simp only [← c51123, ← c51124, k2, ← c51125] at e_818_17
  have hr := e_818_17
  simp only [e_818_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 818 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5914)) := by
  have c51126 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51127 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51128 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51129 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51130 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51131 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_818_18 := arithEq_of_rows h (row := 818) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_18
  simp only [← c51126, ← c51127, ← c51128] at e_818_18
  have e_818_19 := arithEq_of_rows h (row := 818) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_818_19
  simp only [← c51129, ← c51130, k2, ← c51131] at e_818_19
  have hr := e_818_19
  simp only [e_818_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5915)) := by
  have c51132 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51133 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51134 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51135 := (publicBatchWrapper89_copies1597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51136 := (publicBatchWrapper89_copies1598 a h).1
  have c51137 := (publicBatchWrapper89_copies1598 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_0 := arithEq_of_rows h (row := 819) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_0
  simp only [← c51132, ← c51133, ← c51134] at e_819_0
  have e_819_1 := arithEq_of_rows h (row := 819) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_1
  simp only [← c51135, ← c51136, k2, ← c51137] at e_819_1
  have hr := e_819_1
  simp only [e_819_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5916)) := by
  have c51138 := (publicBatchWrapper89_copies1598 a h).2.2.1
  have c51139 := (publicBatchWrapper89_copies1598 a h).2.2.2.1
  have c51140 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.1
  have c51141 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.1
  have c51142 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.1
  have c51143 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_2 := arithEq_of_rows h (row := 819) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_2
  simp only [← c51138, ← c51139, ← c51140] at e_819_2
  have e_819_3 := arithEq_of_rows h (row := 819) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_3
  simp only [← c51141, ← c51142, k2, ← c51143] at e_819_3
  have hr := e_819_3
  simp only [e_819_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5917)) := by
  have c51144 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.1
  have c51145 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.1
  have c51146 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51147 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51148 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51149 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_4 := arithEq_of_rows h (row := 819) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_4
  simp only [← c51144, ← c51145, ← c51146] at e_819_4
  have e_819_5 := arithEq_of_rows h (row := 819) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_5
  simp only [← c51147, ← c51148, k2, ← c51149] at e_819_5
  have hr := e_819_5
  simp only [e_819_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5918)) := by
  have c51150 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51151 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51152 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51153 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51154 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51155 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_6 := arithEq_of_rows h (row := 819) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_6
  simp only [← c51150, ← c51151, ← c51152] at e_819_6
  have e_819_7 := arithEq_of_rows h (row := 819) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_7
  simp only [← c51153, ← c51154, k2, ← c51155] at e_819_7
  have hr := e_819_7
  simp only [e_819_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5919)) := by
  have c51156 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51157 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51158 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51159 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51160 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51161 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_8 := arithEq_of_rows h (row := 819) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_8
  simp only [← c51156, ← c51157, ← c51158] at e_819_8
  have e_819_9 := arithEq_of_rows h (row := 819) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_9
  simp only [← c51159, ← c51160, k2, ← c51161] at e_819_9
  have hr := e_819_9
  simp only [e_819_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5920)) := by
  have c51162 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51163 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51164 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51165 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51166 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51167 := (publicBatchWrapper89_copies1598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_10 := arithEq_of_rows h (row := 819) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_10
  simp only [← c51162, ← c51163, ← c51164] at e_819_10
  have e_819_11 := arithEq_of_rows h (row := 819) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_11
  simp only [← c51165, ← c51166, k2, ← c51167] at e_819_11
  have hr := e_819_11
  simp only [e_819_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5921)) := by
  have c51168 := (publicBatchWrapper89_copies1599 a h).1
  have c51169 := (publicBatchWrapper89_copies1599 a h).2.1
  have c51170 := (publicBatchWrapper89_copies1599 a h).2.2.1
  have c51171 := (publicBatchWrapper89_copies1599 a h).2.2.2.1
  have c51172 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.1
  have c51173 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_12 := arithEq_of_rows h (row := 819) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_12
  simp only [← c51168, ← c51169, ← c51170] at e_819_12
  have e_819_13 := arithEq_of_rows h (row := 819) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_13
  simp only [← c51171, ← c51172, k2, ← c51173] at e_819_13
  have hr := e_819_13
  simp only [e_819_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5922)) := by
  have c51174 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.1
  have c51175 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.1
  have c51176 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.1
  have c51177 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.1
  have c51178 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51179 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_14 := arithEq_of_rows h (row := 819) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_14
  simp only [← c51174, ← c51175, ← c51176] at e_819_14
  have e_819_15 := arithEq_of_rows h (row := 819) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_15
  simp only [← c51177, ← c51178, k2, ← c51179] at e_819_15
  have hr := e_819_15
  simp only [e_819_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5923)) := by
  have c51180 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51181 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51182 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51183 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51184 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51185 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_16 := arithEq_of_rows h (row := 819) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_16
  simp only [← c51180, ← c51181, ← c51182] at e_819_16
  have e_819_17 := arithEq_of_rows h (row := 819) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_17
  simp only [← c51183, ← c51184, k2, ← c51185] at e_819_17
  have hr := e_819_17
  simp only [e_819_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 819 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5924)) := by
  have c51186 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51187 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51188 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51189 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51190 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51191 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_819_18 := arithEq_of_rows h (row := 819) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_18
  simp only [← c51186, ← c51187, ← c51188] at e_819_18
  have e_819_19 := arithEq_of_rows h (row := 819) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_819_19
  simp only [← c51189, ← c51190, k2, ← c51191] at e_819_19
  have hr := e_819_19
  simp only [e_819_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5925)) := by
  have c51192 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51193 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51194 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51195 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51196 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51197 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_0 := arithEq_of_rows h (row := 820) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_0
  simp only [← c51192, ← c51193, ← c51194] at e_820_0
  have e_820_1 := arithEq_of_rows h (row := 820) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_1
  simp only [← c51195, ← c51196, k2, ← c51197] at e_820_1
  have hr := e_820_1
  simp only [e_820_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5926)) := by
  have c51198 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51199 := (publicBatchWrapper89_copies1599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51200 := (publicBatchWrapper89_copies1600 a h).1
  have c51201 := (publicBatchWrapper89_copies1600 a h).2.1
  have c51202 := (publicBatchWrapper89_copies1600 a h).2.2.1
  have c51203 := (publicBatchWrapper89_copies1600 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_2 := arithEq_of_rows h (row := 820) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_2
  simp only [← c51198, ← c51199, ← c51200] at e_820_2
  have e_820_3 := arithEq_of_rows h (row := 820) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_3
  simp only [← c51201, ← c51202, k2, ← c51203] at e_820_3
  have hr := e_820_3
  simp only [e_820_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5927)) := by
  have c51204 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.1
  have c51205 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.1
  have c51206 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.1
  have c51207 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.1
  have c51208 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.1
  have c51209 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_4 := arithEq_of_rows h (row := 820) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_4
  simp only [← c51204, ← c51205, ← c51206] at e_820_4
  have e_820_5 := arithEq_of_rows h (row := 820) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_5
  simp only [← c51207, ← c51208, k2, ← c51209] at e_820_5
  have hr := e_820_5
  simp only [e_820_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5928)) := by
  have c51210 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51211 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51212 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51213 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51214 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51215 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_6 := arithEq_of_rows h (row := 820) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_6
  simp only [← c51210, ← c51211, ← c51212] at e_820_6
  have e_820_7 := arithEq_of_rows h (row := 820) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_7
  simp only [← c51213, ← c51214, k2, ← c51215] at e_820_7
  have hr := e_820_7
  simp only [e_820_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5929)) := by
  have c51216 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51217 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51218 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51219 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51220 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51221 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_8 := arithEq_of_rows h (row := 820) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_8
  simp only [← c51216, ← c51217, ← c51218] at e_820_8
  have e_820_9 := arithEq_of_rows h (row := 820) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_9
  simp only [← c51219, ← c51220, k2, ← c51221] at e_820_9
  have hr := e_820_9
  simp only [e_820_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5930)) := by
  have c51222 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51223 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51224 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51225 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51226 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51227 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_10 := arithEq_of_rows h (row := 820) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_10
  simp only [← c51222, ← c51223, ← c51224] at e_820_10
  have e_820_11 := arithEq_of_rows h (row := 820) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_11
  simp only [← c51225, ← c51226, k2, ← c51227] at e_820_11
  have hr := e_820_11
  simp only [e_820_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5931)) := by
  have c51228 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51229 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51230 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51231 := (publicBatchWrapper89_copies1600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51232 := (publicBatchWrapper89_copies1601 a h).1
  have c51233 := (publicBatchWrapper89_copies1601 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_12 := arithEq_of_rows h (row := 820) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_12
  simp only [← c51228, ← c51229, ← c51230] at e_820_12
  have e_820_13 := arithEq_of_rows h (row := 820) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_13
  simp only [← c51231, ← c51232, k2, ← c51233] at e_820_13
  have hr := e_820_13
  simp only [e_820_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5932)) := by
  have c51234 := (publicBatchWrapper89_copies1601 a h).2.2.1
  have c51235 := (publicBatchWrapper89_copies1601 a h).2.2.2.1
  have c51236 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.1
  have c51237 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.1
  have c51238 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.1
  have c51239 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_14 := arithEq_of_rows h (row := 820) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_14
  simp only [← c51234, ← c51235, ← c51236] at e_820_14
  have e_820_15 := arithEq_of_rows h (row := 820) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_15
  simp only [← c51237, ← c51238, k2, ← c51239] at e_820_15
  have hr := e_820_15
  simp only [e_820_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5933)) := by
  have c51240 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.1
  have c51241 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.1
  have c51242 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51243 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51244 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51245 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_16 := arithEq_of_rows h (row := 820) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_16
  simp only [← c51240, ← c51241, ← c51242] at e_820_16
  have e_820_17 := arithEq_of_rows h (row := 820) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_17
  simp only [← c51243, ← c51244, k2, ← c51245] at e_820_17
  have hr := e_820_17
  simp only [e_820_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 820 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5934)) := by
  have c51246 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51247 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51248 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51249 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51250 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51251 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_820_18 := arithEq_of_rows h (row := 820) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_18
  simp only [← c51246, ← c51247, ← c51248] at e_820_18
  have e_820_19 := arithEq_of_rows h (row := 820) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_820_19
  simp only [← c51249, ← c51250, k2, ← c51251] at e_820_19
  have hr := e_820_19
  simp only [e_820_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5935)) := by
  have c51252 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51253 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51254 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51255 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51256 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51257 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_0 := arithEq_of_rows h (row := 821) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_0
  simp only [← c51252, ← c51253, ← c51254] at e_821_0
  have e_821_1 := arithEq_of_rows h (row := 821) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_1
  simp only [← c51255, ← c51256, k2, ← c51257] at e_821_1
  have hr := e_821_1
  simp only [e_821_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5936)) := by
  have c51258 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51259 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51260 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51261 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51262 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51263 := (publicBatchWrapper89_copies1601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_2 := arithEq_of_rows h (row := 821) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_2
  simp only [← c51258, ← c51259, ← c51260] at e_821_2
  have e_821_3 := arithEq_of_rows h (row := 821) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_3
  simp only [← c51261, ← c51262, k2, ← c51263] at e_821_3
  have hr := e_821_3
  simp only [e_821_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5937)) := by
  have c51264 := (publicBatchWrapper89_copies1602 a h).1
  have c51265 := (publicBatchWrapper89_copies1602 a h).2.1
  have c51266 := (publicBatchWrapper89_copies1602 a h).2.2.1
  have c51267 := (publicBatchWrapper89_copies1602 a h).2.2.2.1
  have c51268 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.1
  have c51269 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_4 := arithEq_of_rows h (row := 821) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_4
  simp only [← c51264, ← c51265, ← c51266] at e_821_4
  have e_821_5 := arithEq_of_rows h (row := 821) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_5
  simp only [← c51267, ← c51268, k2, ← c51269] at e_821_5
  have hr := e_821_5
  simp only [e_821_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5938)) := by
  have c51270 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.1
  have c51271 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.1
  have c51272 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.1
  have c51273 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.1
  have c51274 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51275 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_6 := arithEq_of_rows h (row := 821) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_6
  simp only [← c51270, ← c51271, ← c51272] at e_821_6
  have e_821_7 := arithEq_of_rows h (row := 821) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_7
  simp only [← c51273, ← c51274, k2, ← c51275] at e_821_7
  have hr := e_821_7
  simp only [e_821_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5939)) := by
  have c51276 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51277 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51278 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51279 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51280 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51281 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_8 := arithEq_of_rows h (row := 821) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_8
  simp only [← c51276, ← c51277, ← c51278] at e_821_8
  have e_821_9 := arithEq_of_rows h (row := 821) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_9
  simp only [← c51279, ← c51280, k2, ← c51281] at e_821_9
  have hr := e_821_9
  simp only [e_821_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5940)) := by
  have c51282 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51283 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51284 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51285 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51286 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51287 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_10 := arithEq_of_rows h (row := 821) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_10
  simp only [← c51282, ← c51283, ← c51284] at e_821_10
  have e_821_11 := arithEq_of_rows h (row := 821) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_11
  simp only [← c51285, ← c51286, k2, ← c51287] at e_821_11
  have hr := e_821_11
  simp only [e_821_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5941)) := by
  have c51288 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51289 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51290 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51291 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51292 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51293 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_12 := arithEq_of_rows h (row := 821) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_12
  simp only [← c51288, ← c51289, ← c51290] at e_821_12
  have e_821_13 := arithEq_of_rows h (row := 821) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_13
  simp only [← c51291, ← c51292, k2, ← c51293] at e_821_13
  have hr := e_821_13
  simp only [e_821_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5942)) := by
  have c51294 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51295 := (publicBatchWrapper89_copies1602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51296 := (publicBatchWrapper89_copies1603 a h).1
  have c51297 := (publicBatchWrapper89_copies1603 a h).2.1
  have c51298 := (publicBatchWrapper89_copies1603 a h).2.2.1
  have c51299 := (publicBatchWrapper89_copies1603 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_14 := arithEq_of_rows h (row := 821) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_14
  simp only [← c51294, ← c51295, ← c51296] at e_821_14
  have e_821_15 := arithEq_of_rows h (row := 821) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_15
  simp only [← c51297, ← c51298, k2, ← c51299] at e_821_15
  have hr := e_821_15
  simp only [e_821_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5943)) := by
  have c51300 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.1
  have c51301 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.1
  have c51302 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.1
  have c51303 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.1
  have c51304 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.1
  have c51305 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_16 := arithEq_of_rows h (row := 821) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_16
  simp only [← c51300, ← c51301, ← c51302] at e_821_16
  have e_821_17 := arithEq_of_rows h (row := 821) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_17
  simp only [← c51303, ← c51304, k2, ← c51305] at e_821_17
  have hr := e_821_17
  simp only [e_821_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 821 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5944)) := by
  have c51306 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51307 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51308 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51309 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51310 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51311 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_821_18 := arithEq_of_rows h (row := 821) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_18
  simp only [← c51306, ← c51307, ← c51308] at e_821_18
  have e_821_19 := arithEq_of_rows h (row := 821) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_821_19
  simp only [← c51309, ← c51310, k2, ← c51311] at e_821_19
  have hr := e_821_19
  simp only [e_821_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5945)) := by
  have c51312 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51313 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51314 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51315 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51316 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51317 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_0 := arithEq_of_rows h (row := 822) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_0
  simp only [← c51312, ← c51313, ← c51314] at e_822_0
  have e_822_1 := arithEq_of_rows h (row := 822) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_1
  simp only [← c51315, ← c51316, k2, ← c51317] at e_822_1
  have hr := e_822_1
  simp only [e_822_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5946)) := by
  have c51318 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51319 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51320 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51321 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51322 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51323 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_2 := arithEq_of_rows h (row := 822) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_2
  simp only [← c51318, ← c51319, ← c51320] at e_822_2
  have e_822_3 := arithEq_of_rows h (row := 822) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_3
  simp only [← c51321, ← c51322, k2, ← c51323] at e_822_3
  have hr := e_822_3
  simp only [e_822_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5947)) := by
  have c51324 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51325 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51326 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51327 := (publicBatchWrapper89_copies1603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51328 := (publicBatchWrapper89_copies1604 a h).1
  have c51329 := (publicBatchWrapper89_copies1604 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_4 := arithEq_of_rows h (row := 822) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_4
  simp only [← c51324, ← c51325, ← c51326] at e_822_4
  have e_822_5 := arithEq_of_rows h (row := 822) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_5
  simp only [← c51327, ← c51328, k2, ← c51329] at e_822_5
  have hr := e_822_5
  simp only [e_822_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5948)) := by
  have c51330 := (publicBatchWrapper89_copies1604 a h).2.2.1
  have c51331 := (publicBatchWrapper89_copies1604 a h).2.2.2.1
  have c51332 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.1
  have c51333 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.1
  have c51334 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.1
  have c51335 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_6 := arithEq_of_rows h (row := 822) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_6
  simp only [← c51330, ← c51331, ← c51332] at e_822_6
  have e_822_7 := arithEq_of_rows h (row := 822) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_7
  simp only [← c51333, ← c51334, k2, ← c51335] at e_822_7
  have hr := e_822_7
  simp only [e_822_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5949)) := by
  have c51336 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.1
  have c51337 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.1
  have c51338 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51339 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51340 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51341 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_8 := arithEq_of_rows h (row := 822) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_8
  simp only [← c51336, ← c51337, ← c51338] at e_822_8
  have e_822_9 := arithEq_of_rows h (row := 822) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_9
  simp only [← c51339, ← c51340, k2, ← c51341] at e_822_9
  have hr := e_822_9
  simp only [e_822_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5950)) := by
  have c51342 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51343 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51344 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51345 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51346 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51347 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_10 := arithEq_of_rows h (row := 822) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_10
  simp only [← c51342, ← c51343, ← c51344] at e_822_10
  have e_822_11 := arithEq_of_rows h (row := 822) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_11
  simp only [← c51345, ← c51346, k2, ← c51347] at e_822_11
  have hr := e_822_11
  simp only [e_822_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5951)) := by
  have c51348 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51349 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51350 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51351 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51352 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51353 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_12 := arithEq_of_rows h (row := 822) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_12
  simp only [← c51348, ← c51349, ← c51350] at e_822_12
  have e_822_13 := arithEq_of_rows h (row := 822) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_13
  simp only [← c51351, ← c51352, k2, ← c51353] at e_822_13
  have hr := e_822_13
  simp only [e_822_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5952)) := by
  have c51354 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51355 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51356 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51357 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51358 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51359 := (publicBatchWrapper89_copies1604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_14 := arithEq_of_rows h (row := 822) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_14
  simp only [← c51354, ← c51355, ← c51356] at e_822_14
  have e_822_15 := arithEq_of_rows h (row := 822) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_15
  simp only [← c51357, ← c51358, k2, ← c51359] at e_822_15
  have hr := e_822_15
  simp only [e_822_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5953)) := by
  have c51360 := (publicBatchWrapper89_copies1605 a h).1
  have c51361 := (publicBatchWrapper89_copies1605 a h).2.1
  have c51362 := (publicBatchWrapper89_copies1605 a h).2.2.1
  have c51363 := (publicBatchWrapper89_copies1605 a h).2.2.2.1
  have c51364 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.1
  have c51365 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_16 := arithEq_of_rows h (row := 822) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_16
  simp only [← c51360, ← c51361, ← c51362] at e_822_16
  have e_822_17 := arithEq_of_rows h (row := 822) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_17
  simp only [← c51363, ← c51364, k2, ← c51365] at e_822_17
  have hr := e_822_17
  simp only [e_822_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 822 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5954)) := by
  have c51366 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.1
  have c51367 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.1
  have c51368 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.1
  have c51369 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.1
  have c51370 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51371 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_822_18 := arithEq_of_rows h (row := 822) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_18
  simp only [← c51366, ← c51367, ← c51368] at e_822_18
  have e_822_19 := arithEq_of_rows h (row := 822) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_822_19
  simp only [← c51369, ← c51370, k2, ← c51371] at e_822_19
  have hr := e_822_19
  simp only [e_822_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5955)) := by
  have c51372 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51373 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51374 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51375 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51376 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51377 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_0 := arithEq_of_rows h (row := 823) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_0
  simp only [← c51372, ← c51373, ← c51374] at e_823_0
  have e_823_1 := arithEq_of_rows h (row := 823) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_1
  simp only [← c51375, ← c51376, k2, ← c51377] at e_823_1
  have hr := e_823_1
  simp only [e_823_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5956)) := by
  have c51378 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51379 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51380 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51381 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51382 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51383 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_2 := arithEq_of_rows h (row := 823) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_2
  simp only [← c51378, ← c51379, ← c51380] at e_823_2
  have e_823_3 := arithEq_of_rows h (row := 823) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_3
  simp only [← c51381, ← c51382, k2, ← c51383] at e_823_3
  have hr := e_823_3
  simp only [e_823_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5957)) := by
  have c51384 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51385 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51386 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51387 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51388 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51389 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_4 := arithEq_of_rows h (row := 823) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_4
  simp only [← c51384, ← c51385, ← c51386] at e_823_4
  have e_823_5 := arithEq_of_rows h (row := 823) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_5
  simp only [← c51387, ← c51388, k2, ← c51389] at e_823_5
  have hr := e_823_5
  simp only [e_823_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5958)) := by
  have c51390 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51391 := (publicBatchWrapper89_copies1605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51392 := (publicBatchWrapper89_copies1606 a h).1
  have c51393 := (publicBatchWrapper89_copies1606 a h).2.1
  have c51394 := (publicBatchWrapper89_copies1606 a h).2.2.1
  have c51395 := (publicBatchWrapper89_copies1606 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_6 := arithEq_of_rows h (row := 823) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_6
  simp only [← c51390, ← c51391, ← c51392] at e_823_6
  have e_823_7 := arithEq_of_rows h (row := 823) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_7
  simp only [← c51393, ← c51394, k2, ← c51395] at e_823_7
  have hr := e_823_7
  simp only [e_823_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5959)) := by
  have c51396 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.1
  have c51397 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.1
  have c51398 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.1
  have c51399 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.1
  have c51400 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.1
  have c51401 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_8 := arithEq_of_rows h (row := 823) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_8
  simp only [← c51396, ← c51397, ← c51398] at e_823_8
  have e_823_9 := arithEq_of_rows h (row := 823) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_9
  simp only [← c51399, ← c51400, k2, ← c51401] at e_823_9
  have hr := e_823_9
  simp only [e_823_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5960)) := by
  have c51402 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51403 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51404 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51405 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51406 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51407 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_10 := arithEq_of_rows h (row := 823) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_10
  simp only [← c51402, ← c51403, ← c51404] at e_823_10
  have e_823_11 := arithEq_of_rows h (row := 823) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_11
  simp only [← c51405, ← c51406, k2, ← c51407] at e_823_11
  have hr := e_823_11
  simp only [e_823_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5961)) := by
  have c51408 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51409 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51410 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51411 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51412 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51413 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_12 := arithEq_of_rows h (row := 823) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_12
  simp only [← c51408, ← c51409, ← c51410] at e_823_12
  have e_823_13 := arithEq_of_rows h (row := 823) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_13
  simp only [← c51411, ← c51412, k2, ← c51413] at e_823_13
  have hr := e_823_13
  simp only [e_823_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5962)) := by
  have c51414 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51415 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51416 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51417 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51418 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51419 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_14 := arithEq_of_rows h (row := 823) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_14
  simp only [← c51414, ← c51415, ← c51416] at e_823_14
  have e_823_15 := arithEq_of_rows h (row := 823) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_15
  simp only [← c51417, ← c51418, k2, ← c51419] at e_823_15
  have hr := e_823_15
  simp only [e_823_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5963)) := by
  have c51420 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51421 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51422 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51423 := (publicBatchWrapper89_copies1606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51424 := (publicBatchWrapper89_copies1607 a h).1
  have c51425 := (publicBatchWrapper89_copies1607 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_16 := arithEq_of_rows h (row := 823) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_16
  simp only [← c51420, ← c51421, ← c51422] at e_823_16
  have e_823_17 := arithEq_of_rows h (row := 823) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_17
  simp only [← c51423, ← c51424, k2, ← c51425] at e_823_17
  have hr := e_823_17
  simp only [e_823_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 823 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5964)) := by
  have c51426 := (publicBatchWrapper89_copies1607 a h).2.2.1
  have c51427 := (publicBatchWrapper89_copies1607 a h).2.2.2.1
  have c51428 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.1
  have c51429 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.1
  have c51430 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.1
  have c51431 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_823_18 := arithEq_of_rows h (row := 823) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_18
  simp only [← c51426, ← c51427, ← c51428] at e_823_18
  have e_823_19 := arithEq_of_rows h (row := 823) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_823_19
  simp only [← c51429, ← c51430, k2, ← c51431] at e_823_19
  have hr := e_823_19
  simp only [e_823_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5965)) := by
  have c51432 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.1
  have c51433 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.1
  have c51434 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51435 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51436 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51437 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_0 := arithEq_of_rows h (row := 824) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_0
  simp only [← c51432, ← c51433, ← c51434] at e_824_0
  have e_824_1 := arithEq_of_rows h (row := 824) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_1
  simp only [← c51435, ← c51436, k2, ← c51437] at e_824_1
  have hr := e_824_1
  simp only [e_824_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5966)) := by
  have c51438 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51439 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51440 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51441 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51442 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51443 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_2 := arithEq_of_rows h (row := 824) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_2
  simp only [← c51438, ← c51439, ← c51440] at e_824_2
  have e_824_3 := arithEq_of_rows h (row := 824) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_3
  simp only [← c51441, ← c51442, k2, ← c51443] at e_824_3
  have hr := e_824_3
  simp only [e_824_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5967)) := by
  have c51444 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51445 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51446 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51447 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51448 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51449 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_4 := arithEq_of_rows h (row := 824) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_4
  simp only [← c51444, ← c51445, ← c51446] at e_824_4
  have e_824_5 := arithEq_of_rows h (row := 824) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_5
  simp only [← c51447, ← c51448, k2, ← c51449] at e_824_5
  have hr := e_824_5
  simp only [e_824_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5968)) := by
  have c51450 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51451 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51452 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51453 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51454 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51455 := (publicBatchWrapper89_copies1607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_6 := arithEq_of_rows h (row := 824) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_6
  simp only [← c51450, ← c51451, ← c51452] at e_824_6
  have e_824_7 := arithEq_of_rows h (row := 824) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_7
  simp only [← c51453, ← c51454, k2, ← c51455] at e_824_7
  have hr := e_824_7
  simp only [e_824_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5969)) := by
  have c51456 := (publicBatchWrapper89_copies1608 a h).1
  have c51457 := (publicBatchWrapper89_copies1608 a h).2.1
  have c51458 := (publicBatchWrapper89_copies1608 a h).2.2.1
  have c51459 := (publicBatchWrapper89_copies1608 a h).2.2.2.1
  have c51460 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.1
  have c51461 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_8 := arithEq_of_rows h (row := 824) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_8
  simp only [← c51456, ← c51457, ← c51458] at e_824_8
  have e_824_9 := arithEq_of_rows h (row := 824) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_9
  simp only [← c51459, ← c51460, k2, ← c51461] at e_824_9
  have hr := e_824_9
  simp only [e_824_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6010)) := by
  have c51462 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.1
  have c51463 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.1
  have c51464 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.1
  have c51465 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.1
  have c51466 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51467 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_10 := arithEq_of_rows h (row := 824) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_10
  simp only [← c51462, ← c51463, ← c51464] at e_824_10
  have e_824_11 := arithEq_of_rows h (row := 824) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_11
  simp only [← c51465, ← c51466, k2, ← c51467] at e_824_11
  have hr := e_824_11
  simp only [e_824_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6011)) := by
  have c51468 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51469 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51470 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51471 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51472 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51473 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_12 := arithEq_of_rows h (row := 824) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_12
  simp only [← c51468, ← c51469, ← c51470] at e_824_12
  have e_824_13 := arithEq_of_rows h (row := 824) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_13
  simp only [← c51471, ← c51472, k2, ← c51473] at e_824_13
  have hr := e_824_13
  simp only [e_824_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6012)) := by
  have c51474 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51475 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51476 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51477 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51478 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51479 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_14 := arithEq_of_rows h (row := 824) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_14
  simp only [← c51474, ← c51475, ← c51476] at e_824_14
  have e_824_15 := arithEq_of_rows h (row := 824) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_15
  simp only [← c51477, ← c51478, k2, ← c51479] at e_824_15
  have hr := e_824_15
  simp only [e_824_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6013)) := by
  have c51480 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51481 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51482 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51483 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51484 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51485 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_16 := arithEq_of_rows h (row := 824) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_16
  simp only [← c51480, ← c51481, ← c51482] at e_824_16
  have e_824_17 := arithEq_of_rows h (row := 824) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_17
  simp only [← c51483, ← c51484, k2, ← c51485] at e_824_17
  have hr := e_824_17
  simp only [e_824_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 824 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6014)) := by
  have c51486 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51487 := (publicBatchWrapper89_copies1608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51488 := (publicBatchWrapper89_copies1609 a h).1
  have c51489 := (publicBatchWrapper89_copies1609 a h).2.1
  have c51490 := (publicBatchWrapper89_copies1609 a h).2.2.1
  have c51491 := (publicBatchWrapper89_copies1609 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_824_18 := arithEq_of_rows h (row := 824) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_18
  simp only [← c51486, ← c51487, ← c51488] at e_824_18
  have e_824_19 := arithEq_of_rows h (row := 824) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_824_19
  simp only [← c51489, ← c51490, k2, ← c51491] at e_824_19
  have hr := e_824_19
  simp only [e_824_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6015)) := by
  have c51492 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.1
  have c51493 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.1
  have c51494 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.1
  have c51495 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.1
  have c51496 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.1
  have c51497 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_0 := arithEq_of_rows h (row := 825) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_0
  simp only [← c51492, ← c51493, ← c51494] at e_825_0
  have e_825_1 := arithEq_of_rows h (row := 825) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_1
  simp only [← c51495, ← c51496, k2, ← c51497] at e_825_1
  have hr := e_825_1
  simp only [e_825_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6016)) := by
  have c51498 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51499 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51500 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51501 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51502 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51503 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_2 := arithEq_of_rows h (row := 825) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_2
  simp only [← c51498, ← c51499, ← c51500] at e_825_2
  have e_825_3 := arithEq_of_rows h (row := 825) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_3
  simp only [← c51501, ← c51502, k2, ← c51503] at e_825_3
  have hr := e_825_3
  simp only [e_825_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6017)) := by
  have c51504 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51505 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51506 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51507 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51508 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51509 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_4 := arithEq_of_rows h (row := 825) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_4
  simp only [← c51504, ← c51505, ← c51506] at e_825_4
  have e_825_5 := arithEq_of_rows h (row := 825) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_5
  simp only [← c51507, ← c51508, k2, ← c51509] at e_825_5
  have hr := e_825_5
  simp only [e_825_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6018)) := by
  have c51510 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51511 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51512 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51513 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51514 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51515 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_6 := arithEq_of_rows h (row := 825) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_6
  simp only [← c51510, ← c51511, ← c51512] at e_825_6
  have e_825_7 := arithEq_of_rows h (row := 825) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_7
  simp only [← c51513, ← c51514, k2, ← c51515] at e_825_7
  have hr := e_825_7
  simp only [e_825_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6019)) := by
  have c51516 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51517 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51518 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51519 := (publicBatchWrapper89_copies1609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51520 := (publicBatchWrapper89_copies1610 a h).1
  have c51521 := (publicBatchWrapper89_copies1610 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_8 := arithEq_of_rows h (row := 825) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_8
  simp only [← c51516, ← c51517, ← c51518] at e_825_8
  have e_825_9 := arithEq_of_rows h (row := 825) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_9
  simp only [← c51519, ← c51520, k2, ← c51521] at e_825_9
  have hr := e_825_9
  simp only [e_825_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6020)) := by
  have c51522 := (publicBatchWrapper89_copies1610 a h).2.2.1
  have c51523 := (publicBatchWrapper89_copies1610 a h).2.2.2.1
  have c51524 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.1
  have c51525 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.1
  have c51526 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.1
  have c51527 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_10 := arithEq_of_rows h (row := 825) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_10
  simp only [← c51522, ← c51523, ← c51524] at e_825_10
  have e_825_11 := arithEq_of_rows h (row := 825) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_11
  simp only [← c51525, ← c51526, k2, ← c51527] at e_825_11
  have hr := e_825_11
  simp only [e_825_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6021)) := by
  have c51528 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.1
  have c51529 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.1
  have c51530 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51531 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51532 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51533 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_12 := arithEq_of_rows h (row := 825) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_12
  simp only [← c51528, ← c51529, ← c51530] at e_825_12
  have e_825_13 := arithEq_of_rows h (row := 825) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_13
  simp only [← c51531, ← c51532, k2, ← c51533] at e_825_13
  have hr := e_825_13
  simp only [e_825_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6022)) := by
  have c51534 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51535 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51536 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51537 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51538 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51539 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_14 := arithEq_of_rows h (row := 825) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_14
  simp only [← c51534, ← c51535, ← c51536] at e_825_14
  have e_825_15 := arithEq_of_rows h (row := 825) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_15
  simp only [← c51537, ← c51538, k2, ← c51539] at e_825_15
  have hr := e_825_15
  simp only [e_825_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6023)) := by
  have c51540 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51541 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51542 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51543 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51544 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51545 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_16 := arithEq_of_rows h (row := 825) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_16
  simp only [← c51540, ← c51541, ← c51542] at e_825_16
  have e_825_17 := arithEq_of_rows h (row := 825) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_17
  simp only [← c51543, ← c51544, k2, ← c51545] at e_825_17
  have hr := e_825_17
  simp only [e_825_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 825 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6024)) := by
  have c51546 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51547 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51548 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51549 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51550 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51551 := (publicBatchWrapper89_copies1610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_825_18 := arithEq_of_rows h (row := 825) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_18
  simp only [← c51546, ← c51547, ← c51548] at e_825_18
  have e_825_19 := arithEq_of_rows h (row := 825) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_825_19
  simp only [← c51549, ← c51550, k2, ← c51551] at e_825_19
  have hr := e_825_19
  simp only [e_825_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6025)) := by
  have c51552 := (publicBatchWrapper89_copies1611 a h).1
  have c51553 := (publicBatchWrapper89_copies1611 a h).2.1
  have c51554 := (publicBatchWrapper89_copies1611 a h).2.2.1
  have c51555 := (publicBatchWrapper89_copies1611 a h).2.2.2.1
  have c51556 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.1
  have c51557 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_0 := arithEq_of_rows h (row := 826) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_0
  simp only [← c51552, ← c51553, ← c51554] at e_826_0
  have e_826_1 := arithEq_of_rows h (row := 826) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_1
  simp only [← c51555, ← c51556, k2, ← c51557] at e_826_1
  have hr := e_826_1
  simp only [e_826_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6026)) := by
  have c51558 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.1
  have c51559 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.1
  have c51560 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.1
  have c51561 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.1
  have c51562 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51563 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_2 := arithEq_of_rows h (row := 826) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_2
  simp only [← c51558, ← c51559, ← c51560] at e_826_2
  have e_826_3 := arithEq_of_rows h (row := 826) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_3
  simp only [← c51561, ← c51562, k2, ← c51563] at e_826_3
  have hr := e_826_3
  simp only [e_826_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6027)) := by
  have c51564 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51565 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51566 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51567 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51568 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51569 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_4 := arithEq_of_rows h (row := 826) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_4
  simp only [← c51564, ← c51565, ← c51566] at e_826_4
  have e_826_5 := arithEq_of_rows h (row := 826) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_5
  simp only [← c51567, ← c51568, k2, ← c51569] at e_826_5
  have hr := e_826_5
  simp only [e_826_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6028)) := by
  have c51570 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51571 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51572 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51573 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51574 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51575 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_6 := arithEq_of_rows h (row := 826) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_6
  simp only [← c51570, ← c51571, ← c51572] at e_826_6
  have e_826_7 := arithEq_of_rows h (row := 826) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_7
  simp only [← c51573, ← c51574, k2, ← c51575] at e_826_7
  have hr := e_826_7
  simp only [e_826_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6029)) := by
  have c51576 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51577 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51578 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51579 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51580 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51581 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_8 := arithEq_of_rows h (row := 826) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_8
  simp only [← c51576, ← c51577, ← c51578] at e_826_8
  have e_826_9 := arithEq_of_rows h (row := 826) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_9
  simp only [← c51579, ← c51580, k2, ← c51581] at e_826_9
  have hr := e_826_9
  simp only [e_826_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6030)) := by
  have c51582 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51583 := (publicBatchWrapper89_copies1611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51584 := (publicBatchWrapper89_copies1612 a h).1
  have c51585 := (publicBatchWrapper89_copies1612 a h).2.1
  have c51586 := (publicBatchWrapper89_copies1612 a h).2.2.1
  have c51587 := (publicBatchWrapper89_copies1612 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_10 := arithEq_of_rows h (row := 826) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_10
  simp only [← c51582, ← c51583, ← c51584] at e_826_10
  have e_826_11 := arithEq_of_rows h (row := 826) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_11
  simp only [← c51585, ← c51586, k2, ← c51587] at e_826_11
  have hr := e_826_11
  simp only [e_826_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6031)) := by
  have c51588 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.1
  have c51589 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.1
  have c51590 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.1
  have c51591 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.1
  have c51592 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.1
  have c51593 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_12 := arithEq_of_rows h (row := 826) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_12
  simp only [← c51588, ← c51589, ← c51590] at e_826_12
  have e_826_13 := arithEq_of_rows h (row := 826) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_13
  simp only [← c51591, ← c51592, k2, ← c51593] at e_826_13
  have hr := e_826_13
  simp only [e_826_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6032)) := by
  have c51594 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51595 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51596 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51597 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51598 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51599 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_14 := arithEq_of_rows h (row := 826) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_14
  simp only [← c51594, ← c51595, ← c51596] at e_826_14
  have e_826_15 := arithEq_of_rows h (row := 826) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_15
  simp only [← c51597, ← c51598, k2, ← c51599] at e_826_15
  have hr := e_826_15
  simp only [e_826_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6033)) := by
  have c51600 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51601 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51602 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51603 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51604 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51605 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_16 := arithEq_of_rows h (row := 826) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_16
  simp only [← c51600, ← c51601, ← c51602] at e_826_16
  have e_826_17 := arithEq_of_rows h (row := 826) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_17
  simp only [← c51603, ← c51604, k2, ← c51605] at e_826_17
  have hr := e_826_17
  simp only [e_826_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 826 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6034)) := by
  have c51606 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51607 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51608 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51609 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51610 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51611 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_826_18 := arithEq_of_rows h (row := 826) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_18
  simp only [← c51606, ← c51607, ← c51608] at e_826_18
  have e_826_19 := arithEq_of_rows h (row := 826) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_826_19
  simp only [← c51609, ← c51610, k2, ← c51611] at e_826_19
  have hr := e_826_19
  simp only [e_826_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6035)) := by
  have c51612 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51613 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51614 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51615 := (publicBatchWrapper89_copies1612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51616 := (publicBatchWrapper89_copies1613 a h).1
  have c51617 := (publicBatchWrapper89_copies1613 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_0 := arithEq_of_rows h (row := 827) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_0
  simp only [← c51612, ← c51613, ← c51614] at e_827_0
  have e_827_1 := arithEq_of_rows h (row := 827) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_1
  simp only [← c51615, ← c51616, k2, ← c51617] at e_827_1
  have hr := e_827_1
  simp only [e_827_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6036)) := by
  have c51618 := (publicBatchWrapper89_copies1613 a h).2.2.1
  have c51619 := (publicBatchWrapper89_copies1613 a h).2.2.2.1
  have c51620 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.1
  have c51621 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.1
  have c51622 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.1
  have c51623 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_2 := arithEq_of_rows h (row := 827) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_2
  simp only [← c51618, ← c51619, ← c51620] at e_827_2
  have e_827_3 := arithEq_of_rows h (row := 827) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_3
  simp only [← c51621, ← c51622, k2, ← c51623] at e_827_3
  have hr := e_827_3
  simp only [e_827_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6037)) := by
  have c51624 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.1
  have c51625 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.1
  have c51626 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51627 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51628 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51629 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_4 := arithEq_of_rows h (row := 827) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_4
  simp only [← c51624, ← c51625, ← c51626] at e_827_4
  have e_827_5 := arithEq_of_rows h (row := 827) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_5
  simp only [← c51627, ← c51628, k2, ← c51629] at e_827_5
  have hr := e_827_5
  simp only [e_827_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6038)) := by
  have c51630 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51631 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51632 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51633 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51634 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51635 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_6 := arithEq_of_rows h (row := 827) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_6
  simp only [← c51630, ← c51631, ← c51632] at e_827_6
  have e_827_7 := arithEq_of_rows h (row := 827) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_7
  simp only [← c51633, ← c51634, k2, ← c51635] at e_827_7
  have hr := e_827_7
  simp only [e_827_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6039)) := by
  have c51636 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51637 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51638 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51639 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51640 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51641 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_8 := arithEq_of_rows h (row := 827) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_8
  simp only [← c51636, ← c51637, ← c51638] at e_827_8
  have e_827_9 := arithEq_of_rows h (row := 827) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_9
  simp only [← c51639, ← c51640, k2, ← c51641] at e_827_9
  have hr := e_827_9
  simp only [e_827_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6040)) := by
  have c51642 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51643 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51644 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51645 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51646 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51647 := (publicBatchWrapper89_copies1613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_10 := arithEq_of_rows h (row := 827) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_10
  simp only [← c51642, ← c51643, ← c51644] at e_827_10
  have e_827_11 := arithEq_of_rows h (row := 827) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_11
  simp only [← c51645, ← c51646, k2, ← c51647] at e_827_11
  have hr := e_827_11
  simp only [e_827_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6041)) := by
  have c51648 := (publicBatchWrapper89_copies1614 a h).1
  have c51649 := (publicBatchWrapper89_copies1614 a h).2.1
  have c51650 := (publicBatchWrapper89_copies1614 a h).2.2.1
  have c51651 := (publicBatchWrapper89_copies1614 a h).2.2.2.1
  have c51652 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.1
  have c51653 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_12 := arithEq_of_rows h (row := 827) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_12
  simp only [← c51648, ← c51649, ← c51650] at e_827_12
  have e_827_13 := arithEq_of_rows h (row := 827) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_13
  simp only [← c51651, ← c51652, k2, ← c51653] at e_827_13
  have hr := e_827_13
  simp only [e_827_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6042)) := by
  have c51654 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.1
  have c51655 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.1
  have c51656 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.1
  have c51657 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.1
  have c51658 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51659 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_14 := arithEq_of_rows h (row := 827) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_14
  simp only [← c51654, ← c51655, ← c51656] at e_827_14
  have e_827_15 := arithEq_of_rows h (row := 827) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_15
  simp only [← c51657, ← c51658, k2, ← c51659] at e_827_15
  have hr := e_827_15
  simp only [e_827_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6043)) := by
  have c51660 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51661 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51662 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51663 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51664 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51665 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_16 := arithEq_of_rows h (row := 827) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_16
  simp only [← c51660, ← c51661, ← c51662] at e_827_16
  have e_827_17 := arithEq_of_rows h (row := 827) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_17
  simp only [← c51663, ← c51664, k2, ← c51665] at e_827_17
  have hr := e_827_17
  simp only [e_827_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 827 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6044)) := by
  have c51666 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51667 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51668 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51669 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51670 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51671 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_827_18 := arithEq_of_rows h (row := 827) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_18
  simp only [← c51666, ← c51667, ← c51668] at e_827_18
  have e_827_19 := arithEq_of_rows h (row := 827) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_827_19
  simp only [← c51669, ← c51670, k2, ← c51671] at e_827_19
  have hr := e_827_19
  simp only [e_827_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6045)) := by
  have c51672 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51673 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51674 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51675 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51676 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51677 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_0 := arithEq_of_rows h (row := 828) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_0
  simp only [← c51672, ← c51673, ← c51674] at e_828_0
  have e_828_1 := arithEq_of_rows h (row := 828) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_1
  simp only [← c51675, ← c51676, k2, ← c51677] at e_828_1
  have hr := e_828_1
  simp only [e_828_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6046)) := by
  have c51678 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51679 := (publicBatchWrapper89_copies1614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51680 := (publicBatchWrapper89_copies1615 a h).1
  have c51681 := (publicBatchWrapper89_copies1615 a h).2.1
  have c51682 := (publicBatchWrapper89_copies1615 a h).2.2.1
  have c51683 := (publicBatchWrapper89_copies1615 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_2 := arithEq_of_rows h (row := 828) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_2
  simp only [← c51678, ← c51679, ← c51680] at e_828_2
  have e_828_3 := arithEq_of_rows h (row := 828) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_3
  simp only [← c51681, ← c51682, k2, ← c51683] at e_828_3
  have hr := e_828_3
  simp only [e_828_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6047)) := by
  have c51684 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.1
  have c51685 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.1
  have c51686 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.1
  have c51687 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.1
  have c51688 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.1
  have c51689 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_4 := arithEq_of_rows h (row := 828) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_4
  simp only [← c51684, ← c51685, ← c51686] at e_828_4
  have e_828_5 := arithEq_of_rows h (row := 828) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_5
  simp only [← c51687, ← c51688, k2, ← c51689] at e_828_5
  have hr := e_828_5
  simp only [e_828_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6048)) := by
  have c51690 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51691 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51692 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51693 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51694 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51695 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_6 := arithEq_of_rows h (row := 828) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_6
  simp only [← c51690, ← c51691, ← c51692] at e_828_6
  have e_828_7 := arithEq_of_rows h (row := 828) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_7
  simp only [← c51693, ← c51694, k2, ← c51695] at e_828_7
  have hr := e_828_7
  simp only [e_828_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6049)) := by
  have c51696 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51697 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51698 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51699 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51700 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51701 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_8 := arithEq_of_rows h (row := 828) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_8
  simp only [← c51696, ← c51697, ← c51698] at e_828_8
  have e_828_9 := arithEq_of_rows h (row := 828) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_9
  simp only [← c51699, ← c51700, k2, ← c51701] at e_828_9
  have hr := e_828_9
  simp only [e_828_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6050)) := by
  have c51702 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51703 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51704 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51705 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51706 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51707 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_10 := arithEq_of_rows h (row := 828) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_10
  simp only [← c51702, ← c51703, ← c51704] at e_828_10
  have e_828_11 := arithEq_of_rows h (row := 828) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_11
  simp only [← c51705, ← c51706, k2, ← c51707] at e_828_11
  have hr := e_828_11
  simp only [e_828_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6051)) := by
  have c51708 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51709 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51710 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51711 := (publicBatchWrapper89_copies1615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51712 := (publicBatchWrapper89_copies1616 a h).1
  have c51713 := (publicBatchWrapper89_copies1616 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_12 := arithEq_of_rows h (row := 828) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_12
  simp only [← c51708, ← c51709, ← c51710] at e_828_12
  have e_828_13 := arithEq_of_rows h (row := 828) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_13
  simp only [← c51711, ← c51712, k2, ← c51713] at e_828_13
  have hr := e_828_13
  simp only [e_828_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6052)) := by
  have c51714 := (publicBatchWrapper89_copies1616 a h).2.2.1
  have c51715 := (publicBatchWrapper89_copies1616 a h).2.2.2.1
  have c51716 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.1
  have c51717 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.1
  have c51718 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.1
  have c51719 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_14 := arithEq_of_rows h (row := 828) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_14
  simp only [← c51714, ← c51715, ← c51716] at e_828_14
  have e_828_15 := arithEq_of_rows h (row := 828) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_15
  simp only [← c51717, ← c51718, k2, ← c51719] at e_828_15
  have hr := e_828_15
  simp only [e_828_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6053)) := by
  have c51720 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.1
  have c51721 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.1
  have c51722 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51723 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51724 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51725 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_16 := arithEq_of_rows h (row := 828) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_16
  simp only [← c51720, ← c51721, ← c51722] at e_828_16
  have e_828_17 := arithEq_of_rows h (row := 828) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_17
  simp only [← c51723, ← c51724, k2, ← c51725] at e_828_17
  have hr := e_828_17
  simp only [e_828_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 828 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6054)) := by
  have c51726 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51727 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51728 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51729 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51730 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51731 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_828_18 := arithEq_of_rows h (row := 828) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_18
  simp only [← c51726, ← c51727, ← c51728] at e_828_18
  have e_828_19 := arithEq_of_rows h (row := 828) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_828_19
  simp only [← c51729, ← c51730, k2, ← c51731] at e_828_19
  have hr := e_828_19
  simp only [e_828_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6055)) := by
  have c51732 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51733 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51734 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51735 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51736 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51737 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_0 := arithEq_of_rows h (row := 829) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_0
  simp only [← c51732, ← c51733, ← c51734] at e_829_0
  have e_829_1 := arithEq_of_rows h (row := 829) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_1
  simp only [← c51735, ← c51736, k2, ← c51737] at e_829_1
  have hr := e_829_1
  simp only [e_829_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6056)) := by
  have c51738 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51739 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51740 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51741 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51742 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51743 := (publicBatchWrapper89_copies1616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_2 := arithEq_of_rows h (row := 829) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_2
  simp only [← c51738, ← c51739, ← c51740] at e_829_2
  have e_829_3 := arithEq_of_rows h (row := 829) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_3
  simp only [← c51741, ← c51742, k2, ← c51743] at e_829_3
  have hr := e_829_3
  simp only [e_829_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6057)) := by
  have c51744 := (publicBatchWrapper89_copies1617 a h).1
  have c51745 := (publicBatchWrapper89_copies1617 a h).2.1
  have c51746 := (publicBatchWrapper89_copies1617 a h).2.2.1
  have c51747 := (publicBatchWrapper89_copies1617 a h).2.2.2.1
  have c51748 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.1
  have c51749 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_4 := arithEq_of_rows h (row := 829) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_4
  simp only [← c51744, ← c51745, ← c51746] at e_829_4
  have e_829_5 := arithEq_of_rows h (row := 829) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_5
  simp only [← c51747, ← c51748, k2, ← c51749] at e_829_5
  have hr := e_829_5
  simp only [e_829_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6058)) := by
  have c51750 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.1
  have c51751 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.1
  have c51752 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.1
  have c51753 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.1
  have c51754 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51755 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_6 := arithEq_of_rows h (row := 829) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_6
  simp only [← c51750, ← c51751, ← c51752] at e_829_6
  have e_829_7 := arithEq_of_rows h (row := 829) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_7
  simp only [← c51753, ← c51754, k2, ← c51755] at e_829_7
  have hr := e_829_7
  simp only [e_829_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6059)) := by
  have c51756 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51757 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51758 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51759 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51760 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51761 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_8 := arithEq_of_rows h (row := 829) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_8
  simp only [← c51756, ← c51757, ← c51758] at e_829_8
  have e_829_9 := arithEq_of_rows h (row := 829) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_9
  simp only [← c51759, ← c51760, k2, ← c51761] at e_829_9
  have hr := e_829_9
  simp only [e_829_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6060)) := by
  have c51762 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51763 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51764 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51765 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51766 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51767 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_10 := arithEq_of_rows h (row := 829) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_10
  simp only [← c51762, ← c51763, ← c51764] at e_829_10
  have e_829_11 := arithEq_of_rows h (row := 829) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_11
  simp only [← c51765, ← c51766, k2, ← c51767] at e_829_11
  have hr := e_829_11
  simp only [e_829_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6061)) := by
  have c51768 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51769 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51770 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51771 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51772 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51773 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_12 := arithEq_of_rows h (row := 829) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_12
  simp only [← c51768, ← c51769, ← c51770] at e_829_12
  have e_829_13 := arithEq_of_rows h (row := 829) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_13
  simp only [← c51771, ← c51772, k2, ← c51773] at e_829_13
  have hr := e_829_13
  simp only [e_829_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6062)) := by
  have c51774 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51775 := (publicBatchWrapper89_copies1617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51776 := (publicBatchWrapper89_copies1618 a h).1
  have c51777 := (publicBatchWrapper89_copies1618 a h).2.1
  have c51778 := (publicBatchWrapper89_copies1618 a h).2.2.1
  have c51779 := (publicBatchWrapper89_copies1618 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_14 := arithEq_of_rows h (row := 829) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_14
  simp only [← c51774, ← c51775, ← c51776] at e_829_14
  have e_829_15 := arithEq_of_rows h (row := 829) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_15
  simp only [← c51777, ← c51778, k2, ← c51779] at e_829_15
  have hr := e_829_15
  simp only [e_829_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6063)) := by
  have c51780 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.1
  have c51781 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.1
  have c51782 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.1
  have c51783 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.1
  have c51784 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.1
  have c51785 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_16 := arithEq_of_rows h (row := 829) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_16
  simp only [← c51780, ← c51781, ← c51782] at e_829_16
  have e_829_17 := arithEq_of_rows h (row := 829) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_17
  simp only [← c51783, ← c51784, k2, ← c51785] at e_829_17
  have hr := e_829_17
  simp only [e_829_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 829 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6064)) := by
  have c51786 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51787 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51788 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51789 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51790 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51791 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_829_18 := arithEq_of_rows h (row := 829) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_18
  simp only [← c51786, ← c51787, ← c51788] at e_829_18
  have e_829_19 := arithEq_of_rows h (row := 829) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_829_19
  simp only [← c51789, ← c51790, k2, ← c51791] at e_829_19
  have hr := e_829_19
  simp only [e_829_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6065)) := by
  have c51792 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51793 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51794 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51795 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51796 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51797 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_0 := arithEq_of_rows h (row := 830) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_0
  simp only [← c51792, ← c51793, ← c51794] at e_830_0
  have e_830_1 := arithEq_of_rows h (row := 830) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_1
  simp only [← c51795, ← c51796, k2, ← c51797] at e_830_1
  have hr := e_830_1
  simp only [e_830_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6066)) := by
  have c51798 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51799 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51800 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51801 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51802 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51803 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_2 := arithEq_of_rows h (row := 830) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_2
  simp only [← c51798, ← c51799, ← c51800] at e_830_2
  have e_830_3 := arithEq_of_rows h (row := 830) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_3
  simp only [← c51801, ← c51802, k2, ← c51803] at e_830_3
  have hr := e_830_3
  simp only [e_830_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6067)) := by
  have c51804 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51805 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51806 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51807 := (publicBatchWrapper89_copies1618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51808 := (publicBatchWrapper89_copies1619 a h).1
  have c51809 := (publicBatchWrapper89_copies1619 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_4 := arithEq_of_rows h (row := 830) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_4
  simp only [← c51804, ← c51805, ← c51806] at e_830_4
  have e_830_5 := arithEq_of_rows h (row := 830) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_5
  simp only [← c51807, ← c51808, k2, ← c51809] at e_830_5
  have hr := e_830_5
  simp only [e_830_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6068)) := by
  have c51810 := (publicBatchWrapper89_copies1619 a h).2.2.1
  have c51811 := (publicBatchWrapper89_copies1619 a h).2.2.2.1
  have c51812 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.1
  have c51813 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.1
  have c51814 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.1
  have c51815 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_6 := arithEq_of_rows h (row := 830) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_6
  simp only [← c51810, ← c51811, ← c51812] at e_830_6
  have e_830_7 := arithEq_of_rows h (row := 830) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_7
  simp only [← c51813, ← c51814, k2, ← c51815] at e_830_7
  have hr := e_830_7
  simp only [e_830_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6069)) := by
  have c51816 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.1
  have c51817 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.1
  have c51818 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51819 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51820 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51821 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_8 := arithEq_of_rows h (row := 830) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_8
  simp only [← c51816, ← c51817, ← c51818] at e_830_8
  have e_830_9 := arithEq_of_rows h (row := 830) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_9
  simp only [← c51819, ← c51820, k2, ← c51821] at e_830_9
  have hr := e_830_9
  simp only [e_830_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6070)) := by
  have c51822 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51823 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51824 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51825 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51826 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51827 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_10 := arithEq_of_rows h (row := 830) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_10
  simp only [← c51822, ← c51823, ← c51824] at e_830_10
  have e_830_11 := arithEq_of_rows h (row := 830) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_11
  simp only [← c51825, ← c51826, k2, ← c51827] at e_830_11
  have hr := e_830_11
  simp only [e_830_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6071)) := by
  have c51828 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51829 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51830 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51831 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51832 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51833 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_12 := arithEq_of_rows h (row := 830) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_12
  simp only [← c51828, ← c51829, ← c51830] at e_830_12
  have e_830_13 := arithEq_of_rows h (row := 830) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_13
  simp only [← c51831, ← c51832, k2, ← c51833] at e_830_13
  have hr := e_830_13
  simp only [e_830_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6072)) := by
  have c51834 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51835 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51836 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51837 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51838 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51839 := (publicBatchWrapper89_copies1619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_14 := arithEq_of_rows h (row := 830) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_14
  simp only [← c51834, ← c51835, ← c51836] at e_830_14
  have e_830_15 := arithEq_of_rows h (row := 830) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_15
  simp only [← c51837, ← c51838, k2, ← c51839] at e_830_15
  have hr := e_830_15
  simp only [e_830_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6073)) := by
  have c51840 := (publicBatchWrapper89_copies1620 a h).1
  have c51841 := (publicBatchWrapper89_copies1620 a h).2.1
  have c51842 := (publicBatchWrapper89_copies1620 a h).2.2.1
  have c51843 := (publicBatchWrapper89_copies1620 a h).2.2.2.1
  have c51844 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.1
  have c51845 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_16 := arithEq_of_rows h (row := 830) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_16
  simp only [← c51840, ← c51841, ← c51842] at e_830_16
  have e_830_17 := arithEq_of_rows h (row := 830) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_17
  simp only [← c51843, ← c51844, k2, ← c51845] at e_830_17
  have hr := e_830_17
  simp only [e_830_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 830 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6074)) := by
  have c51846 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.1
  have c51847 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.1
  have c51848 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.1
  have c51849 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.1
  have c51850 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51851 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_830_18 := arithEq_of_rows h (row := 830) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_18
  simp only [← c51846, ← c51847, ← c51848] at e_830_18
  have e_830_19 := arithEq_of_rows h (row := 830) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_830_19
  simp only [← c51849, ← c51850, k2, ← c51851] at e_830_19
  have hr := e_830_19
  simp only [e_830_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6075)) := by
  have c51852 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51853 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51854 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51855 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51856 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51857 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_0 := arithEq_of_rows h (row := 831) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_0
  simp only [← c51852, ← c51853, ← c51854] at e_831_0
  have e_831_1 := arithEq_of_rows h (row := 831) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_1
  simp only [← c51855, ← c51856, k2, ← c51857] at e_831_1
  have hr := e_831_1
  simp only [e_831_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6076)) := by
  have c51858 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51859 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51860 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51861 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51862 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51863 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_2 := arithEq_of_rows h (row := 831) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_2
  simp only [← c51858, ← c51859, ← c51860] at e_831_2
  have e_831_3 := arithEq_of_rows h (row := 831) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_3
  simp only [← c51861, ← c51862, k2, ← c51863] at e_831_3
  have hr := e_831_3
  simp only [e_831_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6077)) := by
  have c51864 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51865 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51866 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51867 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51868 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51869 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_4 := arithEq_of_rows h (row := 831) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_4
  simp only [← c51864, ← c51865, ← c51866] at e_831_4
  have e_831_5 := arithEq_of_rows h (row := 831) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_5
  simp only [← c51867, ← c51868, k2, ← c51869] at e_831_5
  have hr := e_831_5
  simp only [e_831_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6078)) := by
  have c51870 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51871 := (publicBatchWrapper89_copies1620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51872 := (publicBatchWrapper89_copies1621 a h).1
  have c51873 := (publicBatchWrapper89_copies1621 a h).2.1
  have c51874 := (publicBatchWrapper89_copies1621 a h).2.2.1
  have c51875 := (publicBatchWrapper89_copies1621 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_6 := arithEq_of_rows h (row := 831) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_6
  simp only [← c51870, ← c51871, ← c51872] at e_831_6
  have e_831_7 := arithEq_of_rows h (row := 831) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_7
  simp only [← c51873, ← c51874, k2, ← c51875] at e_831_7
  have hr := e_831_7
  simp only [e_831_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6079)) := by
  have c51876 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.1
  have c51877 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.1
  have c51878 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.1
  have c51879 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.1
  have c51880 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.1
  have c51881 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_8 := arithEq_of_rows h (row := 831) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_8
  simp only [← c51876, ← c51877, ← c51878] at e_831_8
  have e_831_9 := arithEq_of_rows h (row := 831) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_9
  simp only [← c51879, ← c51880, k2, ← c51881] at e_831_9
  have hr := e_831_9
  simp only [e_831_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6080)) := by
  have c51882 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51883 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51884 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51885 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51886 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51887 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_10 := arithEq_of_rows h (row := 831) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_10
  simp only [← c51882, ← c51883, ← c51884] at e_831_10
  have e_831_11 := arithEq_of_rows h (row := 831) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_11
  simp only [← c51885, ← c51886, k2, ← c51887] at e_831_11
  have hr := e_831_11
  simp only [e_831_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6081)) := by
  have c51888 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51889 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51890 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51891 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51892 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51893 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_12 := arithEq_of_rows h (row := 831) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_12
  simp only [← c51888, ← c51889, ← c51890] at e_831_12
  have e_831_13 := arithEq_of_rows h (row := 831) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_13
  simp only [← c51891, ← c51892, k2, ← c51893] at e_831_13
  have hr := e_831_13
  simp only [e_831_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6082)) := by
  have c51894 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51895 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51896 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51897 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51898 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51899 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_14 := arithEq_of_rows h (row := 831) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_14
  simp only [← c51894, ← c51895, ← c51896] at e_831_14
  have e_831_15 := arithEq_of_rows h (row := 831) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_15
  simp only [← c51897, ← c51898, k2, ← c51899] at e_831_15
  have hr := e_831_15
  simp only [e_831_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6083)) := by
  have c51900 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51901 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51902 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51903 := (publicBatchWrapper89_copies1621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51904 := (publicBatchWrapper89_copies1622 a h).1
  have c51905 := (publicBatchWrapper89_copies1622 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_16 := arithEq_of_rows h (row := 831) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_16
  simp only [← c51900, ← c51901, ← c51902] at e_831_16
  have e_831_17 := arithEq_of_rows h (row := 831) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_17
  simp only [← c51903, ← c51904, k2, ← c51905] at e_831_17
  have hr := e_831_17
  simp only [e_831_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 831 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6084)) := by
  have c51906 := (publicBatchWrapper89_copies1622 a h).2.2.1
  have c51907 := (publicBatchWrapper89_copies1622 a h).2.2.2.1
  have c51908 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.1
  have c51909 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.1
  have c51910 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.1
  have c51911 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_831_18 := arithEq_of_rows h (row := 831) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_18
  simp only [← c51906, ← c51907, ← c51908] at e_831_18
  have e_831_19 := arithEq_of_rows h (row := 831) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_831_19
  simp only [← c51909, ← c51910, k2, ← c51911] at e_831_19
  have hr := e_831_19
  simp only [e_831_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6085)) := by
  have c51912 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.1
  have c51913 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.1
  have c51914 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51915 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51916 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51917 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_0 := arithEq_of_rows h (row := 832) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_0
  simp only [← c51912, ← c51913, ← c51914] at e_832_0
  have e_832_1 := arithEq_of_rows h (row := 832) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_1
  simp only [← c51915, ← c51916, k2, ← c51917] at e_832_1
  have hr := e_832_1
  simp only [e_832_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6086)) := by
  have c51918 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51919 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51920 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51921 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51922 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51923 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_2 := arithEq_of_rows h (row := 832) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_2
  simp only [← c51918, ← c51919, ← c51920] at e_832_2
  have e_832_3 := arithEq_of_rows h (row := 832) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_3
  simp only [← c51921, ← c51922, k2, ← c51923] at e_832_3
  have hr := e_832_3
  simp only [e_832_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6087)) := by
  have c51924 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51925 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51926 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51927 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51928 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51929 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_4 := arithEq_of_rows h (row := 832) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_4
  simp only [← c51924, ← c51925, ← c51926] at e_832_4
  have e_832_5 := arithEq_of_rows h (row := 832) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_5
  simp only [← c51927, ← c51928, k2, ← c51929] at e_832_5
  have hr := e_832_5
  simp only [e_832_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6088)) := by
  have c51930 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51931 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51932 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51933 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51934 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51935 := (publicBatchWrapper89_copies1622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_6 := arithEq_of_rows h (row := 832) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_6
  simp only [← c51930, ← c51931, ← c51932] at e_832_6
  have e_832_7 := arithEq_of_rows h (row := 832) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_7
  simp only [← c51933, ← c51934, k2, ← c51935] at e_832_7
  have hr := e_832_7
  simp only [e_832_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6089)) := by
  have c51936 := (publicBatchWrapper89_copies1623 a h).1
  have c51937 := (publicBatchWrapper89_copies1623 a h).2.1
  have c51938 := (publicBatchWrapper89_copies1623 a h).2.2.1
  have c51939 := (publicBatchWrapper89_copies1623 a h).2.2.2.1
  have c51940 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.1
  have c51941 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_8 := arithEq_of_rows h (row := 832) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_8
  simp only [← c51936, ← c51937, ← c51938] at e_832_8
  have e_832_9 := arithEq_of_rows h (row := 832) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_9
  simp only [← c51939, ← c51940, k2, ← c51941] at e_832_9
  have hr := e_832_9
  simp only [e_832_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6090)) := by
  have c51942 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.1
  have c51943 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.1
  have c51944 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.1
  have c51945 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.1
  have c51946 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51947 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_10 := arithEq_of_rows h (row := 832) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_10
  simp only [← c51942, ← c51943, ← c51944] at e_832_10
  have e_832_11 := arithEq_of_rows h (row := 832) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_11
  simp only [← c51945, ← c51946, k2, ← c51947] at e_832_11
  have hr := e_832_11
  simp only [e_832_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6091)) := by
  have c51948 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51949 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51950 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51951 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51952 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51953 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_12 := arithEq_of_rows h (row := 832) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_12
  simp only [← c51948, ← c51949, ← c51950] at e_832_12
  have e_832_13 := arithEq_of_rows h (row := 832) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_13
  simp only [← c51951, ← c51952, k2, ← c51953] at e_832_13
  have hr := e_832_13
  simp only [e_832_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6092)) := by
  have c51954 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51955 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51956 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51957 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51958 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51959 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_14 := arithEq_of_rows h (row := 832) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_14
  simp only [← c51954, ← c51955, ← c51956] at e_832_14
  have e_832_15 := arithEq_of_rows h (row := 832) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_15
  simp only [← c51957, ← c51958, k2, ← c51959] at e_832_15
  have hr := e_832_15
  simp only [e_832_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6093)) := by
  have c51960 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51961 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51962 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51963 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51964 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51965 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_16 := arithEq_of_rows h (row := 832) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_16
  simp only [← c51960, ← c51961, ← c51962] at e_832_16
  have e_832_17 := arithEq_of_rows h (row := 832) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_17
  simp only [← c51963, ← c51964, k2, ← c51965] at e_832_17
  have hr := e_832_17
  simp only [e_832_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 832 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6094)) := by
  have c51966 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51967 := (publicBatchWrapper89_copies1623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c51968 := (publicBatchWrapper89_copies1624 a h).1
  have c51969 := (publicBatchWrapper89_copies1624 a h).2.1
  have c51970 := (publicBatchWrapper89_copies1624 a h).2.2.1
  have c51971 := (publicBatchWrapper89_copies1624 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_832_18 := arithEq_of_rows h (row := 832) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_18
  simp only [← c51966, ← c51967, ← c51968] at e_832_18
  have e_832_19 := arithEq_of_rows h (row := 832) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_832_19
  simp only [← c51969, ← c51970, k2, ← c51971] at e_832_19
  have hr := e_832_19
  simp only [e_832_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6095)) := by
  have c51972 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.1
  have c51973 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.1
  have c51974 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.1
  have c51975 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.1
  have c51976 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.1
  have c51977 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_0 := arithEq_of_rows h (row := 833) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_0
  simp only [← c51972, ← c51973, ← c51974] at e_833_0
  have e_833_1 := arithEq_of_rows h (row := 833) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_1
  simp only [← c51975, ← c51976, k2, ← c51977] at e_833_1
  have hr := e_833_1
  simp only [e_833_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6096)) := by
  have c51978 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c51979 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c51980 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51981 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51982 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51983 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_2 := arithEq_of_rows h (row := 833) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_2
  simp only [← c51978, ← c51979, ← c51980] at e_833_2
  have e_833_3 := arithEq_of_rows h (row := 833) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_3
  simp only [← c51981, ← c51982, k2, ← c51983] at e_833_3
  have hr := e_833_3
  simp only [e_833_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6097)) := by
  have c51984 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51985 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51986 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51987 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51988 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51989 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_4 := arithEq_of_rows h (row := 833) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_4
  simp only [← c51984, ← c51985, ← c51986] at e_833_4
  have e_833_5 := arithEq_of_rows h (row := 833) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_5
  simp only [← c51987, ← c51988, k2, ← c51989] at e_833_5
  have hr := e_833_5
  simp only [e_833_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6098)) := by
  have c51990 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51991 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51992 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51993 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51994 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51995 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_6 := arithEq_of_rows h (row := 833) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_6
  simp only [← c51990, ← c51991, ← c51992] at e_833_6
  have e_833_7 := arithEq_of_rows h (row := 833) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_7
  simp only [← c51993, ← c51994, k2, ← c51995] at e_833_7
  have hr := e_833_7
  simp only [e_833_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6099)) := by
  have c51996 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51997 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51998 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51999 := (publicBatchWrapper89_copies1624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52000 := (publicBatchWrapper89_copies1625 a h).1
  have c52001 := (publicBatchWrapper89_copies1625 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_8 := arithEq_of_rows h (row := 833) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_8
  simp only [← c51996, ← c51997, ← c51998] at e_833_8
  have e_833_9 := arithEq_of_rows h (row := 833) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_9
  simp only [← c51999, ← c52000, k2, ← c52001] at e_833_9
  have hr := e_833_9
  simp only [e_833_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6100)) := by
  have c52002 := (publicBatchWrapper89_copies1625 a h).2.2.1
  have c52003 := (publicBatchWrapper89_copies1625 a h).2.2.2.1
  have c52004 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.1
  have c52005 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.1
  have c52006 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.1
  have c52007 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_10 := arithEq_of_rows h (row := 833) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_10
  simp only [← c52002, ← c52003, ← c52004] at e_833_10
  have e_833_11 := arithEq_of_rows h (row := 833) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_11
  simp only [← c52005, ← c52006, k2, ← c52007] at e_833_11
  have hr := e_833_11
  simp only [e_833_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6101)) := by
  have c52008 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.1
  have c52009 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.1
  have c52010 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52011 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52012 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52013 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_12 := arithEq_of_rows h (row := 833) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_12
  simp only [← c52008, ← c52009, ← c52010] at e_833_12
  have e_833_13 := arithEq_of_rows h (row := 833) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_13
  simp only [← c52011, ← c52012, k2, ← c52013] at e_833_13
  have hr := e_833_13
  simp only [e_833_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6102)) := by
  have c52014 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52015 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52016 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52017 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52018 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52019 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_14 := arithEq_of_rows h (row := 833) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_14
  simp only [← c52014, ← c52015, ← c52016] at e_833_14
  have e_833_15 := arithEq_of_rows h (row := 833) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_15
  simp only [← c52017, ← c52018, k2, ← c52019] at e_833_15
  have hr := e_833_15
  simp only [e_833_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6103)) := by
  have c52020 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52021 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52022 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52023 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52024 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52025 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_16 := arithEq_of_rows h (row := 833) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_16
  simp only [← c52020, ← c52021, ← c52022] at e_833_16
  have e_833_17 := arithEq_of_rows h (row := 833) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_17
  simp only [← c52023, ← c52024, k2, ← c52025] at e_833_17
  have hr := e_833_17
  simp only [e_833_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 833 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6104)) := by
  have c52026 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52027 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52028 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52029 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52030 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52031 := (publicBatchWrapper89_copies1625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_833_18 := arithEq_of_rows h (row := 833) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_18
  simp only [← c52026, ← c52027, ← c52028] at e_833_18
  have e_833_19 := arithEq_of_rows h (row := 833) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_833_19
  simp only [← c52029, ← c52030, k2, ← c52031] at e_833_19
  have hr := e_833_19
  simp only [e_833_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6105)) := by
  have c52032 := (publicBatchWrapper89_copies1626 a h).1
  have c52033 := (publicBatchWrapper89_copies1626 a h).2.1
  have c52034 := (publicBatchWrapper89_copies1626 a h).2.2.1
  have c52035 := (publicBatchWrapper89_copies1626 a h).2.2.2.1
  have c52036 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.1
  have c52037 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_0 := arithEq_of_rows h (row := 834) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_0
  simp only [← c52032, ← c52033, ← c52034] at e_834_0
  have e_834_1 := arithEq_of_rows h (row := 834) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_1
  simp only [← c52035, ← c52036, k2, ← c52037] at e_834_1
  have hr := e_834_1
  simp only [e_834_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6106)) := by
  have c52038 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.1
  have c52039 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.1
  have c52040 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.1
  have c52041 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.1
  have c52042 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52043 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_2 := arithEq_of_rows h (row := 834) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_2
  simp only [← c52038, ← c52039, ← c52040] at e_834_2
  have e_834_3 := arithEq_of_rows h (row := 834) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_3
  simp only [← c52041, ← c52042, k2, ← c52043] at e_834_3
  have hr := e_834_3
  simp only [e_834_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6107)) := by
  have c52044 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52045 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52046 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52047 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52048 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52049 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_4 := arithEq_of_rows h (row := 834) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_4
  simp only [← c52044, ← c52045, ← c52046] at e_834_4
  have e_834_5 := arithEq_of_rows h (row := 834) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_5
  simp only [← c52047, ← c52048, k2, ← c52049] at e_834_5
  have hr := e_834_5
  simp only [e_834_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6108)) := by
  have c52050 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52051 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52052 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52053 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52054 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52055 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_6 := arithEq_of_rows h (row := 834) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_6
  simp only [← c52050, ← c52051, ← c52052] at e_834_6
  have e_834_7 := arithEq_of_rows h (row := 834) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_7
  simp only [← c52053, ← c52054, k2, ← c52055] at e_834_7
  have hr := e_834_7
  simp only [e_834_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6109)) := by
  have c52056 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52057 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52058 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52059 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52060 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52061 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_8 := arithEq_of_rows h (row := 834) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_8
  simp only [← c52056, ← c52057, ← c52058] at e_834_8
  have e_834_9 := arithEq_of_rows h (row := 834) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_9
  simp only [← c52059, ← c52060, k2, ← c52061] at e_834_9
  have hr := e_834_9
  simp only [e_834_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6110)) := by
  have c52062 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52063 := (publicBatchWrapper89_copies1626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52064 := (publicBatchWrapper89_copies1627 a h).1
  have c52065 := (publicBatchWrapper89_copies1627 a h).2.1
  have c52066 := (publicBatchWrapper89_copies1627 a h).2.2.1
  have c52067 := (publicBatchWrapper89_copies1627 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_10 := arithEq_of_rows h (row := 834) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_10
  simp only [← c52062, ← c52063, ← c52064] at e_834_10
  have e_834_11 := arithEq_of_rows h (row := 834) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_11
  simp only [← c52065, ← c52066, k2, ← c52067] at e_834_11
  have hr := e_834_11
  simp only [e_834_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6111)) := by
  have c52068 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.1
  have c52069 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.1
  have c52070 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.1
  have c52071 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.1
  have c52072 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.1
  have c52073 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_12 := arithEq_of_rows h (row := 834) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_12
  simp only [← c52068, ← c52069, ← c52070] at e_834_12
  have e_834_13 := arithEq_of_rows h (row := 834) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_13
  simp only [← c52071, ← c52072, k2, ← c52073] at e_834_13
  have hr := e_834_13
  simp only [e_834_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6112)) := by
  have c52074 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52075 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52076 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52077 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52078 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52079 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_14 := arithEq_of_rows h (row := 834) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_14
  simp only [← c52074, ← c52075, ← c52076] at e_834_14
  have e_834_15 := arithEq_of_rows h (row := 834) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_15
  simp only [← c52077, ← c52078, k2, ← c52079] at e_834_15
  have hr := e_834_15
  simp only [e_834_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6113)) := by
  have c52080 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52081 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52082 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52083 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52084 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52085 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_16 := arithEq_of_rows h (row := 834) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_16
  simp only [← c52080, ← c52081, ← c52082] at e_834_16
  have e_834_17 := arithEq_of_rows h (row := 834) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_17
  simp only [← c52083, ← c52084, k2, ← c52085] at e_834_17
  have hr := e_834_17
  simp only [e_834_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 834 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6114)) := by
  have c52086 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52087 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52088 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52089 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52090 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52091 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_834_18 := arithEq_of_rows h (row := 834) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_18
  simp only [← c52086, ← c52087, ← c52088] at e_834_18
  have e_834_19 := arithEq_of_rows h (row := 834) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_834_19
  simp only [← c52089, ← c52090, k2, ← c52091] at e_834_19
  have hr := e_834_19
  simp only [e_834_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6115)) := by
  have c52092 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52093 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52094 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52095 := (publicBatchWrapper89_copies1627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52096 := (publicBatchWrapper89_copies1628 a h).1
  have c52097 := (publicBatchWrapper89_copies1628 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_0 := arithEq_of_rows h (row := 835) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_0
  simp only [← c52092, ← c52093, ← c52094] at e_835_0
  have e_835_1 := arithEq_of_rows h (row := 835) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_1
  simp only [← c52095, ← c52096, k2, ← c52097] at e_835_1
  have hr := e_835_1
  simp only [e_835_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6116)) := by
  have c52098 := (publicBatchWrapper89_copies1628 a h).2.2.1
  have c52099 := (publicBatchWrapper89_copies1628 a h).2.2.2.1
  have c52100 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.1
  have c52101 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.1
  have c52102 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.1
  have c52103 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_2 := arithEq_of_rows h (row := 835) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_2
  simp only [← c52098, ← c52099, ← c52100] at e_835_2
  have e_835_3 := arithEq_of_rows h (row := 835) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_3
  simp only [← c52101, ← c52102, k2, ← c52103] at e_835_3
  have hr := e_835_3
  simp only [e_835_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6117)) := by
  have c52104 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.1
  have c52105 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.1
  have c52106 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52107 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52108 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52109 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_4 := arithEq_of_rows h (row := 835) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_4
  simp only [← c52104, ← c52105, ← c52106] at e_835_4
  have e_835_5 := arithEq_of_rows h (row := 835) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_5
  simp only [← c52107, ← c52108, k2, ← c52109] at e_835_5
  have hr := e_835_5
  simp only [e_835_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6118)) := by
  have c52110 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52111 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52112 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52113 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52114 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52115 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_6 := arithEq_of_rows h (row := 835) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_6
  simp only [← c52110, ← c52111, ← c52112] at e_835_6
  have e_835_7 := arithEq_of_rows h (row := 835) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_7
  simp only [← c52113, ← c52114, k2, ← c52115] at e_835_7
  have hr := e_835_7
  simp only [e_835_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6119)) := by
  have c52116 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52117 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52118 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52119 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52120 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52121 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_8 := arithEq_of_rows h (row := 835) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_8
  simp only [← c52116, ← c52117, ← c52118] at e_835_8
  have e_835_9 := arithEq_of_rows h (row := 835) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_9
  simp only [← c52119, ← c52120, k2, ← c52121] at e_835_9
  have hr := e_835_9
  simp only [e_835_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6120)) := by
  have c52122 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52123 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52124 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52125 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52126 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52127 := (publicBatchWrapper89_copies1628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_10 := arithEq_of_rows h (row := 835) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_10
  simp only [← c52122, ← c52123, ← c52124] at e_835_10
  have e_835_11 := arithEq_of_rows h (row := 835) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_11
  simp only [← c52125, ← c52126, k2, ← c52127] at e_835_11
  have hr := e_835_11
  simp only [e_835_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6121)) := by
  have c52128 := (publicBatchWrapper89_copies1629 a h).1
  have c52129 := (publicBatchWrapper89_copies1629 a h).2.1
  have c52130 := (publicBatchWrapper89_copies1629 a h).2.2.1
  have c52131 := (publicBatchWrapper89_copies1629 a h).2.2.2.1
  have c52132 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.1
  have c52133 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_12 := arithEq_of_rows h (row := 835) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_12
  simp only [← c52128, ← c52129, ← c52130] at e_835_12
  have e_835_13 := arithEq_of_rows h (row := 835) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_13
  simp only [← c52131, ← c52132, k2, ← c52133] at e_835_13
  have hr := e_835_13
  simp only [e_835_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6122)) := by
  have c52134 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.1
  have c52135 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.1
  have c52136 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.1
  have c52137 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.1
  have c52138 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52139 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_14 := arithEq_of_rows h (row := 835) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_14
  simp only [← c52134, ← c52135, ← c52136] at e_835_14
  have e_835_15 := arithEq_of_rows h (row := 835) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_15
  simp only [← c52137, ← c52138, k2, ← c52139] at e_835_15
  have hr := e_835_15
  simp only [e_835_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6123)) := by
  have c52140 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52141 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52142 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52143 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52144 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52145 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_16 := arithEq_of_rows h (row := 835) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_16
  simp only [← c52140, ← c52141, ← c52142] at e_835_16
  have e_835_17 := arithEq_of_rows h (row := 835) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_17
  simp only [← c52143, ← c52144, k2, ← c52145] at e_835_17
  have hr := e_835_17
  simp only [e_835_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 835 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6124)) := by
  have c52146 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52147 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52148 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52149 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52150 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52151 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_835_18 := arithEq_of_rows h (row := 835) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_18
  simp only [← c52146, ← c52147, ← c52148] at e_835_18
  have e_835_19 := arithEq_of_rows h (row := 835) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_835_19
  simp only [← c52149, ← c52150, k2, ← c52151] at e_835_19
  have hr := e_835_19
  simp only [e_835_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6125)) := by
  have c52152 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52153 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52154 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52155 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52156 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52157 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_0 := arithEq_of_rows h (row := 836) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_0
  simp only [← c52152, ← c52153, ← c52154] at e_836_0
  have e_836_1 := arithEq_of_rows h (row := 836) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_1
  simp only [← c52155, ← c52156, k2, ← c52157] at e_836_1
  have hr := e_836_1
  simp only [e_836_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6126)) := by
  have c52158 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52159 := (publicBatchWrapper89_copies1629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52160 := (publicBatchWrapper89_copies1630 a h).1
  have c52161 := (publicBatchWrapper89_copies1630 a h).2.1
  have c52162 := (publicBatchWrapper89_copies1630 a h).2.2.1
  have c52163 := (publicBatchWrapper89_copies1630 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_2 := arithEq_of_rows h (row := 836) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_2
  simp only [← c52158, ← c52159, ← c52160] at e_836_2
  have e_836_3 := arithEq_of_rows h (row := 836) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_3
  simp only [← c52161, ← c52162, k2, ← c52163] at e_836_3
  have hr := e_836_3
  simp only [e_836_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6127)) := by
  have c52164 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.1
  have c52165 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.1
  have c52166 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.1
  have c52167 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.1
  have c52168 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.1
  have c52169 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_4 := arithEq_of_rows h (row := 836) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_4
  simp only [← c52164, ← c52165, ← c52166] at e_836_4
  have e_836_5 := arithEq_of_rows h (row := 836) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_5
  simp only [← c52167, ← c52168, k2, ← c52169] at e_836_5
  have hr := e_836_5
  simp only [e_836_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6128)) := by
  have c52170 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52171 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52172 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52173 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52174 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52175 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_6 := arithEq_of_rows h (row := 836) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_6
  simp only [← c52170, ← c52171, ← c52172] at e_836_6
  have e_836_7 := arithEq_of_rows h (row := 836) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_7
  simp only [← c52173, ← c52174, k2, ← c52175] at e_836_7
  have hr := e_836_7
  simp only [e_836_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6129)) := by
  have c52176 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52177 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52178 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52179 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52180 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52181 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_8 := arithEq_of_rows h (row := 836) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_8
  simp only [← c52176, ← c52177, ← c52178] at e_836_8
  have e_836_9 := arithEq_of_rows h (row := 836) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_9
  simp only [← c52179, ← c52180, k2, ← c52181] at e_836_9
  have hr := e_836_9
  simp only [e_836_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6130)) := by
  have c52182 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52183 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52184 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52185 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52186 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52187 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_10 := arithEq_of_rows h (row := 836) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_10
  simp only [← c52182, ← c52183, ← c52184] at e_836_10
  have e_836_11 := arithEq_of_rows h (row := 836) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_11
  simp only [← c52185, ← c52186, k2, ← c52187] at e_836_11
  have hr := e_836_11
  simp only [e_836_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6131)) := by
  have c52188 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52189 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52190 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52191 := (publicBatchWrapper89_copies1630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52192 := (publicBatchWrapper89_copies1631 a h).1
  have c52193 := (publicBatchWrapper89_copies1631 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_12 := arithEq_of_rows h (row := 836) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_12
  simp only [← c52188, ← c52189, ← c52190] at e_836_12
  have e_836_13 := arithEq_of_rows h (row := 836) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_13
  simp only [← c52191, ← c52192, k2, ← c52193] at e_836_13
  have hr := e_836_13
  simp only [e_836_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6132)) := by
  have c52194 := (publicBatchWrapper89_copies1631 a h).2.2.1
  have c52195 := (publicBatchWrapper89_copies1631 a h).2.2.2.1
  have c52196 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.1
  have c52197 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.1
  have c52198 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.1
  have c52199 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_14 := arithEq_of_rows h (row := 836) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_14
  simp only [← c52194, ← c52195, ← c52196] at e_836_14
  have e_836_15 := arithEq_of_rows h (row := 836) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_15
  simp only [← c52197, ← c52198, k2, ← c52199] at e_836_15
  have hr := e_836_15
  simp only [e_836_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6133)) := by
  have c52200 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.1
  have c52201 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.1
  have c52202 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52203 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52204 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52205 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_16 := arithEq_of_rows h (row := 836) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_16
  simp only [← c52200, ← c52201, ← c52202] at e_836_16
  have e_836_17 := arithEq_of_rows h (row := 836) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_17
  simp only [← c52203, ← c52204, k2, ← c52205] at e_836_17
  have hr := e_836_17
  simp only [e_836_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 836 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6134)) := by
  have c52206 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52207 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52208 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52209 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52210 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52211 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_836_18 := arithEq_of_rows h (row := 836) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_18
  simp only [← c52206, ← c52207, ← c52208] at e_836_18
  have e_836_19 := arithEq_of_rows h (row := 836) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_836_19
  simp only [← c52209, ← c52210, k2, ← c52211] at e_836_19
  have hr := e_836_19
  simp only [e_836_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6135)) := by
  have c52212 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52213 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52214 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52215 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52216 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52217 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_0 := arithEq_of_rows h (row := 837) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_0
  simp only [← c52212, ← c52213, ← c52214] at e_837_0
  have e_837_1 := arithEq_of_rows h (row := 837) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_1
  simp only [← c52215, ← c52216, k2, ← c52217] at e_837_1
  have hr := e_837_1
  simp only [e_837_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6136)) := by
  have c52218 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52219 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52220 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52221 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52222 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52223 := (publicBatchWrapper89_copies1631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_2 := arithEq_of_rows h (row := 837) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_2
  simp only [← c52218, ← c52219, ← c52220] at e_837_2
  have e_837_3 := arithEq_of_rows h (row := 837) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_3
  simp only [← c52221, ← c52222, k2, ← c52223] at e_837_3
  have hr := e_837_3
  simp only [e_837_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6137)) := by
  have c52224 := (publicBatchWrapper89_copies1632 a h).1
  have c52225 := (publicBatchWrapper89_copies1632 a h).2.1
  have c52226 := (publicBatchWrapper89_copies1632 a h).2.2.1
  have c52227 := (publicBatchWrapper89_copies1632 a h).2.2.2.1
  have c52228 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.1
  have c52229 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_4 := arithEq_of_rows h (row := 837) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_4
  simp only [← c52224, ← c52225, ← c52226] at e_837_4
  have e_837_5 := arithEq_of_rows h (row := 837) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_5
  simp only [← c52227, ← c52228, k2, ← c52229] at e_837_5
  have hr := e_837_5
  simp only [e_837_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6138)) := by
  have c52230 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.1
  have c52231 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.1
  have c52232 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.1
  have c52233 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.1
  have c52234 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52235 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_6 := arithEq_of_rows h (row := 837) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_6
  simp only [← c52230, ← c52231, ← c52232] at e_837_6
  have e_837_7 := arithEq_of_rows h (row := 837) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_7
  simp only [← c52233, ← c52234, k2, ← c52235] at e_837_7
  have hr := e_837_7
  simp only [e_837_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6139)) := by
  have c52236 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52237 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52238 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52239 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52240 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52241 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_8 := arithEq_of_rows h (row := 837) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_8
  simp only [← c52236, ← c52237, ← c52238] at e_837_8
  have e_837_9 := arithEq_of_rows h (row := 837) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_9
  simp only [← c52239, ← c52240, k2, ← c52241] at e_837_9
  have hr := e_837_9
  simp only [e_837_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6140)) := by
  have c52242 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52243 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52244 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52245 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52246 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52247 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_10 := arithEq_of_rows h (row := 837) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_10
  simp only [← c52242, ← c52243, ← c52244] at e_837_10
  have e_837_11 := arithEq_of_rows h (row := 837) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_11
  simp only [← c52245, ← c52246, k2, ← c52247] at e_837_11
  have hr := e_837_11
  simp only [e_837_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6141)) := by
  have c52248 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52249 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52250 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52251 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52252 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52253 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_12 := arithEq_of_rows h (row := 837) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_12
  simp only [← c52248, ← c52249, ← c52250] at e_837_12
  have e_837_13 := arithEq_of_rows h (row := 837) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_13
  simp only [← c52251, ← c52252, k2, ← c52253] at e_837_13
  have hr := e_837_13
  simp only [e_837_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6142)) := by
  have c52254 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52255 := (publicBatchWrapper89_copies1632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52256 := (publicBatchWrapper89_copies1633 a h).1
  have c52257 := (publicBatchWrapper89_copies1633 a h).2.1
  have c52258 := (publicBatchWrapper89_copies1633 a h).2.2.1
  have c52259 := (publicBatchWrapper89_copies1633 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_14 := arithEq_of_rows h (row := 837) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_14
  simp only [← c52254, ← c52255, ← c52256] at e_837_14
  have e_837_15 := arithEq_of_rows h (row := 837) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_15
  simp only [← c52257, ← c52258, k2, ← c52259] at e_837_15
  have hr := e_837_15
  simp only [e_837_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6143)) := by
  have c52260 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.1
  have c52261 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.1
  have c52262 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.1
  have c52263 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.1
  have c52264 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.1
  have c52265 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_16 := arithEq_of_rows h (row := 837) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_16
  simp only [← c52260, ← c52261, ← c52262] at e_837_16
  have e_837_17 := arithEq_of_rows h (row := 837) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_17
  simp only [← c52263, ← c52264, k2, ← c52265] at e_837_17
  have hr := e_837_17
  simp only [e_837_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 837 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6144)) := by
  have c52266 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52267 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52268 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52269 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52270 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52271 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_837_18 := arithEq_of_rows h (row := 837) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_18
  simp only [← c52266, ← c52267, ← c52268] at e_837_18
  have e_837_19 := arithEq_of_rows h (row := 837) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_837_19
  simp only [← c52269, ← c52270, k2, ← c52271] at e_837_19
  have hr := e_837_19
  simp only [e_837_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6145)) := by
  have c52272 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52273 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52274 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52275 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52276 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52277 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_0 := arithEq_of_rows h (row := 838) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_0
  simp only [← c52272, ← c52273, ← c52274] at e_838_0
  have e_838_1 := arithEq_of_rows h (row := 838) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_1
  simp only [← c52275, ← c52276, k2, ← c52277] at e_838_1
  have hr := e_838_1
  simp only [e_838_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6146)) := by
  have c52278 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52279 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52280 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52281 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52282 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52283 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_2 := arithEq_of_rows h (row := 838) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_2
  simp only [← c52278, ← c52279, ← c52280] at e_838_2
  have e_838_3 := arithEq_of_rows h (row := 838) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_3
  simp only [← c52281, ← c52282, k2, ← c52283] at e_838_3
  have hr := e_838_3
  simp only [e_838_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6147)) := by
  have c52284 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52285 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52286 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52287 := (publicBatchWrapper89_copies1633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52288 := (publicBatchWrapper89_copies1634 a h).1
  have c52289 := (publicBatchWrapper89_copies1634 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_4 := arithEq_of_rows h (row := 838) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_4
  simp only [← c52284, ← c52285, ← c52286] at e_838_4
  have e_838_5 := arithEq_of_rows h (row := 838) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_5
  simp only [← c52287, ← c52288, k2, ← c52289] at e_838_5
  have hr := e_838_5
  simp only [e_838_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6148)) := by
  have c52290 := (publicBatchWrapper89_copies1634 a h).2.2.1
  have c52291 := (publicBatchWrapper89_copies1634 a h).2.2.2.1
  have c52292 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.1
  have c52293 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.1
  have c52294 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.1
  have c52295 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_6 := arithEq_of_rows h (row := 838) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_6
  simp only [← c52290, ← c52291, ← c52292] at e_838_6
  have e_838_7 := arithEq_of_rows h (row := 838) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_7
  simp only [← c52293, ← c52294, k2, ← c52295] at e_838_7
  have hr := e_838_7
  simp only [e_838_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6149)) := by
  have c52296 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.1
  have c52297 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.1
  have c52298 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52299 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52300 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52301 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_8 := arithEq_of_rows h (row := 838) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_8
  simp only [← c52296, ← c52297, ← c52298] at e_838_8
  have e_838_9 := arithEq_of_rows h (row := 838) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_9
  simp only [← c52299, ← c52300, k2, ← c52301] at e_838_9
  have hr := e_838_9
  simp only [e_838_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6150)) := by
  have c52302 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52303 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52304 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52305 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52306 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52307 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_10 := arithEq_of_rows h (row := 838) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_10
  simp only [← c52302, ← c52303, ← c52304] at e_838_10
  have e_838_11 := arithEq_of_rows h (row := 838) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_11
  simp only [← c52305, ← c52306, k2, ← c52307] at e_838_11
  have hr := e_838_11
  simp only [e_838_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6151)) := by
  have c52308 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52309 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52310 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52311 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52312 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52313 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_12 := arithEq_of_rows h (row := 838) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_12
  simp only [← c52308, ← c52309, ← c52310] at e_838_12
  have e_838_13 := arithEq_of_rows h (row := 838) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_13
  simp only [← c52311, ← c52312, k2, ← c52313] at e_838_13
  have hr := e_838_13
  simp only [e_838_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6152)) := by
  have c52314 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52315 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52316 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52317 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52318 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52319 := (publicBatchWrapper89_copies1634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_14 := arithEq_of_rows h (row := 838) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_14
  simp only [← c52314, ← c52315, ← c52316] at e_838_14
  have e_838_15 := arithEq_of_rows h (row := 838) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_15
  simp only [← c52317, ← c52318, k2, ← c52319] at e_838_15
  have hr := e_838_15
  simp only [e_838_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6153)) := by
  have c52320 := (publicBatchWrapper89_copies1635 a h).1
  have c52321 := (publicBatchWrapper89_copies1635 a h).2.1
  have c52322 := (publicBatchWrapper89_copies1635 a h).2.2.1
  have c52323 := (publicBatchWrapper89_copies1635 a h).2.2.2.1
  have c52324 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.1
  have c52325 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_16 := arithEq_of_rows h (row := 838) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_16
  simp only [← c52320, ← c52321, ← c52322] at e_838_16
  have e_838_17 := arithEq_of_rows h (row := 838) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_17
  simp only [← c52323, ← c52324, k2, ← c52325] at e_838_17
  have hr := e_838_17
  simp only [e_838_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 838 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6154)) := by
  have c52326 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.1
  have c52327 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.1
  have c52328 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.1
  have c52329 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.1
  have c52330 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52331 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_838_18 := arithEq_of_rows h (row := 838) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_18
  simp only [← c52326, ← c52327, ← c52328] at e_838_18
  have e_838_19 := arithEq_of_rows h (row := 838) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_838_19
  simp only [← c52329, ← c52330, k2, ← c52331] at e_838_19
  have hr := e_838_19
  simp only [e_838_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6155)) := by
  have c52332 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52333 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52334 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52335 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52336 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52337 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_0 := arithEq_of_rows h (row := 839) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_0
  simp only [← c52332, ← c52333, ← c52334] at e_839_0
  have e_839_1 := arithEq_of_rows h (row := 839) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_1
  simp only [← c52335, ← c52336, k2, ← c52337] at e_839_1
  have hr := e_839_1
  simp only [e_839_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6156)) := by
  have c52338 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52339 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52340 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52341 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52342 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52343 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_2 := arithEq_of_rows h (row := 839) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_2
  simp only [← c52338, ← c52339, ← c52340] at e_839_2
  have e_839_3 := arithEq_of_rows h (row := 839) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_3
  simp only [← c52341, ← c52342, k2, ← c52343] at e_839_3
  have hr := e_839_3
  simp only [e_839_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6157)) := by
  have c52344 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52345 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52346 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52347 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52348 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52349 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_4 := arithEq_of_rows h (row := 839) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_4
  simp only [← c52344, ← c52345, ← c52346] at e_839_4
  have e_839_5 := arithEq_of_rows h (row := 839) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_5
  simp only [← c52347, ← c52348, k2, ← c52349] at e_839_5
  have hr := e_839_5
  simp only [e_839_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6158)) := by
  have c52350 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52351 := (publicBatchWrapper89_copies1635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52352 := (publicBatchWrapper89_copies1636 a h).1
  have c52353 := (publicBatchWrapper89_copies1636 a h).2.1
  have c52354 := (publicBatchWrapper89_copies1636 a h).2.2.1
  have c52355 := (publicBatchWrapper89_copies1636 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_6 := arithEq_of_rows h (row := 839) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_6
  simp only [← c52350, ← c52351, ← c52352] at e_839_6
  have e_839_7 := arithEq_of_rows h (row := 839) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_7
  simp only [← c52353, ← c52354, k2, ← c52355] at e_839_7
  have hr := e_839_7
  simp only [e_839_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6159)) := by
  have c52356 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.1
  have c52357 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.1
  have c52358 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.1
  have c52359 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.1
  have c52360 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.1
  have c52361 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_8 := arithEq_of_rows h (row := 839) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_8
  simp only [← c52356, ← c52357, ← c52358] at e_839_8
  have e_839_9 := arithEq_of_rows h (row := 839) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_9
  simp only [← c52359, ← c52360, k2, ← c52361] at e_839_9
  have hr := e_839_9
  simp only [e_839_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6160)) := by
  have c52362 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52363 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52364 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52365 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52366 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52367 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_10 := arithEq_of_rows h (row := 839) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_10
  simp only [← c52362, ← c52363, ← c52364] at e_839_10
  have e_839_11 := arithEq_of_rows h (row := 839) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_11
  simp only [← c52365, ← c52366, k2, ← c52367] at e_839_11
  have hr := e_839_11
  simp only [e_839_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6161)) := by
  have c52368 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52369 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52370 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52371 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52372 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52373 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_12 := arithEq_of_rows h (row := 839) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_12
  simp only [← c52368, ← c52369, ← c52370] at e_839_12
  have e_839_13 := arithEq_of_rows h (row := 839) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_13
  simp only [← c52371, ← c52372, k2, ← c52373] at e_839_13
  have hr := e_839_13
  simp only [e_839_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6162)) := by
  have c52374 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52375 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52376 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52377 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52378 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52379 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_14 := arithEq_of_rows h (row := 839) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_14
  simp only [← c52374, ← c52375, ← c52376] at e_839_14
  have e_839_15 := arithEq_of_rows h (row := 839) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_15
  simp only [← c52377, ← c52378, k2, ← c52379] at e_839_15
  have hr := e_839_15
  simp only [e_839_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6163)) := by
  have c52380 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52381 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52382 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52383 := (publicBatchWrapper89_copies1636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52384 := (publicBatchWrapper89_copies1637 a h).1
  have c52385 := (publicBatchWrapper89_copies1637 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_16 := arithEq_of_rows h (row := 839) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_16
  simp only [← c52380, ← c52381, ← c52382] at e_839_16
  have e_839_17 := arithEq_of_rows h (row := 839) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_17
  simp only [← c52383, ← c52384, k2, ← c52385] at e_839_17
  have hr := e_839_17
  simp only [e_839_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 839 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6164)) := by
  have c52386 := (publicBatchWrapper89_copies1637 a h).2.2.1
  have c52387 := (publicBatchWrapper89_copies1637 a h).2.2.2.1
  have c52388 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.1
  have c52389 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.1
  have c52390 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.1
  have c52391 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_839_18 := arithEq_of_rows h (row := 839) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_18
  simp only [← c52386, ← c52387, ← c52388] at e_839_18
  have e_839_19 := arithEq_of_rows h (row := 839) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_839_19
  simp only [← c52389, ← c52390, k2, ← c52391] at e_839_19
  have hr := e_839_19
  simp only [e_839_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6165)) := by
  have c52392 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.1
  have c52393 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.1
  have c52394 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52395 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52396 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52397 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_0 := arithEq_of_rows h (row := 840) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_0
  simp only [← c52392, ← c52393, ← c52394] at e_840_0
  have e_840_1 := arithEq_of_rows h (row := 840) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_1
  simp only [← c52395, ← c52396, k2, ← c52397] at e_840_1
  have hr := e_840_1
  simp only [e_840_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6166)) := by
  have c52398 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52399 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52400 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52401 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52402 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52403 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_2 := arithEq_of_rows h (row := 840) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_2
  simp only [← c52398, ← c52399, ← c52400] at e_840_2
  have e_840_3 := arithEq_of_rows h (row := 840) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_3
  simp only [← c52401, ← c52402, k2, ← c52403] at e_840_3
  have hr := e_840_3
  simp only [e_840_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6167)) := by
  have c52404 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52405 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52406 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52407 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52408 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52409 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_4 := arithEq_of_rows h (row := 840) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_4
  simp only [← c52404, ← c52405, ← c52406] at e_840_4
  have e_840_5 := arithEq_of_rows h (row := 840) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_5
  simp only [← c52407, ← c52408, k2, ← c52409] at e_840_5
  have hr := e_840_5
  simp only [e_840_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6168)) := by
  have c52410 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52411 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52412 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52413 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52414 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52415 := (publicBatchWrapper89_copies1637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_6 := arithEq_of_rows h (row := 840) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_6
  simp only [← c52410, ← c52411, ← c52412] at e_840_6
  have e_840_7 := arithEq_of_rows h (row := 840) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_7
  simp only [← c52413, ← c52414, k2, ← c52415] at e_840_7
  have hr := e_840_7
  simp only [e_840_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6169)) := by
  have c52416 := (publicBatchWrapper89_copies1638 a h).1
  have c52417 := (publicBatchWrapper89_copies1638 a h).2.1
  have c52418 := (publicBatchWrapper89_copies1638 a h).2.2.1
  have c52419 := (publicBatchWrapper89_copies1638 a h).2.2.2.1
  have c52420 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.1
  have c52421 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_8 := arithEq_of_rows h (row := 840) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_8
  simp only [← c52416, ← c52417, ← c52418] at e_840_8
  have e_840_9 := arithEq_of_rows h (row := 840) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_9
  simp only [← c52419, ← c52420, k2, ← c52421] at e_840_9
  have hr := e_840_9
  simp only [e_840_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6210)) := by
  have c52422 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.1
  have c52423 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.1
  have c52424 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.1
  have c52425 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.1
  have c52426 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52427 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_10 := arithEq_of_rows h (row := 840) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_10
  simp only [← c52422, ← c52423, ← c52424] at e_840_10
  have e_840_11 := arithEq_of_rows h (row := 840) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_11
  simp only [← c52425, ← c52426, k2, ← c52427] at e_840_11
  have hr := e_840_11
  simp only [e_840_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6211)) := by
  have c52428 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52429 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52430 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52431 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52432 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52433 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_12 := arithEq_of_rows h (row := 840) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_12
  simp only [← c52428, ← c52429, ← c52430] at e_840_12
  have e_840_13 := arithEq_of_rows h (row := 840) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_13
  simp only [← c52431, ← c52432, k2, ← c52433] at e_840_13
  have hr := e_840_13
  simp only [e_840_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6212)) := by
  have c52434 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52435 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52436 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52437 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52438 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52439 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_14 := arithEq_of_rows h (row := 840) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_14
  simp only [← c52434, ← c52435, ← c52436] at e_840_14
  have e_840_15 := arithEq_of_rows h (row := 840) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_15
  simp only [← c52437, ← c52438, k2, ← c52439] at e_840_15
  have hr := e_840_15
  simp only [e_840_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6213)) := by
  have c52440 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52441 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52442 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52443 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52444 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52445 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_16 := arithEq_of_rows h (row := 840) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_16
  simp only [← c52440, ← c52441, ← c52442] at e_840_16
  have e_840_17 := arithEq_of_rows h (row := 840) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_17
  simp only [← c52443, ← c52444, k2, ← c52445] at e_840_17
  have hr := e_840_17
  simp only [e_840_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 840 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6214)) := by
  have c52446 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52447 := (publicBatchWrapper89_copies1638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52448 := (publicBatchWrapper89_copies1639 a h).1
  have c52449 := (publicBatchWrapper89_copies1639 a h).2.1
  have c52450 := (publicBatchWrapper89_copies1639 a h).2.2.1
  have c52451 := (publicBatchWrapper89_copies1639 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_840_18 := arithEq_of_rows h (row := 840) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_18
  simp only [← c52446, ← c52447, ← c52448] at e_840_18
  have e_840_19 := arithEq_of_rows h (row := 840) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_840_19
  simp only [← c52449, ← c52450, k2, ← c52451] at e_840_19
  have hr := e_840_19
  simp only [e_840_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6215)) := by
  have c52452 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.1
  have c52453 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.1
  have c52454 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.1
  have c52455 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.1
  have c52456 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.1
  have c52457 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_0 := arithEq_of_rows h (row := 841) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_0
  simp only [← c52452, ← c52453, ← c52454] at e_841_0
  have e_841_1 := arithEq_of_rows h (row := 841) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_1
  simp only [← c52455, ← c52456, k2, ← c52457] at e_841_1
  have hr := e_841_1
  simp only [e_841_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6216)) := by
  have c52458 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52459 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52460 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52461 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52462 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52463 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_2 := arithEq_of_rows h (row := 841) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_2
  simp only [← c52458, ← c52459, ← c52460] at e_841_2
  have e_841_3 := arithEq_of_rows h (row := 841) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_3
  simp only [← c52461, ← c52462, k2, ← c52463] at e_841_3
  have hr := e_841_3
  simp only [e_841_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6217)) := by
  have c52464 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52465 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52466 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52467 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52468 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52469 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_4 := arithEq_of_rows h (row := 841) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_4
  simp only [← c52464, ← c52465, ← c52466] at e_841_4
  have e_841_5 := arithEq_of_rows h (row := 841) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_5
  simp only [← c52467, ← c52468, k2, ← c52469] at e_841_5
  have hr := e_841_5
  simp only [e_841_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6218)) := by
  have c52470 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52471 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52472 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52473 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52474 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52475 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_6 := arithEq_of_rows h (row := 841) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_6
  simp only [← c52470, ← c52471, ← c52472] at e_841_6
  have e_841_7 := arithEq_of_rows h (row := 841) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_7
  simp only [← c52473, ← c52474, k2, ← c52475] at e_841_7
  have hr := e_841_7
  simp only [e_841_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6219)) := by
  have c52476 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52477 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52478 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52479 := (publicBatchWrapper89_copies1639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52480 := (publicBatchWrapper89_copies1640 a h).1
  have c52481 := (publicBatchWrapper89_copies1640 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_8 := arithEq_of_rows h (row := 841) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_8
  simp only [← c52476, ← c52477, ← c52478] at e_841_8
  have e_841_9 := arithEq_of_rows h (row := 841) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_9
  simp only [← c52479, ← c52480, k2, ← c52481] at e_841_9
  have hr := e_841_9
  simp only [e_841_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6220)) := by
  have c52482 := (publicBatchWrapper89_copies1640 a h).2.2.1
  have c52483 := (publicBatchWrapper89_copies1640 a h).2.2.2.1
  have c52484 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.1
  have c52485 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.1
  have c52486 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.1
  have c52487 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_10 := arithEq_of_rows h (row := 841) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_10
  simp only [← c52482, ← c52483, ← c52484] at e_841_10
  have e_841_11 := arithEq_of_rows h (row := 841) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_11
  simp only [← c52485, ← c52486, k2, ← c52487] at e_841_11
  have hr := e_841_11
  simp only [e_841_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6221)) := by
  have c52488 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.1
  have c52489 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.1
  have c52490 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52491 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52492 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52493 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_12 := arithEq_of_rows h (row := 841) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_12
  simp only [← c52488, ← c52489, ← c52490] at e_841_12
  have e_841_13 := arithEq_of_rows h (row := 841) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_13
  simp only [← c52491, ← c52492, k2, ← c52493] at e_841_13
  have hr := e_841_13
  simp only [e_841_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6222)) := by
  have c52494 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52495 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52496 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52497 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52498 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52499 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_14 := arithEq_of_rows h (row := 841) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_14
  simp only [← c52494, ← c52495, ← c52496] at e_841_14
  have e_841_15 := arithEq_of_rows h (row := 841) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_15
  simp only [← c52497, ← c52498, k2, ← c52499] at e_841_15
  have hr := e_841_15
  simp only [e_841_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6223)) := by
  have c52500 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52501 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52502 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52503 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52504 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52505 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_16 := arithEq_of_rows h (row := 841) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_16
  simp only [← c52500, ← c52501, ← c52502] at e_841_16
  have e_841_17 := arithEq_of_rows h (row := 841) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_17
  simp only [← c52503, ← c52504, k2, ← c52505] at e_841_17
  have hr := e_841_17
  simp only [e_841_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 841 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6224)) := by
  have c52506 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52507 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52508 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52509 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52510 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52511 := (publicBatchWrapper89_copies1640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_841_18 := arithEq_of_rows h (row := 841) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_18
  simp only [← c52506, ← c52507, ← c52508] at e_841_18
  have e_841_19 := arithEq_of_rows h (row := 841) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_841_19
  simp only [← c52509, ← c52510, k2, ← c52511] at e_841_19
  have hr := e_841_19
  simp only [e_841_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6225)) := by
  have c52512 := (publicBatchWrapper89_copies1641 a h).1
  have c52513 := (publicBatchWrapper89_copies1641 a h).2.1
  have c52514 := (publicBatchWrapper89_copies1641 a h).2.2.1
  have c52515 := (publicBatchWrapper89_copies1641 a h).2.2.2.1
  have c52516 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.1
  have c52517 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_0 := arithEq_of_rows h (row := 842) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_0
  simp only [← c52512, ← c52513, ← c52514] at e_842_0
  have e_842_1 := arithEq_of_rows h (row := 842) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_1
  simp only [← c52515, ← c52516, k2, ← c52517] at e_842_1
  have hr := e_842_1
  simp only [e_842_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6226)) := by
  have c52518 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.1
  have c52519 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.1
  have c52520 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.1
  have c52521 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.1
  have c52522 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52523 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_2 := arithEq_of_rows h (row := 842) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_2
  simp only [← c52518, ← c52519, ← c52520] at e_842_2
  have e_842_3 := arithEq_of_rows h (row := 842) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_3
  simp only [← c52521, ← c52522, k2, ← c52523] at e_842_3
  have hr := e_842_3
  simp only [e_842_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6227)) := by
  have c52524 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52525 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52526 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52527 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52528 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52529 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_4 := arithEq_of_rows h (row := 842) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_4
  simp only [← c52524, ← c52525, ← c52526] at e_842_4
  have e_842_5 := arithEq_of_rows h (row := 842) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_5
  simp only [← c52527, ← c52528, k2, ← c52529] at e_842_5
  have hr := e_842_5
  simp only [e_842_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6228)) := by
  have c52530 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52531 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52532 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52533 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52534 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52535 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_6 := arithEq_of_rows h (row := 842) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_6
  simp only [← c52530, ← c52531, ← c52532] at e_842_6
  have e_842_7 := arithEq_of_rows h (row := 842) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_7
  simp only [← c52533, ← c52534, k2, ← c52535] at e_842_7
  have hr := e_842_7
  simp only [e_842_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6229)) := by
  have c52536 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52537 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52538 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52539 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52540 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52541 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_8 := arithEq_of_rows h (row := 842) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_8
  simp only [← c52536, ← c52537, ← c52538] at e_842_8
  have e_842_9 := arithEq_of_rows h (row := 842) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_9
  simp only [← c52539, ← c52540, k2, ← c52541] at e_842_9
  have hr := e_842_9
  simp only [e_842_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6230)) := by
  have c52542 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52543 := (publicBatchWrapper89_copies1641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52544 := (publicBatchWrapper89_copies1642 a h).1
  have c52545 := (publicBatchWrapper89_copies1642 a h).2.1
  have c52546 := (publicBatchWrapper89_copies1642 a h).2.2.1
  have c52547 := (publicBatchWrapper89_copies1642 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_10 := arithEq_of_rows h (row := 842) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_10
  simp only [← c52542, ← c52543, ← c52544] at e_842_10
  have e_842_11 := arithEq_of_rows h (row := 842) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_11
  simp only [← c52545, ← c52546, k2, ← c52547] at e_842_11
  have hr := e_842_11
  simp only [e_842_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6231)) := by
  have c52548 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.1
  have c52549 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.1
  have c52550 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.1
  have c52551 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.1
  have c52552 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.1
  have c52553 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_12 := arithEq_of_rows h (row := 842) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_12
  simp only [← c52548, ← c52549, ← c52550] at e_842_12
  have e_842_13 := arithEq_of_rows h (row := 842) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_13
  simp only [← c52551, ← c52552, k2, ← c52553] at e_842_13
  have hr := e_842_13
  simp only [e_842_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6232)) := by
  have c52554 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52555 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52556 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52557 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52558 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52559 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_14 := arithEq_of_rows h (row := 842) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_14
  simp only [← c52554, ← c52555, ← c52556] at e_842_14
  have e_842_15 := arithEq_of_rows h (row := 842) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_15
  simp only [← c52557, ← c52558, k2, ← c52559] at e_842_15
  have hr := e_842_15
  simp only [e_842_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6233)) := by
  have c52560 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52561 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52562 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52563 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52564 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52565 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_16 := arithEq_of_rows h (row := 842) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_16
  simp only [← c52560, ← c52561, ← c52562] at e_842_16
  have e_842_17 := arithEq_of_rows h (row := 842) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_17
  simp only [← c52563, ← c52564, k2, ← c52565] at e_842_17
  have hr := e_842_17
  simp only [e_842_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 842 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6234)) := by
  have c52566 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52567 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52568 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52569 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52570 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52571 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_842_18 := arithEq_of_rows h (row := 842) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_18
  simp only [← c52566, ← c52567, ← c52568] at e_842_18
  have e_842_19 := arithEq_of_rows h (row := 842) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_842_19
  simp only [← c52569, ← c52570, k2, ← c52571] at e_842_19
  have hr := e_842_19
  simp only [e_842_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6235)) := by
  have c52572 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52573 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52574 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52575 := (publicBatchWrapper89_copies1642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52576 := (publicBatchWrapper89_copies1643 a h).1
  have c52577 := (publicBatchWrapper89_copies1643 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_0 := arithEq_of_rows h (row := 843) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_0
  simp only [← c52572, ← c52573, ← c52574] at e_843_0
  have e_843_1 := arithEq_of_rows h (row := 843) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_1
  simp only [← c52575, ← c52576, k2, ← c52577] at e_843_1
  have hr := e_843_1
  simp only [e_843_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6236)) := by
  have c52578 := (publicBatchWrapper89_copies1643 a h).2.2.1
  have c52579 := (publicBatchWrapper89_copies1643 a h).2.2.2.1
  have c52580 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.1
  have c52581 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.1
  have c52582 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.1
  have c52583 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_2 := arithEq_of_rows h (row := 843) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_2
  simp only [← c52578, ← c52579, ← c52580] at e_843_2
  have e_843_3 := arithEq_of_rows h (row := 843) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_3
  simp only [← c52581, ← c52582, k2, ← c52583] at e_843_3
  have hr := e_843_3
  simp only [e_843_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6237)) := by
  have c52584 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.1
  have c52585 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.1
  have c52586 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52587 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52588 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52589 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_4 := arithEq_of_rows h (row := 843) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_4
  simp only [← c52584, ← c52585, ← c52586] at e_843_4
  have e_843_5 := arithEq_of_rows h (row := 843) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_5
  simp only [← c52587, ← c52588, k2, ← c52589] at e_843_5
  have hr := e_843_5
  simp only [e_843_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6238)) := by
  have c52590 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52591 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52592 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52593 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52594 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52595 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_6 := arithEq_of_rows h (row := 843) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_6
  simp only [← c52590, ← c52591, ← c52592] at e_843_6
  have e_843_7 := arithEq_of_rows h (row := 843) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_7
  simp only [← c52593, ← c52594, k2, ← c52595] at e_843_7
  have hr := e_843_7
  simp only [e_843_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6239)) := by
  have c52596 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52597 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52598 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52599 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52600 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52601 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_8 := arithEq_of_rows h (row := 843) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_8
  simp only [← c52596, ← c52597, ← c52598] at e_843_8
  have e_843_9 := arithEq_of_rows h (row := 843) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_9
  simp only [← c52599, ← c52600, k2, ← c52601] at e_843_9
  have hr := e_843_9
  simp only [e_843_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6240)) := by
  have c52602 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52603 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52604 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52605 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52606 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52607 := (publicBatchWrapper89_copies1643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_10 := arithEq_of_rows h (row := 843) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_10
  simp only [← c52602, ← c52603, ← c52604] at e_843_10
  have e_843_11 := arithEq_of_rows h (row := 843) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_11
  simp only [← c52605, ← c52606, k2, ← c52607] at e_843_11
  have hr := e_843_11
  simp only [e_843_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6241)) := by
  have c52608 := (publicBatchWrapper89_copies1644 a h).1
  have c52609 := (publicBatchWrapper89_copies1644 a h).2.1
  have c52610 := (publicBatchWrapper89_copies1644 a h).2.2.1
  have c52611 := (publicBatchWrapper89_copies1644 a h).2.2.2.1
  have c52612 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.1
  have c52613 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_12 := arithEq_of_rows h (row := 843) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_12
  simp only [← c52608, ← c52609, ← c52610] at e_843_12
  have e_843_13 := arithEq_of_rows h (row := 843) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_13
  simp only [← c52611, ← c52612, k2, ← c52613] at e_843_13
  have hr := e_843_13
  simp only [e_843_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6242)) := by
  have c52614 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.1
  have c52615 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.1
  have c52616 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.1
  have c52617 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.1
  have c52618 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52619 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_14 := arithEq_of_rows h (row := 843) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_14
  simp only [← c52614, ← c52615, ← c52616] at e_843_14
  have e_843_15 := arithEq_of_rows h (row := 843) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_15
  simp only [← c52617, ← c52618, k2, ← c52619] at e_843_15
  have hr := e_843_15
  simp only [e_843_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6243)) := by
  have c52620 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52621 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52622 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52623 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52624 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52625 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_16 := arithEq_of_rows h (row := 843) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_16
  simp only [← c52620, ← c52621, ← c52622] at e_843_16
  have e_843_17 := arithEq_of_rows h (row := 843) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_17
  simp only [← c52623, ← c52624, k2, ← c52625] at e_843_17
  have hr := e_843_17
  simp only [e_843_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 843 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6244)) := by
  have c52626 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52627 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52628 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52629 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52630 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52631 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_843_18 := arithEq_of_rows h (row := 843) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_18
  simp only [← c52626, ← c52627, ← c52628] at e_843_18
  have e_843_19 := arithEq_of_rows h (row := 843) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_843_19
  simp only [← c52629, ← c52630, k2, ← c52631] at e_843_19
  have hr := e_843_19
  simp only [e_843_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6245)) := by
  have c52632 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52633 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52634 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52635 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52636 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52637 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_0 := arithEq_of_rows h (row := 844) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_0
  simp only [← c52632, ← c52633, ← c52634] at e_844_0
  have e_844_1 := arithEq_of_rows h (row := 844) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_1
  simp only [← c52635, ← c52636, k2, ← c52637] at e_844_1
  have hr := e_844_1
  simp only [e_844_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6246)) := by
  have c52638 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52639 := (publicBatchWrapper89_copies1644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52640 := (publicBatchWrapper89_copies1645 a h).1
  have c52641 := (publicBatchWrapper89_copies1645 a h).2.1
  have c52642 := (publicBatchWrapper89_copies1645 a h).2.2.1
  have c52643 := (publicBatchWrapper89_copies1645 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_2 := arithEq_of_rows h (row := 844) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_2
  simp only [← c52638, ← c52639, ← c52640] at e_844_2
  have e_844_3 := arithEq_of_rows h (row := 844) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_3
  simp only [← c52641, ← c52642, k2, ← c52643] at e_844_3
  have hr := e_844_3
  simp only [e_844_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6247)) := by
  have c52644 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.1
  have c52645 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.1
  have c52646 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.1
  have c52647 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.1
  have c52648 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.1
  have c52649 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_4 := arithEq_of_rows h (row := 844) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_4
  simp only [← c52644, ← c52645, ← c52646] at e_844_4
  have e_844_5 := arithEq_of_rows h (row := 844) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_5
  simp only [← c52647, ← c52648, k2, ← c52649] at e_844_5
  have hr := e_844_5
  simp only [e_844_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6248)) := by
  have c52650 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52651 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52652 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52653 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52654 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52655 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_6 := arithEq_of_rows h (row := 844) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_6
  simp only [← c52650, ← c52651, ← c52652] at e_844_6
  have e_844_7 := arithEq_of_rows h (row := 844) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_7
  simp only [← c52653, ← c52654, k2, ← c52655] at e_844_7
  have hr := e_844_7
  simp only [e_844_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6249)) := by
  have c52656 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52657 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52658 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52659 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52660 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52661 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_8 := arithEq_of_rows h (row := 844) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_8
  simp only [← c52656, ← c52657, ← c52658] at e_844_8
  have e_844_9 := arithEq_of_rows h (row := 844) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_9
  simp only [← c52659, ← c52660, k2, ← c52661] at e_844_9
  have hr := e_844_9
  simp only [e_844_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6250)) := by
  have c52662 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52663 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52664 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52665 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52666 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52667 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_10 := arithEq_of_rows h (row := 844) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_10
  simp only [← c52662, ← c52663, ← c52664] at e_844_10
  have e_844_11 := arithEq_of_rows h (row := 844) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_11
  simp only [← c52665, ← c52666, k2, ← c52667] at e_844_11
  have hr := e_844_11
  simp only [e_844_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6251)) := by
  have c52668 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52669 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52670 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52671 := (publicBatchWrapper89_copies1645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52672 := (publicBatchWrapper89_copies1646 a h).1
  have c52673 := (publicBatchWrapper89_copies1646 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_12 := arithEq_of_rows h (row := 844) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_12
  simp only [← c52668, ← c52669, ← c52670] at e_844_12
  have e_844_13 := arithEq_of_rows h (row := 844) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_13
  simp only [← c52671, ← c52672, k2, ← c52673] at e_844_13
  have hr := e_844_13
  simp only [e_844_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6252)) := by
  have c52674 := (publicBatchWrapper89_copies1646 a h).2.2.1
  have c52675 := (publicBatchWrapper89_copies1646 a h).2.2.2.1
  have c52676 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.1
  have c52677 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.1
  have c52678 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.1
  have c52679 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_14 := arithEq_of_rows h (row := 844) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_14
  simp only [← c52674, ← c52675, ← c52676] at e_844_14
  have e_844_15 := arithEq_of_rows h (row := 844) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_15
  simp only [← c52677, ← c52678, k2, ← c52679] at e_844_15
  have hr := e_844_15
  simp only [e_844_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6253)) := by
  have c52680 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.1
  have c52681 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.1
  have c52682 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52683 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52684 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52685 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_16 := arithEq_of_rows h (row := 844) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_16
  simp only [← c52680, ← c52681, ← c52682] at e_844_16
  have e_844_17 := arithEq_of_rows h (row := 844) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_17
  simp only [← c52683, ← c52684, k2, ← c52685] at e_844_17
  have hr := e_844_17
  simp only [e_844_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 844 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6254)) := by
  have c52686 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52687 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52688 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52689 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52690 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52691 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_844_18 := arithEq_of_rows h (row := 844) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_18
  simp only [← c52686, ← c52687, ← c52688] at e_844_18
  have e_844_19 := arithEq_of_rows h (row := 844) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_844_19
  simp only [← c52689, ← c52690, k2, ← c52691] at e_844_19
  have hr := e_844_19
  simp only [e_844_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6255)) := by
  have c52692 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52693 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52694 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52695 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52696 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52697 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_0 := arithEq_of_rows h (row := 845) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_0
  simp only [← c52692, ← c52693, ← c52694] at e_845_0
  have e_845_1 := arithEq_of_rows h (row := 845) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_1
  simp only [← c52695, ← c52696, k2, ← c52697] at e_845_1
  have hr := e_845_1
  simp only [e_845_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6256)) := by
  have c52698 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52699 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52700 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52701 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52702 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52703 := (publicBatchWrapper89_copies1646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_2 := arithEq_of_rows h (row := 845) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_2
  simp only [← c52698, ← c52699, ← c52700] at e_845_2
  have e_845_3 := arithEq_of_rows h (row := 845) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_3
  simp only [← c52701, ← c52702, k2, ← c52703] at e_845_3
  have hr := e_845_3
  simp only [e_845_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6257)) := by
  have c52704 := (publicBatchWrapper89_copies1647 a h).1
  have c52705 := (publicBatchWrapper89_copies1647 a h).2.1
  have c52706 := (publicBatchWrapper89_copies1647 a h).2.2.1
  have c52707 := (publicBatchWrapper89_copies1647 a h).2.2.2.1
  have c52708 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.1
  have c52709 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_4 := arithEq_of_rows h (row := 845) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_4
  simp only [← c52704, ← c52705, ← c52706] at e_845_4
  have e_845_5 := arithEq_of_rows h (row := 845) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_5
  simp only [← c52707, ← c52708, k2, ← c52709] at e_845_5
  have hr := e_845_5
  simp only [e_845_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6258)) := by
  have c52710 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.1
  have c52711 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.1
  have c52712 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.1
  have c52713 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.1
  have c52714 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52715 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_6 := arithEq_of_rows h (row := 845) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_6
  simp only [← c52710, ← c52711, ← c52712] at e_845_6
  have e_845_7 := arithEq_of_rows h (row := 845) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_7
  simp only [← c52713, ← c52714, k2, ← c52715] at e_845_7
  have hr := e_845_7
  simp only [e_845_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6259)) := by
  have c52716 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52717 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52718 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52719 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52720 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52721 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_8 := arithEq_of_rows h (row := 845) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_8
  simp only [← c52716, ← c52717, ← c52718] at e_845_8
  have e_845_9 := arithEq_of_rows h (row := 845) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_9
  simp only [← c52719, ← c52720, k2, ← c52721] at e_845_9
  have hr := e_845_9
  simp only [e_845_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6260)) := by
  have c52722 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52723 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52724 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52725 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52726 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52727 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_10 := arithEq_of_rows h (row := 845) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_10
  simp only [← c52722, ← c52723, ← c52724] at e_845_10
  have e_845_11 := arithEq_of_rows h (row := 845) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_11
  simp only [← c52725, ← c52726, k2, ← c52727] at e_845_11
  have hr := e_845_11
  simp only [e_845_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6261)) := by
  have c52728 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52729 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52730 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52731 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52732 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52733 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_12 := arithEq_of_rows h (row := 845) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_12
  simp only [← c52728, ← c52729, ← c52730] at e_845_12
  have e_845_13 := arithEq_of_rows h (row := 845) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_13
  simp only [← c52731, ← c52732, k2, ← c52733] at e_845_13
  have hr := e_845_13
  simp only [e_845_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6262)) := by
  have c52734 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52735 := (publicBatchWrapper89_copies1647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52736 := (publicBatchWrapper89_copies1648 a h).1
  have c52737 := (publicBatchWrapper89_copies1648 a h).2.1
  have c52738 := (publicBatchWrapper89_copies1648 a h).2.2.1
  have c52739 := (publicBatchWrapper89_copies1648 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_14 := arithEq_of_rows h (row := 845) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_14
  simp only [← c52734, ← c52735, ← c52736] at e_845_14
  have e_845_15 := arithEq_of_rows h (row := 845) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_15
  simp only [← c52737, ← c52738, k2, ← c52739] at e_845_15
  have hr := e_845_15
  simp only [e_845_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6263)) := by
  have c52740 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.1
  have c52741 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.1
  have c52742 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.1
  have c52743 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.1
  have c52744 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.1
  have c52745 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_16 := arithEq_of_rows h (row := 845) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_16
  simp only [← c52740, ← c52741, ← c52742] at e_845_16
  have e_845_17 := arithEq_of_rows h (row := 845) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_17
  simp only [← c52743, ← c52744, k2, ← c52745] at e_845_17
  have hr := e_845_17
  simp only [e_845_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 845 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6264)) := by
  have c52746 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52747 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52748 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52749 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52750 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52751 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_845_18 := arithEq_of_rows h (row := 845) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_18
  simp only [← c52746, ← c52747, ← c52748] at e_845_18
  have e_845_19 := arithEq_of_rows h (row := 845) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_845_19
  simp only [← c52749, ← c52750, k2, ← c52751] at e_845_19
  have hr := e_845_19
  simp only [e_845_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6265)) := by
  have c52752 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52753 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52754 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52755 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52756 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52757 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_0 := arithEq_of_rows h (row := 846) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_0
  simp only [← c52752, ← c52753, ← c52754] at e_846_0
  have e_846_1 := arithEq_of_rows h (row := 846) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_1
  simp only [← c52755, ← c52756, k2, ← c52757] at e_846_1
  have hr := e_846_1
  simp only [e_846_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6266)) := by
  have c52758 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52759 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52760 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52761 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52762 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52763 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_2 := arithEq_of_rows h (row := 846) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_2
  simp only [← c52758, ← c52759, ← c52760] at e_846_2
  have e_846_3 := arithEq_of_rows h (row := 846) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_3
  simp only [← c52761, ← c52762, k2, ← c52763] at e_846_3
  have hr := e_846_3
  simp only [e_846_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6267)) := by
  have c52764 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52765 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52766 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52767 := (publicBatchWrapper89_copies1648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52768 := (publicBatchWrapper89_copies1649 a h).1
  have c52769 := (publicBatchWrapper89_copies1649 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_4 := arithEq_of_rows h (row := 846) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_4
  simp only [← c52764, ← c52765, ← c52766] at e_846_4
  have e_846_5 := arithEq_of_rows h (row := 846) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_5
  simp only [← c52767, ← c52768, k2, ← c52769] at e_846_5
  have hr := e_846_5
  simp only [e_846_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6268)) := by
  have c52770 := (publicBatchWrapper89_copies1649 a h).2.2.1
  have c52771 := (publicBatchWrapper89_copies1649 a h).2.2.2.1
  have c52772 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.1
  have c52773 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.1
  have c52774 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.1
  have c52775 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_6 := arithEq_of_rows h (row := 846) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_6
  simp only [← c52770, ← c52771, ← c52772] at e_846_6
  have e_846_7 := arithEq_of_rows h (row := 846) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_7
  simp only [← c52773, ← c52774, k2, ← c52775] at e_846_7
  have hr := e_846_7
  simp only [e_846_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6269)) := by
  have c52776 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.1
  have c52777 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.1
  have c52778 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52779 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52780 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52781 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_8 := arithEq_of_rows h (row := 846) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_8
  simp only [← c52776, ← c52777, ← c52778] at e_846_8
  have e_846_9 := arithEq_of_rows h (row := 846) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_9
  simp only [← c52779, ← c52780, k2, ← c52781] at e_846_9
  have hr := e_846_9
  simp only [e_846_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6270)) := by
  have c52782 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52783 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52784 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52785 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52786 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52787 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_10 := arithEq_of_rows h (row := 846) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_10
  simp only [← c52782, ← c52783, ← c52784] at e_846_10
  have e_846_11 := arithEq_of_rows h (row := 846) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_11
  simp only [← c52785, ← c52786, k2, ← c52787] at e_846_11
  have hr := e_846_11
  simp only [e_846_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6271)) := by
  have c52788 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52789 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52790 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52791 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52792 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52793 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_12 := arithEq_of_rows h (row := 846) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_12
  simp only [← c52788, ← c52789, ← c52790] at e_846_12
  have e_846_13 := arithEq_of_rows h (row := 846) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_13
  simp only [← c52791, ← c52792, k2, ← c52793] at e_846_13
  have hr := e_846_13
  simp only [e_846_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6272)) := by
  have c52794 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52795 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52796 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52797 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52798 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52799 := (publicBatchWrapper89_copies1649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_14 := arithEq_of_rows h (row := 846) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_14
  simp only [← c52794, ← c52795, ← c52796] at e_846_14
  have e_846_15 := arithEq_of_rows h (row := 846) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_15
  simp only [← c52797, ← c52798, k2, ← c52799] at e_846_15
  have hr := e_846_15
  simp only [e_846_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6273)) := by
  have c52800 := (publicBatchWrapper89_copies1650 a h).1
  have c52801 := (publicBatchWrapper89_copies1650 a h).2.1
  have c52802 := (publicBatchWrapper89_copies1650 a h).2.2.1
  have c52803 := (publicBatchWrapper89_copies1650 a h).2.2.2.1
  have c52804 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.1
  have c52805 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_16 := arithEq_of_rows h (row := 846) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_16
  simp only [← c52800, ← c52801, ← c52802] at e_846_16
  have e_846_17 := arithEq_of_rows h (row := 846) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_17
  simp only [← c52803, ← c52804, k2, ← c52805] at e_846_17
  have hr := e_846_17
  simp only [e_846_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 846 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6274)) := by
  have c52806 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.1
  have c52807 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.1
  have c52808 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.1
  have c52809 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.1
  have c52810 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52811 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_846_18 := arithEq_of_rows h (row := 846) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_18
  simp only [← c52806, ← c52807, ← c52808] at e_846_18
  have e_846_19 := arithEq_of_rows h (row := 846) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_846_19
  simp only [← c52809, ← c52810, k2, ← c52811] at e_846_19
  have hr := e_846_19
  simp only [e_846_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6275)) := by
  have c52812 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52813 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52814 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52815 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52816 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52817 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_0 := arithEq_of_rows h (row := 847) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_0
  simp only [← c52812, ← c52813, ← c52814] at e_847_0
  have e_847_1 := arithEq_of_rows h (row := 847) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_1
  simp only [← c52815, ← c52816, k2, ← c52817] at e_847_1
  have hr := e_847_1
  simp only [e_847_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6276)) := by
  have c52818 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52819 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52820 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52821 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52822 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52823 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_2 := arithEq_of_rows h (row := 847) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_2
  simp only [← c52818, ← c52819, ← c52820] at e_847_2
  have e_847_3 := arithEq_of_rows h (row := 847) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_3
  simp only [← c52821, ← c52822, k2, ← c52823] at e_847_3
  have hr := e_847_3
  simp only [e_847_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6277)) := by
  have c52824 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52825 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52826 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52827 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52828 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52829 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_4 := arithEq_of_rows h (row := 847) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_4
  simp only [← c52824, ← c52825, ← c52826] at e_847_4
  have e_847_5 := arithEq_of_rows h (row := 847) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_5
  simp only [← c52827, ← c52828, k2, ← c52829] at e_847_5
  have hr := e_847_5
  simp only [e_847_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6278)) := by
  have c52830 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52831 := (publicBatchWrapper89_copies1650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52832 := (publicBatchWrapper89_copies1651 a h).1
  have c52833 := (publicBatchWrapper89_copies1651 a h).2.1
  have c52834 := (publicBatchWrapper89_copies1651 a h).2.2.1
  have c52835 := (publicBatchWrapper89_copies1651 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_6 := arithEq_of_rows h (row := 847) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_6
  simp only [← c52830, ← c52831, ← c52832] at e_847_6
  have e_847_7 := arithEq_of_rows h (row := 847) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_7
  simp only [← c52833, ← c52834, k2, ← c52835] at e_847_7
  have hr := e_847_7
  simp only [e_847_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6279)) := by
  have c52836 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.1
  have c52837 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.1
  have c52838 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.1
  have c52839 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.1
  have c52840 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.1
  have c52841 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_8 := arithEq_of_rows h (row := 847) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_8
  simp only [← c52836, ← c52837, ← c52838] at e_847_8
  have e_847_9 := arithEq_of_rows h (row := 847) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_9
  simp only [← c52839, ← c52840, k2, ← c52841] at e_847_9
  have hr := e_847_9
  simp only [e_847_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6280)) := by
  have c52842 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52843 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52844 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52845 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52846 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52847 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_10 := arithEq_of_rows h (row := 847) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_10
  simp only [← c52842, ← c52843, ← c52844] at e_847_10
  have e_847_11 := arithEq_of_rows h (row := 847) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_11
  simp only [← c52845, ← c52846, k2, ← c52847] at e_847_11
  have hr := e_847_11
  simp only [e_847_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6281)) := by
  have c52848 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52849 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52850 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52851 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52852 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52853 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_12 := arithEq_of_rows h (row := 847) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_12
  simp only [← c52848, ← c52849, ← c52850] at e_847_12
  have e_847_13 := arithEq_of_rows h (row := 847) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_13
  simp only [← c52851, ← c52852, k2, ← c52853] at e_847_13
  have hr := e_847_13
  simp only [e_847_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6282)) := by
  have c52854 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52855 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52856 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52857 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52858 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52859 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_14 := arithEq_of_rows h (row := 847) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_14
  simp only [← c52854, ← c52855, ← c52856] at e_847_14
  have e_847_15 := arithEq_of_rows h (row := 847) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_15
  simp only [← c52857, ← c52858, k2, ← c52859] at e_847_15
  have hr := e_847_15
  simp only [e_847_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6283)) := by
  have c52860 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52861 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52862 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52863 := (publicBatchWrapper89_copies1651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52864 := (publicBatchWrapper89_copies1652 a h).1
  have c52865 := (publicBatchWrapper89_copies1652 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_16 := arithEq_of_rows h (row := 847) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_16
  simp only [← c52860, ← c52861, ← c52862] at e_847_16
  have e_847_17 := arithEq_of_rows h (row := 847) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_17
  simp only [← c52863, ← c52864, k2, ← c52865] at e_847_17
  have hr := e_847_17
  simp only [e_847_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 847 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6284)) := by
  have c52866 := (publicBatchWrapper89_copies1652 a h).2.2.1
  have c52867 := (publicBatchWrapper89_copies1652 a h).2.2.2.1
  have c52868 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.1
  have c52869 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.1
  have c52870 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.1
  have c52871 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_847_18 := arithEq_of_rows h (row := 847) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_18
  simp only [← c52866, ← c52867, ← c52868] at e_847_18
  have e_847_19 := arithEq_of_rows h (row := 847) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_847_19
  simp only [← c52869, ← c52870, k2, ← c52871] at e_847_19
  have hr := e_847_19
  simp only [e_847_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6285)) := by
  have c52872 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.1
  have c52873 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.1
  have c52874 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52875 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52876 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52877 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_0 := arithEq_of_rows h (row := 848) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_0
  simp only [← c52872, ← c52873, ← c52874] at e_848_0
  have e_848_1 := arithEq_of_rows h (row := 848) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_1
  simp only [← c52875, ← c52876, k2, ← c52877] at e_848_1
  have hr := e_848_1
  simp only [e_848_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6286)) := by
  have c52878 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52879 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52880 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52881 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52882 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52883 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_2 := arithEq_of_rows h (row := 848) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_2
  simp only [← c52878, ← c52879, ← c52880] at e_848_2
  have e_848_3 := arithEq_of_rows h (row := 848) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_3
  simp only [← c52881, ← c52882, k2, ← c52883] at e_848_3
  have hr := e_848_3
  simp only [e_848_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6287)) := by
  have c52884 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52885 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52886 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52887 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52888 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52889 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_4 := arithEq_of_rows h (row := 848) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_4
  simp only [← c52884, ← c52885, ← c52886] at e_848_4
  have e_848_5 := arithEq_of_rows h (row := 848) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_5
  simp only [← c52887, ← c52888, k2, ← c52889] at e_848_5
  have hr := e_848_5
  simp only [e_848_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6288)) := by
  have c52890 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52891 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52892 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52893 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52894 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52895 := (publicBatchWrapper89_copies1652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_6 := arithEq_of_rows h (row := 848) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_6
  simp only [← c52890, ← c52891, ← c52892] at e_848_6
  have e_848_7 := arithEq_of_rows h (row := 848) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_7
  simp only [← c52893, ← c52894, k2, ← c52895] at e_848_7
  have hr := e_848_7
  simp only [e_848_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6289)) := by
  have c52896 := (publicBatchWrapper89_copies1653 a h).1
  have c52897 := (publicBatchWrapper89_copies1653 a h).2.1
  have c52898 := (publicBatchWrapper89_copies1653 a h).2.2.1
  have c52899 := (publicBatchWrapper89_copies1653 a h).2.2.2.1
  have c52900 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.1
  have c52901 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_8 := arithEq_of_rows h (row := 848) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_8
  simp only [← c52896, ← c52897, ← c52898] at e_848_8
  have e_848_9 := arithEq_of_rows h (row := 848) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_9
  simp only [← c52899, ← c52900, k2, ← c52901] at e_848_9
  have hr := e_848_9
  simp only [e_848_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6290)) := by
  have c52902 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.1
  have c52903 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.1
  have c52904 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.1
  have c52905 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.1
  have c52906 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52907 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_10 := arithEq_of_rows h (row := 848) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_10
  simp only [← c52902, ← c52903, ← c52904] at e_848_10
  have e_848_11 := arithEq_of_rows h (row := 848) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_11
  simp only [← c52905, ← c52906, k2, ← c52907] at e_848_11
  have hr := e_848_11
  simp only [e_848_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6291)) := by
  have c52908 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52909 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52910 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52911 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52912 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52913 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_12 := arithEq_of_rows h (row := 848) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_12
  simp only [← c52908, ← c52909, ← c52910] at e_848_12
  have e_848_13 := arithEq_of_rows h (row := 848) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_13
  simp only [← c52911, ← c52912, k2, ← c52913] at e_848_13
  have hr := e_848_13
  simp only [e_848_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6292)) := by
  have c52914 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52915 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52916 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52917 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52918 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52919 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_14 := arithEq_of_rows h (row := 848) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_14
  simp only [← c52914, ← c52915, ← c52916] at e_848_14
  have e_848_15 := arithEq_of_rows h (row := 848) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_15
  simp only [← c52917, ← c52918, k2, ← c52919] at e_848_15
  have hr := e_848_15
  simp only [e_848_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6293)) := by
  have c52920 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52921 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52922 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52923 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52924 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52925 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_16 := arithEq_of_rows h (row := 848) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_16
  simp only [← c52920, ← c52921, ← c52922] at e_848_16
  have e_848_17 := arithEq_of_rows h (row := 848) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_17
  simp only [← c52923, ← c52924, k2, ← c52925] at e_848_17
  have hr := e_848_17
  simp only [e_848_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 848 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6294)) := by
  have c52926 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52927 := (publicBatchWrapper89_copies1653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52928 := (publicBatchWrapper89_copies1654 a h).1
  have c52929 := (publicBatchWrapper89_copies1654 a h).2.1
  have c52930 := (publicBatchWrapper89_copies1654 a h).2.2.1
  have c52931 := (publicBatchWrapper89_copies1654 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_848_18 := arithEq_of_rows h (row := 848) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_18
  simp only [← c52926, ← c52927, ← c52928] at e_848_18
  have e_848_19 := arithEq_of_rows h (row := 848) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_848_19
  simp only [← c52929, ← c52930, k2, ← c52931] at e_848_19
  have hr := e_848_19
  simp only [e_848_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6295)) := by
  have c52932 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.1
  have c52933 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.1
  have c52934 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.1
  have c52935 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.1
  have c52936 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.1
  have c52937 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_0 := arithEq_of_rows h (row := 849) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_0
  simp only [← c52932, ← c52933, ← c52934] at e_849_0
  have e_849_1 := arithEq_of_rows h (row := 849) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_1
  simp only [← c52935, ← c52936, k2, ← c52937] at e_849_1
  have hr := e_849_1
  simp only [e_849_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6296)) := by
  have c52938 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52939 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52940 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52941 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52942 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52943 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_2 := arithEq_of_rows h (row := 849) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_2
  simp only [← c52938, ← c52939, ← c52940] at e_849_2
  have e_849_3 := arithEq_of_rows h (row := 849) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_3
  simp only [← c52941, ← c52942, k2, ← c52943] at e_849_3
  have hr := e_849_3
  simp only [e_849_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6297)) := by
  have c52944 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52945 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52946 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52947 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52948 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52949 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_4 := arithEq_of_rows h (row := 849) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_4
  simp only [← c52944, ← c52945, ← c52946] at e_849_4
  have e_849_5 := arithEq_of_rows h (row := 849) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_5
  simp only [← c52947, ← c52948, k2, ← c52949] at e_849_5
  have hr := e_849_5
  simp only [e_849_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6298)) := by
  have c52950 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52951 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52952 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52953 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52954 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52955 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_6 := arithEq_of_rows h (row := 849) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_6
  simp only [← c52950, ← c52951, ← c52952] at e_849_6
  have e_849_7 := arithEq_of_rows h (row := 849) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_7
  simp only [← c52953, ← c52954, k2, ← c52955] at e_849_7
  have hr := e_849_7
  simp only [e_849_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6299)) := by
  have c52956 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52957 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52958 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52959 := (publicBatchWrapper89_copies1654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c52960 := (publicBatchWrapper89_copies1655 a h).1
  have c52961 := (publicBatchWrapper89_copies1655 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_8 := arithEq_of_rows h (row := 849) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_8
  simp only [← c52956, ← c52957, ← c52958] at e_849_8
  have e_849_9 := arithEq_of_rows h (row := 849) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_9
  simp only [← c52959, ← c52960, k2, ← c52961] at e_849_9
  have hr := e_849_9
  simp only [e_849_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6300)) := by
  have c52962 := (publicBatchWrapper89_copies1655 a h).2.2.1
  have c52963 := (publicBatchWrapper89_copies1655 a h).2.2.2.1
  have c52964 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.1
  have c52965 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.1
  have c52966 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.1
  have c52967 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_10 := arithEq_of_rows h (row := 849) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_10
  simp only [← c52962, ← c52963, ← c52964] at e_849_10
  have e_849_11 := arithEq_of_rows h (row := 849) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_11
  simp only [← c52965, ← c52966, k2, ← c52967] at e_849_11
  have hr := e_849_11
  simp only [e_849_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6301)) := by
  have c52968 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.1
  have c52969 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.1
  have c52970 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c52971 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c52972 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52973 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_12 := arithEq_of_rows h (row := 849) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_12
  simp only [← c52968, ← c52969, ← c52970] at e_849_12
  have e_849_13 := arithEq_of_rows h (row := 849) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_13
  simp only [← c52971, ← c52972, k2, ← c52973] at e_849_13
  have hr := e_849_13
  simp only [e_849_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6302)) := by
  have c52974 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52975 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52976 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52977 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52978 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52979 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_14 := arithEq_of_rows h (row := 849) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_14
  simp only [← c52974, ← c52975, ← c52976] at e_849_14
  have e_849_15 := arithEq_of_rows h (row := 849) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_15
  simp only [← c52977, ← c52978, k2, ← c52979] at e_849_15
  have hr := e_849_15
  simp only [e_849_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6303)) := by
  have c52980 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52981 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52982 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52983 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52984 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52985 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_16 := arithEq_of_rows h (row := 849) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_16
  simp only [← c52980, ← c52981, ← c52982] at e_849_16
  have e_849_17 := arithEq_of_rows h (row := 849) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_17
  simp only [← c52983, ← c52984, k2, ← c52985] at e_849_17
  have hr := e_849_17
  simp only [e_849_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 849 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6304)) := by
  have c52986 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52987 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52988 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52989 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52990 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52991 := (publicBatchWrapper89_copies1655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_849_18 := arithEq_of_rows h (row := 849) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_18
  simp only [← c52986, ← c52987, ← c52988] at e_849_18
  have e_849_19 := arithEq_of_rows h (row := 849) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_849_19
  simp only [← c52989, ← c52990, k2, ← c52991] at e_849_19
  have hr := e_849_19
  simp only [e_849_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6305)) := by
  have c52992 := (publicBatchWrapper89_copies1656 a h).1
  have c52993 := (publicBatchWrapper89_copies1656 a h).2.1
  have c52994 := (publicBatchWrapper89_copies1656 a h).2.2.1
  have c52995 := (publicBatchWrapper89_copies1656 a h).2.2.2.1
  have c52996 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.1
  have c52997 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_0 := arithEq_of_rows h (row := 850) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_0
  simp only [← c52992, ← c52993, ← c52994] at e_850_0
  have e_850_1 := arithEq_of_rows h (row := 850) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_1
  simp only [← c52995, ← c52996, k2, ← c52997] at e_850_1
  have hr := e_850_1
  simp only [e_850_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6306)) := by
  have c52998 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.1
  have c52999 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.1
  have c53000 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.1
  have c53001 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.1
  have c53002 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53003 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_2 := arithEq_of_rows h (row := 850) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_2
  simp only [← c52998, ← c52999, ← c53000] at e_850_2
  have e_850_3 := arithEq_of_rows h (row := 850) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_3
  simp only [← c53001, ← c53002, k2, ← c53003] at e_850_3
  have hr := e_850_3
  simp only [e_850_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6307)) := by
  have c53004 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53005 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53006 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53007 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53008 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53009 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_4 := arithEq_of_rows h (row := 850) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_4
  simp only [← c53004, ← c53005, ← c53006] at e_850_4
  have e_850_5 := arithEq_of_rows h (row := 850) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_5
  simp only [← c53007, ← c53008, k2, ← c53009] at e_850_5
  have hr := e_850_5
  simp only [e_850_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6308)) := by
  have c53010 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53011 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53012 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53013 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53014 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53015 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_6 := arithEq_of_rows h (row := 850) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_6
  simp only [← c53010, ← c53011, ← c53012] at e_850_6
  have e_850_7 := arithEq_of_rows h (row := 850) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_7
  simp only [← c53013, ← c53014, k2, ← c53015] at e_850_7
  have hr := e_850_7
  simp only [e_850_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6309)) := by
  have c53016 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53017 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53018 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53019 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53020 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53021 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_8 := arithEq_of_rows h (row := 850) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_8
  simp only [← c53016, ← c53017, ← c53018] at e_850_8
  have e_850_9 := arithEq_of_rows h (row := 850) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_9
  simp only [← c53019, ← c53020, k2, ← c53021] at e_850_9
  have hr := e_850_9
  simp only [e_850_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6310)) := by
  have c53022 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53023 := (publicBatchWrapper89_copies1656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53024 := (publicBatchWrapper89_copies1657 a h).1
  have c53025 := (publicBatchWrapper89_copies1657 a h).2.1
  have c53026 := (publicBatchWrapper89_copies1657 a h).2.2.1
  have c53027 := (publicBatchWrapper89_copies1657 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_10 := arithEq_of_rows h (row := 850) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_10
  simp only [← c53022, ← c53023, ← c53024] at e_850_10
  have e_850_11 := arithEq_of_rows h (row := 850) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_11
  simp only [← c53025, ← c53026, k2, ← c53027] at e_850_11
  have hr := e_850_11
  simp only [e_850_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6311)) := by
  have c53028 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.1
  have c53029 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.1
  have c53030 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.1
  have c53031 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.1
  have c53032 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.1
  have c53033 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_12 := arithEq_of_rows h (row := 850) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_12
  simp only [← c53028, ← c53029, ← c53030] at e_850_12
  have e_850_13 := arithEq_of_rows h (row := 850) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_13
  simp only [← c53031, ← c53032, k2, ← c53033] at e_850_13
  have hr := e_850_13
  simp only [e_850_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6312)) := by
  have c53034 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53035 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53036 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53037 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53038 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53039 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_14 := arithEq_of_rows h (row := 850) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_14
  simp only [← c53034, ← c53035, ← c53036] at e_850_14
  have e_850_15 := arithEq_of_rows h (row := 850) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_15
  simp only [← c53037, ← c53038, k2, ← c53039] at e_850_15
  have hr := e_850_15
  simp only [e_850_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6313)) := by
  have c53040 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53041 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53042 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53043 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53044 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53045 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_16 := arithEq_of_rows h (row := 850) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_16
  simp only [← c53040, ← c53041, ← c53042] at e_850_16
  have e_850_17 := arithEq_of_rows h (row := 850) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_17
  simp only [← c53043, ← c53044, k2, ← c53045] at e_850_17
  have hr := e_850_17
  simp only [e_850_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 850 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6314)) := by
  have c53046 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53047 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53048 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53049 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53050 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53051 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_850_18 := arithEq_of_rows h (row := 850) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_18
  simp only [← c53046, ← c53047, ← c53048] at e_850_18
  have e_850_19 := arithEq_of_rows h (row := 850) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_850_19
  simp only [← c53049, ← c53050, k2, ← c53051] at e_850_19
  have hr := e_850_19
  simp only [e_850_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6315)) := by
  have c53052 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53053 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53054 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53055 := (publicBatchWrapper89_copies1657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53056 := (publicBatchWrapper89_copies1658 a h).1
  have c53057 := (publicBatchWrapper89_copies1658 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_0 := arithEq_of_rows h (row := 851) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_0
  simp only [← c53052, ← c53053, ← c53054] at e_851_0
  have e_851_1 := arithEq_of_rows h (row := 851) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_1
  simp only [← c53055, ← c53056, k2, ← c53057] at e_851_1
  have hr := e_851_1
  simp only [e_851_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6316)) := by
  have c53058 := (publicBatchWrapper89_copies1658 a h).2.2.1
  have c53059 := (publicBatchWrapper89_copies1658 a h).2.2.2.1
  have c53060 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.1
  have c53061 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.1
  have c53062 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.1
  have c53063 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_2 := arithEq_of_rows h (row := 851) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_2
  simp only [← c53058, ← c53059, ← c53060] at e_851_2
  have e_851_3 := arithEq_of_rows h (row := 851) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_3
  simp only [← c53061, ← c53062, k2, ← c53063] at e_851_3
  have hr := e_851_3
  simp only [e_851_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6317)) := by
  have c53064 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.1
  have c53065 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.1
  have c53066 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53067 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53068 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53069 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_4 := arithEq_of_rows h (row := 851) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_4
  simp only [← c53064, ← c53065, ← c53066] at e_851_4
  have e_851_5 := arithEq_of_rows h (row := 851) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_5
  simp only [← c53067, ← c53068, k2, ← c53069] at e_851_5
  have hr := e_851_5
  simp only [e_851_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6318)) := by
  have c53070 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53071 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53072 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53073 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53074 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53075 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_6 := arithEq_of_rows h (row := 851) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_6
  simp only [← c53070, ← c53071, ← c53072] at e_851_6
  have e_851_7 := arithEq_of_rows h (row := 851) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_7
  simp only [← c53073, ← c53074, k2, ← c53075] at e_851_7
  have hr := e_851_7
  simp only [e_851_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6319)) := by
  have c53076 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53077 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53078 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53079 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53080 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53081 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_8 := arithEq_of_rows h (row := 851) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_8
  simp only [← c53076, ← c53077, ← c53078] at e_851_8
  have e_851_9 := arithEq_of_rows h (row := 851) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_9
  simp only [← c53079, ← c53080, k2, ← c53081] at e_851_9
  have hr := e_851_9
  simp only [e_851_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6320)) := by
  have c53082 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53083 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53084 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53085 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53086 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53087 := (publicBatchWrapper89_copies1658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_10 := arithEq_of_rows h (row := 851) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_10
  simp only [← c53082, ← c53083, ← c53084] at e_851_10
  have e_851_11 := arithEq_of_rows h (row := 851) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_11
  simp only [← c53085, ← c53086, k2, ← c53087] at e_851_11
  have hr := e_851_11
  simp only [e_851_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6321)) := by
  have c53088 := (publicBatchWrapper89_copies1659 a h).1
  have c53089 := (publicBatchWrapper89_copies1659 a h).2.1
  have c53090 := (publicBatchWrapper89_copies1659 a h).2.2.1
  have c53091 := (publicBatchWrapper89_copies1659 a h).2.2.2.1
  have c53092 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.1
  have c53093 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_12 := arithEq_of_rows h (row := 851) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_12
  simp only [← c53088, ← c53089, ← c53090] at e_851_12
  have e_851_13 := arithEq_of_rows h (row := 851) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_13
  simp only [← c53091, ← c53092, k2, ← c53093] at e_851_13
  have hr := e_851_13
  simp only [e_851_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6322)) := by
  have c53094 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.1
  have c53095 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.1
  have c53096 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.1
  have c53097 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.1
  have c53098 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53099 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_14 := arithEq_of_rows h (row := 851) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_14
  simp only [← c53094, ← c53095, ← c53096] at e_851_14
  have e_851_15 := arithEq_of_rows h (row := 851) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_15
  simp only [← c53097, ← c53098, k2, ← c53099] at e_851_15
  have hr := e_851_15
  simp only [e_851_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6323)) := by
  have c53100 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53101 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53102 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53103 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53104 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53105 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_16 := arithEq_of_rows h (row := 851) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_16
  simp only [← c53100, ← c53101, ← c53102] at e_851_16
  have e_851_17 := arithEq_of_rows h (row := 851) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_17
  simp only [← c53103, ← c53104, k2, ← c53105] at e_851_17
  have hr := e_851_17
  simp only [e_851_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 851 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6324)) := by
  have c53106 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53107 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53108 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53109 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53110 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53111 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_851_18 := arithEq_of_rows h (row := 851) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_18
  simp only [← c53106, ← c53107, ← c53108] at e_851_18
  have e_851_19 := arithEq_of_rows h (row := 851) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_851_19
  simp only [← c53109, ← c53110, k2, ← c53111] at e_851_19
  have hr := e_851_19
  simp only [e_851_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6325)) := by
  have c53112 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53113 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53114 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53115 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53116 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53117 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_0 := arithEq_of_rows h (row := 852) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_0
  simp only [← c53112, ← c53113, ← c53114] at e_852_0
  have e_852_1 := arithEq_of_rows h (row := 852) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_1
  simp only [← c53115, ← c53116, k2, ← c53117] at e_852_1
  have hr := e_852_1
  simp only [e_852_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6326)) := by
  have c53118 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53119 := (publicBatchWrapper89_copies1659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53120 := (publicBatchWrapper89_copies1660 a h).1
  have c53121 := (publicBatchWrapper89_copies1660 a h).2.1
  have c53122 := (publicBatchWrapper89_copies1660 a h).2.2.1
  have c53123 := (publicBatchWrapper89_copies1660 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_2 := arithEq_of_rows h (row := 852) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_2
  simp only [← c53118, ← c53119, ← c53120] at e_852_2
  have e_852_3 := arithEq_of_rows h (row := 852) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_3
  simp only [← c53121, ← c53122, k2, ← c53123] at e_852_3
  have hr := e_852_3
  simp only [e_852_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6327)) := by
  have c53124 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.1
  have c53125 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.1
  have c53126 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.1
  have c53127 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.1
  have c53128 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.1
  have c53129 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_4 := arithEq_of_rows h (row := 852) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_4
  simp only [← c53124, ← c53125, ← c53126] at e_852_4
  have e_852_5 := arithEq_of_rows h (row := 852) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_5
  simp only [← c53127, ← c53128, k2, ← c53129] at e_852_5
  have hr := e_852_5
  simp only [e_852_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6328)) := by
  have c53130 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53131 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53132 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53133 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53134 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53135 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_6 := arithEq_of_rows h (row := 852) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_6
  simp only [← c53130, ← c53131, ← c53132] at e_852_6
  have e_852_7 := arithEq_of_rows h (row := 852) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_7
  simp only [← c53133, ← c53134, k2, ← c53135] at e_852_7
  have hr := e_852_7
  simp only [e_852_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6329)) := by
  have c53136 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53137 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53138 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53139 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53140 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53141 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_8 := arithEq_of_rows h (row := 852) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_8
  simp only [← c53136, ← c53137, ← c53138] at e_852_8
  have e_852_9 := arithEq_of_rows h (row := 852) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_9
  simp only [← c53139, ← c53140, k2, ← c53141] at e_852_9
  have hr := e_852_9
  simp only [e_852_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6330)) := by
  have c53142 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53143 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53144 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53145 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53146 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53147 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_10 := arithEq_of_rows h (row := 852) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_10
  simp only [← c53142, ← c53143, ← c53144] at e_852_10
  have e_852_11 := arithEq_of_rows h (row := 852) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_11
  simp only [← c53145, ← c53146, k2, ← c53147] at e_852_11
  have hr := e_852_11
  simp only [e_852_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6331)) := by
  have c53148 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53149 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53150 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53151 := (publicBatchWrapper89_copies1660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53152 := (publicBatchWrapper89_copies1661 a h).1
  have c53153 := (publicBatchWrapper89_copies1661 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_12 := arithEq_of_rows h (row := 852) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_12
  simp only [← c53148, ← c53149, ← c53150] at e_852_12
  have e_852_13 := arithEq_of_rows h (row := 852) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_13
  simp only [← c53151, ← c53152, k2, ← c53153] at e_852_13
  have hr := e_852_13
  simp only [e_852_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6332)) := by
  have c53154 := (publicBatchWrapper89_copies1661 a h).2.2.1
  have c53155 := (publicBatchWrapper89_copies1661 a h).2.2.2.1
  have c53156 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.1
  have c53157 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.1
  have c53158 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.1
  have c53159 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_14 := arithEq_of_rows h (row := 852) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_14
  simp only [← c53154, ← c53155, ← c53156] at e_852_14
  have e_852_15 := arithEq_of_rows h (row := 852) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_15
  simp only [← c53157, ← c53158, k2, ← c53159] at e_852_15
  have hr := e_852_15
  simp only [e_852_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6333)) := by
  have c53160 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.1
  have c53161 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.1
  have c53162 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53163 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53164 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53165 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_16 := arithEq_of_rows h (row := 852) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_16
  simp only [← c53160, ← c53161, ← c53162] at e_852_16
  have e_852_17 := arithEq_of_rows h (row := 852) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_17
  simp only [← c53163, ← c53164, k2, ← c53165] at e_852_17
  have hr := e_852_17
  simp only [e_852_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 852 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6334)) := by
  have c53166 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53167 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53168 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53169 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53170 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53171 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_852_18 := arithEq_of_rows h (row := 852) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_18
  simp only [← c53166, ← c53167, ← c53168] at e_852_18
  have e_852_19 := arithEq_of_rows h (row := 852) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_852_19
  simp only [← c53169, ← c53170, k2, ← c53171] at e_852_19
  have hr := e_852_19
  simp only [e_852_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6335)) := by
  have c53172 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53173 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53174 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53175 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53176 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53177 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_0 := arithEq_of_rows h (row := 853) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_0
  simp only [← c53172, ← c53173, ← c53174] at e_853_0
  have e_853_1 := arithEq_of_rows h (row := 853) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_1
  simp only [← c53175, ← c53176, k2, ← c53177] at e_853_1
  have hr := e_853_1
  simp only [e_853_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6336)) := by
  have c53178 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53179 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53180 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53181 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53182 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53183 := (publicBatchWrapper89_copies1661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_2 := arithEq_of_rows h (row := 853) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_2
  simp only [← c53178, ← c53179, ← c53180] at e_853_2
  have e_853_3 := arithEq_of_rows h (row := 853) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_3
  simp only [← c53181, ← c53182, k2, ← c53183] at e_853_3
  have hr := e_853_3
  simp only [e_853_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6337)) := by
  have c53184 := (publicBatchWrapper89_copies1662 a h).1
  have c53185 := (publicBatchWrapper89_copies1662 a h).2.1
  have c53186 := (publicBatchWrapper89_copies1662 a h).2.2.1
  have c53187 := (publicBatchWrapper89_copies1662 a h).2.2.2.1
  have c53188 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.1
  have c53189 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_4 := arithEq_of_rows h (row := 853) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_4
  simp only [← c53184, ← c53185, ← c53186] at e_853_4
  have e_853_5 := arithEq_of_rows h (row := 853) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_5
  simp only [← c53187, ← c53188, k2, ← c53189] at e_853_5
  have hr := e_853_5
  simp only [e_853_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6338)) := by
  have c53190 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.1
  have c53191 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.1
  have c53192 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.1
  have c53193 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.1
  have c53194 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53195 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_6 := arithEq_of_rows h (row := 853) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_6
  simp only [← c53190, ← c53191, ← c53192] at e_853_6
  have e_853_7 := arithEq_of_rows h (row := 853) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_7
  simp only [← c53193, ← c53194, k2, ← c53195] at e_853_7
  have hr := e_853_7
  simp only [e_853_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6339)) := by
  have c53196 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53197 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53198 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53199 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53200 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53201 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_8 := arithEq_of_rows h (row := 853) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_8
  simp only [← c53196, ← c53197, ← c53198] at e_853_8
  have e_853_9 := arithEq_of_rows h (row := 853) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_9
  simp only [← c53199, ← c53200, k2, ← c53201] at e_853_9
  have hr := e_853_9
  simp only [e_853_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6340)) := by
  have c53202 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53203 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53204 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53205 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53206 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53207 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_10 := arithEq_of_rows h (row := 853) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_10
  simp only [← c53202, ← c53203, ← c53204] at e_853_10
  have e_853_11 := arithEq_of_rows h (row := 853) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_11
  simp only [← c53205, ← c53206, k2, ← c53207] at e_853_11
  have hr := e_853_11
  simp only [e_853_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6341)) := by
  have c53208 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53209 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53210 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53211 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53212 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53213 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_12 := arithEq_of_rows h (row := 853) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_12
  simp only [← c53208, ← c53209, ← c53210] at e_853_12
  have e_853_13 := arithEq_of_rows h (row := 853) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_13
  simp only [← c53211, ← c53212, k2, ← c53213] at e_853_13
  have hr := e_853_13
  simp only [e_853_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6342)) := by
  have c53214 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53215 := (publicBatchWrapper89_copies1662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53216 := (publicBatchWrapper89_copies1663 a h).1
  have c53217 := (publicBatchWrapper89_copies1663 a h).2.1
  have c53218 := (publicBatchWrapper89_copies1663 a h).2.2.1
  have c53219 := (publicBatchWrapper89_copies1663 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_14 := arithEq_of_rows h (row := 853) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_14
  simp only [← c53214, ← c53215, ← c53216] at e_853_14
  have e_853_15 := arithEq_of_rows h (row := 853) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_15
  simp only [← c53217, ← c53218, k2, ← c53219] at e_853_15
  have hr := e_853_15
  simp only [e_853_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6343)) := by
  have c53220 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.1
  have c53221 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.1
  have c53222 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.1
  have c53223 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.1
  have c53224 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.1
  have c53225 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_16 := arithEq_of_rows h (row := 853) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_16
  simp only [← c53220, ← c53221, ← c53222] at e_853_16
  have e_853_17 := arithEq_of_rows h (row := 853) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_17
  simp only [← c53223, ← c53224, k2, ← c53225] at e_853_17
  have hr := e_853_17
  simp only [e_853_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 853 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6344)) := by
  have c53226 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53227 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53228 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53229 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53230 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53231 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_853_18 := arithEq_of_rows h (row := 853) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_18
  simp only [← c53226, ← c53227, ← c53228] at e_853_18
  have e_853_19 := arithEq_of_rows h (row := 853) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_853_19
  simp only [← c53229, ← c53230, k2, ← c53231] at e_853_19
  have hr := e_853_19
  simp only [e_853_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6345)) := by
  have c53232 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53233 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53234 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53235 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53236 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53237 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_0 := arithEq_of_rows h (row := 854) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_0
  simp only [← c53232, ← c53233, ← c53234] at e_854_0
  have e_854_1 := arithEq_of_rows h (row := 854) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_1
  simp only [← c53235, ← c53236, k2, ← c53237] at e_854_1
  have hr := e_854_1
  simp only [e_854_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6346)) := by
  have c53238 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53239 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53240 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53241 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53242 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53243 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_2 := arithEq_of_rows h (row := 854) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_2
  simp only [← c53238, ← c53239, ← c53240] at e_854_2
  have e_854_3 := arithEq_of_rows h (row := 854) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_3
  simp only [← c53241, ← c53242, k2, ← c53243] at e_854_3
  have hr := e_854_3
  simp only [e_854_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6347)) := by
  have c53244 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53245 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53246 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53247 := (publicBatchWrapper89_copies1663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53248 := (publicBatchWrapper89_copies1664 a h).1
  have c53249 := (publicBatchWrapper89_copies1664 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_4 := arithEq_of_rows h (row := 854) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_4
  simp only [← c53244, ← c53245, ← c53246] at e_854_4
  have e_854_5 := arithEq_of_rows h (row := 854) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_5
  simp only [← c53247, ← c53248, k2, ← c53249] at e_854_5
  have hr := e_854_5
  simp only [e_854_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6348)) := by
  have c53250 := (publicBatchWrapper89_copies1664 a h).2.2.1
  have c53251 := (publicBatchWrapper89_copies1664 a h).2.2.2.1
  have c53252 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.1
  have c53253 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.1
  have c53254 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.1
  have c53255 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_6 := arithEq_of_rows h (row := 854) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_6
  simp only [← c53250, ← c53251, ← c53252] at e_854_6
  have e_854_7 := arithEq_of_rows h (row := 854) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_7
  simp only [← c53253, ← c53254, k2, ← c53255] at e_854_7
  have hr := e_854_7
  simp only [e_854_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6349)) := by
  have c53256 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.1
  have c53257 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.1
  have c53258 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53259 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53260 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53261 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_8 := arithEq_of_rows h (row := 854) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_8
  simp only [← c53256, ← c53257, ← c53258] at e_854_8
  have e_854_9 := arithEq_of_rows h (row := 854) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_9
  simp only [← c53259, ← c53260, k2, ← c53261] at e_854_9
  have hr := e_854_9
  simp only [e_854_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6350)) := by
  have c53262 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53263 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53264 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53265 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53266 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53267 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_10 := arithEq_of_rows h (row := 854) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_10
  simp only [← c53262, ← c53263, ← c53264] at e_854_10
  have e_854_11 := arithEq_of_rows h (row := 854) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_11
  simp only [← c53265, ← c53266, k2, ← c53267] at e_854_11
  have hr := e_854_11
  simp only [e_854_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6351)) := by
  have c53268 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53269 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53270 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53271 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53272 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53273 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_12 := arithEq_of_rows h (row := 854) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_12
  simp only [← c53268, ← c53269, ← c53270] at e_854_12
  have e_854_13 := arithEq_of_rows h (row := 854) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_13
  simp only [← c53271, ← c53272, k2, ← c53273] at e_854_13
  have hr := e_854_13
  simp only [e_854_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6352)) := by
  have c53274 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53275 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53276 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53277 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53278 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53279 := (publicBatchWrapper89_copies1664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_14 := arithEq_of_rows h (row := 854) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_14
  simp only [← c53274, ← c53275, ← c53276] at e_854_14
  have e_854_15 := arithEq_of_rows h (row := 854) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_15
  simp only [← c53277, ← c53278, k2, ← c53279] at e_854_15
  have hr := e_854_15
  simp only [e_854_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6353)) := by
  have c53280 := (publicBatchWrapper89_copies1665 a h).1
  have c53281 := (publicBatchWrapper89_copies1665 a h).2.1
  have c53282 := (publicBatchWrapper89_copies1665 a h).2.2.1
  have c53283 := (publicBatchWrapper89_copies1665 a h).2.2.2.1
  have c53284 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.1
  have c53285 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_16 := arithEq_of_rows h (row := 854) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_16
  simp only [← c53280, ← c53281, ← c53282] at e_854_16
  have e_854_17 := arithEq_of_rows h (row := 854) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_17
  simp only [← c53283, ← c53284, k2, ← c53285] at e_854_17
  have hr := e_854_17
  simp only [e_854_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 854 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6354)) := by
  have c53286 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.1
  have c53287 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.1
  have c53288 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.1
  have c53289 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.1
  have c53290 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53291 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_854_18 := arithEq_of_rows h (row := 854) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_18
  simp only [← c53286, ← c53287, ← c53288] at e_854_18
  have e_854_19 := arithEq_of_rows h (row := 854) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_854_19
  simp only [← c53289, ← c53290, k2, ← c53291] at e_854_19
  have hr := e_854_19
  simp only [e_854_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6355)) := by
  have c53292 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53293 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53294 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53295 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53296 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53297 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_0 := arithEq_of_rows h (row := 855) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_0
  simp only [← c53292, ← c53293, ← c53294] at e_855_0
  have e_855_1 := arithEq_of_rows h (row := 855) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_1
  simp only [← c53295, ← c53296, k2, ← c53297] at e_855_1
  have hr := e_855_1
  simp only [e_855_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6356)) := by
  have c53298 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53299 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53300 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53301 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53302 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53303 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_2 := arithEq_of_rows h (row := 855) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_2
  simp only [← c53298, ← c53299, ← c53300] at e_855_2
  have e_855_3 := arithEq_of_rows h (row := 855) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_3
  simp only [← c53301, ← c53302, k2, ← c53303] at e_855_3
  have hr := e_855_3
  simp only [e_855_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6357)) := by
  have c53304 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53305 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53306 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53307 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53308 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53309 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_4 := arithEq_of_rows h (row := 855) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_4
  simp only [← c53304, ← c53305, ← c53306] at e_855_4
  have e_855_5 := arithEq_of_rows h (row := 855) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_5
  simp only [← c53307, ← c53308, k2, ← c53309] at e_855_5
  have hr := e_855_5
  simp only [e_855_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6358)) := by
  have c53310 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53311 := (publicBatchWrapper89_copies1665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53312 := (publicBatchWrapper89_copies1666 a h).1
  have c53313 := (publicBatchWrapper89_copies1666 a h).2.1
  have c53314 := (publicBatchWrapper89_copies1666 a h).2.2.1
  have c53315 := (publicBatchWrapper89_copies1666 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_6 := arithEq_of_rows h (row := 855) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_6
  simp only [← c53310, ← c53311, ← c53312] at e_855_6
  have e_855_7 := arithEq_of_rows h (row := 855) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_7
  simp only [← c53313, ← c53314, k2, ← c53315] at e_855_7
  have hr := e_855_7
  simp only [e_855_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6359)) := by
  have c53316 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.1
  have c53317 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.1
  have c53318 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.1
  have c53319 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.1
  have c53320 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.1
  have c53321 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_8 := arithEq_of_rows h (row := 855) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_8
  simp only [← c53316, ← c53317, ← c53318] at e_855_8
  have e_855_9 := arithEq_of_rows h (row := 855) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_9
  simp only [← c53319, ← c53320, k2, ← c53321] at e_855_9
  have hr := e_855_9
  simp only [e_855_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6360)) := by
  have c53322 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53323 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53324 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53325 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53326 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53327 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_10 := arithEq_of_rows h (row := 855) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_10
  simp only [← c53322, ← c53323, ← c53324] at e_855_10
  have e_855_11 := arithEq_of_rows h (row := 855) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_11
  simp only [← c53325, ← c53326, k2, ← c53327] at e_855_11
  have hr := e_855_11
  simp only [e_855_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6361)) := by
  have c53328 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53329 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53330 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53331 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53332 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53333 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_12 := arithEq_of_rows h (row := 855) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_12
  simp only [← c53328, ← c53329, ← c53330] at e_855_12
  have e_855_13 := arithEq_of_rows h (row := 855) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_13
  simp only [← c53331, ← c53332, k2, ← c53333] at e_855_13
  have hr := e_855_13
  simp only [e_855_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6362)) := by
  have c53334 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53335 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53336 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53337 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53338 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53339 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_14 := arithEq_of_rows h (row := 855) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_14
  simp only [← c53334, ← c53335, ← c53336] at e_855_14
  have e_855_15 := arithEq_of_rows h (row := 855) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_15
  simp only [← c53337, ← c53338, k2, ← c53339] at e_855_15
  have hr := e_855_15
  simp only [e_855_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6363)) := by
  have c53340 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53341 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53342 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53343 := (publicBatchWrapper89_copies1666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53344 := (publicBatchWrapper89_copies1667 a h).1
  have c53345 := (publicBatchWrapper89_copies1667 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_16 := arithEq_of_rows h (row := 855) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_16
  simp only [← c53340, ← c53341, ← c53342] at e_855_16
  have e_855_17 := arithEq_of_rows h (row := 855) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_17
  simp only [← c53343, ← c53344, k2, ← c53345] at e_855_17
  have hr := e_855_17
  simp only [e_855_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 855 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6364)) := by
  have c53346 := (publicBatchWrapper89_copies1667 a h).2.2.1
  have c53347 := (publicBatchWrapper89_copies1667 a h).2.2.2.1
  have c53348 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.1
  have c53349 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.1
  have c53350 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.1
  have c53351 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_855_18 := arithEq_of_rows h (row := 855) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_18
  simp only [← c53346, ← c53347, ← c53348] at e_855_18
  have e_855_19 := arithEq_of_rows h (row := 855) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_855_19
  simp only [← c53349, ← c53350, k2, ← c53351] at e_855_19
  have hr := e_855_19
  simp only [e_855_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6365)) := by
  have c53352 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.1
  have c53353 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.1
  have c53354 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53355 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53356 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53357 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_0 := arithEq_of_rows h (row := 856) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_0
  simp only [← c53352, ← c53353, ← c53354] at e_856_0
  have e_856_1 := arithEq_of_rows h (row := 856) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_1
  simp only [← c53355, ← c53356, k2, ← c53357] at e_856_1
  have hr := e_856_1
  simp only [e_856_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6366)) := by
  have c53358 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53359 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53360 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53361 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53362 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53363 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_2 := arithEq_of_rows h (row := 856) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_2
  simp only [← c53358, ← c53359, ← c53360] at e_856_2
  have e_856_3 := arithEq_of_rows h (row := 856) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_3
  simp only [← c53361, ← c53362, k2, ← c53363] at e_856_3
  have hr := e_856_3
  simp only [e_856_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6367)) := by
  have c53364 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53365 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53366 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53367 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53368 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53369 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_4 := arithEq_of_rows h (row := 856) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_4
  simp only [← c53364, ← c53365, ← c53366] at e_856_4
  have e_856_5 := arithEq_of_rows h (row := 856) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_5
  simp only [← c53367, ← c53368, k2, ← c53369] at e_856_5
  have hr := e_856_5
  simp only [e_856_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6368)) := by
  have c53370 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53371 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53372 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53373 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53374 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53375 := (publicBatchWrapper89_copies1667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_6 := arithEq_of_rows h (row := 856) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_6
  simp only [← c53370, ← c53371, ← c53372] at e_856_6
  have e_856_7 := arithEq_of_rows h (row := 856) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_7
  simp only [← c53373, ← c53374, k2, ← c53375] at e_856_7
  have hr := e_856_7
  simp only [e_856_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6369)) := by
  have c53376 := (publicBatchWrapper89_copies1668 a h).1
  have c53377 := (publicBatchWrapper89_copies1668 a h).2.1
  have c53378 := (publicBatchWrapper89_copies1668 a h).2.2.1
  have c53379 := (publicBatchWrapper89_copies1668 a h).2.2.2.1
  have c53380 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.1
  have c53381 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_8 := arithEq_of_rows h (row := 856) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_8
  simp only [← c53376, ← c53377, ← c53378] at e_856_8
  have e_856_9 := arithEq_of_rows h (row := 856) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_9
  simp only [← c53379, ← c53380, k2, ← c53381] at e_856_9
  have hr := e_856_9
  simp only [e_856_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6410)) := by
  have c53382 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.1
  have c53383 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.1
  have c53384 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.1
  have c53385 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.1
  have c53386 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53387 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_10 := arithEq_of_rows h (row := 856) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_10
  simp only [← c53382, ← c53383, ← c53384] at e_856_10
  have e_856_11 := arithEq_of_rows h (row := 856) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_11
  simp only [← c53385, ← c53386, k2, ← c53387] at e_856_11
  have hr := e_856_11
  simp only [e_856_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6411)) := by
  have c53388 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53389 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53390 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53391 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53392 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53393 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_12 := arithEq_of_rows h (row := 856) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_12
  simp only [← c53388, ← c53389, ← c53390] at e_856_12
  have e_856_13 := arithEq_of_rows h (row := 856) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_13
  simp only [← c53391, ← c53392, k2, ← c53393] at e_856_13
  have hr := e_856_13
  simp only [e_856_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6412)) := by
  have c53394 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53395 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53396 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53397 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53398 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53399 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_14 := arithEq_of_rows h (row := 856) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_14
  simp only [← c53394, ← c53395, ← c53396] at e_856_14
  have e_856_15 := arithEq_of_rows h (row := 856) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_15
  simp only [← c53397, ← c53398, k2, ← c53399] at e_856_15
  have hr := e_856_15
  simp only [e_856_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6413)) := by
  have c53400 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53401 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53402 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53403 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53404 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53405 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_16 := arithEq_of_rows h (row := 856) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_16
  simp only [← c53400, ← c53401, ← c53402] at e_856_16
  have e_856_17 := arithEq_of_rows h (row := 856) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_17
  simp only [← c53403, ← c53404, k2, ← c53405] at e_856_17
  have hr := e_856_17
  simp only [e_856_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 856 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6414)) := by
  have c53406 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53407 := (publicBatchWrapper89_copies1668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53408 := (publicBatchWrapper89_copies1669 a h).1
  have c53409 := (publicBatchWrapper89_copies1669 a h).2.1
  have c53410 := (publicBatchWrapper89_copies1669 a h).2.2.1
  have c53411 := (publicBatchWrapper89_copies1669 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_856_18 := arithEq_of_rows h (row := 856) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_18
  simp only [← c53406, ← c53407, ← c53408] at e_856_18
  have e_856_19 := arithEq_of_rows h (row := 856) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_856_19
  simp only [← c53409, ← c53410, k2, ← c53411] at e_856_19
  have hr := e_856_19
  simp only [e_856_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6415)) := by
  have c53412 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.1
  have c53413 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.1
  have c53414 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.1
  have c53415 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.1
  have c53416 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.1
  have c53417 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_0 := arithEq_of_rows h (row := 857) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_0
  simp only [← c53412, ← c53413, ← c53414] at e_857_0
  have e_857_1 := arithEq_of_rows h (row := 857) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_1
  simp only [← c53415, ← c53416, k2, ← c53417] at e_857_1
  have hr := e_857_1
  simp only [e_857_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6416)) := by
  have c53418 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53419 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53420 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53421 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53422 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53423 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_2 := arithEq_of_rows h (row := 857) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_2
  simp only [← c53418, ← c53419, ← c53420] at e_857_2
  have e_857_3 := arithEq_of_rows h (row := 857) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_3
  simp only [← c53421, ← c53422, k2, ← c53423] at e_857_3
  have hr := e_857_3
  simp only [e_857_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6417)) := by
  have c53424 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53425 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53426 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53427 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53428 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53429 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_4 := arithEq_of_rows h (row := 857) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_4
  simp only [← c53424, ← c53425, ← c53426] at e_857_4
  have e_857_5 := arithEq_of_rows h (row := 857) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_5
  simp only [← c53427, ← c53428, k2, ← c53429] at e_857_5
  have hr := e_857_5
  simp only [e_857_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6418)) := by
  have c53430 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53431 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53432 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53433 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53434 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53435 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_6 := arithEq_of_rows h (row := 857) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_6
  simp only [← c53430, ← c53431, ← c53432] at e_857_6
  have e_857_7 := arithEq_of_rows h (row := 857) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_7
  simp only [← c53433, ← c53434, k2, ← c53435] at e_857_7
  have hr := e_857_7
  simp only [e_857_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6419)) := by
  have c53436 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53437 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53438 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53439 := (publicBatchWrapper89_copies1669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53440 := (publicBatchWrapper89_copies1670 a h).1
  have c53441 := (publicBatchWrapper89_copies1670 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_8 := arithEq_of_rows h (row := 857) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_8
  simp only [← c53436, ← c53437, ← c53438] at e_857_8
  have e_857_9 := arithEq_of_rows h (row := 857) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_9
  simp only [← c53439, ← c53440, k2, ← c53441] at e_857_9
  have hr := e_857_9
  simp only [e_857_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6420)) := by
  have c53442 := (publicBatchWrapper89_copies1670 a h).2.2.1
  have c53443 := (publicBatchWrapper89_copies1670 a h).2.2.2.1
  have c53444 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.1
  have c53445 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.1
  have c53446 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.1
  have c53447 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_10 := arithEq_of_rows h (row := 857) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_10
  simp only [← c53442, ← c53443, ← c53444] at e_857_10
  have e_857_11 := arithEq_of_rows h (row := 857) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_11
  simp only [← c53445, ← c53446, k2, ← c53447] at e_857_11
  have hr := e_857_11
  simp only [e_857_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6421)) := by
  have c53448 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.1
  have c53449 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.1
  have c53450 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53451 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53452 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53453 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_12 := arithEq_of_rows h (row := 857) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_12
  simp only [← c53448, ← c53449, ← c53450] at e_857_12
  have e_857_13 := arithEq_of_rows h (row := 857) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_13
  simp only [← c53451, ← c53452, k2, ← c53453] at e_857_13
  have hr := e_857_13
  simp only [e_857_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6422)) := by
  have c53454 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53455 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53456 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53457 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53458 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53459 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_14 := arithEq_of_rows h (row := 857) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_14
  simp only [← c53454, ← c53455, ← c53456] at e_857_14
  have e_857_15 := arithEq_of_rows h (row := 857) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_15
  simp only [← c53457, ← c53458, k2, ← c53459] at e_857_15
  have hr := e_857_15
  simp only [e_857_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6423)) := by
  have c53460 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53461 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53462 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53463 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53464 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53465 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_16 := arithEq_of_rows h (row := 857) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_16
  simp only [← c53460, ← c53461, ← c53462] at e_857_16
  have e_857_17 := arithEq_of_rows h (row := 857) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_17
  simp only [← c53463, ← c53464, k2, ← c53465] at e_857_17
  have hr := e_857_17
  simp only [e_857_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 857 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6424)) := by
  have c53466 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53467 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53468 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53469 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53470 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53471 := (publicBatchWrapper89_copies1670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_857_18 := arithEq_of_rows h (row := 857) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_18
  simp only [← c53466, ← c53467, ← c53468] at e_857_18
  have e_857_19 := arithEq_of_rows h (row := 857) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_857_19
  simp only [← c53469, ← c53470, k2, ← c53471] at e_857_19
  have hr := e_857_19
  simp only [e_857_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6425)) := by
  have c53472 := (publicBatchWrapper89_copies1671 a h).1
  have c53473 := (publicBatchWrapper89_copies1671 a h).2.1
  have c53474 := (publicBatchWrapper89_copies1671 a h).2.2.1
  have c53475 := (publicBatchWrapper89_copies1671 a h).2.2.2.1
  have c53476 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.1
  have c53477 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_0 := arithEq_of_rows h (row := 858) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_0
  simp only [← c53472, ← c53473, ← c53474] at e_858_0
  have e_858_1 := arithEq_of_rows h (row := 858) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_1
  simp only [← c53475, ← c53476, k2, ← c53477] at e_858_1
  have hr := e_858_1
  simp only [e_858_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6426)) := by
  have c53478 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.1
  have c53479 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.1
  have c53480 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.1
  have c53481 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.1
  have c53482 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53483 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_2 := arithEq_of_rows h (row := 858) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_2
  simp only [← c53478, ← c53479, ← c53480] at e_858_2
  have e_858_3 := arithEq_of_rows h (row := 858) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_3
  simp only [← c53481, ← c53482, k2, ← c53483] at e_858_3
  have hr := e_858_3
  simp only [e_858_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6427)) := by
  have c53484 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53485 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53486 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53487 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53488 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53489 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_4 := arithEq_of_rows h (row := 858) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_4
  simp only [← c53484, ← c53485, ← c53486] at e_858_4
  have e_858_5 := arithEq_of_rows h (row := 858) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_5
  simp only [← c53487, ← c53488, k2, ← c53489] at e_858_5
  have hr := e_858_5
  simp only [e_858_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6428)) := by
  have c53490 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53491 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53492 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53493 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53494 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53495 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_6 := arithEq_of_rows h (row := 858) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_6
  simp only [← c53490, ← c53491, ← c53492] at e_858_6
  have e_858_7 := arithEq_of_rows h (row := 858) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_7
  simp only [← c53493, ← c53494, k2, ← c53495] at e_858_7
  have hr := e_858_7
  simp only [e_858_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6429)) := by
  have c53496 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53497 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53498 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53499 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53500 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53501 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_8 := arithEq_of_rows h (row := 858) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_8
  simp only [← c53496, ← c53497, ← c53498] at e_858_8
  have e_858_9 := arithEq_of_rows h (row := 858) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_9
  simp only [← c53499, ← c53500, k2, ← c53501] at e_858_9
  have hr := e_858_9
  simp only [e_858_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6430)) := by
  have c53502 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53503 := (publicBatchWrapper89_copies1671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53504 := (publicBatchWrapper89_copies1672 a h).1
  have c53505 := (publicBatchWrapper89_copies1672 a h).2.1
  have c53506 := (publicBatchWrapper89_copies1672 a h).2.2.1
  have c53507 := (publicBatchWrapper89_copies1672 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_10 := arithEq_of_rows h (row := 858) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_10
  simp only [← c53502, ← c53503, ← c53504] at e_858_10
  have e_858_11 := arithEq_of_rows h (row := 858) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_11
  simp only [← c53505, ← c53506, k2, ← c53507] at e_858_11
  have hr := e_858_11
  simp only [e_858_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6431)) := by
  have c53508 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.1
  have c53509 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.1
  have c53510 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.1
  have c53511 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.1
  have c53512 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.1
  have c53513 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_12 := arithEq_of_rows h (row := 858) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_12
  simp only [← c53508, ← c53509, ← c53510] at e_858_12
  have e_858_13 := arithEq_of_rows h (row := 858) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_13
  simp only [← c53511, ← c53512, k2, ← c53513] at e_858_13
  have hr := e_858_13
  simp only [e_858_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6432)) := by
  have c53514 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53515 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53516 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53517 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53518 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53519 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_14 := arithEq_of_rows h (row := 858) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_14
  simp only [← c53514, ← c53515, ← c53516] at e_858_14
  have e_858_15 := arithEq_of_rows h (row := 858) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_15
  simp only [← c53517, ← c53518, k2, ← c53519] at e_858_15
  have hr := e_858_15
  simp only [e_858_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6433)) := by
  have c53520 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53521 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53522 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53523 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53524 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53525 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_16 := arithEq_of_rows h (row := 858) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_16
  simp only [← c53520, ← c53521, ← c53522] at e_858_16
  have e_858_17 := arithEq_of_rows h (row := 858) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_17
  simp only [← c53523, ← c53524, k2, ← c53525] at e_858_17
  have hr := e_858_17
  simp only [e_858_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 858 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6434)) := by
  have c53526 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53527 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53528 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53529 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53530 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53531 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_858_18 := arithEq_of_rows h (row := 858) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_18
  simp only [← c53526, ← c53527, ← c53528] at e_858_18
  have e_858_19 := arithEq_of_rows h (row := 858) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_858_19
  simp only [← c53529, ← c53530, k2, ← c53531] at e_858_19
  have hr := e_858_19
  simp only [e_858_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6435)) := by
  have c53532 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53533 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53534 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53535 := (publicBatchWrapper89_copies1672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53536 := (publicBatchWrapper89_copies1673 a h).1
  have c53537 := (publicBatchWrapper89_copies1673 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_0 := arithEq_of_rows h (row := 859) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_0
  simp only [← c53532, ← c53533, ← c53534] at e_859_0
  have e_859_1 := arithEq_of_rows h (row := 859) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_1
  simp only [← c53535, ← c53536, k2, ← c53537] at e_859_1
  have hr := e_859_1
  simp only [e_859_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6436)) := by
  have c53538 := (publicBatchWrapper89_copies1673 a h).2.2.1
  have c53539 := (publicBatchWrapper89_copies1673 a h).2.2.2.1
  have c53540 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.1
  have c53541 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.1
  have c53542 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.1
  have c53543 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_2 := arithEq_of_rows h (row := 859) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_2
  simp only [← c53538, ← c53539, ← c53540] at e_859_2
  have e_859_3 := arithEq_of_rows h (row := 859) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_3
  simp only [← c53541, ← c53542, k2, ← c53543] at e_859_3
  have hr := e_859_3
  simp only [e_859_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6437)) := by
  have c53544 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.1
  have c53545 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.1
  have c53546 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53547 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53548 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53549 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_4 := arithEq_of_rows h (row := 859) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_4
  simp only [← c53544, ← c53545, ← c53546] at e_859_4
  have e_859_5 := arithEq_of_rows h (row := 859) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_5
  simp only [← c53547, ← c53548, k2, ← c53549] at e_859_5
  have hr := e_859_5
  simp only [e_859_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6438)) := by
  have c53550 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53551 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53552 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53553 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53554 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53555 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_6 := arithEq_of_rows h (row := 859) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_6
  simp only [← c53550, ← c53551, ← c53552] at e_859_6
  have e_859_7 := arithEq_of_rows h (row := 859) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_7
  simp only [← c53553, ← c53554, k2, ← c53555] at e_859_7
  have hr := e_859_7
  simp only [e_859_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6439)) := by
  have c53556 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53557 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53558 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53559 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53560 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53561 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_8 := arithEq_of_rows h (row := 859) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_8
  simp only [← c53556, ← c53557, ← c53558] at e_859_8
  have e_859_9 := arithEq_of_rows h (row := 859) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_9
  simp only [← c53559, ← c53560, k2, ← c53561] at e_859_9
  have hr := e_859_9
  simp only [e_859_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6440)) := by
  have c53562 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53563 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53564 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53565 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53566 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53567 := (publicBatchWrapper89_copies1673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_10 := arithEq_of_rows h (row := 859) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_10
  simp only [← c53562, ← c53563, ← c53564] at e_859_10
  have e_859_11 := arithEq_of_rows h (row := 859) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_11
  simp only [← c53565, ← c53566, k2, ← c53567] at e_859_11
  have hr := e_859_11
  simp only [e_859_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6441)) := by
  have c53568 := (publicBatchWrapper89_copies1674 a h).1
  have c53569 := (publicBatchWrapper89_copies1674 a h).2.1
  have c53570 := (publicBatchWrapper89_copies1674 a h).2.2.1
  have c53571 := (publicBatchWrapper89_copies1674 a h).2.2.2.1
  have c53572 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.1
  have c53573 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_12 := arithEq_of_rows h (row := 859) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_12
  simp only [← c53568, ← c53569, ← c53570] at e_859_12
  have e_859_13 := arithEq_of_rows h (row := 859) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_13
  simp only [← c53571, ← c53572, k2, ← c53573] at e_859_13
  have hr := e_859_13
  simp only [e_859_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6442)) := by
  have c53574 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.1
  have c53575 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.1
  have c53576 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.1
  have c53577 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.1
  have c53578 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53579 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_14 := arithEq_of_rows h (row := 859) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_14
  simp only [← c53574, ← c53575, ← c53576] at e_859_14
  have e_859_15 := arithEq_of_rows h (row := 859) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_15
  simp only [← c53577, ← c53578, k2, ← c53579] at e_859_15
  have hr := e_859_15
  simp only [e_859_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6443)) := by
  have c53580 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53581 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53582 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53583 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53584 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53585 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_16 := arithEq_of_rows h (row := 859) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_16
  simp only [← c53580, ← c53581, ← c53582] at e_859_16
  have e_859_17 := arithEq_of_rows h (row := 859) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_17
  simp only [← c53583, ← c53584, k2, ← c53585] at e_859_17
  have hr := e_859_17
  simp only [e_859_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 859 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6444)) := by
  have c53586 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53587 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53588 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53589 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53590 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53591 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_859_18 := arithEq_of_rows h (row := 859) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_18
  simp only [← c53586, ← c53587, ← c53588] at e_859_18
  have e_859_19 := arithEq_of_rows h (row := 859) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_859_19
  simp only [← c53589, ← c53590, k2, ← c53591] at e_859_19
  have hr := e_859_19
  simp only [e_859_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6445)) := by
  have c53592 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53593 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53594 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53595 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53596 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53597 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_0 := arithEq_of_rows h (row := 860) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_0
  simp only [← c53592, ← c53593, ← c53594] at e_860_0
  have e_860_1 := arithEq_of_rows h (row := 860) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_1
  simp only [← c53595, ← c53596, k2, ← c53597] at e_860_1
  have hr := e_860_1
  simp only [e_860_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6446)) := by
  have c53598 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53599 := (publicBatchWrapper89_copies1674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53600 := (publicBatchWrapper89_copies1675 a h).1
  have c53601 := (publicBatchWrapper89_copies1675 a h).2.1
  have c53602 := (publicBatchWrapper89_copies1675 a h).2.2.1
  have c53603 := (publicBatchWrapper89_copies1675 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_2 := arithEq_of_rows h (row := 860) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_2
  simp only [← c53598, ← c53599, ← c53600] at e_860_2
  have e_860_3 := arithEq_of_rows h (row := 860) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_3
  simp only [← c53601, ← c53602, k2, ← c53603] at e_860_3
  have hr := e_860_3
  simp only [e_860_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6447)) := by
  have c53604 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.1
  have c53605 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.1
  have c53606 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.1
  have c53607 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.1
  have c53608 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.1
  have c53609 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_4 := arithEq_of_rows h (row := 860) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_4
  simp only [← c53604, ← c53605, ← c53606] at e_860_4
  have e_860_5 := arithEq_of_rows h (row := 860) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_5
  simp only [← c53607, ← c53608, k2, ← c53609] at e_860_5
  have hr := e_860_5
  simp only [e_860_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6448)) := by
  have c53610 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53611 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53612 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53613 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53614 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53615 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_6 := arithEq_of_rows h (row := 860) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_6
  simp only [← c53610, ← c53611, ← c53612] at e_860_6
  have e_860_7 := arithEq_of_rows h (row := 860) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_7
  simp only [← c53613, ← c53614, k2, ← c53615] at e_860_7
  have hr := e_860_7
  simp only [e_860_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6449)) := by
  have c53616 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53617 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53618 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53619 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53620 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53621 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_8 := arithEq_of_rows h (row := 860) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_8
  simp only [← c53616, ← c53617, ← c53618] at e_860_8
  have e_860_9 := arithEq_of_rows h (row := 860) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_9
  simp only [← c53619, ← c53620, k2, ← c53621] at e_860_9
  have hr := e_860_9
  simp only [e_860_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6450)) := by
  have c53622 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53623 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53624 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53625 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53626 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53627 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_10 := arithEq_of_rows h (row := 860) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_10
  simp only [← c53622, ← c53623, ← c53624] at e_860_10
  have e_860_11 := arithEq_of_rows h (row := 860) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_11
  simp only [← c53625, ← c53626, k2, ← c53627] at e_860_11
  have hr := e_860_11
  simp only [e_860_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6451)) := by
  have c53628 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53629 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53630 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53631 := (publicBatchWrapper89_copies1675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53632 := (publicBatchWrapper89_copies1676 a h).1
  have c53633 := (publicBatchWrapper89_copies1676 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_12 := arithEq_of_rows h (row := 860) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_12
  simp only [← c53628, ← c53629, ← c53630] at e_860_12
  have e_860_13 := arithEq_of_rows h (row := 860) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_13
  simp only [← c53631, ← c53632, k2, ← c53633] at e_860_13
  have hr := e_860_13
  simp only [e_860_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6452)) := by
  have c53634 := (publicBatchWrapper89_copies1676 a h).2.2.1
  have c53635 := (publicBatchWrapper89_copies1676 a h).2.2.2.1
  have c53636 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.1
  have c53637 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.1
  have c53638 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.1
  have c53639 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_14 := arithEq_of_rows h (row := 860) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_14
  simp only [← c53634, ← c53635, ← c53636] at e_860_14
  have e_860_15 := arithEq_of_rows h (row := 860) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_15
  simp only [← c53637, ← c53638, k2, ← c53639] at e_860_15
  have hr := e_860_15
  simp only [e_860_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6453)) := by
  have c53640 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.1
  have c53641 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.1
  have c53642 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53643 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53644 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53645 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_16 := arithEq_of_rows h (row := 860) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_16
  simp only [← c53640, ← c53641, ← c53642] at e_860_16
  have e_860_17 := arithEq_of_rows h (row := 860) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_17
  simp only [← c53643, ← c53644, k2, ← c53645] at e_860_17
  have hr := e_860_17
  simp only [e_860_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 860 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6454)) := by
  have c53646 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53647 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53648 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53649 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53650 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53651 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_860_18 := arithEq_of_rows h (row := 860) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_18
  simp only [← c53646, ← c53647, ← c53648] at e_860_18
  have e_860_19 := arithEq_of_rows h (row := 860) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_860_19
  simp only [← c53649, ← c53650, k2, ← c53651] at e_860_19
  have hr := e_860_19
  simp only [e_860_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6455)) := by
  have c53652 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53653 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53654 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53655 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53656 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53657 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_0 := arithEq_of_rows h (row := 861) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_0
  simp only [← c53652, ← c53653, ← c53654] at e_861_0
  have e_861_1 := arithEq_of_rows h (row := 861) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_1
  simp only [← c53655, ← c53656, k2, ← c53657] at e_861_1
  have hr := e_861_1
  simp only [e_861_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6456)) := by
  have c53658 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53659 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53660 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53661 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53662 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53663 := (publicBatchWrapper89_copies1676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_2 := arithEq_of_rows h (row := 861) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_2
  simp only [← c53658, ← c53659, ← c53660] at e_861_2
  have e_861_3 := arithEq_of_rows h (row := 861) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_3
  simp only [← c53661, ← c53662, k2, ← c53663] at e_861_3
  have hr := e_861_3
  simp only [e_861_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6457)) := by
  have c53664 := (publicBatchWrapper89_copies1677 a h).1
  have c53665 := (publicBatchWrapper89_copies1677 a h).2.1
  have c53666 := (publicBatchWrapper89_copies1677 a h).2.2.1
  have c53667 := (publicBatchWrapper89_copies1677 a h).2.2.2.1
  have c53668 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.1
  have c53669 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_4 := arithEq_of_rows h (row := 861) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_4
  simp only [← c53664, ← c53665, ← c53666] at e_861_4
  have e_861_5 := arithEq_of_rows h (row := 861) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_5
  simp only [← c53667, ← c53668, k2, ← c53669] at e_861_5
  have hr := e_861_5
  simp only [e_861_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6458)) := by
  have c53670 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.1
  have c53671 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.1
  have c53672 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.1
  have c53673 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.1
  have c53674 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53675 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_6 := arithEq_of_rows h (row := 861) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_6
  simp only [← c53670, ← c53671, ← c53672] at e_861_6
  have e_861_7 := arithEq_of_rows h (row := 861) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_7
  simp only [← c53673, ← c53674, k2, ← c53675] at e_861_7
  have hr := e_861_7
  simp only [e_861_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6459)) := by
  have c53676 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53677 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53678 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53679 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53680 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53681 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_8 := arithEq_of_rows h (row := 861) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_8
  simp only [← c53676, ← c53677, ← c53678] at e_861_8
  have e_861_9 := arithEq_of_rows h (row := 861) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_9
  simp only [← c53679, ← c53680, k2, ← c53681] at e_861_9
  have hr := e_861_9
  simp only [e_861_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6460)) := by
  have c53682 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53683 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53684 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53685 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53686 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53687 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_10 := arithEq_of_rows h (row := 861) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_10
  simp only [← c53682, ← c53683, ← c53684] at e_861_10
  have e_861_11 := arithEq_of_rows h (row := 861) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_11
  simp only [← c53685, ← c53686, k2, ← c53687] at e_861_11
  have hr := e_861_11
  simp only [e_861_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6461)) := by
  have c53688 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53689 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53690 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53691 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53692 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53693 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_12 := arithEq_of_rows h (row := 861) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_12
  simp only [← c53688, ← c53689, ← c53690] at e_861_12
  have e_861_13 := arithEq_of_rows h (row := 861) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_13
  simp only [← c53691, ← c53692, k2, ← c53693] at e_861_13
  have hr := e_861_13
  simp only [e_861_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6462)) := by
  have c53694 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53695 := (publicBatchWrapper89_copies1677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53696 := (publicBatchWrapper89_copies1678 a h).1
  have c53697 := (publicBatchWrapper89_copies1678 a h).2.1
  have c53698 := (publicBatchWrapper89_copies1678 a h).2.2.1
  have c53699 := (publicBatchWrapper89_copies1678 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_14 := arithEq_of_rows h (row := 861) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_14
  simp only [← c53694, ← c53695, ← c53696] at e_861_14
  have e_861_15 := arithEq_of_rows h (row := 861) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_15
  simp only [← c53697, ← c53698, k2, ← c53699] at e_861_15
  have hr := e_861_15
  simp only [e_861_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6463)) := by
  have c53700 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.1
  have c53701 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.1
  have c53702 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.1
  have c53703 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.1
  have c53704 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.1
  have c53705 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_16 := arithEq_of_rows h (row := 861) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_16
  simp only [← c53700, ← c53701, ← c53702] at e_861_16
  have e_861_17 := arithEq_of_rows h (row := 861) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_17
  simp only [← c53703, ← c53704, k2, ← c53705] at e_861_17
  have hr := e_861_17
  simp only [e_861_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 861 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6464)) := by
  have c53706 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53707 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53708 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53709 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53710 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53711 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_861_18 := arithEq_of_rows h (row := 861) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_18
  simp only [← c53706, ← c53707, ← c53708] at e_861_18
  have e_861_19 := arithEq_of_rows h (row := 861) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_861_19
  simp only [← c53709, ← c53710, k2, ← c53711] at e_861_19
  have hr := e_861_19
  simp only [e_861_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6465)) := by
  have c53712 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53713 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53714 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53715 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53716 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53717 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_0 := arithEq_of_rows h (row := 862) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_0
  simp only [← c53712, ← c53713, ← c53714] at e_862_0
  have e_862_1 := arithEq_of_rows h (row := 862) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_1
  simp only [← c53715, ← c53716, k2, ← c53717] at e_862_1
  have hr := e_862_1
  simp only [e_862_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6466)) := by
  have c53718 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53719 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53720 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53721 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53722 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53723 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_2 := arithEq_of_rows h (row := 862) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_2
  simp only [← c53718, ← c53719, ← c53720] at e_862_2
  have e_862_3 := arithEq_of_rows h (row := 862) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_3
  simp only [← c53721, ← c53722, k2, ← c53723] at e_862_3
  have hr := e_862_3
  simp only [e_862_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6467)) := by
  have c53724 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53725 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53726 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53727 := (publicBatchWrapper89_copies1678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53728 := (publicBatchWrapper89_copies1679 a h).1
  have c53729 := (publicBatchWrapper89_copies1679 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_4 := arithEq_of_rows h (row := 862) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_4
  simp only [← c53724, ← c53725, ← c53726] at e_862_4
  have e_862_5 := arithEq_of_rows h (row := 862) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_5
  simp only [← c53727, ← c53728, k2, ← c53729] at e_862_5
  have hr := e_862_5
  simp only [e_862_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6468)) := by
  have c53730 := (publicBatchWrapper89_copies1679 a h).2.2.1
  have c53731 := (publicBatchWrapper89_copies1679 a h).2.2.2.1
  have c53732 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.1
  have c53733 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.1
  have c53734 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.1
  have c53735 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_6 := arithEq_of_rows h (row := 862) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_6
  simp only [← c53730, ← c53731, ← c53732] at e_862_6
  have e_862_7 := arithEq_of_rows h (row := 862) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_7
  simp only [← c53733, ← c53734, k2, ← c53735] at e_862_7
  have hr := e_862_7
  simp only [e_862_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6469)) := by
  have c53736 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.1
  have c53737 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.1
  have c53738 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53739 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53740 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53741 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_8 := arithEq_of_rows h (row := 862) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_8
  simp only [← c53736, ← c53737, ← c53738] at e_862_8
  have e_862_9 := arithEq_of_rows h (row := 862) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_9
  simp only [← c53739, ← c53740, k2, ← c53741] at e_862_9
  have hr := e_862_9
  simp only [e_862_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6470)) := by
  have c53742 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53743 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53744 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53745 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53746 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53747 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_10 := arithEq_of_rows h (row := 862) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_10
  simp only [← c53742, ← c53743, ← c53744] at e_862_10
  have e_862_11 := arithEq_of_rows h (row := 862) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_11
  simp only [← c53745, ← c53746, k2, ← c53747] at e_862_11
  have hr := e_862_11
  simp only [e_862_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6471)) := by
  have c53748 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53749 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53750 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53751 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53752 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53753 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_12 := arithEq_of_rows h (row := 862) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_12
  simp only [← c53748, ← c53749, ← c53750] at e_862_12
  have e_862_13 := arithEq_of_rows h (row := 862) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_13
  simp only [← c53751, ← c53752, k2, ← c53753] at e_862_13
  have hr := e_862_13
  simp only [e_862_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6472)) := by
  have c53754 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53755 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53756 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53757 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53758 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53759 := (publicBatchWrapper89_copies1679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_14 := arithEq_of_rows h (row := 862) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_14
  simp only [← c53754, ← c53755, ← c53756] at e_862_14
  have e_862_15 := arithEq_of_rows h (row := 862) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_15
  simp only [← c53757, ← c53758, k2, ← c53759] at e_862_15
  have hr := e_862_15
  simp only [e_862_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6473)) := by
  have c53760 := (publicBatchWrapper89_copies1680 a h).1
  have c53761 := (publicBatchWrapper89_copies1680 a h).2.1
  have c53762 := (publicBatchWrapper89_copies1680 a h).2.2.1
  have c53763 := (publicBatchWrapper89_copies1680 a h).2.2.2.1
  have c53764 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.1
  have c53765 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_16 := arithEq_of_rows h (row := 862) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_16
  simp only [← c53760, ← c53761, ← c53762] at e_862_16
  have e_862_17 := arithEq_of_rows h (row := 862) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_17
  simp only [← c53763, ← c53764, k2, ← c53765] at e_862_17
  have hr := e_862_17
  simp only [e_862_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 862 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6474)) := by
  have c53766 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.1
  have c53767 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.1
  have c53768 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.1
  have c53769 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.1
  have c53770 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53771 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_862_18 := arithEq_of_rows h (row := 862) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_18
  simp only [← c53766, ← c53767, ← c53768] at e_862_18
  have e_862_19 := arithEq_of_rows h (row := 862) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_862_19
  simp only [← c53769, ← c53770, k2, ← c53771] at e_862_19
  have hr := e_862_19
  simp only [e_862_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6475)) := by
  have c53772 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53773 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53774 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53775 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53776 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53777 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_0 := arithEq_of_rows h (row := 863) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_0
  simp only [← c53772, ← c53773, ← c53774] at e_863_0
  have e_863_1 := arithEq_of_rows h (row := 863) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_1
  simp only [← c53775, ← c53776, k2, ← c53777] at e_863_1
  have hr := e_863_1
  simp only [e_863_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6476)) := by
  have c53778 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53779 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53780 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53781 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53782 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53783 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_2 := arithEq_of_rows h (row := 863) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_2
  simp only [← c53778, ← c53779, ← c53780] at e_863_2
  have e_863_3 := arithEq_of_rows h (row := 863) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_3
  simp only [← c53781, ← c53782, k2, ← c53783] at e_863_3
  have hr := e_863_3
  simp only [e_863_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6477)) := by
  have c53784 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53785 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53786 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53787 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53788 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53789 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_4 := arithEq_of_rows h (row := 863) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_4
  simp only [← c53784, ← c53785, ← c53786] at e_863_4
  have e_863_5 := arithEq_of_rows h (row := 863) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_5
  simp only [← c53787, ← c53788, k2, ← c53789] at e_863_5
  have hr := e_863_5
  simp only [e_863_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6478)) := by
  have c53790 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53791 := (publicBatchWrapper89_copies1680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53792 := (publicBatchWrapper89_copies1681 a h).1
  have c53793 := (publicBatchWrapper89_copies1681 a h).2.1
  have c53794 := (publicBatchWrapper89_copies1681 a h).2.2.1
  have c53795 := (publicBatchWrapper89_copies1681 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_6 := arithEq_of_rows h (row := 863) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_6
  simp only [← c53790, ← c53791, ← c53792] at e_863_6
  have e_863_7 := arithEq_of_rows h (row := 863) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_7
  simp only [← c53793, ← c53794, k2, ← c53795] at e_863_7
  have hr := e_863_7
  simp only [e_863_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6479)) := by
  have c53796 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.1
  have c53797 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.1
  have c53798 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.1
  have c53799 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.1
  have c53800 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.1
  have c53801 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_8 := arithEq_of_rows h (row := 863) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_8
  simp only [← c53796, ← c53797, ← c53798] at e_863_8
  have e_863_9 := arithEq_of_rows h (row := 863) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_9
  simp only [← c53799, ← c53800, k2, ← c53801] at e_863_9
  have hr := e_863_9
  simp only [e_863_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6480)) := by
  have c53802 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53803 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53804 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53805 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53806 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53807 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_10 := arithEq_of_rows h (row := 863) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_10
  simp only [← c53802, ← c53803, ← c53804] at e_863_10
  have e_863_11 := arithEq_of_rows h (row := 863) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_11
  simp only [← c53805, ← c53806, k2, ← c53807] at e_863_11
  have hr := e_863_11
  simp only [e_863_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6481)) := by
  have c53808 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53809 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53810 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53811 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53812 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53813 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_12 := arithEq_of_rows h (row := 863) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_12
  simp only [← c53808, ← c53809, ← c53810] at e_863_12
  have e_863_13 := arithEq_of_rows h (row := 863) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_13
  simp only [← c53811, ← c53812, k2, ← c53813] at e_863_13
  have hr := e_863_13
  simp only [e_863_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6482)) := by
  have c53814 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53815 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53816 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53817 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53818 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53819 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_14 := arithEq_of_rows h (row := 863) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_14
  simp only [← c53814, ← c53815, ← c53816] at e_863_14
  have e_863_15 := arithEq_of_rows h (row := 863) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_15
  simp only [← c53817, ← c53818, k2, ← c53819] at e_863_15
  have hr := e_863_15
  simp only [e_863_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6483)) := by
  have c53820 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53821 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53822 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53823 := (publicBatchWrapper89_copies1681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53824 := (publicBatchWrapper89_copies1682 a h).1
  have c53825 := (publicBatchWrapper89_copies1682 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_16 := arithEq_of_rows h (row := 863) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_16
  simp only [← c53820, ← c53821, ← c53822] at e_863_16
  have e_863_17 := arithEq_of_rows h (row := 863) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_17
  simp only [← c53823, ← c53824, k2, ← c53825] at e_863_17
  have hr := e_863_17
  simp only [e_863_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 863 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6484)) := by
  have c53826 := (publicBatchWrapper89_copies1682 a h).2.2.1
  have c53827 := (publicBatchWrapper89_copies1682 a h).2.2.2.1
  have c53828 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.1
  have c53829 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.1
  have c53830 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.1
  have c53831 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_863_18 := arithEq_of_rows h (row := 863) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_18
  simp only [← c53826, ← c53827, ← c53828] at e_863_18
  have e_863_19 := arithEq_of_rows h (row := 863) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_863_19
  simp only [← c53829, ← c53830, k2, ← c53831] at e_863_19
  have hr := e_863_19
  simp only [e_863_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6485)) := by
  have c53832 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.1
  have c53833 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.1
  have c53834 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53835 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53836 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53837 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_0 := arithEq_of_rows h (row := 864) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_0
  simp only [← c53832, ← c53833, ← c53834] at e_864_0
  have e_864_1 := arithEq_of_rows h (row := 864) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_1
  simp only [← c53835, ← c53836, k2, ← c53837] at e_864_1
  have hr := e_864_1
  simp only [e_864_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6486)) := by
  have c53838 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53839 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53840 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53841 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53842 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53843 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_2 := arithEq_of_rows h (row := 864) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_2
  simp only [← c53838, ← c53839, ← c53840] at e_864_2
  have e_864_3 := arithEq_of_rows h (row := 864) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_3
  simp only [← c53841, ← c53842, k2, ← c53843] at e_864_3
  have hr := e_864_3
  simp only [e_864_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6487)) := by
  have c53844 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53845 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53846 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53847 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53848 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53849 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_4 := arithEq_of_rows h (row := 864) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_4
  simp only [← c53844, ← c53845, ← c53846] at e_864_4
  have e_864_5 := arithEq_of_rows h (row := 864) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_5
  simp only [← c53847, ← c53848, k2, ← c53849] at e_864_5
  have hr := e_864_5
  simp only [e_864_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6488)) := by
  have c53850 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53851 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53852 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53853 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53854 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53855 := (publicBatchWrapper89_copies1682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_6 := arithEq_of_rows h (row := 864) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_6
  simp only [← c53850, ← c53851, ← c53852] at e_864_6
  have e_864_7 := arithEq_of_rows h (row := 864) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_7
  simp only [← c53853, ← c53854, k2, ← c53855] at e_864_7
  have hr := e_864_7
  simp only [e_864_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6489)) := by
  have c53856 := (publicBatchWrapper89_copies1683 a h).1
  have c53857 := (publicBatchWrapper89_copies1683 a h).2.1
  have c53858 := (publicBatchWrapper89_copies1683 a h).2.2.1
  have c53859 := (publicBatchWrapper89_copies1683 a h).2.2.2.1
  have c53860 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.1
  have c53861 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_8 := arithEq_of_rows h (row := 864) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_8
  simp only [← c53856, ← c53857, ← c53858] at e_864_8
  have e_864_9 := arithEq_of_rows h (row := 864) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_9
  simp only [← c53859, ← c53860, k2, ← c53861] at e_864_9
  have hr := e_864_9
  simp only [e_864_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6490)) := by
  have c53862 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.1
  have c53863 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.1
  have c53864 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.1
  have c53865 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.1
  have c53866 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53867 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_10 := arithEq_of_rows h (row := 864) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_10
  simp only [← c53862, ← c53863, ← c53864] at e_864_10
  have e_864_11 := arithEq_of_rows h (row := 864) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_11
  simp only [← c53865, ← c53866, k2, ← c53867] at e_864_11
  have hr := e_864_11
  simp only [e_864_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6491)) := by
  have c53868 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53869 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53870 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53871 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53872 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53873 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_12 := arithEq_of_rows h (row := 864) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_12
  simp only [← c53868, ← c53869, ← c53870] at e_864_12
  have e_864_13 := arithEq_of_rows h (row := 864) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_13
  simp only [← c53871, ← c53872, k2, ← c53873] at e_864_13
  have hr := e_864_13
  simp only [e_864_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6492)) := by
  have c53874 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53875 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53876 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53877 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53878 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53879 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_14 := arithEq_of_rows h (row := 864) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_14
  simp only [← c53874, ← c53875, ← c53876] at e_864_14
  have e_864_15 := arithEq_of_rows h (row := 864) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_15
  simp only [← c53877, ← c53878, k2, ← c53879] at e_864_15
  have hr := e_864_15
  simp only [e_864_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6493)) := by
  have c53880 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53881 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53882 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53883 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53884 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53885 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_16 := arithEq_of_rows h (row := 864) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_16
  simp only [← c53880, ← c53881, ← c53882] at e_864_16
  have e_864_17 := arithEq_of_rows h (row := 864) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_17
  simp only [← c53883, ← c53884, k2, ← c53885] at e_864_17
  have hr := e_864_17
  simp only [e_864_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 864 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6494)) := by
  have c53886 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53887 := (publicBatchWrapper89_copies1683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53888 := (publicBatchWrapper89_copies1684 a h).1
  have c53889 := (publicBatchWrapper89_copies1684 a h).2.1
  have c53890 := (publicBatchWrapper89_copies1684 a h).2.2.1
  have c53891 := (publicBatchWrapper89_copies1684 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_864_18 := arithEq_of_rows h (row := 864) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_18
  simp only [← c53886, ← c53887, ← c53888] at e_864_18
  have e_864_19 := arithEq_of_rows h (row := 864) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_864_19
  simp only [← c53889, ← c53890, k2, ← c53891] at e_864_19
  have hr := e_864_19
  simp only [e_864_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6495)) := by
  have c53892 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.1
  have c53893 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.1
  have c53894 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.1
  have c53895 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.1
  have c53896 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.1
  have c53897 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_0 := arithEq_of_rows h (row := 865) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_0
  simp only [← c53892, ← c53893, ← c53894] at e_865_0
  have e_865_1 := arithEq_of_rows h (row := 865) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_1
  simp only [← c53895, ← c53896, k2, ← c53897] at e_865_1
  have hr := e_865_1
  simp only [e_865_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6496)) := by
  have c53898 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53899 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53900 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53901 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53902 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53903 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_2 := arithEq_of_rows h (row := 865) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_2
  simp only [← c53898, ← c53899, ← c53900] at e_865_2
  have e_865_3 := arithEq_of_rows h (row := 865) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_3
  simp only [← c53901, ← c53902, k2, ← c53903] at e_865_3
  have hr := e_865_3
  simp only [e_865_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6497)) := by
  have c53904 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53905 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53906 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53907 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53908 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53909 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_4 := arithEq_of_rows h (row := 865) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_4
  simp only [← c53904, ← c53905, ← c53906] at e_865_4
  have e_865_5 := arithEq_of_rows h (row := 865) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_5
  simp only [← c53907, ← c53908, k2, ← c53909] at e_865_5
  have hr := e_865_5
  simp only [e_865_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6498)) := by
  have c53910 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53911 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53912 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53913 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53914 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53915 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_6 := arithEq_of_rows h (row := 865) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_6
  simp only [← c53910, ← c53911, ← c53912] at e_865_6
  have e_865_7 := arithEq_of_rows h (row := 865) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_7
  simp only [← c53913, ← c53914, k2, ← c53915] at e_865_7
  have hr := e_865_7
  simp only [e_865_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6499)) := by
  have c53916 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53917 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53918 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53919 := (publicBatchWrapper89_copies1684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53920 := (publicBatchWrapper89_copies1685 a h).1
  have c53921 := (publicBatchWrapper89_copies1685 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_8 := arithEq_of_rows h (row := 865) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_8
  simp only [← c53916, ← c53917, ← c53918] at e_865_8
  have e_865_9 := arithEq_of_rows h (row := 865) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_9
  simp only [← c53919, ← c53920, k2, ← c53921] at e_865_9
  have hr := e_865_9
  simp only [e_865_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6500)) := by
  have c53922 := (publicBatchWrapper89_copies1685 a h).2.2.1
  have c53923 := (publicBatchWrapper89_copies1685 a h).2.2.2.1
  have c53924 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.1
  have c53925 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.1
  have c53926 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.1
  have c53927 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_10 := arithEq_of_rows h (row := 865) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_10
  simp only [← c53922, ← c53923, ← c53924] at e_865_10
  have e_865_11 := arithEq_of_rows h (row := 865) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_11
  simp only [← c53925, ← c53926, k2, ← c53927] at e_865_11
  have hr := e_865_11
  simp only [e_865_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6501)) := by
  have c53928 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.1
  have c53929 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.1
  have c53930 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53931 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53932 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53933 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_12 := arithEq_of_rows h (row := 865) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_12
  simp only [← c53928, ← c53929, ← c53930] at e_865_12
  have e_865_13 := arithEq_of_rows h (row := 865) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_13
  simp only [← c53931, ← c53932, k2, ← c53933] at e_865_13
  have hr := e_865_13
  simp only [e_865_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6502)) := by
  have c53934 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53935 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53936 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53937 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53938 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53939 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_14 := arithEq_of_rows h (row := 865) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_14
  simp only [← c53934, ← c53935, ← c53936] at e_865_14
  have e_865_15 := arithEq_of_rows h (row := 865) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_15
  simp only [← c53937, ← c53938, k2, ← c53939] at e_865_15
  have hr := e_865_15
  simp only [e_865_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6503)) := by
  have c53940 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53941 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53942 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53943 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53944 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53945 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_16 := arithEq_of_rows h (row := 865) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_16
  simp only [← c53940, ← c53941, ← c53942] at e_865_16
  have e_865_17 := arithEq_of_rows h (row := 865) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_17
  simp only [← c53943, ← c53944, k2, ← c53945] at e_865_17
  have hr := e_865_17
  simp only [e_865_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6504)) := by
  have c53946 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53947 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53948 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53949 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53950 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53951 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_18 := arithEq_of_rows h (row := 865) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_18
  simp only [← c53946, ← c53947, ← c53948] at e_865_18
  have e_865_19 := arithEq_of_rows h (row := 865) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_19
  simp only [← c53949, ← c53950, k2, ← c53951] at e_865_19
  have hr := e_865_19
  simp only [e_865_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6505)) := by
  have c53952 := (publicBatchWrapper89_copies1686 a h).1
  have c53953 := (publicBatchWrapper89_copies1686 a h).2.1
  have c53954 := (publicBatchWrapper89_copies1686 a h).2.2.1
  have c53955 := (publicBatchWrapper89_copies1686 a h).2.2.2.1
  have c53956 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.1
  have c53957 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_0 := arithEq_of_rows h (row := 866) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_0
  simp only [← c53952, ← c53953, ← c53954] at e_866_0
  have e_866_1 := arithEq_of_rows h (row := 866) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_1
  simp only [← c53955, ← c53956, k2, ← c53957] at e_866_1
  have hr := e_866_1
  simp only [e_866_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6506)) := by
  have c53958 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.1
  have c53959 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.1
  have c53960 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.1
  have c53961 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.1
  have c53962 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53963 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_2 := arithEq_of_rows h (row := 866) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_2
  simp only [← c53958, ← c53959, ← c53960] at e_866_2
  have e_866_3 := arithEq_of_rows h (row := 866) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_3
  simp only [← c53961, ← c53962, k2, ← c53963] at e_866_3
  have hr := e_866_3
  simp only [e_866_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6507)) := by
  have c53964 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53965 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53966 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53967 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53968 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53969 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_4 := arithEq_of_rows h (row := 866) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_4
  simp only [← c53964, ← c53965, ← c53966] at e_866_4
  have e_866_5 := arithEq_of_rows h (row := 866) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_5
  simp only [← c53967, ← c53968, k2, ← c53969] at e_866_5
  have hr := e_866_5
  simp only [e_866_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6508)) := by
  have c53970 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53971 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53972 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53973 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53974 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53975 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_6 := arithEq_of_rows h (row := 866) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_6
  simp only [← c53970, ← c53971, ← c53972] at e_866_6
  have e_866_7 := arithEq_of_rows h (row := 866) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_7
  simp only [← c53973, ← c53974, k2, ← c53975] at e_866_7
  have hr := e_866_7
  simp only [e_866_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6509)) := by
  have c53976 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53977 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53978 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53979 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53980 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53981 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_8 := arithEq_of_rows h (row := 866) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_8
  simp only [← c53976, ← c53977, ← c53978] at e_866_8
  have e_866_9 := arithEq_of_rows h (row := 866) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_9
  simp only [← c53979, ← c53980, k2, ← c53981] at e_866_9
  have hr := e_866_9
  simp only [e_866_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6510)) := by
  have c53982 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53983 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53984 := (publicBatchWrapper89_copies1687 a h).1
  have c53985 := (publicBatchWrapper89_copies1687 a h).2.1
  have c53986 := (publicBatchWrapper89_copies1687 a h).2.2.1
  have c53987 := (publicBatchWrapper89_copies1687 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_10 := arithEq_of_rows h (row := 866) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_10
  simp only [← c53982, ← c53983, ← c53984] at e_866_10
  have e_866_11 := arithEq_of_rows h (row := 866) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_11
  simp only [← c53985, ← c53986, k2, ← c53987] at e_866_11
  have hr := e_866_11
  simp only [e_866_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6511)) := by
  have c53988 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.1
  have c53989 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.1
  have c53990 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.1
  have c53991 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.1
  have c53992 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.1
  have c53993 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_12 := arithEq_of_rows h (row := 866) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_12
  simp only [← c53988, ← c53989, ← c53990] at e_866_12
  have e_866_13 := arithEq_of_rows h (row := 866) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_13
  simp only [← c53991, ← c53992, k2, ← c53993] at e_866_13
  have hr := e_866_13
  simp only [e_866_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6512)) := by
  have c53994 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53995 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53996 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53997 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53998 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53999 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_14 := arithEq_of_rows h (row := 866) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_14
  simp only [← c53994, ← c53995, ← c53996] at e_866_14
  have e_866_15 := arithEq_of_rows h (row := 866) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_15
  simp only [← c53997, ← c53998, k2, ← c53999] at e_866_15
  have hr := e_866_15
  simp only [e_866_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6513)) := by
  have c54000 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54001 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54002 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54003 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54004 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54005 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_16 := arithEq_of_rows h (row := 866) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_16
  simp only [← c54000, ← c54001, ← c54002] at e_866_16
  have e_866_17 := arithEq_of_rows h (row := 866) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_17
  simp only [← c54003, ← c54004, k2, ← c54005] at e_866_17
  have hr := e_866_17
  simp only [e_866_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6514)) := by
  have c54006 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54007 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54008 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54009 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54010 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54011 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_18 := arithEq_of_rows h (row := 866) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_18
  simp only [← c54006, ← c54007, ← c54008] at e_866_18
  have e_866_19 := arithEq_of_rows h (row := 866) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_19
  simp only [← c54009, ← c54010, k2, ← c54011] at e_866_19
  have hr := e_866_19
  simp only [e_866_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6515)) := by
  have c54012 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54013 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54014 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54015 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54016 := (publicBatchWrapper89_copies1688 a h).1
  have c54017 := (publicBatchWrapper89_copies1688 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_0 := arithEq_of_rows h (row := 867) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_0
  simp only [← c54012, ← c54013, ← c54014] at e_867_0
  have e_867_1 := arithEq_of_rows h (row := 867) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_1
  simp only [← c54015, ← c54016, k2, ← c54017] at e_867_1
  have hr := e_867_1
  simp only [e_867_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6516)) := by
  have c54018 := (publicBatchWrapper89_copies1688 a h).2.2.1
  have c54019 := (publicBatchWrapper89_copies1688 a h).2.2.2.1
  have c54020 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.1
  have c54021 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.1
  have c54022 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.1
  have c54023 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_2 := arithEq_of_rows h (row := 867) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_2
  simp only [← c54018, ← c54019, ← c54020] at e_867_2
  have e_867_3 := arithEq_of_rows h (row := 867) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_3
  simp only [← c54021, ← c54022, k2, ← c54023] at e_867_3
  have hr := e_867_3
  simp only [e_867_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6517)) := by
  have c54024 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.1
  have c54025 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.1
  have c54026 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54027 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54028 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54029 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_4 := arithEq_of_rows h (row := 867) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_4
  simp only [← c54024, ← c54025, ← c54026] at e_867_4
  have e_867_5 := arithEq_of_rows h (row := 867) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_5
  simp only [← c54027, ← c54028, k2, ← c54029] at e_867_5
  have hr := e_867_5
  simp only [e_867_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6518)) := by
  have c54030 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54031 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54032 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54033 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54034 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54035 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_6 := arithEq_of_rows h (row := 867) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_6
  simp only [← c54030, ← c54031, ← c54032] at e_867_6
  have e_867_7 := arithEq_of_rows h (row := 867) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_7
  simp only [← c54033, ← c54034, k2, ← c54035] at e_867_7
  have hr := e_867_7
  simp only [e_867_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6519)) := by
  have c54036 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54037 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54038 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54039 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54040 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54041 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_8 := arithEq_of_rows h (row := 867) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_8
  simp only [← c54036, ← c54037, ← c54038] at e_867_8
  have e_867_9 := arithEq_of_rows h (row := 867) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_9
  simp only [← c54039, ← c54040, k2, ← c54041] at e_867_9
  have hr := e_867_9
  simp only [e_867_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6520)) := by
  have c54042 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54043 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54044 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54045 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54046 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54047 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_10 := arithEq_of_rows h (row := 867) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_10
  simp only [← c54042, ← c54043, ← c54044] at e_867_10
  have e_867_11 := arithEq_of_rows h (row := 867) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_11
  simp only [← c54045, ← c54046, k2, ← c54047] at e_867_11
  have hr := e_867_11
  simp only [e_867_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6521)) := by
  have c54048 := (publicBatchWrapper89_copies1689 a h).1
  have c54049 := (publicBatchWrapper89_copies1689 a h).2.1
  have c54050 := (publicBatchWrapper89_copies1689 a h).2.2.1
  have c54051 := (publicBatchWrapper89_copies1689 a h).2.2.2.1
  have c54052 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.1
  have c54053 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_12 := arithEq_of_rows h (row := 867) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_12
  simp only [← c54048, ← c54049, ← c54050] at e_867_12
  have e_867_13 := arithEq_of_rows h (row := 867) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_13
  simp only [← c54051, ← c54052, k2, ← c54053] at e_867_13
  have hr := e_867_13
  simp only [e_867_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6522)) := by
  have c54054 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.1
  have c54055 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.1
  have c54056 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.1
  have c54057 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.1
  have c54058 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54059 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_14 := arithEq_of_rows h (row := 867) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_14
  simp only [← c54054, ← c54055, ← c54056] at e_867_14
  have e_867_15 := arithEq_of_rows h (row := 867) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_15
  simp only [← c54057, ← c54058, k2, ← c54059] at e_867_15
  have hr := e_867_15
  simp only [e_867_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6523)) := by
  have c54060 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54061 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54062 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54063 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54064 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54065 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_16 := arithEq_of_rows h (row := 867) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_16
  simp only [← c54060, ← c54061, ← c54062] at e_867_16
  have e_867_17 := arithEq_of_rows h (row := 867) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_17
  simp only [← c54063, ← c54064, k2, ← c54065] at e_867_17
  have hr := e_867_17
  simp only [e_867_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6524)) := by
  have c54066 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54067 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54068 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54069 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54070 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54071 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_18 := arithEq_of_rows h (row := 867) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_18
  simp only [← c54066, ← c54067, ← c54068] at e_867_18
  have e_867_19 := arithEq_of_rows h (row := 867) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_19
  simp only [← c54069, ← c54070, k2, ← c54071] at e_867_19
  have hr := e_867_19
  simp only [e_867_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6525)) := by
  have c54072 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54073 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54074 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54075 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54076 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54077 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_0 := arithEq_of_rows h (row := 868) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_0
  simp only [← c54072, ← c54073, ← c54074] at e_868_0
  have e_868_1 := arithEq_of_rows h (row := 868) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_1
  simp only [← c54075, ← c54076, k2, ← c54077] at e_868_1
  have hr := e_868_1
  simp only [e_868_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6526)) := by
  have c54078 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54079 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54080 := (publicBatchWrapper89_copies1690 a h).1
  have c54081 := (publicBatchWrapper89_copies1690 a h).2.1
  have c54082 := (publicBatchWrapper89_copies1690 a h).2.2.1
  have c54083 := (publicBatchWrapper89_copies1690 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_2 := arithEq_of_rows h (row := 868) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_2
  simp only [← c54078, ← c54079, ← c54080] at e_868_2
  have e_868_3 := arithEq_of_rows h (row := 868) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_3
  simp only [← c54081, ← c54082, k2, ← c54083] at e_868_3
  have hr := e_868_3
  simp only [e_868_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6527)) := by
  have c54084 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.1
  have c54085 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.1
  have c54086 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.1
  have c54087 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.1
  have c54088 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.1
  have c54089 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_4 := arithEq_of_rows h (row := 868) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_4
  simp only [← c54084, ← c54085, ← c54086] at e_868_4
  have e_868_5 := arithEq_of_rows h (row := 868) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_5
  simp only [← c54087, ← c54088, k2, ← c54089] at e_868_5
  have hr := e_868_5
  simp only [e_868_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6528)) := by
  have c54090 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54091 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54092 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54093 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54094 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54095 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_6 := arithEq_of_rows h (row := 868) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_6
  simp only [← c54090, ← c54091, ← c54092] at e_868_6
  have e_868_7 := arithEq_of_rows h (row := 868) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_7
  simp only [← c54093, ← c54094, k2, ← c54095] at e_868_7
  have hr := e_868_7
  simp only [e_868_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6529)) := by
  have c54096 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54097 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54098 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54099 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54100 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54101 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_8 := arithEq_of_rows h (row := 868) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_8
  simp only [← c54096, ← c54097, ← c54098] at e_868_8
  have e_868_9 := arithEq_of_rows h (row := 868) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_9
  simp only [← c54099, ← c54100, k2, ← c54101] at e_868_9
  have hr := e_868_9
  simp only [e_868_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6530)) := by
  have c54102 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54103 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54104 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54105 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54106 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54107 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_10 := arithEq_of_rows h (row := 868) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_10
  simp only [← c54102, ← c54103, ← c54104] at e_868_10
  have e_868_11 := arithEq_of_rows h (row := 868) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_11
  simp only [← c54105, ← c54106, k2, ← c54107] at e_868_11
  have hr := e_868_11
  simp only [e_868_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6531)) := by
  have c54108 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54109 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54110 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54111 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54112 := (publicBatchWrapper89_copies1691 a h).1
  have c54113 := (publicBatchWrapper89_copies1691 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_12 := arithEq_of_rows h (row := 868) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_12
  simp only [← c54108, ← c54109, ← c54110] at e_868_12
  have e_868_13 := arithEq_of_rows h (row := 868) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_13
  simp only [← c54111, ← c54112, k2, ← c54113] at e_868_13
  have hr := e_868_13
  simp only [e_868_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6532)) := by
  have c54114 := (publicBatchWrapper89_copies1691 a h).2.2.1
  have c54115 := (publicBatchWrapper89_copies1691 a h).2.2.2.1
  have c54116 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.1
  have c54117 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.1
  have c54118 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.1
  have c54119 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_14 := arithEq_of_rows h (row := 868) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_14
  simp only [← c54114, ← c54115, ← c54116] at e_868_14
  have e_868_15 := arithEq_of_rows h (row := 868) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_15
  simp only [← c54117, ← c54118, k2, ← c54119] at e_868_15
  have hr := e_868_15
  simp only [e_868_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6533)) := by
  have c54120 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.1
  have c54121 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.1
  have c54122 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54123 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54124 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54125 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_16 := arithEq_of_rows h (row := 868) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_16
  simp only [← c54120, ← c54121, ← c54122] at e_868_16
  have e_868_17 := arithEq_of_rows h (row := 868) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_17
  simp only [← c54123, ← c54124, k2, ← c54125] at e_868_17
  have hr := e_868_17
  simp only [e_868_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6534)) := by
  have c54126 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54127 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54128 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54129 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54130 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54131 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_18 := arithEq_of_rows h (row := 868) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_18
  simp only [← c54126, ← c54127, ← c54128] at e_868_18
  have e_868_19 := arithEq_of_rows h (row := 868) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_19
  simp only [← c54129, ← c54130, k2, ← c54131] at e_868_19
  have hr := e_868_19
  simp only [e_868_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6535)) := by
  have c54132 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54133 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54134 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54135 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54136 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54137 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_0 := arithEq_of_rows h (row := 869) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_0
  simp only [← c54132, ← c54133, ← c54134] at e_869_0
  have e_869_1 := arithEq_of_rows h (row := 869) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_1
  simp only [← c54135, ← c54136, k2, ← c54137] at e_869_1
  have hr := e_869_1
  simp only [e_869_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6536)) := by
  have c54138 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54139 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54140 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54141 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54142 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54143 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_2 := arithEq_of_rows h (row := 869) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_2
  simp only [← c54138, ← c54139, ← c54140] at e_869_2
  have e_869_3 := arithEq_of_rows h (row := 869) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_3
  simp only [← c54141, ← c54142, k2, ← c54143] at e_869_3
  have hr := e_869_3
  simp only [e_869_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6537)) := by
  have c54144 := (publicBatchWrapper89_copies1692 a h).1
  have c54145 := (publicBatchWrapper89_copies1692 a h).2.1
  have c54146 := (publicBatchWrapper89_copies1692 a h).2.2.1
  have c54147 := (publicBatchWrapper89_copies1692 a h).2.2.2.1
  have c54148 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.1
  have c54149 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_4 := arithEq_of_rows h (row := 869) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_4
  simp only [← c54144, ← c54145, ← c54146] at e_869_4
  have e_869_5 := arithEq_of_rows h (row := 869) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_5
  simp only [← c54147, ← c54148, k2, ← c54149] at e_869_5
  have hr := e_869_5
  simp only [e_869_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6538)) := by
  have c54150 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.1
  have c54151 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.1
  have c54152 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.1
  have c54153 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.1
  have c54154 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54155 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_6 := arithEq_of_rows h (row := 869) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_6
  simp only [← c54150, ← c54151, ← c54152] at e_869_6
  have e_869_7 := arithEq_of_rows h (row := 869) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_7
  simp only [← c54153, ← c54154, k2, ← c54155] at e_869_7
  have hr := e_869_7
  simp only [e_869_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6539)) := by
  have c54156 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54157 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54158 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54159 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54160 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54161 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_8 := arithEq_of_rows h (row := 869) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_8
  simp only [← c54156, ← c54157, ← c54158] at e_869_8
  have e_869_9 := arithEq_of_rows h (row := 869) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_9
  simp only [← c54159, ← c54160, k2, ← c54161] at e_869_9
  have hr := e_869_9
  simp only [e_869_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6540)) := by
  have c54162 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54163 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54164 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54165 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54166 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54167 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_10 := arithEq_of_rows h (row := 869) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_10
  simp only [← c54162, ← c54163, ← c54164] at e_869_10
  have e_869_11 := arithEq_of_rows h (row := 869) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_11
  simp only [← c54165, ← c54166, k2, ← c54167] at e_869_11
  have hr := e_869_11
  simp only [e_869_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6541)) := by
  have c54168 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54169 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54170 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54171 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54172 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54173 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_12 := arithEq_of_rows h (row := 869) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_12
  simp only [← c54168, ← c54169, ← c54170] at e_869_12
  have e_869_13 := arithEq_of_rows h (row := 869) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_13
  simp only [← c54171, ← c54172, k2, ← c54173] at e_869_13
  have hr := e_869_13
  simp only [e_869_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6542)) := by
  have c54174 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54175 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54176 := (publicBatchWrapper89_copies1693 a h).1
  have c54177 := (publicBatchWrapper89_copies1693 a h).2.1
  have c54178 := (publicBatchWrapper89_copies1693 a h).2.2.1
  have c54179 := (publicBatchWrapper89_copies1693 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_14 := arithEq_of_rows h (row := 869) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_14
  simp only [← c54174, ← c54175, ← c54176] at e_869_14
  have e_869_15 := arithEq_of_rows h (row := 869) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_15
  simp only [← c54177, ← c54178, k2, ← c54179] at e_869_15
  have hr := e_869_15
  simp only [e_869_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6543)) := by
  have c54180 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.1
  have c54181 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.1
  have c54182 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.1
  have c54183 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.1
  have c54184 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.1
  have c54185 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_16 := arithEq_of_rows h (row := 869) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_16
  simp only [← c54180, ← c54181, ← c54182] at e_869_16
  have e_869_17 := arithEq_of_rows h (row := 869) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_17
  simp only [← c54183, ← c54184, k2, ← c54185] at e_869_17
  have hr := e_869_17
  simp only [e_869_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6544)) := by
  have c54186 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54187 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54188 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54189 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54190 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54191 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_18 := arithEq_of_rows h (row := 869) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_18
  simp only [← c54186, ← c54187, ← c54188] at e_869_18
  have e_869_19 := arithEq_of_rows h (row := 869) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_19
  simp only [← c54189, ← c54190, k2, ← c54191] at e_869_19
  have hr := e_869_19
  simp only [e_869_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6545)) := by
  have c54192 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54193 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54194 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54195 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54196 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54197 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_0 := arithEq_of_rows h (row := 870) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_0
  simp only [← c54192, ← c54193, ← c54194] at e_870_0
  have e_870_1 := arithEq_of_rows h (row := 870) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_1
  simp only [← c54195, ← c54196, k2, ← c54197] at e_870_1
  have hr := e_870_1
  simp only [e_870_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6546)) := by
  have c54198 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54199 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54200 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54201 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54202 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54203 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_2 := arithEq_of_rows h (row := 870) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_2
  simp only [← c54198, ← c54199, ← c54200] at e_870_2
  have e_870_3 := arithEq_of_rows h (row := 870) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_3
  simp only [← c54201, ← c54202, k2, ← c54203] at e_870_3
  have hr := e_870_3
  simp only [e_870_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6547)) := by
  have c54204 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54205 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54206 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54207 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54208 := (publicBatchWrapper89_copies1694 a h).1
  have c54209 := (publicBatchWrapper89_copies1694 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_4 := arithEq_of_rows h (row := 870) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_4
  simp only [← c54204, ← c54205, ← c54206] at e_870_4
  have e_870_5 := arithEq_of_rows h (row := 870) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_5
  simp only [← c54207, ← c54208, k2, ← c54209] at e_870_5
  have hr := e_870_5
  simp only [e_870_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6548)) := by
  have c54210 := (publicBatchWrapper89_copies1694 a h).2.2.1
  have c54211 := (publicBatchWrapper89_copies1694 a h).2.2.2.1
  have c54212 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.1
  have c54213 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.1
  have c54214 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.1
  have c54215 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_6 := arithEq_of_rows h (row := 870) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_6
  simp only [← c54210, ← c54211, ← c54212] at e_870_6
  have e_870_7 := arithEq_of_rows h (row := 870) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_7
  simp only [← c54213, ← c54214, k2, ← c54215] at e_870_7
  have hr := e_870_7
  simp only [e_870_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6549)) := by
  have c54216 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.1
  have c54217 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.1
  have c54218 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54219 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54220 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54221 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_8 := arithEq_of_rows h (row := 870) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_8
  simp only [← c54216, ← c54217, ← c54218] at e_870_8
  have e_870_9 := arithEq_of_rows h (row := 870) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_9
  simp only [← c54219, ← c54220, k2, ← c54221] at e_870_9
  have hr := e_870_9
  simp only [e_870_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6550)) := by
  have c54222 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54223 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54224 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54225 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54226 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54227 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_10 := arithEq_of_rows h (row := 870) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_10
  simp only [← c54222, ← c54223, ← c54224] at e_870_10
  have e_870_11 := arithEq_of_rows h (row := 870) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_11
  simp only [← c54225, ← c54226, k2, ← c54227] at e_870_11
  have hr := e_870_11
  simp only [e_870_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6551)) := by
  have c54228 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54229 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54230 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54231 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54232 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54233 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_12 := arithEq_of_rows h (row := 870) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_12
  simp only [← c54228, ← c54229, ← c54230] at e_870_12
  have e_870_13 := arithEq_of_rows h (row := 870) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_13
  simp only [← c54231, ← c54232, k2, ← c54233] at e_870_13
  have hr := e_870_13
  simp only [e_870_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6552)) := by
  have c54234 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54235 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54236 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54237 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54238 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54239 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_14 := arithEq_of_rows h (row := 870) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_14
  simp only [← c54234, ← c54235, ← c54236] at e_870_14
  have e_870_15 := arithEq_of_rows h (row := 870) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_15
  simp only [← c54237, ← c54238, k2, ← c54239] at e_870_15
  have hr := e_870_15
  simp only [e_870_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6553)) := by
  have c54240 := (publicBatchWrapper89_copies1695 a h).1
  have c54241 := (publicBatchWrapper89_copies1695 a h).2.1
  have c54242 := (publicBatchWrapper89_copies1695 a h).2.2.1
  have c54243 := (publicBatchWrapper89_copies1695 a h).2.2.2.1
  have c54244 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.1
  have c54245 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_16 := arithEq_of_rows h (row := 870) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_16
  simp only [← c54240, ← c54241, ← c54242] at e_870_16
  have e_870_17 := arithEq_of_rows h (row := 870) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_17
  simp only [← c54243, ← c54244, k2, ← c54245] at e_870_17
  have hr := e_870_17
  simp only [e_870_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6554)) := by
  have c54246 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.1
  have c54247 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.1
  have c54248 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.1
  have c54249 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.1
  have c54250 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54251 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_18 := arithEq_of_rows h (row := 870) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_18
  simp only [← c54246, ← c54247, ← c54248] at e_870_18
  have e_870_19 := arithEq_of_rows h (row := 870) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_19
  simp only [← c54249, ← c54250, k2, ← c54251] at e_870_19
  have hr := e_870_19
  simp only [e_870_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6555)) := by
  have c54252 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54253 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54254 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54255 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54256 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54257 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_0 := arithEq_of_rows h (row := 871) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_0
  simp only [← c54252, ← c54253, ← c54254] at e_871_0
  have e_871_1 := arithEq_of_rows h (row := 871) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_1
  simp only [← c54255, ← c54256, k2, ← c54257] at e_871_1
  have hr := e_871_1
  simp only [e_871_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6556)) := by
  have c54258 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54259 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54260 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54261 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54262 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54263 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_2 := arithEq_of_rows h (row := 871) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_2
  simp only [← c54258, ← c54259, ← c54260] at e_871_2
  have e_871_3 := arithEq_of_rows h (row := 871) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_3
  simp only [← c54261, ← c54262, k2, ← c54263] at e_871_3
  have hr := e_871_3
  simp only [e_871_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6557)) := by
  have c54264 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54265 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54266 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54267 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54268 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54269 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_4 := arithEq_of_rows h (row := 871) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_4
  simp only [← c54264, ← c54265, ← c54266] at e_871_4
  have e_871_5 := arithEq_of_rows h (row := 871) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_5
  simp only [← c54267, ← c54268, k2, ← c54269] at e_871_5
  have hr := e_871_5
  simp only [e_871_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6558)) := by
  have c54270 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54271 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54272 := (publicBatchWrapper89_copies1696 a h).1
  have c54273 := (publicBatchWrapper89_copies1696 a h).2.1
  have c54274 := (publicBatchWrapper89_copies1696 a h).2.2.1
  have c54275 := (publicBatchWrapper89_copies1696 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_6 := arithEq_of_rows h (row := 871) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_6
  simp only [← c54270, ← c54271, ← c54272] at e_871_6
  have e_871_7 := arithEq_of_rows h (row := 871) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_7
  simp only [← c54273, ← c54274, k2, ← c54275] at e_871_7
  have hr := e_871_7
  simp only [e_871_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6559)) := by
  have c54276 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.1
  have c54277 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.1
  have c54278 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.1
  have c54279 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.1
  have c54280 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.1
  have c54281 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_8 := arithEq_of_rows h (row := 871) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_8
  simp only [← c54276, ← c54277, ← c54278] at e_871_8
  have e_871_9 := arithEq_of_rows h (row := 871) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_9
  simp only [← c54279, ← c54280, k2, ← c54281] at e_871_9
  have hr := e_871_9
  simp only [e_871_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6560)) := by
  have c54282 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54283 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54284 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54285 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54286 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54287 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_10 := arithEq_of_rows h (row := 871) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_10
  simp only [← c54282, ← c54283, ← c54284] at e_871_10
  have e_871_11 := arithEq_of_rows h (row := 871) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_11
  simp only [← c54285, ← c54286, k2, ← c54287] at e_871_11
  have hr := e_871_11
  simp only [e_871_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6561)) := by
  have c54288 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54289 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54290 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54291 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54292 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54293 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_12 := arithEq_of_rows h (row := 871) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_12
  simp only [← c54288, ← c54289, ← c54290] at e_871_12
  have e_871_13 := arithEq_of_rows h (row := 871) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_13
  simp only [← c54291, ← c54292, k2, ← c54293] at e_871_13
  have hr := e_871_13
  simp only [e_871_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6562)) := by
  have c54294 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54295 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54296 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54297 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54298 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54299 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_14 := arithEq_of_rows h (row := 871) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_14
  simp only [← c54294, ← c54295, ← c54296] at e_871_14
  have e_871_15 := arithEq_of_rows h (row := 871) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_15
  simp only [← c54297, ← c54298, k2, ← c54299] at e_871_15
  have hr := e_871_15
  simp only [e_871_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6563)) := by
  have c54300 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54301 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54302 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54303 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54304 := (publicBatchWrapper89_copies1697 a h).1
  have c54305 := (publicBatchWrapper89_copies1697 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_16 := arithEq_of_rows h (row := 871) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_16
  simp only [← c54300, ← c54301, ← c54302] at e_871_16
  have e_871_17 := arithEq_of_rows h (row := 871) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_17
  simp only [← c54303, ← c54304, k2, ← c54305] at e_871_17
  have hr := e_871_17
  simp only [e_871_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6564)) := by
  have c54306 := (publicBatchWrapper89_copies1697 a h).2.2.1
  have c54307 := (publicBatchWrapper89_copies1697 a h).2.2.2.1
  have c54308 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.1
  have c54309 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.1
  have c54310 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.1
  have c54311 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_18 := arithEq_of_rows h (row := 871) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_18
  simp only [← c54306, ← c54307, ← c54308] at e_871_18
  have e_871_19 := arithEq_of_rows h (row := 871) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_19
  simp only [← c54309, ← c54310, k2, ← c54311] at e_871_19
  have hr := e_871_19
  simp only [e_871_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6565)) := by
  have c54312 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.1
  have c54313 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.1
  have c54314 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54315 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54316 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54317 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_0 := arithEq_of_rows h (row := 872) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_0
  simp only [← c54312, ← c54313, ← c54314] at e_872_0
  have e_872_1 := arithEq_of_rows h (row := 872) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_1
  simp only [← c54315, ← c54316, k2, ← c54317] at e_872_1
  have hr := e_872_1
  simp only [e_872_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6566)) := by
  have c54318 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54319 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54320 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54321 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54322 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54323 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_2 := arithEq_of_rows h (row := 872) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_2
  simp only [← c54318, ← c54319, ← c54320] at e_872_2
  have e_872_3 := arithEq_of_rows h (row := 872) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_3
  simp only [← c54321, ← c54322, k2, ← c54323] at e_872_3
  have hr := e_872_3
  simp only [e_872_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6567)) := by
  have c54324 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54325 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54326 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54327 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54328 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54329 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_4 := arithEq_of_rows h (row := 872) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_4
  simp only [← c54324, ← c54325, ← c54326] at e_872_4
  have e_872_5 := arithEq_of_rows h (row := 872) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_5
  simp only [← c54327, ← c54328, k2, ← c54329] at e_872_5
  have hr := e_872_5
  simp only [e_872_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6568)) := by
  have c54330 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54331 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54332 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54333 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54334 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54335 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_6 := arithEq_of_rows h (row := 872) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_6
  simp only [← c54330, ← c54331, ← c54332] at e_872_6
  have e_872_7 := arithEq_of_rows h (row := 872) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_7
  simp only [← c54333, ← c54334, k2, ← c54335] at e_872_7
  have hr := e_872_7
  simp only [e_872_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6569)) := by
  have c54336 := (publicBatchWrapper89_copies1698 a h).1
  have c54337 := (publicBatchWrapper89_copies1698 a h).2.1
  have c54338 := (publicBatchWrapper89_copies1698 a h).2.2.1
  have c54339 := (publicBatchWrapper89_copies1698 a h).2.2.2.1
  have c54340 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.1
  have c54341 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_8 := arithEq_of_rows h (row := 872) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_8
  simp only [← c54336, ← c54337, ← c54338] at e_872_8
  have e_872_9 := arithEq_of_rows h (row := 872) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_9
  simp only [← c54339, ← c54340, k2, ← c54341] at e_872_9
  have hr := e_872_9
  simp only [e_872_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6610)) := by
  have c54342 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.1
  have c54343 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.1
  have c54344 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.1
  have c54345 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.1
  have c54346 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54347 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_10 := arithEq_of_rows h (row := 872) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_10
  simp only [← c54342, ← c54343, ← c54344] at e_872_10
  have e_872_11 := arithEq_of_rows h (row := 872) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_11
  simp only [← c54345, ← c54346, k2, ← c54347] at e_872_11
  have hr := e_872_11
  simp only [e_872_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6611)) := by
  have c54348 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54349 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54350 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54351 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54352 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54353 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_12 := arithEq_of_rows h (row := 872) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_12
  simp only [← c54348, ← c54349, ← c54350] at e_872_12
  have e_872_13 := arithEq_of_rows h (row := 872) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_13
  simp only [← c54351, ← c54352, k2, ← c54353] at e_872_13
  have hr := e_872_13
  simp only [e_872_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6612)) := by
  have c54354 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54355 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54356 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54357 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54358 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54359 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_14 := arithEq_of_rows h (row := 872) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_14
  simp only [← c54354, ← c54355, ← c54356] at e_872_14
  have e_872_15 := arithEq_of_rows h (row := 872) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_15
  simp only [← c54357, ← c54358, k2, ← c54359] at e_872_15
  have hr := e_872_15
  simp only [e_872_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6613)) := by
  have c54360 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54361 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54362 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54363 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54364 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54365 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_16 := arithEq_of_rows h (row := 872) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_16
  simp only [← c54360, ← c54361, ← c54362] at e_872_16
  have e_872_17 := arithEq_of_rows h (row := 872) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_17
  simp only [← c54363, ← c54364, k2, ← c54365] at e_872_17
  have hr := e_872_17
  simp only [e_872_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6614)) := by
  have c54366 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54367 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54368 := (publicBatchWrapper89_copies1699 a h).1
  have c54369 := (publicBatchWrapper89_copies1699 a h).2.1
  have c54370 := (publicBatchWrapper89_copies1699 a h).2.2.1
  have c54371 := (publicBatchWrapper89_copies1699 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_18 := arithEq_of_rows h (row := 872) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_18
  simp only [← c54366, ← c54367, ← c54368] at e_872_18
  have e_872_19 := arithEq_of_rows h (row := 872) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_19
  simp only [← c54369, ← c54370, k2, ← c54371] at e_872_19
  have hr := e_872_19
  simp only [e_872_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6615)) := by
  have c54372 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.1
  have c54373 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.1
  have c54374 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.1
  have c54375 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.1
  have c54376 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.1
  have c54377 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_0 := arithEq_of_rows h (row := 873) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_0
  simp only [← c54372, ← c54373, ← c54374] at e_873_0
  have e_873_1 := arithEq_of_rows h (row := 873) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_1
  simp only [← c54375, ← c54376, k2, ← c54377] at e_873_1
  have hr := e_873_1
  simp only [e_873_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6616)) := by
  have c54378 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54379 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54380 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54381 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54382 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54383 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_2 := arithEq_of_rows h (row := 873) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_2
  simp only [← c54378, ← c54379, ← c54380] at e_873_2
  have e_873_3 := arithEq_of_rows h (row := 873) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_3
  simp only [← c54381, ← c54382, k2, ← c54383] at e_873_3
  have hr := e_873_3
  simp only [e_873_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6617)) := by
  have c54384 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54385 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54386 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54387 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54388 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54389 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_4 := arithEq_of_rows h (row := 873) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_4
  simp only [← c54384, ← c54385, ← c54386] at e_873_4
  have e_873_5 := arithEq_of_rows h (row := 873) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_5
  simp only [← c54387, ← c54388, k2, ← c54389] at e_873_5
  have hr := e_873_5
  simp only [e_873_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6618)) := by
  have c54390 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54391 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54392 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54393 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54394 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54395 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_6 := arithEq_of_rows h (row := 873) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_6
  simp only [← c54390, ← c54391, ← c54392] at e_873_6
  have e_873_7 := arithEq_of_rows h (row := 873) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_7
  simp only [← c54393, ← c54394, k2, ← c54395] at e_873_7
  have hr := e_873_7
  simp only [e_873_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6619)) := by
  have c54396 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54397 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54398 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54399 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54400 := (publicBatchWrapper89_copies1700 a h).1
  have c54401 := (publicBatchWrapper89_copies1700 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_8 := arithEq_of_rows h (row := 873) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_8
  simp only [← c54396, ← c54397, ← c54398] at e_873_8
  have e_873_9 := arithEq_of_rows h (row := 873) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_9
  simp only [← c54399, ← c54400, k2, ← c54401] at e_873_9
  have hr := e_873_9
  simp only [e_873_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6620)) := by
  have c54402 := (publicBatchWrapper89_copies1700 a h).2.2.1
  have c54403 := (publicBatchWrapper89_copies1700 a h).2.2.2.1
  have c54404 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.1
  have c54405 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.1
  have c54406 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.1
  have c54407 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_10 := arithEq_of_rows h (row := 873) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_10
  simp only [← c54402, ← c54403, ← c54404] at e_873_10
  have e_873_11 := arithEq_of_rows h (row := 873) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_11
  simp only [← c54405, ← c54406, k2, ← c54407] at e_873_11
  have hr := e_873_11
  simp only [e_873_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6621)) := by
  have c54408 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.1
  have c54409 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.1
  have c54410 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54411 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54412 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54413 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_12 := arithEq_of_rows h (row := 873) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_12
  simp only [← c54408, ← c54409, ← c54410] at e_873_12
  have e_873_13 := arithEq_of_rows h (row := 873) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_13
  simp only [← c54411, ← c54412, k2, ← c54413] at e_873_13
  have hr := e_873_13
  simp only [e_873_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6622)) := by
  have c54414 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54415 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54416 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54417 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54418 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54419 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_14 := arithEq_of_rows h (row := 873) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_14
  simp only [← c54414, ← c54415, ← c54416] at e_873_14
  have e_873_15 := arithEq_of_rows h (row := 873) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_15
  simp only [← c54417, ← c54418, k2, ← c54419] at e_873_15
  have hr := e_873_15
  simp only [e_873_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6623)) := by
  have c54420 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54421 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54422 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54423 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54424 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54425 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_16 := arithEq_of_rows h (row := 873) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_16
  simp only [← c54420, ← c54421, ← c54422] at e_873_16
  have e_873_17 := arithEq_of_rows h (row := 873) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_17
  simp only [← c54423, ← c54424, k2, ← c54425] at e_873_17
  have hr := e_873_17
  simp only [e_873_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6624)) := by
  have c54426 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54427 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54428 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54429 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54430 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54431 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_18 := arithEq_of_rows h (row := 873) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_18
  simp only [← c54426, ← c54427, ← c54428] at e_873_18
  have e_873_19 := arithEq_of_rows h (row := 873) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_19
  simp only [← c54429, ← c54430, k2, ← c54431] at e_873_19
  have hr := e_873_19
  simp only [e_873_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6625)) := by
  have c54432 := (publicBatchWrapper89_copies1701 a h).1
  have c54433 := (publicBatchWrapper89_copies1701 a h).2.1
  have c54434 := (publicBatchWrapper89_copies1701 a h).2.2.1
  have c54435 := (publicBatchWrapper89_copies1701 a h).2.2.2.1
  have c54436 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.1
  have c54437 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_0 := arithEq_of_rows h (row := 874) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_0
  simp only [← c54432, ← c54433, ← c54434] at e_874_0
  have e_874_1 := arithEq_of_rows h (row := 874) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_1
  simp only [← c54435, ← c54436, k2, ← c54437] at e_874_1
  have hr := e_874_1
  simp only [e_874_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6626)) := by
  have c54438 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.1
  have c54439 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.1
  have c54440 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.1
  have c54441 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.1
  have c54442 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54443 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_2 := arithEq_of_rows h (row := 874) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_2
  simp only [← c54438, ← c54439, ← c54440] at e_874_2
  have e_874_3 := arithEq_of_rows h (row := 874) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_3
  simp only [← c54441, ← c54442, k2, ← c54443] at e_874_3
  have hr := e_874_3
  simp only [e_874_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6627)) := by
  have c54444 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54445 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54446 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54447 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54448 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54449 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_4 := arithEq_of_rows h (row := 874) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_4
  simp only [← c54444, ← c54445, ← c54446] at e_874_4
  have e_874_5 := arithEq_of_rows h (row := 874) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_5
  simp only [← c54447, ← c54448, k2, ← c54449] at e_874_5
  have hr := e_874_5
  simp only [e_874_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6628)) := by
  have c54450 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54451 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54452 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54453 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54454 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54455 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_6 := arithEq_of_rows h (row := 874) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_6
  simp only [← c54450, ← c54451, ← c54452] at e_874_6
  have e_874_7 := arithEq_of_rows h (row := 874) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_7
  simp only [← c54453, ← c54454, k2, ← c54455] at e_874_7
  have hr := e_874_7
  simp only [e_874_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6629)) := by
  have c54456 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54457 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54458 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54459 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54460 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54461 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_8 := arithEq_of_rows h (row := 874) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_8
  simp only [← c54456, ← c54457, ← c54458] at e_874_8
  have e_874_9 := arithEq_of_rows h (row := 874) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_9
  simp only [← c54459, ← c54460, k2, ← c54461] at e_874_9
  have hr := e_874_9
  simp only [e_874_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6630)) := by
  have c54462 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54463 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54464 := (publicBatchWrapper89_copies1702 a h).1
  have c54465 := (publicBatchWrapper89_copies1702 a h).2.1
  have c54466 := (publicBatchWrapper89_copies1702 a h).2.2.1
  have c54467 := (publicBatchWrapper89_copies1702 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_10 := arithEq_of_rows h (row := 874) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_10
  simp only [← c54462, ← c54463, ← c54464] at e_874_10
  have e_874_11 := arithEq_of_rows h (row := 874) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_11
  simp only [← c54465, ← c54466, k2, ← c54467] at e_874_11
  have hr := e_874_11
  simp only [e_874_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6631)) := by
  have c54468 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.1
  have c54469 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.1
  have c54470 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.1
  have c54471 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.1
  have c54472 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.1
  have c54473 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_12 := arithEq_of_rows h (row := 874) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_12
  simp only [← c54468, ← c54469, ← c54470] at e_874_12
  have e_874_13 := arithEq_of_rows h (row := 874) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_13
  simp only [← c54471, ← c54472, k2, ← c54473] at e_874_13
  have hr := e_874_13
  simp only [e_874_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6632)) := by
  have c54474 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54475 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54476 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54477 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54478 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54479 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_14 := arithEq_of_rows h (row := 874) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_14
  simp only [← c54474, ← c54475, ← c54476] at e_874_14
  have e_874_15 := arithEq_of_rows h (row := 874) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_15
  simp only [← c54477, ← c54478, k2, ← c54479] at e_874_15
  have hr := e_874_15
  simp only [e_874_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6633)) := by
  have c54480 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54481 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54482 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54483 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54484 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54485 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_16 := arithEq_of_rows h (row := 874) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_16
  simp only [← c54480, ← c54481, ← c54482] at e_874_16
  have e_874_17 := arithEq_of_rows h (row := 874) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_17
  simp only [← c54483, ← c54484, k2, ← c54485] at e_874_17
  have hr := e_874_17
  simp only [e_874_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6634)) := by
  have c54486 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54487 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54488 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54489 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54490 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54491 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_18 := arithEq_of_rows h (row := 874) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_18
  simp only [← c54486, ← c54487, ← c54488] at e_874_18
  have e_874_19 := arithEq_of_rows h (row := 874) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_19
  simp only [← c54489, ← c54490, k2, ← c54491] at e_874_19
  have hr := e_874_19
  simp only [e_874_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6635)) := by
  have c54492 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54493 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54494 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54495 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54496 := (publicBatchWrapper89_copies1703 a h).1
  have c54497 := (publicBatchWrapper89_copies1703 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_0 := arithEq_of_rows h (row := 875) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_0
  simp only [← c54492, ← c54493, ← c54494] at e_875_0
  have e_875_1 := arithEq_of_rows h (row := 875) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_1
  simp only [← c54495, ← c54496, k2, ← c54497] at e_875_1
  have hr := e_875_1
  simp only [e_875_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6636)) := by
  have c54498 := (publicBatchWrapper89_copies1703 a h).2.2.1
  have c54499 := (publicBatchWrapper89_copies1703 a h).2.2.2.1
  have c54500 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.1
  have c54501 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.1
  have c54502 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.1
  have c54503 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_2 := arithEq_of_rows h (row := 875) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_2
  simp only [← c54498, ← c54499, ← c54500] at e_875_2
  have e_875_3 := arithEq_of_rows h (row := 875) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_3
  simp only [← c54501, ← c54502, k2, ← c54503] at e_875_3
  have hr := e_875_3
  simp only [e_875_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6637)) := by
  have c54504 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.1
  have c54505 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.1
  have c54506 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54507 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54508 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54509 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_4 := arithEq_of_rows h (row := 875) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_4
  simp only [← c54504, ← c54505, ← c54506] at e_875_4
  have e_875_5 := arithEq_of_rows h (row := 875) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_5
  simp only [← c54507, ← c54508, k2, ← c54509] at e_875_5
  have hr := e_875_5
  simp only [e_875_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6638)) := by
  have c54510 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54511 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54512 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54513 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54514 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54515 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_6 := arithEq_of_rows h (row := 875) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_6
  simp only [← c54510, ← c54511, ← c54512] at e_875_6
  have e_875_7 := arithEq_of_rows h (row := 875) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_7
  simp only [← c54513, ← c54514, k2, ← c54515] at e_875_7
  have hr := e_875_7
  simp only [e_875_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6639)) := by
  have c54516 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54517 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54518 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54519 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54520 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54521 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_8 := arithEq_of_rows h (row := 875) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_8
  simp only [← c54516, ← c54517, ← c54518] at e_875_8
  have e_875_9 := arithEq_of_rows h (row := 875) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_9
  simp only [← c54519, ← c54520, k2, ← c54521] at e_875_9
  have hr := e_875_9
  simp only [e_875_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6640)) := by
  have c54522 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54523 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54524 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54525 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54526 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54527 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_10 := arithEq_of_rows h (row := 875) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_10
  simp only [← c54522, ← c54523, ← c54524] at e_875_10
  have e_875_11 := arithEq_of_rows h (row := 875) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_11
  simp only [← c54525, ← c54526, k2, ← c54527] at e_875_11
  have hr := e_875_11
  simp only [e_875_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6641)) := by
  have c54528 := (publicBatchWrapper89_copies1704 a h).1
  have c54529 := (publicBatchWrapper89_copies1704 a h).2.1
  have c54530 := (publicBatchWrapper89_copies1704 a h).2.2.1
  have c54531 := (publicBatchWrapper89_copies1704 a h).2.2.2.1
  have c54532 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.1
  have c54533 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_12 := arithEq_of_rows h (row := 875) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_12
  simp only [← c54528, ← c54529, ← c54530] at e_875_12
  have e_875_13 := arithEq_of_rows h (row := 875) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_13
  simp only [← c54531, ← c54532, k2, ← c54533] at e_875_13
  have hr := e_875_13
  simp only [e_875_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6642)) := by
  have c54534 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.1
  have c54535 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.1
  have c54536 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.1
  have c54537 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.1
  have c54538 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54539 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_14 := arithEq_of_rows h (row := 875) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_14
  simp only [← c54534, ← c54535, ← c54536] at e_875_14
  have e_875_15 := arithEq_of_rows h (row := 875) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_15
  simp only [← c54537, ← c54538, k2, ← c54539] at e_875_15
  have hr := e_875_15
  simp only [e_875_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6643)) := by
  have c54540 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54541 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54542 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54543 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54544 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54545 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_16 := arithEq_of_rows h (row := 875) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_16
  simp only [← c54540, ← c54541, ← c54542] at e_875_16
  have e_875_17 := arithEq_of_rows h (row := 875) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_17
  simp only [← c54543, ← c54544, k2, ← c54545] at e_875_17
  have hr := e_875_17
  simp only [e_875_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6644)) := by
  have c54546 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54547 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54548 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54549 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54550 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54551 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_18 := arithEq_of_rows h (row := 875) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_18
  simp only [← c54546, ← c54547, ← c54548] at e_875_18
  have e_875_19 := arithEq_of_rows h (row := 875) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_19
  simp only [← c54549, ← c54550, k2, ← c54551] at e_875_19
  have hr := e_875_19
  simp only [e_875_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6645)) := by
  have c54552 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54553 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54554 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54555 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54556 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54557 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_0 := arithEq_of_rows h (row := 876) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_0
  simp only [← c54552, ← c54553, ← c54554] at e_876_0
  have e_876_1 := arithEq_of_rows h (row := 876) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_1
  simp only [← c54555, ← c54556, k2, ← c54557] at e_876_1
  have hr := e_876_1
  simp only [e_876_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6646)) := by
  have c54558 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54559 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54560 := (publicBatchWrapper89_copies1705 a h).1
  have c54561 := (publicBatchWrapper89_copies1705 a h).2.1
  have c54562 := (publicBatchWrapper89_copies1705 a h).2.2.1
  have c54563 := (publicBatchWrapper89_copies1705 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_2 := arithEq_of_rows h (row := 876) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_2
  simp only [← c54558, ← c54559, ← c54560] at e_876_2
  have e_876_3 := arithEq_of_rows h (row := 876) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_3
  simp only [← c54561, ← c54562, k2, ← c54563] at e_876_3
  have hr := e_876_3
  simp only [e_876_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6647)) := by
  have c54564 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.1
  have c54565 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.1
  have c54566 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.1
  have c54567 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.1
  have c54568 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.1
  have c54569 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_4 := arithEq_of_rows h (row := 876) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_4
  simp only [← c54564, ← c54565, ← c54566] at e_876_4
  have e_876_5 := arithEq_of_rows h (row := 876) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_5
  simp only [← c54567, ← c54568, k2, ← c54569] at e_876_5
  have hr := e_876_5
  simp only [e_876_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6648)) := by
  have c54570 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54571 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54572 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54573 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54574 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54575 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_6 := arithEq_of_rows h (row := 876) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_6
  simp only [← c54570, ← c54571, ← c54572] at e_876_6
  have e_876_7 := arithEq_of_rows h (row := 876) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_7
  simp only [← c54573, ← c54574, k2, ← c54575] at e_876_7
  have hr := e_876_7
  simp only [e_876_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6649)) := by
  have c54576 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54577 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54578 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54579 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54580 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54581 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_8 := arithEq_of_rows h (row := 876) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_8
  simp only [← c54576, ← c54577, ← c54578] at e_876_8
  have e_876_9 := arithEq_of_rows h (row := 876) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_9
  simp only [← c54579, ← c54580, k2, ← c54581] at e_876_9
  have hr := e_876_9
  simp only [e_876_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6650)) := by
  have c54582 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54583 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54584 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54585 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54586 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54587 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_10 := arithEq_of_rows h (row := 876) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_10
  simp only [← c54582, ← c54583, ← c54584] at e_876_10
  have e_876_11 := arithEq_of_rows h (row := 876) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_11
  simp only [← c54585, ← c54586, k2, ← c54587] at e_876_11
  have hr := e_876_11
  simp only [e_876_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6651)) := by
  have c54588 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54589 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54590 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54591 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54592 := (publicBatchWrapper89_copies1706 a h).1
  have c54593 := (publicBatchWrapper89_copies1706 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_12 := arithEq_of_rows h (row := 876) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_12
  simp only [← c54588, ← c54589, ← c54590] at e_876_12
  have e_876_13 := arithEq_of_rows h (row := 876) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_13
  simp only [← c54591, ← c54592, k2, ← c54593] at e_876_13
  have hr := e_876_13
  simp only [e_876_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6652)) := by
  have c54594 := (publicBatchWrapper89_copies1706 a h).2.2.1
  have c54595 := (publicBatchWrapper89_copies1706 a h).2.2.2.1
  have c54596 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.1
  have c54597 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.1
  have c54598 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.1
  have c54599 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_14 := arithEq_of_rows h (row := 876) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_14
  simp only [← c54594, ← c54595, ← c54596] at e_876_14
  have e_876_15 := arithEq_of_rows h (row := 876) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_15
  simp only [← c54597, ← c54598, k2, ← c54599] at e_876_15
  have hr := e_876_15
  simp only [e_876_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6653)) := by
  have c54600 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.1
  have c54601 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.1
  have c54602 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54603 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54604 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54605 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_16 := arithEq_of_rows h (row := 876) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_16
  simp only [← c54600, ← c54601, ← c54602] at e_876_16
  have e_876_17 := arithEq_of_rows h (row := 876) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_17
  simp only [← c54603, ← c54604, k2, ← c54605] at e_876_17
  have hr := e_876_17
  simp only [e_876_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6654)) := by
  have c54606 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54607 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54608 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54609 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54610 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54611 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_18 := arithEq_of_rows h (row := 876) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_18
  simp only [← c54606, ← c54607, ← c54608] at e_876_18
  have e_876_19 := arithEq_of_rows h (row := 876) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_19
  simp only [← c54609, ← c54610, k2, ← c54611] at e_876_19
  have hr := e_876_19
  simp only [e_876_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6655)) := by
  have c54612 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54613 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54614 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54615 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54616 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54617 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_0 := arithEq_of_rows h (row := 877) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_0
  simp only [← c54612, ← c54613, ← c54614] at e_877_0
  have e_877_1 := arithEq_of_rows h (row := 877) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_1
  simp only [← c54615, ← c54616, k2, ← c54617] at e_877_1
  have hr := e_877_1
  simp only [e_877_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6656)) := by
  have c54618 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54619 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54620 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54621 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54622 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54623 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_2 := arithEq_of_rows h (row := 877) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_2
  simp only [← c54618, ← c54619, ← c54620] at e_877_2
  have e_877_3 := arithEq_of_rows h (row := 877) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_3
  simp only [← c54621, ← c54622, k2, ← c54623] at e_877_3
  have hr := e_877_3
  simp only [e_877_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6657)) := by
  have c54624 := (publicBatchWrapper89_copies1707 a h).1
  have c54625 := (publicBatchWrapper89_copies1707 a h).2.1
  have c54626 := (publicBatchWrapper89_copies1707 a h).2.2.1
  have c54627 := (publicBatchWrapper89_copies1707 a h).2.2.2.1
  have c54628 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.1
  have c54629 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_4 := arithEq_of_rows h (row := 877) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_4
  simp only [← c54624, ← c54625, ← c54626] at e_877_4
  have e_877_5 := arithEq_of_rows h (row := 877) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_5
  simp only [← c54627, ← c54628, k2, ← c54629] at e_877_5
  have hr := e_877_5
  simp only [e_877_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6658)) := by
  have c54630 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.1
  have c54631 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.1
  have c54632 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.1
  have c54633 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.1
  have c54634 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54635 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_6 := arithEq_of_rows h (row := 877) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_6
  simp only [← c54630, ← c54631, ← c54632] at e_877_6
  have e_877_7 := arithEq_of_rows h (row := 877) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_7
  simp only [← c54633, ← c54634, k2, ← c54635] at e_877_7
  have hr := e_877_7
  simp only [e_877_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6659)) := by
  have c54636 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54637 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54638 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54639 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54640 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54641 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_8 := arithEq_of_rows h (row := 877) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_8
  simp only [← c54636, ← c54637, ← c54638] at e_877_8
  have e_877_9 := arithEq_of_rows h (row := 877) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_9
  simp only [← c54639, ← c54640, k2, ← c54641] at e_877_9
  have hr := e_877_9
  simp only [e_877_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6660)) := by
  have c54642 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54643 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54644 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54645 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54646 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54647 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_10 := arithEq_of_rows h (row := 877) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_10
  simp only [← c54642, ← c54643, ← c54644] at e_877_10
  have e_877_11 := arithEq_of_rows h (row := 877) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_11
  simp only [← c54645, ← c54646, k2, ← c54647] at e_877_11
  have hr := e_877_11
  simp only [e_877_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6661)) := by
  have c54648 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54649 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54650 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54651 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54652 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54653 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_12 := arithEq_of_rows h (row := 877) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_12
  simp only [← c54648, ← c54649, ← c54650] at e_877_12
  have e_877_13 := arithEq_of_rows h (row := 877) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_13
  simp only [← c54651, ← c54652, k2, ← c54653] at e_877_13
  have hr := e_877_13
  simp only [e_877_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6662)) := by
  have c54654 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54655 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54656 := (publicBatchWrapper89_copies1708 a h).1
  have c54657 := (publicBatchWrapper89_copies1708 a h).2.1
  have c54658 := (publicBatchWrapper89_copies1708 a h).2.2.1
  have c54659 := (publicBatchWrapper89_copies1708 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_14 := arithEq_of_rows h (row := 877) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_14
  simp only [← c54654, ← c54655, ← c54656] at e_877_14
  have e_877_15 := arithEq_of_rows h (row := 877) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_15
  simp only [← c54657, ← c54658, k2, ← c54659] at e_877_15
  have hr := e_877_15
  simp only [e_877_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6663)) := by
  have c54660 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.1
  have c54661 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.1
  have c54662 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.1
  have c54663 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.1
  have c54664 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.1
  have c54665 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_16 := arithEq_of_rows h (row := 877) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_16
  simp only [← c54660, ← c54661, ← c54662] at e_877_16
  have e_877_17 := arithEq_of_rows h (row := 877) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_17
  simp only [← c54663, ← c54664, k2, ← c54665] at e_877_17
  have hr := e_877_17
  simp only [e_877_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6664)) := by
  have c54666 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54667 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54668 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54669 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54670 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54671 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_18 := arithEq_of_rows h (row := 877) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_18
  simp only [← c54666, ← c54667, ← c54668] at e_877_18
  have e_877_19 := arithEq_of_rows h (row := 877) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_19
  simp only [← c54669, ← c54670, k2, ← c54671] at e_877_19
  have hr := e_877_19
  simp only [e_877_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6665)) := by
  have c54672 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54673 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54674 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54675 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54676 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54677 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_0 := arithEq_of_rows h (row := 878) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_0
  simp only [← c54672, ← c54673, ← c54674] at e_878_0
  have e_878_1 := arithEq_of_rows h (row := 878) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_1
  simp only [← c54675, ← c54676, k2, ← c54677] at e_878_1
  have hr := e_878_1
  simp only [e_878_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6666)) := by
  have c54678 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54679 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54680 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54681 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54682 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54683 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_2 := arithEq_of_rows h (row := 878) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_2
  simp only [← c54678, ← c54679, ← c54680] at e_878_2
  have e_878_3 := arithEq_of_rows h (row := 878) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_3
  simp only [← c54681, ← c54682, k2, ← c54683] at e_878_3
  have hr := e_878_3
  simp only [e_878_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6667)) := by
  have c54684 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54685 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54686 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54687 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54688 := (publicBatchWrapper89_copies1709 a h).1
  have c54689 := (publicBatchWrapper89_copies1709 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_4 := arithEq_of_rows h (row := 878) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_4
  simp only [← c54684, ← c54685, ← c54686] at e_878_4
  have e_878_5 := arithEq_of_rows h (row := 878) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_5
  simp only [← c54687, ← c54688, k2, ← c54689] at e_878_5
  have hr := e_878_5
  simp only [e_878_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6668)) := by
  have c54690 := (publicBatchWrapper89_copies1709 a h).2.2.1
  have c54691 := (publicBatchWrapper89_copies1709 a h).2.2.2.1
  have c54692 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.1
  have c54693 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.1
  have c54694 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.1
  have c54695 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_6 := arithEq_of_rows h (row := 878) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_6
  simp only [← c54690, ← c54691, ← c54692] at e_878_6
  have e_878_7 := arithEq_of_rows h (row := 878) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_7
  simp only [← c54693, ← c54694, k2, ← c54695] at e_878_7
  have hr := e_878_7
  simp only [e_878_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6669)) := by
  have c54696 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.1
  have c54697 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.1
  have c54698 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54699 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54700 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54701 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_8 := arithEq_of_rows h (row := 878) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_8
  simp only [← c54696, ← c54697, ← c54698] at e_878_8
  have e_878_9 := arithEq_of_rows h (row := 878) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_9
  simp only [← c54699, ← c54700, k2, ← c54701] at e_878_9
  have hr := e_878_9
  simp only [e_878_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6670)) := by
  have c54702 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54703 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54704 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54705 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54706 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54707 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_10 := arithEq_of_rows h (row := 878) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_10
  simp only [← c54702, ← c54703, ← c54704] at e_878_10
  have e_878_11 := arithEq_of_rows h (row := 878) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_11
  simp only [← c54705, ← c54706, k2, ← c54707] at e_878_11
  have hr := e_878_11
  simp only [e_878_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6671)) := by
  have c54708 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54709 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54710 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54711 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54712 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54713 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_12 := arithEq_of_rows h (row := 878) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_12
  simp only [← c54708, ← c54709, ← c54710] at e_878_12
  have e_878_13 := arithEq_of_rows h (row := 878) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_13
  simp only [← c54711, ← c54712, k2, ← c54713] at e_878_13
  have hr := e_878_13
  simp only [e_878_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6672)) := by
  have c54714 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54715 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54716 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54717 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54718 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54719 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_14 := arithEq_of_rows h (row := 878) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_14
  simp only [← c54714, ← c54715, ← c54716] at e_878_14
  have e_878_15 := arithEq_of_rows h (row := 878) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_15
  simp only [← c54717, ← c54718, k2, ← c54719] at e_878_15
  have hr := e_878_15
  simp only [e_878_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6673)) := by
  have c54720 := (publicBatchWrapper89_copies1710 a h).1
  have c54721 := (publicBatchWrapper89_copies1710 a h).2.1
  have c54722 := (publicBatchWrapper89_copies1710 a h).2.2.1
  have c54723 := (publicBatchWrapper89_copies1710 a h).2.2.2.1
  have c54724 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.1
  have c54725 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_16 := arithEq_of_rows h (row := 878) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_16
  simp only [← c54720, ← c54721, ← c54722] at e_878_16
  have e_878_17 := arithEq_of_rows h (row := 878) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_17
  simp only [← c54723, ← c54724, k2, ← c54725] at e_878_17
  have hr := e_878_17
  simp only [e_878_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6674)) := by
  have c54726 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.1
  have c54727 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.1
  have c54728 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.1
  have c54729 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.1
  have c54730 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54731 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_18 := arithEq_of_rows h (row := 878) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_18
  simp only [← c54726, ← c54727, ← c54728] at e_878_18
  have e_878_19 := arithEq_of_rows h (row := 878) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_19
  simp only [← c54729, ← c54730, k2, ← c54731] at e_878_19
  have hr := e_878_19
  simp only [e_878_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6675)) := by
  have c54732 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54733 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54734 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54735 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54736 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54737 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_0 := arithEq_of_rows h (row := 879) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_0
  simp only [← c54732, ← c54733, ← c54734] at e_879_0
  have e_879_1 := arithEq_of_rows h (row := 879) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_1
  simp only [← c54735, ← c54736, k2, ← c54737] at e_879_1
  have hr := e_879_1
  simp only [e_879_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6676)) := by
  have c54738 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54739 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54740 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54741 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54742 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54743 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_2 := arithEq_of_rows h (row := 879) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_2
  simp only [← c54738, ← c54739, ← c54740] at e_879_2
  have e_879_3 := arithEq_of_rows h (row := 879) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_3
  simp only [← c54741, ← c54742, k2, ← c54743] at e_879_3
  have hr := e_879_3
  simp only [e_879_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6677)) := by
  have c54744 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54745 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54746 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54747 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54748 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54749 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_4 := arithEq_of_rows h (row := 879) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_4
  simp only [← c54744, ← c54745, ← c54746] at e_879_4
  have e_879_5 := arithEq_of_rows h (row := 879) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_5
  simp only [← c54747, ← c54748, k2, ← c54749] at e_879_5
  have hr := e_879_5
  simp only [e_879_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6678)) := by
  have c54750 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54751 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54752 := (publicBatchWrapper89_copies1711 a h).1
  have c54753 := (publicBatchWrapper89_copies1711 a h).2.1
  have c54754 := (publicBatchWrapper89_copies1711 a h).2.2.1
  have c54755 := (publicBatchWrapper89_copies1711 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_6 := arithEq_of_rows h (row := 879) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_6
  simp only [← c54750, ← c54751, ← c54752] at e_879_6
  have e_879_7 := arithEq_of_rows h (row := 879) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_7
  simp only [← c54753, ← c54754, k2, ← c54755] at e_879_7
  have hr := e_879_7
  simp only [e_879_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6679)) := by
  have c54756 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.1
  have c54757 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.1
  have c54758 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.1
  have c54759 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.1
  have c54760 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.1
  have c54761 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_8 := arithEq_of_rows h (row := 879) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_8
  simp only [← c54756, ← c54757, ← c54758] at e_879_8
  have e_879_9 := arithEq_of_rows h (row := 879) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_9
  simp only [← c54759, ← c54760, k2, ← c54761] at e_879_9
  have hr := e_879_9
  simp only [e_879_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6680)) := by
  have c54762 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54763 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54764 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54765 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54766 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54767 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_10 := arithEq_of_rows h (row := 879) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_10
  simp only [← c54762, ← c54763, ← c54764] at e_879_10
  have e_879_11 := arithEq_of_rows h (row := 879) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_11
  simp only [← c54765, ← c54766, k2, ← c54767] at e_879_11
  have hr := e_879_11
  simp only [e_879_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6681)) := by
  have c54768 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54769 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54770 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54771 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54772 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54773 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_12 := arithEq_of_rows h (row := 879) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_12
  simp only [← c54768, ← c54769, ← c54770] at e_879_12
  have e_879_13 := arithEq_of_rows h (row := 879) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_13
  simp only [← c54771, ← c54772, k2, ← c54773] at e_879_13
  have hr := e_879_13
  simp only [e_879_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6682)) := by
  have c54774 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54775 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54776 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54777 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54778 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54779 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_14 := arithEq_of_rows h (row := 879) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_14
  simp only [← c54774, ← c54775, ← c54776] at e_879_14
  have e_879_15 := arithEq_of_rows h (row := 879) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_15
  simp only [← c54777, ← c54778, k2, ← c54779] at e_879_15
  have hr := e_879_15
  simp only [e_879_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6683)) := by
  have c54780 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54781 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54782 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54783 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54784 := (publicBatchWrapper89_copies1712 a h).1
  have c54785 := (publicBatchWrapper89_copies1712 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_16 := arithEq_of_rows h (row := 879) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_16
  simp only [← c54780, ← c54781, ← c54782] at e_879_16
  have e_879_17 := arithEq_of_rows h (row := 879) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_17
  simp only [← c54783, ← c54784, k2, ← c54785] at e_879_17
  have hr := e_879_17
  simp only [e_879_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6684)) := by
  have c54786 := (publicBatchWrapper89_copies1712 a h).2.2.1
  have c54787 := (publicBatchWrapper89_copies1712 a h).2.2.2.1
  have c54788 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.1
  have c54789 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.1
  have c54790 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.1
  have c54791 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_18 := arithEq_of_rows h (row := 879) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_18
  simp only [← c54786, ← c54787, ← c54788] at e_879_18
  have e_879_19 := arithEq_of_rows h (row := 879) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_19
  simp only [← c54789, ← c54790, k2, ← c54791] at e_879_19
  have hr := e_879_19
  simp only [e_879_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6685)) := by
  have c54792 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.1
  have c54793 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.1
  have c54794 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54795 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54796 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54797 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_0 := arithEq_of_rows h (row := 880) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_0
  simp only [← c54792, ← c54793, ← c54794] at e_880_0
  have e_880_1 := arithEq_of_rows h (row := 880) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_1
  simp only [← c54795, ← c54796, k2, ← c54797] at e_880_1
  have hr := e_880_1
  simp only [e_880_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6686)) := by
  have c54798 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54799 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54800 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54801 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54802 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54803 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_2 := arithEq_of_rows h (row := 880) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_2
  simp only [← c54798, ← c54799, ← c54800] at e_880_2
  have e_880_3 := arithEq_of_rows h (row := 880) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_3
  simp only [← c54801, ← c54802, k2, ← c54803] at e_880_3
  have hr := e_880_3
  simp only [e_880_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
