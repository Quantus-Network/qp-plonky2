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

theorem publicBatchWrapper86_f13103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12709)) := by
  have c82971 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82972 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82973 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82974 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82975 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82976 := (publicBatchWrapper86_copies2593 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_2 := arithEq_of_rows h (row := 1351) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_2
  simp only [← c82971, ← c82972, ← c82973] at e_1351_2
  have e_1351_3 := arithEq_of_rows h (row := 1351) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_3
  simp only [← c82974, ← c82975, k2, ← c82976] at e_1351_3
  have hr := e_1351_3
  simp only [e_1351_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12710)) := by
  have c82977 := (publicBatchWrapper86_copies2593 a h).2.1
  have c82978 := (publicBatchWrapper86_copies2593 a h).2.2.1
  have c82979 := (publicBatchWrapper86_copies2593 a h).2.2.2.1
  have c82980 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.1
  have c82981 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.1
  have c82982 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_4 := arithEq_of_rows h (row := 1351) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_4
  simp only [← c82977, ← c82978, ← c82979] at e_1351_4
  have e_1351_5 := arithEq_of_rows h (row := 1351) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_5
  simp only [← c82980, ← c82981, k2, ← c82982] at e_1351_5
  have hr := e_1351_5
  simp only [e_1351_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12711)) := by
  have c82983 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.1
  have c82984 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.1
  have c82985 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.1
  have c82986 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82987 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82988 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_6 := arithEq_of_rows h (row := 1351) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_6
  simp only [← c82983, ← c82984, ← c82985] at e_1351_6
  have e_1351_7 := arithEq_of_rows h (row := 1351) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_7
  simp only [← c82986, ← c82987, k2, ← c82988] at e_1351_7
  have hr := e_1351_7
  simp only [e_1351_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12712)) := by
  have c82989 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82990 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82991 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82992 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82993 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82994 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_8 := arithEq_of_rows h (row := 1351) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_8
  simp only [← c82989, ← c82990, ← c82991] at e_1351_8
  have e_1351_9 := arithEq_of_rows h (row := 1351) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_9
  simp only [← c82992, ← c82993, k2, ← c82994] at e_1351_9
  have hr := e_1351_9
  simp only [e_1351_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12713)) := by
  have c82995 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82996 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82997 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82998 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82999 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83000 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_10 := arithEq_of_rows h (row := 1351) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_10
  simp only [← c82995, ← c82996, ← c82997] at e_1351_10
  have e_1351_11 := arithEq_of_rows h (row := 1351) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_11
  simp only [← c82998, ← c82999, k2, ← c83000] at e_1351_11
  have hr := e_1351_11
  simp only [e_1351_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12714)) := by
  have c83001 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83002 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83003 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83004 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83005 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83006 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_12 := arithEq_of_rows h (row := 1351) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_12
  simp only [← c83001, ← c83002, ← c83003] at e_1351_12
  have e_1351_13 := arithEq_of_rows h (row := 1351) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_13
  simp only [← c83004, ← c83005, k2, ← c83006] at e_1351_13
  have hr := e_1351_13
  simp only [e_1351_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12715)) := by
  have c83007 := (publicBatchWrapper86_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83008 := (publicBatchWrapper86_copies2594 a h).1
  have c83009 := (publicBatchWrapper86_copies2594 a h).2.1
  have c83010 := (publicBatchWrapper86_copies2594 a h).2.2.1
  have c83011 := (publicBatchWrapper86_copies2594 a h).2.2.2.1
  have c83012 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_14 := arithEq_of_rows h (row := 1351) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_14
  simp only [← c83007, ← c83008, ← c83009] at e_1351_14
  have e_1351_15 := arithEq_of_rows h (row := 1351) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_15
  simp only [← c83010, ← c83011, k2, ← c83012] at e_1351_15
  have hr := e_1351_15
  simp only [e_1351_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12716)) := by
  have c83013 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.1
  have c83014 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.1
  have c83015 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.1
  have c83016 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.1
  have c83017 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.1
  have c83018 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_16 := arithEq_of_rows h (row := 1351) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_16
  simp only [← c83013, ← c83014, ← c83015] at e_1351_16
  have e_1351_17 := arithEq_of_rows h (row := 1351) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_17
  simp only [← c83016, ← c83017, k2, ← c83018] at e_1351_17
  have hr := e_1351_17
  simp only [e_1351_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12717)) := by
  have c83019 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83020 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83021 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83022 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83023 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83024 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_18 := arithEq_of_rows h (row := 1351) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_18
  simp only [← c83019, ← c83020, ← c83021] at e_1351_18
  have e_1351_19 := arithEq_of_rows h (row := 1351) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_19
  simp only [← c83022, ← c83023, k2, ← c83024] at e_1351_19
  have hr := e_1351_19
  simp only [e_1351_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12718)) := by
  have c83025 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83026 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83027 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83028 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83029 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83030 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_0 := arithEq_of_rows h (row := 1352) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_0
  simp only [← c83025, ← c83026, ← c83027] at e_1352_0
  have e_1352_1 := arithEq_of_rows h (row := 1352) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_1
  simp only [← c83028, ← c83029, k2, ← c83030] at e_1352_1
  have hr := e_1352_1
  simp only [e_1352_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12719)) := by
  have c83031 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83032 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83033 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83034 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83035 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83036 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_2 := arithEq_of_rows h (row := 1352) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_2
  simp only [← c83031, ← c83032, ← c83033] at e_1352_2
  have e_1352_3 := arithEq_of_rows h (row := 1352) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_3
  simp only [← c83034, ← c83035, k2, ← c83036] at e_1352_3
  have hr := e_1352_3
  simp only [e_1352_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12720)) := by
  have c83037 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83038 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83039 := (publicBatchWrapper86_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83040 := (publicBatchWrapper86_copies2595 a h).1
  have c83041 := (publicBatchWrapper86_copies2595 a h).2.1
  have c83042 := (publicBatchWrapper86_copies2595 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_4 := arithEq_of_rows h (row := 1352) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_4
  simp only [← c83037, ← c83038, ← c83039] at e_1352_4
  have e_1352_5 := arithEq_of_rows h (row := 1352) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_5
  simp only [← c83040, ← c83041, k2, ← c83042] at e_1352_5
  have hr := e_1352_5
  simp only [e_1352_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12721)) := by
  have c83043 := (publicBatchWrapper86_copies2595 a h).2.2.2.1
  have c83044 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.1
  have c83045 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.1
  have c83046 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.1
  have c83047 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.1
  have c83048 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_6 := arithEq_of_rows h (row := 1352) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_6
  simp only [← c83043, ← c83044, ← c83045] at e_1352_6
  have e_1352_7 := arithEq_of_rows h (row := 1352) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_7
  simp only [← c83046, ← c83047, k2, ← c83048] at e_1352_7
  have hr := e_1352_7
  simp only [e_1352_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12722)) := by
  have c83049 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.1
  have c83050 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83051 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83052 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83053 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83054 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_8 := arithEq_of_rows h (row := 1352) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_8
  simp only [← c83049, ← c83050, ← c83051] at e_1352_8
  have e_1352_9 := arithEq_of_rows h (row := 1352) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_9
  simp only [← c83052, ← c83053, k2, ← c83054] at e_1352_9
  have hr := e_1352_9
  simp only [e_1352_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12723)) := by
  have c83055 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83056 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83057 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83058 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83059 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83060 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_10 := arithEq_of_rows h (row := 1352) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_10
  simp only [← c83055, ← c83056, ← c83057] at e_1352_10
  have e_1352_11 := arithEq_of_rows h (row := 1352) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_11
  simp only [← c83058, ← c83059, k2, ← c83060] at e_1352_11
  have hr := e_1352_11
  simp only [e_1352_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12724)) := by
  have c83061 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83062 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83063 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83064 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83065 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83066 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_12 := arithEq_of_rows h (row := 1352) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_12
  simp only [← c83061, ← c83062, ← c83063] at e_1352_12
  have e_1352_13 := arithEq_of_rows h (row := 1352) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_13
  simp only [← c83064, ← c83065, k2, ← c83066] at e_1352_13
  have hr := e_1352_13
  simp only [e_1352_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12725)) := by
  have c83067 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83068 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83069 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83070 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83071 := (publicBatchWrapper86_copies2595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83072 := (publicBatchWrapper86_copies2596 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_14 := arithEq_of_rows h (row := 1352) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_14
  simp only [← c83067, ← c83068, ← c83069] at e_1352_14
  have e_1352_15 := arithEq_of_rows h (row := 1352) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_15
  simp only [← c83070, ← c83071, k2, ← c83072] at e_1352_15
  have hr := e_1352_15
  simp only [e_1352_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12726)) := by
  have c83073 := (publicBatchWrapper86_copies2596 a h).2.1
  have c83074 := (publicBatchWrapper86_copies2596 a h).2.2.1
  have c83075 := (publicBatchWrapper86_copies2596 a h).2.2.2.1
  have c83076 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.1
  have c83077 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.1
  have c83078 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_16 := arithEq_of_rows h (row := 1352) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_16
  simp only [← c83073, ← c83074, ← c83075] at e_1352_16
  have e_1352_17 := arithEq_of_rows h (row := 1352) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_17
  simp only [← c83076, ← c83077, k2, ← c83078] at e_1352_17
  have hr := e_1352_17
  simp only [e_1352_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1352 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12727)) := by
  have c83079 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.1
  have c83080 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.1
  have c83081 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.1
  have c83082 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83083 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83084 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1352_18 := arithEq_of_rows h (row := 1352) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_18
  simp only [← c83079, ← c83080, ← c83081] at e_1352_18
  have e_1352_19 := arithEq_of_rows h (row := 1352) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1352_19
  simp only [← c83082, ← c83083, k2, ← c83084] at e_1352_19
  have hr := e_1352_19
  simp only [e_1352_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12728)) := by
  have c83085 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83086 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83087 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83088 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83089 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83090 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_0 := arithEq_of_rows h (row := 1353) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_0
  simp only [← c83085, ← c83086, ← c83087] at e_1353_0
  have e_1353_1 := arithEq_of_rows h (row := 1353) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_1
  simp only [← c83088, ← c83089, k2, ← c83090] at e_1353_1
  have hr := e_1353_1
  simp only [e_1353_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12729)) := by
  have c83091 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83092 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83093 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83094 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83095 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83096 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_2 := arithEq_of_rows h (row := 1353) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_2
  simp only [← c83091, ← c83092, ← c83093] at e_1353_2
  have e_1353_3 := arithEq_of_rows h (row := 1353) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_3
  simp only [← c83094, ← c83095, k2, ← c83096] at e_1353_3
  have hr := e_1353_3
  simp only [e_1353_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12730)) := by
  have c83097 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83098 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83099 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83100 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83101 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83102 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_4 := arithEq_of_rows h (row := 1353) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_4
  simp only [← c83097, ← c83098, ← c83099] at e_1353_4
  have e_1353_5 := arithEq_of_rows h (row := 1353) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_5
  simp only [← c83100, ← c83101, k2, ← c83102] at e_1353_5
  have hr := e_1353_5
  simp only [e_1353_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12731)) := by
  have c83103 := (publicBatchWrapper86_copies2596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83104 := (publicBatchWrapper86_copies2597 a h).1
  have c83105 := (publicBatchWrapper86_copies2597 a h).2.1
  have c83106 := (publicBatchWrapper86_copies2597 a h).2.2.1
  have c83107 := (publicBatchWrapper86_copies2597 a h).2.2.2.1
  have c83108 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_6 := arithEq_of_rows h (row := 1353) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_6
  simp only [← c83103, ← c83104, ← c83105] at e_1353_6
  have e_1353_7 := arithEq_of_rows h (row := 1353) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_7
  simp only [← c83106, ← c83107, k2, ← c83108] at e_1353_7
  have hr := e_1353_7
  simp only [e_1353_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12732)) := by
  have c83109 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.1
  have c83110 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.1
  have c83111 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.1
  have c83112 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.1
  have c83113 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.1
  have c83114 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_8 := arithEq_of_rows h (row := 1353) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_8
  simp only [← c83109, ← c83110, ← c83111] at e_1353_8
  have e_1353_9 := arithEq_of_rows h (row := 1353) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_9
  simp only [← c83112, ← c83113, k2, ← c83114] at e_1353_9
  have hr := e_1353_9
  simp only [e_1353_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12733)) := by
  have c83115 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83116 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83117 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83118 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83119 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83120 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_10 := arithEq_of_rows h (row := 1353) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_10
  simp only [← c83115, ← c83116, ← c83117] at e_1353_10
  have e_1353_11 := arithEq_of_rows h (row := 1353) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_11
  simp only [← c83118, ← c83119, k2, ← c83120] at e_1353_11
  have hr := e_1353_11
  simp only [e_1353_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12734)) := by
  have c83121 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83122 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83123 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83124 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83125 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83126 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_12 := arithEq_of_rows h (row := 1353) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_12
  simp only [← c83121, ← c83122, ← c83123] at e_1353_12
  have e_1353_13 := arithEq_of_rows h (row := 1353) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_13
  simp only [← c83124, ← c83125, k2, ← c83126] at e_1353_13
  have hr := e_1353_13
  simp only [e_1353_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12735)) := by
  have c83127 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83128 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83129 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83130 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83131 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83132 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_14 := arithEq_of_rows h (row := 1353) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_14
  simp only [← c83127, ← c83128, ← c83129] at e_1353_14
  have e_1353_15 := arithEq_of_rows h (row := 1353) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_15
  simp only [← c83130, ← c83131, k2, ← c83132] at e_1353_15
  have hr := e_1353_15
  simp only [e_1353_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12736)) := by
  have c83133 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83134 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83135 := (publicBatchWrapper86_copies2597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83136 := (publicBatchWrapper86_copies2598 a h).1
  have c83137 := (publicBatchWrapper86_copies2598 a h).2.1
  have c83138 := (publicBatchWrapper86_copies2598 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_16 := arithEq_of_rows h (row := 1353) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_16
  simp only [← c83133, ← c83134, ← c83135] at e_1353_16
  have e_1353_17 := arithEq_of_rows h (row := 1353) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_17
  simp only [← c83136, ← c83137, k2, ← c83138] at e_1353_17
  have hr := e_1353_17
  simp only [e_1353_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1353 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12737)) := by
  have c83139 := (publicBatchWrapper86_copies2598 a h).2.2.2.1
  have c83140 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.1
  have c83141 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.1
  have c83142 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.1
  have c83143 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.1
  have c83144 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1353_18 := arithEq_of_rows h (row := 1353) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_18
  simp only [← c83139, ← c83140, ← c83141] at e_1353_18
  have e_1353_19 := arithEq_of_rows h (row := 1353) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1353_19
  simp only [← c83142, ← c83143, k2, ← c83144] at e_1353_19
  have hr := e_1353_19
  simp only [e_1353_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12738)) := by
  have c83145 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.1
  have c83146 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83147 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83148 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83149 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83150 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_0 := arithEq_of_rows h (row := 1354) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_0
  simp only [← c83145, ← c83146, ← c83147] at e_1354_0
  have e_1354_1 := arithEq_of_rows h (row := 1354) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_1
  simp only [← c83148, ← c83149, k2, ← c83150] at e_1354_1
  have hr := e_1354_1
  simp only [e_1354_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12739)) := by
  have c83151 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83152 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83153 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83154 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83155 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83156 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_2 := arithEq_of_rows h (row := 1354) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_2
  simp only [← c83151, ← c83152, ← c83153] at e_1354_2
  have e_1354_3 := arithEq_of_rows h (row := 1354) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_3
  simp only [← c83154, ← c83155, k2, ← c83156] at e_1354_3
  have hr := e_1354_3
  simp only [e_1354_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12740)) := by
  have c83157 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83158 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83159 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83160 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83161 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83162 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_4 := arithEq_of_rows h (row := 1354) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_4
  simp only [← c83157, ← c83158, ← c83159] at e_1354_4
  have e_1354_5 := arithEq_of_rows h (row := 1354) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_5
  simp only [← c83160, ← c83161, k2, ← c83162] at e_1354_5
  have hr := e_1354_5
  simp only [e_1354_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12741)) := by
  have c83163 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83164 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83165 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83166 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83167 := (publicBatchWrapper86_copies2598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83168 := (publicBatchWrapper86_copies2599 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_6 := arithEq_of_rows h (row := 1354) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_6
  simp only [← c83163, ← c83164, ← c83165] at e_1354_6
  have e_1354_7 := arithEq_of_rows h (row := 1354) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_7
  simp only [← c83166, ← c83167, k2, ← c83168] at e_1354_7
  have hr := e_1354_7
  simp only [e_1354_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12742)) := by
  have c83169 := (publicBatchWrapper86_copies2599 a h).2.1
  have c83170 := (publicBatchWrapper86_copies2599 a h).2.2.1
  have c83171 := (publicBatchWrapper86_copies2599 a h).2.2.2.1
  have c83172 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.1
  have c83173 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.1
  have c83174 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_8 := arithEq_of_rows h (row := 1354) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_8
  simp only [← c83169, ← c83170, ← c83171] at e_1354_8
  have e_1354_9 := arithEq_of_rows h (row := 1354) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_9
  simp only [← c83172, ← c83173, k2, ← c83174] at e_1354_9
  have hr := e_1354_9
  simp only [e_1354_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12743)) := by
  have c83175 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.1
  have c83176 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.1
  have c83177 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.1
  have c83178 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83179 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83180 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_10 := arithEq_of_rows h (row := 1354) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_10
  simp only [← c83175, ← c83176, ← c83177] at e_1354_10
  have e_1354_11 := arithEq_of_rows h (row := 1354) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_11
  simp only [← c83178, ← c83179, k2, ← c83180] at e_1354_11
  have hr := e_1354_11
  simp only [e_1354_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12744)) := by
  have c83181 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83182 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83183 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83184 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83185 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83186 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_12 := arithEq_of_rows h (row := 1354) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_12
  simp only [← c83181, ← c83182, ← c83183] at e_1354_12
  have e_1354_13 := arithEq_of_rows h (row := 1354) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_13
  simp only [← c83184, ← c83185, k2, ← c83186] at e_1354_13
  have hr := e_1354_13
  simp only [e_1354_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12745)) := by
  have c83187 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83188 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83189 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83190 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83191 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83192 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_14 := arithEq_of_rows h (row := 1354) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_14
  simp only [← c83187, ← c83188, ← c83189] at e_1354_14
  have e_1354_15 := arithEq_of_rows h (row := 1354) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_15
  simp only [← c83190, ← c83191, k2, ← c83192] at e_1354_15
  have hr := e_1354_15
  simp only [e_1354_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12746)) := by
  have c83193 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83194 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83195 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83196 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83197 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83198 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_16 := arithEq_of_rows h (row := 1354) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_16
  simp only [← c83193, ← c83194, ← c83195] at e_1354_16
  have e_1354_17 := arithEq_of_rows h (row := 1354) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_17
  simp only [← c83196, ← c83197, k2, ← c83198] at e_1354_17
  have hr := e_1354_17
  simp only [e_1354_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1354 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12747)) := by
  have c83199 := (publicBatchWrapper86_copies2599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83200 := (publicBatchWrapper86_copies2600 a h).1
  have c83201 := (publicBatchWrapper86_copies2600 a h).2.1
  have c83202 := (publicBatchWrapper86_copies2600 a h).2.2.1
  have c83203 := (publicBatchWrapper86_copies2600 a h).2.2.2.1
  have c83204 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1354_18 := arithEq_of_rows h (row := 1354) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_18
  simp only [← c83199, ← c83200, ← c83201] at e_1354_18
  have e_1354_19 := arithEq_of_rows h (row := 1354) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1354_19
  simp only [← c83202, ← c83203, k2, ← c83204] at e_1354_19
  have hr := e_1354_19
  simp only [e_1354_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12748)) := by
  have c83205 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.1
  have c83206 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.1
  have c83207 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.1
  have c83208 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.1
  have c83209 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.1
  have c83210 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_0 := arithEq_of_rows h (row := 1355) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_0
  simp only [← c83205, ← c83206, ← c83207] at e_1355_0
  have e_1355_1 := arithEq_of_rows h (row := 1355) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_1
  simp only [← c83208, ← c83209, k2, ← c83210] at e_1355_1
  have hr := e_1355_1
  simp only [e_1355_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12749)) := by
  have c83211 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83212 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83213 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83214 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83215 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83216 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_2 := arithEq_of_rows h (row := 1355) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_2
  simp only [← c83211, ← c83212, ← c83213] at e_1355_2
  have e_1355_3 := arithEq_of_rows h (row := 1355) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_3
  simp only [← c83214, ← c83215, k2, ← c83216] at e_1355_3
  have hr := e_1355_3
  simp only [e_1355_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12750)) := by
  have c83217 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83218 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83219 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83220 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83221 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83222 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_4 := arithEq_of_rows h (row := 1355) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_4
  simp only [← c83217, ← c83218, ← c83219] at e_1355_4
  have e_1355_5 := arithEq_of_rows h (row := 1355) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_5
  simp only [← c83220, ← c83221, k2, ← c83222] at e_1355_5
  have hr := e_1355_5
  simp only [e_1355_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12751)) := by
  have c83223 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83224 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83225 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83226 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83227 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83228 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_6 := arithEq_of_rows h (row := 1355) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_6
  simp only [← c83223, ← c83224, ← c83225] at e_1355_6
  have e_1355_7 := arithEq_of_rows h (row := 1355) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_7
  simp only [← c83226, ← c83227, k2, ← c83228] at e_1355_7
  have hr := e_1355_7
  simp only [e_1355_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12752)) := by
  have c83229 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83230 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83231 := (publicBatchWrapper86_copies2600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83232 := (publicBatchWrapper86_copies2601 a h).1
  have c83233 := (publicBatchWrapper86_copies2601 a h).2.1
  have c83234 := (publicBatchWrapper86_copies2601 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_8 := arithEq_of_rows h (row := 1355) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_8
  simp only [← c83229, ← c83230, ← c83231] at e_1355_8
  have e_1355_9 := arithEq_of_rows h (row := 1355) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_9
  simp only [← c83232, ← c83233, k2, ← c83234] at e_1355_9
  have hr := e_1355_9
  simp only [e_1355_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12753)) := by
  have c83235 := (publicBatchWrapper86_copies2601 a h).2.2.2.1
  have c83236 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.1
  have c83237 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.1
  have c83238 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.1
  have c83239 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.1
  have c83240 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_10 := arithEq_of_rows h (row := 1355) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_10
  simp only [← c83235, ← c83236, ← c83237] at e_1355_10
  have e_1355_11 := arithEq_of_rows h (row := 1355) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_11
  simp only [← c83238, ← c83239, k2, ← c83240] at e_1355_11
  have hr := e_1355_11
  simp only [e_1355_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12754)) := by
  have c83241 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.1
  have c83242 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83243 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83244 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83245 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83246 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_12 := arithEq_of_rows h (row := 1355) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_12
  simp only [← c83241, ← c83242, ← c83243] at e_1355_12
  have e_1355_13 := arithEq_of_rows h (row := 1355) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_13
  simp only [← c83244, ← c83245, k2, ← c83246] at e_1355_13
  have hr := e_1355_13
  simp only [e_1355_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12755)) := by
  have c83247 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83248 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83249 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83250 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83251 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83252 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_14 := arithEq_of_rows h (row := 1355) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_14
  simp only [← c83247, ← c83248, ← c83249] at e_1355_14
  have e_1355_15 := arithEq_of_rows h (row := 1355) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_15
  simp only [← c83250, ← c83251, k2, ← c83252] at e_1355_15
  have hr := e_1355_15
  simp only [e_1355_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12756)) := by
  have c83253 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83254 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83255 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83256 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83257 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83258 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_16 := arithEq_of_rows h (row := 1355) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_16
  simp only [← c83253, ← c83254, ← c83255] at e_1355_16
  have e_1355_17 := arithEq_of_rows h (row := 1355) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_17
  simp only [← c83256, ← c83257, k2, ← c83258] at e_1355_17
  have hr := e_1355_17
  simp only [e_1355_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1355 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12757)) := by
  have c83259 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83260 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83261 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83262 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83263 := (publicBatchWrapper86_copies2601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83264 := (publicBatchWrapper86_copies2602 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1355_18 := arithEq_of_rows h (row := 1355) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_18
  simp only [← c83259, ← c83260, ← c83261] at e_1355_18
  have e_1355_19 := arithEq_of_rows h (row := 1355) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1355_19
  simp only [← c83262, ← c83263, k2, ← c83264] at e_1355_19
  have hr := e_1355_19
  simp only [e_1355_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12758)) := by
  have c83265 := (publicBatchWrapper86_copies2602 a h).2.1
  have c83266 := (publicBatchWrapper86_copies2602 a h).2.2.1
  have c83267 := (publicBatchWrapper86_copies2602 a h).2.2.2.1
  have c83268 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.1
  have c83269 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.1
  have c83270 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_0 := arithEq_of_rows h (row := 1356) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_0
  simp only [← c83265, ← c83266, ← c83267] at e_1356_0
  have e_1356_1 := arithEq_of_rows h (row := 1356) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_1
  simp only [← c83268, ← c83269, k2, ← c83270] at e_1356_1
  have hr := e_1356_1
  simp only [e_1356_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12759)) := by
  have c83271 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.1
  have c83272 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.1
  have c83273 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.1
  have c83274 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83275 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83276 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_2 := arithEq_of_rows h (row := 1356) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_2
  simp only [← c83271, ← c83272, ← c83273] at e_1356_2
  have e_1356_3 := arithEq_of_rows h (row := 1356) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_3
  simp only [← c83274, ← c83275, k2, ← c83276] at e_1356_3
  have hr := e_1356_3
  simp only [e_1356_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12760)) := by
  have c83277 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83278 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83279 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83280 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83281 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83282 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_4 := arithEq_of_rows h (row := 1356) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_4
  simp only [← c83277, ← c83278, ← c83279] at e_1356_4
  have e_1356_5 := arithEq_of_rows h (row := 1356) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_5
  simp only [← c83280, ← c83281, k2, ← c83282] at e_1356_5
  have hr := e_1356_5
  simp only [e_1356_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12761)) := by
  have c83283 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83284 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83285 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83286 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83287 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83288 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_6 := arithEq_of_rows h (row := 1356) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_6
  simp only [← c83283, ← c83284, ← c83285] at e_1356_6
  have e_1356_7 := arithEq_of_rows h (row := 1356) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_7
  simp only [← c83286, ← c83287, k2, ← c83288] at e_1356_7
  have hr := e_1356_7
  simp only [e_1356_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12762)) := by
  have c83289 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83290 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83291 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83292 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83293 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83294 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_8 := arithEq_of_rows h (row := 1356) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_8
  simp only [← c83289, ← c83290, ← c83291] at e_1356_8
  have e_1356_9 := arithEq_of_rows h (row := 1356) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_9
  simp only [← c83292, ← c83293, k2, ← c83294] at e_1356_9
  have hr := e_1356_9
  simp only [e_1356_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12763)) := by
  have c83295 := (publicBatchWrapper86_copies2602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83296 := (publicBatchWrapper86_copies2603 a h).1
  have c83297 := (publicBatchWrapper86_copies2603 a h).2.1
  have c83298 := (publicBatchWrapper86_copies2603 a h).2.2.1
  have c83299 := (publicBatchWrapper86_copies2603 a h).2.2.2.1
  have c83300 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_10 := arithEq_of_rows h (row := 1356) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_10
  simp only [← c83295, ← c83296, ← c83297] at e_1356_10
  have e_1356_11 := arithEq_of_rows h (row := 1356) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_11
  simp only [← c83298, ← c83299, k2, ← c83300] at e_1356_11
  have hr := e_1356_11
  simp only [e_1356_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12764)) := by
  have c83301 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.1
  have c83302 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.1
  have c83303 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.1
  have c83304 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.1
  have c83305 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.1
  have c83306 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_12 := arithEq_of_rows h (row := 1356) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_12
  simp only [← c83301, ← c83302, ← c83303] at e_1356_12
  have e_1356_13 := arithEq_of_rows h (row := 1356) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_13
  simp only [← c83304, ← c83305, k2, ← c83306] at e_1356_13
  have hr := e_1356_13
  simp only [e_1356_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12765)) := by
  have c83307 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83308 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83309 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83310 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83311 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83312 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_14 := arithEq_of_rows h (row := 1356) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_14
  simp only [← c83307, ← c83308, ← c83309] at e_1356_14
  have e_1356_15 := arithEq_of_rows h (row := 1356) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_15
  simp only [← c83310, ← c83311, k2, ← c83312] at e_1356_15
  have hr := e_1356_15
  simp only [e_1356_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12766)) := by
  have c83313 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83314 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83315 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83316 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83317 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83318 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_16 := arithEq_of_rows h (row := 1356) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_16
  simp only [← c83313, ← c83314, ← c83315] at e_1356_16
  have e_1356_17 := arithEq_of_rows h (row := 1356) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_17
  simp only [← c83316, ← c83317, k2, ← c83318] at e_1356_17
  have hr := e_1356_17
  simp only [e_1356_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1356 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12767)) := by
  have c83319 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83320 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83321 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83322 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83323 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83324 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1356_18 := arithEq_of_rows h (row := 1356) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_18
  simp only [← c83319, ← c83320, ← c83321] at e_1356_18
  have e_1356_19 := arithEq_of_rows h (row := 1356) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1356_19
  simp only [← c83322, ← c83323, k2, ← c83324] at e_1356_19
  have hr := e_1356_19
  simp only [e_1356_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12768)) := by
  have c83325 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83326 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83327 := (publicBatchWrapper86_copies2603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83328 := (publicBatchWrapper86_copies2604 a h).1
  have c83329 := (publicBatchWrapper86_copies2604 a h).2.1
  have c83330 := (publicBatchWrapper86_copies2604 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_0 := arithEq_of_rows h (row := 1357) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_0
  simp only [← c83325, ← c83326, ← c83327] at e_1357_0
  have e_1357_1 := arithEq_of_rows h (row := 1357) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_1
  simp only [← c83328, ← c83329, k2, ← c83330] at e_1357_1
  have hr := e_1357_1
  simp only [e_1357_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12769)) := by
  have c83331 := (publicBatchWrapper86_copies2604 a h).2.2.2.1
  have c83332 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.1
  have c83333 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.1
  have c83334 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.1
  have c83335 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.1
  have c83336 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_2 := arithEq_of_rows h (row := 1357) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_2
  simp only [← c83331, ← c83332, ← c83333] at e_1357_2
  have e_1357_3 := arithEq_of_rows h (row := 1357) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_3
  simp only [← c83334, ← c83335, k2, ← c83336] at e_1357_3
  have hr := e_1357_3
  simp only [e_1357_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12810)) := by
  have c83337 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.1
  have c83338 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83339 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83340 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83341 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83342 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_4 := arithEq_of_rows h (row := 1357) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_4
  simp only [← c83337, ← c83338, ← c83339] at e_1357_4
  have e_1357_5 := arithEq_of_rows h (row := 1357) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_5
  simp only [← c83340, ← c83341, k2, ← c83342] at e_1357_5
  have hr := e_1357_5
  simp only [e_1357_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12811)) := by
  have c83343 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83344 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83345 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83346 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83347 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83348 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_6 := arithEq_of_rows h (row := 1357) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_6
  simp only [← c83343, ← c83344, ← c83345] at e_1357_6
  have e_1357_7 := arithEq_of_rows h (row := 1357) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_7
  simp only [← c83346, ← c83347, k2, ← c83348] at e_1357_7
  have hr := e_1357_7
  simp only [e_1357_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12812)) := by
  have c83349 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83350 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83351 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83352 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83353 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83354 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_8 := arithEq_of_rows h (row := 1357) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_8
  simp only [← c83349, ← c83350, ← c83351] at e_1357_8
  have e_1357_9 := arithEq_of_rows h (row := 1357) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_9
  simp only [← c83352, ← c83353, k2, ← c83354] at e_1357_9
  have hr := e_1357_9
  simp only [e_1357_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12813)) := by
  have c83355 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83356 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83357 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83358 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83359 := (publicBatchWrapper86_copies2604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83360 := (publicBatchWrapper86_copies2605 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_10 := arithEq_of_rows h (row := 1357) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_10
  simp only [← c83355, ← c83356, ← c83357] at e_1357_10
  have e_1357_11 := arithEq_of_rows h (row := 1357) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_11
  simp only [← c83358, ← c83359, k2, ← c83360] at e_1357_11
  have hr := e_1357_11
  simp only [e_1357_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12814)) := by
  have c83361 := (publicBatchWrapper86_copies2605 a h).2.1
  have c83362 := (publicBatchWrapper86_copies2605 a h).2.2.1
  have c83363 := (publicBatchWrapper86_copies2605 a h).2.2.2.1
  have c83364 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.1
  have c83365 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.1
  have c83366 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_12 := arithEq_of_rows h (row := 1357) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_12
  simp only [← c83361, ← c83362, ← c83363] at e_1357_12
  have e_1357_13 := arithEq_of_rows h (row := 1357) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_13
  simp only [← c83364, ← c83365, k2, ← c83366] at e_1357_13
  have hr := e_1357_13
  simp only [e_1357_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12815)) := by
  have c83367 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.1
  have c83368 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.1
  have c83369 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.1
  have c83370 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83371 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83372 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_14 := arithEq_of_rows h (row := 1357) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_14
  simp only [← c83367, ← c83368, ← c83369] at e_1357_14
  have e_1357_15 := arithEq_of_rows h (row := 1357) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_15
  simp only [← c83370, ← c83371, k2, ← c83372] at e_1357_15
  have hr := e_1357_15
  simp only [e_1357_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12816)) := by
  have c83373 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83374 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83375 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83376 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83377 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83378 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_16 := arithEq_of_rows h (row := 1357) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_16
  simp only [← c83373, ← c83374, ← c83375] at e_1357_16
  have e_1357_17 := arithEq_of_rows h (row := 1357) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_17
  simp only [← c83376, ← c83377, k2, ← c83378] at e_1357_17
  have hr := e_1357_17
  simp only [e_1357_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1357 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12817)) := by
  have c83379 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83380 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83381 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83382 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83383 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83384 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1357_18 := arithEq_of_rows h (row := 1357) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_18
  simp only [← c83379, ← c83380, ← c83381] at e_1357_18
  have e_1357_19 := arithEq_of_rows h (row := 1357) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1357_19
  simp only [← c83382, ← c83383, k2, ← c83384] at e_1357_19
  have hr := e_1357_19
  simp only [e_1357_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12818)) := by
  have c83385 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83386 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83387 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83388 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83389 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83390 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_0 := arithEq_of_rows h (row := 1358) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_0
  simp only [← c83385, ← c83386, ← c83387] at e_1358_0
  have e_1358_1 := arithEq_of_rows h (row := 1358) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_1
  simp only [← c83388, ← c83389, k2, ← c83390] at e_1358_1
  have hr := e_1358_1
  simp only [e_1358_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12819)) := by
  have c83391 := (publicBatchWrapper86_copies2605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83392 := (publicBatchWrapper86_copies2606 a h).1
  have c83393 := (publicBatchWrapper86_copies2606 a h).2.1
  have c83394 := (publicBatchWrapper86_copies2606 a h).2.2.1
  have c83395 := (publicBatchWrapper86_copies2606 a h).2.2.2.1
  have c83396 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_2 := arithEq_of_rows h (row := 1358) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_2
  simp only [← c83391, ← c83392, ← c83393] at e_1358_2
  have e_1358_3 := arithEq_of_rows h (row := 1358) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_3
  simp only [← c83394, ← c83395, k2, ← c83396] at e_1358_3
  have hr := e_1358_3
  simp only [e_1358_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12820)) := by
  have c83397 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.1
  have c83398 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.1
  have c83399 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.1
  have c83400 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.1
  have c83401 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.1
  have c83402 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_4 := arithEq_of_rows h (row := 1358) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_4
  simp only [← c83397, ← c83398, ← c83399] at e_1358_4
  have e_1358_5 := arithEq_of_rows h (row := 1358) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_5
  simp only [← c83400, ← c83401, k2, ← c83402] at e_1358_5
  have hr := e_1358_5
  simp only [e_1358_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12821)) := by
  have c83403 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83404 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83405 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83406 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83407 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83408 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_6 := arithEq_of_rows h (row := 1358) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_6
  simp only [← c83403, ← c83404, ← c83405] at e_1358_6
  have e_1358_7 := arithEq_of_rows h (row := 1358) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_7
  simp only [← c83406, ← c83407, k2, ← c83408] at e_1358_7
  have hr := e_1358_7
  simp only [e_1358_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12822)) := by
  have c83409 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83410 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83411 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83412 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83413 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83414 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_8 := arithEq_of_rows h (row := 1358) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_8
  simp only [← c83409, ← c83410, ← c83411] at e_1358_8
  have e_1358_9 := arithEq_of_rows h (row := 1358) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_9
  simp only [← c83412, ← c83413, k2, ← c83414] at e_1358_9
  have hr := e_1358_9
  simp only [e_1358_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12823)) := by
  have c83415 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83416 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83417 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83418 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83419 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83420 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_10 := arithEq_of_rows h (row := 1358) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_10
  simp only [← c83415, ← c83416, ← c83417] at e_1358_10
  have e_1358_11 := arithEq_of_rows h (row := 1358) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_11
  simp only [← c83418, ← c83419, k2, ← c83420] at e_1358_11
  have hr := e_1358_11
  simp only [e_1358_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12824)) := by
  have c83421 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83422 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83423 := (publicBatchWrapper86_copies2606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83424 := (publicBatchWrapper86_copies2607 a h).1
  have c83425 := (publicBatchWrapper86_copies2607 a h).2.1
  have c83426 := (publicBatchWrapper86_copies2607 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_12 := arithEq_of_rows h (row := 1358) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_12
  simp only [← c83421, ← c83422, ← c83423] at e_1358_12
  have e_1358_13 := arithEq_of_rows h (row := 1358) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_13
  simp only [← c83424, ← c83425, k2, ← c83426] at e_1358_13
  have hr := e_1358_13
  simp only [e_1358_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12825)) := by
  have c83427 := (publicBatchWrapper86_copies2607 a h).2.2.2.1
  have c83428 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.1
  have c83429 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.1
  have c83430 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.1
  have c83431 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.1
  have c83432 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_14 := arithEq_of_rows h (row := 1358) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_14
  simp only [← c83427, ← c83428, ← c83429] at e_1358_14
  have e_1358_15 := arithEq_of_rows h (row := 1358) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_15
  simp only [← c83430, ← c83431, k2, ← c83432] at e_1358_15
  have hr := e_1358_15
  simp only [e_1358_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12826)) := by
  have c83433 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.1
  have c83434 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83435 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83436 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83437 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83438 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_16 := arithEq_of_rows h (row := 1358) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_16
  simp only [← c83433, ← c83434, ← c83435] at e_1358_16
  have e_1358_17 := arithEq_of_rows h (row := 1358) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_17
  simp only [← c83436, ← c83437, k2, ← c83438] at e_1358_17
  have hr := e_1358_17
  simp only [e_1358_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1358 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12827)) := by
  have c83439 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83440 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83441 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83442 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83443 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83444 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1358_18 := arithEq_of_rows h (row := 1358) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_18
  simp only [← c83439, ← c83440, ← c83441] at e_1358_18
  have e_1358_19 := arithEq_of_rows h (row := 1358) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1358_19
  simp only [← c83442, ← c83443, k2, ← c83444] at e_1358_19
  have hr := e_1358_19
  simp only [e_1358_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12828)) := by
  have c83445 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83446 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83447 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83448 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83449 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83450 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_0 := arithEq_of_rows h (row := 1359) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_0
  simp only [← c83445, ← c83446, ← c83447] at e_1359_0
  have e_1359_1 := arithEq_of_rows h (row := 1359) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_1
  simp only [← c83448, ← c83449, k2, ← c83450] at e_1359_1
  have hr := e_1359_1
  simp only [e_1359_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12829)) := by
  have c83451 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83452 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83453 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83454 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83455 := (publicBatchWrapper86_copies2607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83456 := (publicBatchWrapper86_copies2608 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_2 := arithEq_of_rows h (row := 1359) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_2
  simp only [← c83451, ← c83452, ← c83453] at e_1359_2
  have e_1359_3 := arithEq_of_rows h (row := 1359) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_3
  simp only [← c83454, ← c83455, k2, ← c83456] at e_1359_3
  have hr := e_1359_3
  simp only [e_1359_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12830)) := by
  have c83457 := (publicBatchWrapper86_copies2608 a h).2.1
  have c83458 := (publicBatchWrapper86_copies2608 a h).2.2.1
  have c83459 := (publicBatchWrapper86_copies2608 a h).2.2.2.1
  have c83460 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.1
  have c83461 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.1
  have c83462 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_4 := arithEq_of_rows h (row := 1359) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_4
  simp only [← c83457, ← c83458, ← c83459] at e_1359_4
  have e_1359_5 := arithEq_of_rows h (row := 1359) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_5
  simp only [← c83460, ← c83461, k2, ← c83462] at e_1359_5
  have hr := e_1359_5
  simp only [e_1359_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12831)) := by
  have c83463 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.1
  have c83464 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.1
  have c83465 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.1
  have c83466 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83467 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83468 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_6 := arithEq_of_rows h (row := 1359) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_6
  simp only [← c83463, ← c83464, ← c83465] at e_1359_6
  have e_1359_7 := arithEq_of_rows h (row := 1359) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_7
  simp only [← c83466, ← c83467, k2, ← c83468] at e_1359_7
  have hr := e_1359_7
  simp only [e_1359_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12832)) := by
  have c83469 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83470 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83471 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83472 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83473 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83474 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_8 := arithEq_of_rows h (row := 1359) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_8
  simp only [← c83469, ← c83470, ← c83471] at e_1359_8
  have e_1359_9 := arithEq_of_rows h (row := 1359) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_9
  simp only [← c83472, ← c83473, k2, ← c83474] at e_1359_9
  have hr := e_1359_9
  simp only [e_1359_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12833)) := by
  have c83475 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83476 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83477 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83478 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83479 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83480 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_10 := arithEq_of_rows h (row := 1359) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_10
  simp only [← c83475, ← c83476, ← c83477] at e_1359_10
  have e_1359_11 := arithEq_of_rows h (row := 1359) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_11
  simp only [← c83478, ← c83479, k2, ← c83480] at e_1359_11
  have hr := e_1359_11
  simp only [e_1359_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12834)) := by
  have c83481 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83482 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83483 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83484 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83485 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83486 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_12 := arithEq_of_rows h (row := 1359) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_12
  simp only [← c83481, ← c83482, ← c83483] at e_1359_12
  have e_1359_13 := arithEq_of_rows h (row := 1359) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_13
  simp only [← c83484, ← c83485, k2, ← c83486] at e_1359_13
  have hr := e_1359_13
  simp only [e_1359_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12835)) := by
  have c83487 := (publicBatchWrapper86_copies2608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83488 := (publicBatchWrapper86_copies2609 a h).1
  have c83489 := (publicBatchWrapper86_copies2609 a h).2.1
  have c83490 := (publicBatchWrapper86_copies2609 a h).2.2.1
  have c83491 := (publicBatchWrapper86_copies2609 a h).2.2.2.1
  have c83492 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_14 := arithEq_of_rows h (row := 1359) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_14
  simp only [← c83487, ← c83488, ← c83489] at e_1359_14
  have e_1359_15 := arithEq_of_rows h (row := 1359) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_15
  simp only [← c83490, ← c83491, k2, ← c83492] at e_1359_15
  have hr := e_1359_15
  simp only [e_1359_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12836)) := by
  have c83493 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.1
  have c83494 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.1
  have c83495 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.1
  have c83496 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.1
  have c83497 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.1
  have c83498 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_16 := arithEq_of_rows h (row := 1359) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_16
  simp only [← c83493, ← c83494, ← c83495] at e_1359_16
  have e_1359_17 := arithEq_of_rows h (row := 1359) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_17
  simp only [← c83496, ← c83497, k2, ← c83498] at e_1359_17
  have hr := e_1359_17
  simp only [e_1359_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1359 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12837)) := by
  have c83499 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83500 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83501 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83502 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83503 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83504 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1359_18 := arithEq_of_rows h (row := 1359) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_18
  simp only [← c83499, ← c83500, ← c83501] at e_1359_18
  have e_1359_19 := arithEq_of_rows h (row := 1359) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1359_19
  simp only [← c83502, ← c83503, k2, ← c83504] at e_1359_19
  have hr := e_1359_19
  simp only [e_1359_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12838)) := by
  have c83505 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83506 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83507 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83508 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83509 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83510 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_0 := arithEq_of_rows h (row := 1360) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_0
  simp only [← c83505, ← c83506, ← c83507] at e_1360_0
  have e_1360_1 := arithEq_of_rows h (row := 1360) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_1
  simp only [← c83508, ← c83509, k2, ← c83510] at e_1360_1
  have hr := e_1360_1
  simp only [e_1360_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12839)) := by
  have c83511 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83512 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83513 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83514 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83515 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83516 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_2 := arithEq_of_rows h (row := 1360) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_2
  simp only [← c83511, ← c83512, ← c83513] at e_1360_2
  have e_1360_3 := arithEq_of_rows h (row := 1360) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_3
  simp only [← c83514, ← c83515, k2, ← c83516] at e_1360_3
  have hr := e_1360_3
  simp only [e_1360_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12840)) := by
  have c83517 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83518 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83519 := (publicBatchWrapper86_copies2609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83520 := (publicBatchWrapper86_copies2610 a h).1
  have c83521 := (publicBatchWrapper86_copies2610 a h).2.1
  have c83522 := (publicBatchWrapper86_copies2610 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_4 := arithEq_of_rows h (row := 1360) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_4
  simp only [← c83517, ← c83518, ← c83519] at e_1360_4
  have e_1360_5 := arithEq_of_rows h (row := 1360) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_5
  simp only [← c83520, ← c83521, k2, ← c83522] at e_1360_5
  have hr := e_1360_5
  simp only [e_1360_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12841)) := by
  have c83523 := (publicBatchWrapper86_copies2610 a h).2.2.2.1
  have c83524 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.1
  have c83525 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.1
  have c83526 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.1
  have c83527 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.1
  have c83528 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_6 := arithEq_of_rows h (row := 1360) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_6
  simp only [← c83523, ← c83524, ← c83525] at e_1360_6
  have e_1360_7 := arithEq_of_rows h (row := 1360) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_7
  simp only [← c83526, ← c83527, k2, ← c83528] at e_1360_7
  have hr := e_1360_7
  simp only [e_1360_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12842)) := by
  have c83529 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.1
  have c83530 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83531 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83532 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83533 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83534 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_8 := arithEq_of_rows h (row := 1360) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_8
  simp only [← c83529, ← c83530, ← c83531] at e_1360_8
  have e_1360_9 := arithEq_of_rows h (row := 1360) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_9
  simp only [← c83532, ← c83533, k2, ← c83534] at e_1360_9
  have hr := e_1360_9
  simp only [e_1360_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12843)) := by
  have c83535 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83536 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83537 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83538 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83539 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83540 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_10 := arithEq_of_rows h (row := 1360) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_10
  simp only [← c83535, ← c83536, ← c83537] at e_1360_10
  have e_1360_11 := arithEq_of_rows h (row := 1360) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_11
  simp only [← c83538, ← c83539, k2, ← c83540] at e_1360_11
  have hr := e_1360_11
  simp only [e_1360_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12844)) := by
  have c83541 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83542 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83543 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83544 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83545 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83546 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_12 := arithEq_of_rows h (row := 1360) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_12
  simp only [← c83541, ← c83542, ← c83543] at e_1360_12
  have e_1360_13 := arithEq_of_rows h (row := 1360) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_13
  simp only [← c83544, ← c83545, k2, ← c83546] at e_1360_13
  have hr := e_1360_13
  simp only [e_1360_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12845)) := by
  have c83547 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83548 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83549 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83550 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83551 := (publicBatchWrapper86_copies2610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83552 := (publicBatchWrapper86_copies2611 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_14 := arithEq_of_rows h (row := 1360) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_14
  simp only [← c83547, ← c83548, ← c83549] at e_1360_14
  have e_1360_15 := arithEq_of_rows h (row := 1360) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_15
  simp only [← c83550, ← c83551, k2, ← c83552] at e_1360_15
  have hr := e_1360_15
  simp only [e_1360_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12846)) := by
  have c83553 := (publicBatchWrapper86_copies2611 a h).2.1
  have c83554 := (publicBatchWrapper86_copies2611 a h).2.2.1
  have c83555 := (publicBatchWrapper86_copies2611 a h).2.2.2.1
  have c83556 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.1
  have c83557 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.1
  have c83558 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_16 := arithEq_of_rows h (row := 1360) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_16
  simp only [← c83553, ← c83554, ← c83555] at e_1360_16
  have e_1360_17 := arithEq_of_rows h (row := 1360) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_17
  simp only [← c83556, ← c83557, k2, ← c83558] at e_1360_17
  have hr := e_1360_17
  simp only [e_1360_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1360 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12847)) := by
  have c83559 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.1
  have c83560 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.1
  have c83561 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.1
  have c83562 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83563 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83564 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1360_18 := arithEq_of_rows h (row := 1360) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_18
  simp only [← c83559, ← c83560, ← c83561] at e_1360_18
  have e_1360_19 := arithEq_of_rows h (row := 1360) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1360_19
  simp only [← c83562, ← c83563, k2, ← c83564] at e_1360_19
  have hr := e_1360_19
  simp only [e_1360_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12848)) := by
  have c83565 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83566 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83567 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83568 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83569 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83570 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_0 := arithEq_of_rows h (row := 1361) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_0
  simp only [← c83565, ← c83566, ← c83567] at e_1361_0
  have e_1361_1 := arithEq_of_rows h (row := 1361) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_1
  simp only [← c83568, ← c83569, k2, ← c83570] at e_1361_1
  have hr := e_1361_1
  simp only [e_1361_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12849)) := by
  have c83571 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83572 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83573 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83574 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83575 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83576 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_2 := arithEq_of_rows h (row := 1361) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_2
  simp only [← c83571, ← c83572, ← c83573] at e_1361_2
  have e_1361_3 := arithEq_of_rows h (row := 1361) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_3
  simp only [← c83574, ← c83575, k2, ← c83576] at e_1361_3
  have hr := e_1361_3
  simp only [e_1361_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12850)) := by
  have c83577 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83578 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83579 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83580 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83581 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83582 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_4 := arithEq_of_rows h (row := 1361) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_4
  simp only [← c83577, ← c83578, ← c83579] at e_1361_4
  have e_1361_5 := arithEq_of_rows h (row := 1361) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_5
  simp only [← c83580, ← c83581, k2, ← c83582] at e_1361_5
  have hr := e_1361_5
  simp only [e_1361_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12851)) := by
  have c83583 := (publicBatchWrapper86_copies2611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83584 := (publicBatchWrapper86_copies2612 a h).1
  have c83585 := (publicBatchWrapper86_copies2612 a h).2.1
  have c83586 := (publicBatchWrapper86_copies2612 a h).2.2.1
  have c83587 := (publicBatchWrapper86_copies2612 a h).2.2.2.1
  have c83588 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_6 := arithEq_of_rows h (row := 1361) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_6
  simp only [← c83583, ← c83584, ← c83585] at e_1361_6
  have e_1361_7 := arithEq_of_rows h (row := 1361) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_7
  simp only [← c83586, ← c83587, k2, ← c83588] at e_1361_7
  have hr := e_1361_7
  simp only [e_1361_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12852)) := by
  have c83589 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.1
  have c83590 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.1
  have c83591 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.1
  have c83592 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.1
  have c83593 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.1
  have c83594 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_8 := arithEq_of_rows h (row := 1361) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_8
  simp only [← c83589, ← c83590, ← c83591] at e_1361_8
  have e_1361_9 := arithEq_of_rows h (row := 1361) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_9
  simp only [← c83592, ← c83593, k2, ← c83594] at e_1361_9
  have hr := e_1361_9
  simp only [e_1361_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12853)) := by
  have c83595 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83596 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83597 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83598 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83599 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83600 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_10 := arithEq_of_rows h (row := 1361) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_10
  simp only [← c83595, ← c83596, ← c83597] at e_1361_10
  have e_1361_11 := arithEq_of_rows h (row := 1361) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_11
  simp only [← c83598, ← c83599, k2, ← c83600] at e_1361_11
  have hr := e_1361_11
  simp only [e_1361_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12854)) := by
  have c83601 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83602 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83603 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83604 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83605 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83606 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_12 := arithEq_of_rows h (row := 1361) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_12
  simp only [← c83601, ← c83602, ← c83603] at e_1361_12
  have e_1361_13 := arithEq_of_rows h (row := 1361) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_13
  simp only [← c83604, ← c83605, k2, ← c83606] at e_1361_13
  have hr := e_1361_13
  simp only [e_1361_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12855)) := by
  have c83607 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83608 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83609 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83610 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83611 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83612 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_14 := arithEq_of_rows h (row := 1361) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_14
  simp only [← c83607, ← c83608, ← c83609] at e_1361_14
  have e_1361_15 := arithEq_of_rows h (row := 1361) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_15
  simp only [← c83610, ← c83611, k2, ← c83612] at e_1361_15
  have hr := e_1361_15
  simp only [e_1361_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12856)) := by
  have c83613 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83614 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83615 := (publicBatchWrapper86_copies2612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83616 := (publicBatchWrapper86_copies2613 a h).1
  have c83617 := (publicBatchWrapper86_copies2613 a h).2.1
  have c83618 := (publicBatchWrapper86_copies2613 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_16 := arithEq_of_rows h (row := 1361) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_16
  simp only [← c83613, ← c83614, ← c83615] at e_1361_16
  have e_1361_17 := arithEq_of_rows h (row := 1361) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_17
  simp only [← c83616, ← c83617, k2, ← c83618] at e_1361_17
  have hr := e_1361_17
  simp only [e_1361_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1361 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12857)) := by
  have c83619 := (publicBatchWrapper86_copies2613 a h).2.2.2.1
  have c83620 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.1
  have c83621 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.1
  have c83622 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.1
  have c83623 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.1
  have c83624 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1361_18 := arithEq_of_rows h (row := 1361) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_18
  simp only [← c83619, ← c83620, ← c83621] at e_1361_18
  have e_1361_19 := arithEq_of_rows h (row := 1361) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1361_19
  simp only [← c83622, ← c83623, k2, ← c83624] at e_1361_19
  have hr := e_1361_19
  simp only [e_1361_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12858)) := by
  have c83625 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.1
  have c83626 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83627 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83628 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83629 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83630 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_0 := arithEq_of_rows h (row := 1362) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_0
  simp only [← c83625, ← c83626, ← c83627] at e_1362_0
  have e_1362_1 := arithEq_of_rows h (row := 1362) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_1
  simp only [← c83628, ← c83629, k2, ← c83630] at e_1362_1
  have hr := e_1362_1
  simp only [e_1362_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12859)) := by
  have c83631 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83632 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83633 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83634 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83635 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83636 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_2 := arithEq_of_rows h (row := 1362) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_2
  simp only [← c83631, ← c83632, ← c83633] at e_1362_2
  have e_1362_3 := arithEq_of_rows h (row := 1362) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_3
  simp only [← c83634, ← c83635, k2, ← c83636] at e_1362_3
  have hr := e_1362_3
  simp only [e_1362_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12860)) := by
  have c83637 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83638 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83639 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83640 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83641 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83642 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_4 := arithEq_of_rows h (row := 1362) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_4
  simp only [← c83637, ← c83638, ← c83639] at e_1362_4
  have e_1362_5 := arithEq_of_rows h (row := 1362) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_5
  simp only [← c83640, ← c83641, k2, ← c83642] at e_1362_5
  have hr := e_1362_5
  simp only [e_1362_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12861)) := by
  have c83643 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83644 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83645 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83646 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83647 := (publicBatchWrapper86_copies2613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83648 := (publicBatchWrapper86_copies2614 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_6 := arithEq_of_rows h (row := 1362) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_6
  simp only [← c83643, ← c83644, ← c83645] at e_1362_6
  have e_1362_7 := arithEq_of_rows h (row := 1362) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_7
  simp only [← c83646, ← c83647, k2, ← c83648] at e_1362_7
  have hr := e_1362_7
  simp only [e_1362_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12862)) := by
  have c83649 := (publicBatchWrapper86_copies2614 a h).2.1
  have c83650 := (publicBatchWrapper86_copies2614 a h).2.2.1
  have c83651 := (publicBatchWrapper86_copies2614 a h).2.2.2.1
  have c83652 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.1
  have c83653 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.1
  have c83654 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_8 := arithEq_of_rows h (row := 1362) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_8
  simp only [← c83649, ← c83650, ← c83651] at e_1362_8
  have e_1362_9 := arithEq_of_rows h (row := 1362) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_9
  simp only [← c83652, ← c83653, k2, ← c83654] at e_1362_9
  have hr := e_1362_9
  simp only [e_1362_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12863)) := by
  have c83655 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.1
  have c83656 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.1
  have c83657 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.1
  have c83658 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83659 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83660 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_10 := arithEq_of_rows h (row := 1362) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_10
  simp only [← c83655, ← c83656, ← c83657] at e_1362_10
  have e_1362_11 := arithEq_of_rows h (row := 1362) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_11
  simp only [← c83658, ← c83659, k2, ← c83660] at e_1362_11
  have hr := e_1362_11
  simp only [e_1362_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12864)) := by
  have c83661 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83662 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83663 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83664 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83665 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83666 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_12 := arithEq_of_rows h (row := 1362) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_12
  simp only [← c83661, ← c83662, ← c83663] at e_1362_12
  have e_1362_13 := arithEq_of_rows h (row := 1362) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_13
  simp only [← c83664, ← c83665, k2, ← c83666] at e_1362_13
  have hr := e_1362_13
  simp only [e_1362_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12865)) := by
  have c83667 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83668 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83669 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83670 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83671 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83672 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_14 := arithEq_of_rows h (row := 1362) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_14
  simp only [← c83667, ← c83668, ← c83669] at e_1362_14
  have e_1362_15 := arithEq_of_rows h (row := 1362) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_15
  simp only [← c83670, ← c83671, k2, ← c83672] at e_1362_15
  have hr := e_1362_15
  simp only [e_1362_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12866)) := by
  have c83673 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83674 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83675 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83676 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83677 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83678 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_16 := arithEq_of_rows h (row := 1362) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_16
  simp only [← c83673, ← c83674, ← c83675] at e_1362_16
  have e_1362_17 := arithEq_of_rows h (row := 1362) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_17
  simp only [← c83676, ← c83677, k2, ← c83678] at e_1362_17
  have hr := e_1362_17
  simp only [e_1362_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1362 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12867)) := by
  have c83679 := (publicBatchWrapper86_copies2614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83680 := (publicBatchWrapper86_copies2615 a h).1
  have c83681 := (publicBatchWrapper86_copies2615 a h).2.1
  have c83682 := (publicBatchWrapper86_copies2615 a h).2.2.1
  have c83683 := (publicBatchWrapper86_copies2615 a h).2.2.2.1
  have c83684 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1362_18 := arithEq_of_rows h (row := 1362) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_18
  simp only [← c83679, ← c83680, ← c83681] at e_1362_18
  have e_1362_19 := arithEq_of_rows h (row := 1362) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1362_19
  simp only [← c83682, ← c83683, k2, ← c83684] at e_1362_19
  have hr := e_1362_19
  simp only [e_1362_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12868)) := by
  have c83685 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.1
  have c83686 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.1
  have c83687 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.1
  have c83688 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.1
  have c83689 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.1
  have c83690 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_0 := arithEq_of_rows h (row := 1363) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_0
  simp only [← c83685, ← c83686, ← c83687] at e_1363_0
  have e_1363_1 := arithEq_of_rows h (row := 1363) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_1
  simp only [← c83688, ← c83689, k2, ← c83690] at e_1363_1
  have hr := e_1363_1
  simp only [e_1363_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12869)) := by
  have c83691 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83692 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83693 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83694 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83695 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83696 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_2 := arithEq_of_rows h (row := 1363) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_2
  simp only [← c83691, ← c83692, ← c83693] at e_1363_2
  have e_1363_3 := arithEq_of_rows h (row := 1363) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_3
  simp only [← c83694, ← c83695, k2, ← c83696] at e_1363_3
  have hr := e_1363_3
  simp only [e_1363_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12870)) := by
  have c83697 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83698 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83699 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83700 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83701 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83702 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_4 := arithEq_of_rows h (row := 1363) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_4
  simp only [← c83697, ← c83698, ← c83699] at e_1363_4
  have e_1363_5 := arithEq_of_rows h (row := 1363) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_5
  simp only [← c83700, ← c83701, k2, ← c83702] at e_1363_5
  have hr := e_1363_5
  simp only [e_1363_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12871)) := by
  have c83703 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83704 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83705 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83706 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83707 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83708 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_6 := arithEq_of_rows h (row := 1363) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_6
  simp only [← c83703, ← c83704, ← c83705] at e_1363_6
  have e_1363_7 := arithEq_of_rows h (row := 1363) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_7
  simp only [← c83706, ← c83707, k2, ← c83708] at e_1363_7
  have hr := e_1363_7
  simp only [e_1363_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12872)) := by
  have c83709 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83710 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83711 := (publicBatchWrapper86_copies2615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83712 := (publicBatchWrapper86_copies2616 a h).1
  have c83713 := (publicBatchWrapper86_copies2616 a h).2.1
  have c83714 := (publicBatchWrapper86_copies2616 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_8 := arithEq_of_rows h (row := 1363) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_8
  simp only [← c83709, ← c83710, ← c83711] at e_1363_8
  have e_1363_9 := arithEq_of_rows h (row := 1363) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_9
  simp only [← c83712, ← c83713, k2, ← c83714] at e_1363_9
  have hr := e_1363_9
  simp only [e_1363_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12873)) := by
  have c83715 := (publicBatchWrapper86_copies2616 a h).2.2.2.1
  have c83716 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.1
  have c83717 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.1
  have c83718 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.1
  have c83719 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.1
  have c83720 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_10 := arithEq_of_rows h (row := 1363) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_10
  simp only [← c83715, ← c83716, ← c83717] at e_1363_10
  have e_1363_11 := arithEq_of_rows h (row := 1363) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_11
  simp only [← c83718, ← c83719, k2, ← c83720] at e_1363_11
  have hr := e_1363_11
  simp only [e_1363_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12874)) := by
  have c83721 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.1
  have c83722 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83723 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83724 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83725 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83726 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_12 := arithEq_of_rows h (row := 1363) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_12
  simp only [← c83721, ← c83722, ← c83723] at e_1363_12
  have e_1363_13 := arithEq_of_rows h (row := 1363) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_13
  simp only [← c83724, ← c83725, k2, ← c83726] at e_1363_13
  have hr := e_1363_13
  simp only [e_1363_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12875)) := by
  have c83727 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83728 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83729 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83730 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83731 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83732 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_14 := arithEq_of_rows h (row := 1363) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_14
  simp only [← c83727, ← c83728, ← c83729] at e_1363_14
  have e_1363_15 := arithEq_of_rows h (row := 1363) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_15
  simp only [← c83730, ← c83731, k2, ← c83732] at e_1363_15
  have hr := e_1363_15
  simp only [e_1363_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12876)) := by
  have c83733 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83734 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83735 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83736 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83737 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83738 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_16 := arithEq_of_rows h (row := 1363) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_16
  simp only [← c83733, ← c83734, ← c83735] at e_1363_16
  have e_1363_17 := arithEq_of_rows h (row := 1363) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_17
  simp only [← c83736, ← c83737, k2, ← c83738] at e_1363_17
  have hr := e_1363_17
  simp only [e_1363_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1363 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12877)) := by
  have c83739 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83740 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83741 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83742 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83743 := (publicBatchWrapper86_copies2616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83744 := (publicBatchWrapper86_copies2617 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1363_18 := arithEq_of_rows h (row := 1363) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_18
  simp only [← c83739, ← c83740, ← c83741] at e_1363_18
  have e_1363_19 := arithEq_of_rows h (row := 1363) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1363_19
  simp only [← c83742, ← c83743, k2, ← c83744] at e_1363_19
  have hr := e_1363_19
  simp only [e_1363_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12878)) := by
  have c83745 := (publicBatchWrapper86_copies2617 a h).2.1
  have c83746 := (publicBatchWrapper86_copies2617 a h).2.2.1
  have c83747 := (publicBatchWrapper86_copies2617 a h).2.2.2.1
  have c83748 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.1
  have c83749 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.1
  have c83750 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_0 := arithEq_of_rows h (row := 1364) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_0
  simp only [← c83745, ← c83746, ← c83747] at e_1364_0
  have e_1364_1 := arithEq_of_rows h (row := 1364) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_1
  simp only [← c83748, ← c83749, k2, ← c83750] at e_1364_1
  have hr := e_1364_1
  simp only [e_1364_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12879)) := by
  have c83751 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.1
  have c83752 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.1
  have c83753 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.1
  have c83754 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83755 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83756 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_2 := arithEq_of_rows h (row := 1364) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_2
  simp only [← c83751, ← c83752, ← c83753] at e_1364_2
  have e_1364_3 := arithEq_of_rows h (row := 1364) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_3
  simp only [← c83754, ← c83755, k2, ← c83756] at e_1364_3
  have hr := e_1364_3
  simp only [e_1364_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12880)) := by
  have c83757 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83758 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83759 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83760 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83761 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83762 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_4 := arithEq_of_rows h (row := 1364) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_4
  simp only [← c83757, ← c83758, ← c83759] at e_1364_4
  have e_1364_5 := arithEq_of_rows h (row := 1364) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_5
  simp only [← c83760, ← c83761, k2, ← c83762] at e_1364_5
  have hr := e_1364_5
  simp only [e_1364_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12881)) := by
  have c83763 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83764 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83765 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83766 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83767 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83768 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_6 := arithEq_of_rows h (row := 1364) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_6
  simp only [← c83763, ← c83764, ← c83765] at e_1364_6
  have e_1364_7 := arithEq_of_rows h (row := 1364) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_7
  simp only [← c83766, ← c83767, k2, ← c83768] at e_1364_7
  have hr := e_1364_7
  simp only [e_1364_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12882)) := by
  have c83769 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83770 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83771 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83772 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83773 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83774 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_8 := arithEq_of_rows h (row := 1364) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_8
  simp only [← c83769, ← c83770, ← c83771] at e_1364_8
  have e_1364_9 := arithEq_of_rows h (row := 1364) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_9
  simp only [← c83772, ← c83773, k2, ← c83774] at e_1364_9
  have hr := e_1364_9
  simp only [e_1364_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12883)) := by
  have c83775 := (publicBatchWrapper86_copies2617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83776 := (publicBatchWrapper86_copies2618 a h).1
  have c83777 := (publicBatchWrapper86_copies2618 a h).2.1
  have c83778 := (publicBatchWrapper86_copies2618 a h).2.2.1
  have c83779 := (publicBatchWrapper86_copies2618 a h).2.2.2.1
  have c83780 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_10 := arithEq_of_rows h (row := 1364) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_10
  simp only [← c83775, ← c83776, ← c83777] at e_1364_10
  have e_1364_11 := arithEq_of_rows h (row := 1364) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_11
  simp only [← c83778, ← c83779, k2, ← c83780] at e_1364_11
  have hr := e_1364_11
  simp only [e_1364_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12884)) := by
  have c83781 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.1
  have c83782 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.1
  have c83783 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.1
  have c83784 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.1
  have c83785 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.1
  have c83786 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_12 := arithEq_of_rows h (row := 1364) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_12
  simp only [← c83781, ← c83782, ← c83783] at e_1364_12
  have e_1364_13 := arithEq_of_rows h (row := 1364) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_13
  simp only [← c83784, ← c83785, k2, ← c83786] at e_1364_13
  have hr := e_1364_13
  simp only [e_1364_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12885)) := by
  have c83787 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83788 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83789 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83790 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83791 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83792 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_14 := arithEq_of_rows h (row := 1364) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_14
  simp only [← c83787, ← c83788, ← c83789] at e_1364_14
  have e_1364_15 := arithEq_of_rows h (row := 1364) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_15
  simp only [← c83790, ← c83791, k2, ← c83792] at e_1364_15
  have hr := e_1364_15
  simp only [e_1364_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12886)) := by
  have c83793 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83794 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83795 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83796 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83797 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83798 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_16 := arithEq_of_rows h (row := 1364) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_16
  simp only [← c83793, ← c83794, ← c83795] at e_1364_16
  have e_1364_17 := arithEq_of_rows h (row := 1364) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_17
  simp only [← c83796, ← c83797, k2, ← c83798] at e_1364_17
  have hr := e_1364_17
  simp only [e_1364_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1364 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12887)) := by
  have c83799 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83800 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83801 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83802 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83803 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83804 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1364_18 := arithEq_of_rows h (row := 1364) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_18
  simp only [← c83799, ← c83800, ← c83801] at e_1364_18
  have e_1364_19 := arithEq_of_rows h (row := 1364) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1364_19
  simp only [← c83802, ← c83803, k2, ← c83804] at e_1364_19
  have hr := e_1364_19
  simp only [e_1364_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12888)) := by
  have c83805 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83806 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83807 := (publicBatchWrapper86_copies2618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83808 := (publicBatchWrapper86_copies2619 a h).1
  have c83809 := (publicBatchWrapper86_copies2619 a h).2.1
  have c83810 := (publicBatchWrapper86_copies2619 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_0 := arithEq_of_rows h (row := 1365) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_0
  simp only [← c83805, ← c83806, ← c83807] at e_1365_0
  have e_1365_1 := arithEq_of_rows h (row := 1365) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_1
  simp only [← c83808, ← c83809, k2, ← c83810] at e_1365_1
  have hr := e_1365_1
  simp only [e_1365_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12889)) := by
  have c83811 := (publicBatchWrapper86_copies2619 a h).2.2.2.1
  have c83812 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.1
  have c83813 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.1
  have c83814 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.1
  have c83815 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.1
  have c83816 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_2 := arithEq_of_rows h (row := 1365) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_2
  simp only [← c83811, ← c83812, ← c83813] at e_1365_2
  have e_1365_3 := arithEq_of_rows h (row := 1365) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_3
  simp only [← c83814, ← c83815, k2, ← c83816] at e_1365_3
  have hr := e_1365_3
  simp only [e_1365_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12890)) := by
  have c83817 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.1
  have c83818 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83819 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83820 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83821 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83822 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_4 := arithEq_of_rows h (row := 1365) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_4
  simp only [← c83817, ← c83818, ← c83819] at e_1365_4
  have e_1365_5 := arithEq_of_rows h (row := 1365) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_5
  simp only [← c83820, ← c83821, k2, ← c83822] at e_1365_5
  have hr := e_1365_5
  simp only [e_1365_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12891)) := by
  have c83823 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83824 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83825 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83826 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83827 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83828 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_6 := arithEq_of_rows h (row := 1365) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_6
  simp only [← c83823, ← c83824, ← c83825] at e_1365_6
  have e_1365_7 := arithEq_of_rows h (row := 1365) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_7
  simp only [← c83826, ← c83827, k2, ← c83828] at e_1365_7
  have hr := e_1365_7
  simp only [e_1365_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12892)) := by
  have c83829 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83830 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83831 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83832 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83833 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83834 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_8 := arithEq_of_rows h (row := 1365) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_8
  simp only [← c83829, ← c83830, ← c83831] at e_1365_8
  have e_1365_9 := arithEq_of_rows h (row := 1365) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_9
  simp only [← c83832, ← c83833, k2, ← c83834] at e_1365_9
  have hr := e_1365_9
  simp only [e_1365_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12893)) := by
  have c83835 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83836 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83837 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83838 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83839 := (publicBatchWrapper86_copies2619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83840 := (publicBatchWrapper86_copies2620 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_10 := arithEq_of_rows h (row := 1365) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_10
  simp only [← c83835, ← c83836, ← c83837] at e_1365_10
  have e_1365_11 := arithEq_of_rows h (row := 1365) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_11
  simp only [← c83838, ← c83839, k2, ← c83840] at e_1365_11
  have hr := e_1365_11
  simp only [e_1365_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12894)) := by
  have c83841 := (publicBatchWrapper86_copies2620 a h).2.1
  have c83842 := (publicBatchWrapper86_copies2620 a h).2.2.1
  have c83843 := (publicBatchWrapper86_copies2620 a h).2.2.2.1
  have c83844 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.1
  have c83845 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.1
  have c83846 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_12 := arithEq_of_rows h (row := 1365) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_12
  simp only [← c83841, ← c83842, ← c83843] at e_1365_12
  have e_1365_13 := arithEq_of_rows h (row := 1365) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_13
  simp only [← c83844, ← c83845, k2, ← c83846] at e_1365_13
  have hr := e_1365_13
  simp only [e_1365_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12895)) := by
  have c83847 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.1
  have c83848 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.1
  have c83849 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.1
  have c83850 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83851 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83852 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_14 := arithEq_of_rows h (row := 1365) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_14
  simp only [← c83847, ← c83848, ← c83849] at e_1365_14
  have e_1365_15 := arithEq_of_rows h (row := 1365) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_15
  simp only [← c83850, ← c83851, k2, ← c83852] at e_1365_15
  have hr := e_1365_15
  simp only [e_1365_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12896)) := by
  have c83853 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83854 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83855 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83856 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83857 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83858 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_16 := arithEq_of_rows h (row := 1365) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_16
  simp only [← c83853, ← c83854, ← c83855] at e_1365_16
  have e_1365_17 := arithEq_of_rows h (row := 1365) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_17
  simp only [← c83856, ← c83857, k2, ← c83858] at e_1365_17
  have hr := e_1365_17
  simp only [e_1365_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1365 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12897)) := by
  have c83859 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83860 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83861 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83862 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83863 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83864 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1365_18 := arithEq_of_rows h (row := 1365) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_18
  simp only [← c83859, ← c83860, ← c83861] at e_1365_18
  have e_1365_19 := arithEq_of_rows h (row := 1365) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1365_19
  simp only [← c83862, ← c83863, k2, ← c83864] at e_1365_19
  have hr := e_1365_19
  simp only [e_1365_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12898)) := by
  have c83865 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83866 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83867 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83868 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83869 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83870 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_0 := arithEq_of_rows h (row := 1366) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_0
  simp only [← c83865, ← c83866, ← c83867] at e_1366_0
  have e_1366_1 := arithEq_of_rows h (row := 1366) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_1
  simp only [← c83868, ← c83869, k2, ← c83870] at e_1366_1
  have hr := e_1366_1
  simp only [e_1366_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12899)) := by
  have c83871 := (publicBatchWrapper86_copies2620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83872 := (publicBatchWrapper86_copies2621 a h).1
  have c83873 := (publicBatchWrapper86_copies2621 a h).2.1
  have c83874 := (publicBatchWrapper86_copies2621 a h).2.2.1
  have c83875 := (publicBatchWrapper86_copies2621 a h).2.2.2.1
  have c83876 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_2 := arithEq_of_rows h (row := 1366) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_2
  simp only [← c83871, ← c83872, ← c83873] at e_1366_2
  have e_1366_3 := arithEq_of_rows h (row := 1366) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_3
  simp only [← c83874, ← c83875, k2, ← c83876] at e_1366_3
  have hr := e_1366_3
  simp only [e_1366_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12900)) := by
  have c83877 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.1
  have c83878 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.1
  have c83879 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.1
  have c83880 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.1
  have c83881 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.1
  have c83882 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_4 := arithEq_of_rows h (row := 1366) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_4
  simp only [← c83877, ← c83878, ← c83879] at e_1366_4
  have e_1366_5 := arithEq_of_rows h (row := 1366) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_5
  simp only [← c83880, ← c83881, k2, ← c83882] at e_1366_5
  have hr := e_1366_5
  simp only [e_1366_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12901)) := by
  have c83883 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83884 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83885 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83886 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83887 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83888 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_6 := arithEq_of_rows h (row := 1366) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_6
  simp only [← c83883, ← c83884, ← c83885] at e_1366_6
  have e_1366_7 := arithEq_of_rows h (row := 1366) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_7
  simp only [← c83886, ← c83887, k2, ← c83888] at e_1366_7
  have hr := e_1366_7
  simp only [e_1366_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12902)) := by
  have c83889 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83890 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83891 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83892 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83893 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83894 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_8 := arithEq_of_rows h (row := 1366) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_8
  simp only [← c83889, ← c83890, ← c83891] at e_1366_8
  have e_1366_9 := arithEq_of_rows h (row := 1366) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_9
  simp only [← c83892, ← c83893, k2, ← c83894] at e_1366_9
  have hr := e_1366_9
  simp only [e_1366_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12903)) := by
  have c83895 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83896 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83897 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83898 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83899 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83900 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_10 := arithEq_of_rows h (row := 1366) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_10
  simp only [← c83895, ← c83896, ← c83897] at e_1366_10
  have e_1366_11 := arithEq_of_rows h (row := 1366) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_11
  simp only [← c83898, ← c83899, k2, ← c83900] at e_1366_11
  have hr := e_1366_11
  simp only [e_1366_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12904)) := by
  have c83901 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83902 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83903 := (publicBatchWrapper86_copies2621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83904 := (publicBatchWrapper86_copies2622 a h).1
  have c83905 := (publicBatchWrapper86_copies2622 a h).2.1
  have c83906 := (publicBatchWrapper86_copies2622 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_12 := arithEq_of_rows h (row := 1366) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_12
  simp only [← c83901, ← c83902, ← c83903] at e_1366_12
  have e_1366_13 := arithEq_of_rows h (row := 1366) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_13
  simp only [← c83904, ← c83905, k2, ← c83906] at e_1366_13
  have hr := e_1366_13
  simp only [e_1366_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12905)) := by
  have c83907 := (publicBatchWrapper86_copies2622 a h).2.2.2.1
  have c83908 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.1
  have c83909 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.1
  have c83910 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.1
  have c83911 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.1
  have c83912 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_14 := arithEq_of_rows h (row := 1366) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_14
  simp only [← c83907, ← c83908, ← c83909] at e_1366_14
  have e_1366_15 := arithEq_of_rows h (row := 1366) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_15
  simp only [← c83910, ← c83911, k2, ← c83912] at e_1366_15
  have hr := e_1366_15
  simp only [e_1366_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12906)) := by
  have c83913 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.1
  have c83914 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83915 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83916 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83917 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83918 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_16 := arithEq_of_rows h (row := 1366) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_16
  simp only [← c83913, ← c83914, ← c83915] at e_1366_16
  have e_1366_17 := arithEq_of_rows h (row := 1366) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_17
  simp only [← c83916, ← c83917, k2, ← c83918] at e_1366_17
  have hr := e_1366_17
  simp only [e_1366_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1366 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12907)) := by
  have c83919 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83920 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83921 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83922 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83923 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83924 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1366_18 := arithEq_of_rows h (row := 1366) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_18
  simp only [← c83919, ← c83920, ← c83921] at e_1366_18
  have e_1366_19 := arithEq_of_rows h (row := 1366) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1366_19
  simp only [← c83922, ← c83923, k2, ← c83924] at e_1366_19
  have hr := e_1366_19
  simp only [e_1366_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12908)) := by
  have c83925 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83926 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83927 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83928 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83929 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83930 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_0 := arithEq_of_rows h (row := 1367) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_0
  simp only [← c83925, ← c83926, ← c83927] at e_1367_0
  have e_1367_1 := arithEq_of_rows h (row := 1367) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_1
  simp only [← c83928, ← c83929, k2, ← c83930] at e_1367_1
  have hr := e_1367_1
  simp only [e_1367_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12909)) := by
  have c83931 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83932 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83933 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83934 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83935 := (publicBatchWrapper86_copies2622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83936 := (publicBatchWrapper86_copies2623 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_2 := arithEq_of_rows h (row := 1367) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_2
  simp only [← c83931, ← c83932, ← c83933] at e_1367_2
  have e_1367_3 := arithEq_of_rows h (row := 1367) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_3
  simp only [← c83934, ← c83935, k2, ← c83936] at e_1367_3
  have hr := e_1367_3
  simp only [e_1367_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12910)) := by
  have c83937 := (publicBatchWrapper86_copies2623 a h).2.1
  have c83938 := (publicBatchWrapper86_copies2623 a h).2.2.1
  have c83939 := (publicBatchWrapper86_copies2623 a h).2.2.2.1
  have c83940 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.1
  have c83941 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.1
  have c83942 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_4 := arithEq_of_rows h (row := 1367) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_4
  simp only [← c83937, ← c83938, ← c83939] at e_1367_4
  have e_1367_5 := arithEq_of_rows h (row := 1367) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_5
  simp only [← c83940, ← c83941, k2, ← c83942] at e_1367_5
  have hr := e_1367_5
  simp only [e_1367_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12911)) := by
  have c83943 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.1
  have c83944 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.1
  have c83945 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.1
  have c83946 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83947 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83948 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_6 := arithEq_of_rows h (row := 1367) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_6
  simp only [← c83943, ← c83944, ← c83945] at e_1367_6
  have e_1367_7 := arithEq_of_rows h (row := 1367) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_7
  simp only [← c83946, ← c83947, k2, ← c83948] at e_1367_7
  have hr := e_1367_7
  simp only [e_1367_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12912)) := by
  have c83949 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83950 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83951 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83952 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83953 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83954 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_8 := arithEq_of_rows h (row := 1367) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_8
  simp only [← c83949, ← c83950, ← c83951] at e_1367_8
  have e_1367_9 := arithEq_of_rows h (row := 1367) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_9
  simp only [← c83952, ← c83953, k2, ← c83954] at e_1367_9
  have hr := e_1367_9
  simp only [e_1367_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12913)) := by
  have c83955 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83956 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83957 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83958 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83959 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83960 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_10 := arithEq_of_rows h (row := 1367) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_10
  simp only [← c83955, ← c83956, ← c83957] at e_1367_10
  have e_1367_11 := arithEq_of_rows h (row := 1367) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_11
  simp only [← c83958, ← c83959, k2, ← c83960] at e_1367_11
  have hr := e_1367_11
  simp only [e_1367_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12914)) := by
  have c83961 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83962 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83963 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83964 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83965 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83966 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_12 := arithEq_of_rows h (row := 1367) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_12
  simp only [← c83961, ← c83962, ← c83963] at e_1367_12
  have e_1367_13 := arithEq_of_rows h (row := 1367) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_13
  simp only [← c83964, ← c83965, k2, ← c83966] at e_1367_13
  have hr := e_1367_13
  simp only [e_1367_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12915)) := by
  have c83967 := (publicBatchWrapper86_copies2623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83968 := (publicBatchWrapper86_copies2624 a h).1
  have c83969 := (publicBatchWrapper86_copies2624 a h).2.1
  have c83970 := (publicBatchWrapper86_copies2624 a h).2.2.1
  have c83971 := (publicBatchWrapper86_copies2624 a h).2.2.2.1
  have c83972 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_14 := arithEq_of_rows h (row := 1367) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_14
  simp only [← c83967, ← c83968, ← c83969] at e_1367_14
  have e_1367_15 := arithEq_of_rows h (row := 1367) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_15
  simp only [← c83970, ← c83971, k2, ← c83972] at e_1367_15
  have hr := e_1367_15
  simp only [e_1367_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12916)) := by
  have c83973 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.1
  have c83974 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.1
  have c83975 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.1
  have c83976 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.1
  have c83977 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.1
  have c83978 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_16 := arithEq_of_rows h (row := 1367) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_16
  simp only [← c83973, ← c83974, ← c83975] at e_1367_16
  have e_1367_17 := arithEq_of_rows h (row := 1367) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_17
  simp only [← c83976, ← c83977, k2, ← c83978] at e_1367_17
  have hr := e_1367_17
  simp only [e_1367_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1367 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12917)) := by
  have c83979 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83980 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83981 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83982 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83983 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83984 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1367_18 := arithEq_of_rows h (row := 1367) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_18
  simp only [← c83979, ← c83980, ← c83981] at e_1367_18
  have e_1367_19 := arithEq_of_rows h (row := 1367) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1367_19
  simp only [← c83982, ← c83983, k2, ← c83984] at e_1367_19
  have hr := e_1367_19
  simp only [e_1367_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12918)) := by
  have c83985 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83986 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83987 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83988 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83989 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83990 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_0 := arithEq_of_rows h (row := 1368) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_0
  simp only [← c83985, ← c83986, ← c83987] at e_1368_0
  have e_1368_1 := arithEq_of_rows h (row := 1368) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_1
  simp only [← c83988, ← c83989, k2, ← c83990] at e_1368_1
  have hr := e_1368_1
  simp only [e_1368_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12919)) := by
  have c83991 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83992 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83993 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83994 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83995 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83996 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_2 := arithEq_of_rows h (row := 1368) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_2
  simp only [← c83991, ← c83992, ← c83993] at e_1368_2
  have e_1368_3 := arithEq_of_rows h (row := 1368) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_3
  simp only [← c83994, ← c83995, k2, ← c83996] at e_1368_3
  have hr := e_1368_3
  simp only [e_1368_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12920)) := by
  have c83997 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83998 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83999 := (publicBatchWrapper86_copies2624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84000 := (publicBatchWrapper86_copies2625 a h).1
  have c84001 := (publicBatchWrapper86_copies2625 a h).2.1
  have c84002 := (publicBatchWrapper86_copies2625 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_4 := arithEq_of_rows h (row := 1368) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_4
  simp only [← c83997, ← c83998, ← c83999] at e_1368_4
  have e_1368_5 := arithEq_of_rows h (row := 1368) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_5
  simp only [← c84000, ← c84001, k2, ← c84002] at e_1368_5
  have hr := e_1368_5
  simp only [e_1368_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12921)) := by
  have c84003 := (publicBatchWrapper86_copies2625 a h).2.2.2.1
  have c84004 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.1
  have c84005 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.1
  have c84006 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.1
  have c84007 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.1
  have c84008 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_6 := arithEq_of_rows h (row := 1368) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_6
  simp only [← c84003, ← c84004, ← c84005] at e_1368_6
  have e_1368_7 := arithEq_of_rows h (row := 1368) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_7
  simp only [← c84006, ← c84007, k2, ← c84008] at e_1368_7
  have hr := e_1368_7
  simp only [e_1368_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12922)) := by
  have c84009 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.1
  have c84010 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84011 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84012 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84013 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84014 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_8 := arithEq_of_rows h (row := 1368) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_8
  simp only [← c84009, ← c84010, ← c84011] at e_1368_8
  have e_1368_9 := arithEq_of_rows h (row := 1368) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_9
  simp only [← c84012, ← c84013, k2, ← c84014] at e_1368_9
  have hr := e_1368_9
  simp only [e_1368_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12923)) := by
  have c84015 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84016 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84017 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84018 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84019 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84020 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_10 := arithEq_of_rows h (row := 1368) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_10
  simp only [← c84015, ← c84016, ← c84017] at e_1368_10
  have e_1368_11 := arithEq_of_rows h (row := 1368) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_11
  simp only [← c84018, ← c84019, k2, ← c84020] at e_1368_11
  have hr := e_1368_11
  simp only [e_1368_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12924)) := by
  have c84021 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84022 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84023 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84024 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84025 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84026 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_12 := arithEq_of_rows h (row := 1368) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_12
  simp only [← c84021, ← c84022, ← c84023] at e_1368_12
  have e_1368_13 := arithEq_of_rows h (row := 1368) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_13
  simp only [← c84024, ← c84025, k2, ← c84026] at e_1368_13
  have hr := e_1368_13
  simp only [e_1368_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12925)) := by
  have c84027 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84028 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84029 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84030 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84031 := (publicBatchWrapper86_copies2625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84032 := (publicBatchWrapper86_copies2626 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_14 := arithEq_of_rows h (row := 1368) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_14
  simp only [← c84027, ← c84028, ← c84029] at e_1368_14
  have e_1368_15 := arithEq_of_rows h (row := 1368) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_15
  simp only [← c84030, ← c84031, k2, ← c84032] at e_1368_15
  have hr := e_1368_15
  simp only [e_1368_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12926)) := by
  have c84033 := (publicBatchWrapper86_copies2626 a h).2.1
  have c84034 := (publicBatchWrapper86_copies2626 a h).2.2.1
  have c84035 := (publicBatchWrapper86_copies2626 a h).2.2.2.1
  have c84036 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.1
  have c84037 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.1
  have c84038 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_16 := arithEq_of_rows h (row := 1368) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_16
  simp only [← c84033, ← c84034, ← c84035] at e_1368_16
  have e_1368_17 := arithEq_of_rows h (row := 1368) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_17
  simp only [← c84036, ← c84037, k2, ← c84038] at e_1368_17
  have hr := e_1368_17
  simp only [e_1368_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1368 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12927)) := by
  have c84039 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.1
  have c84040 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.1
  have c84041 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.1
  have c84042 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84043 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84044 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1368_18 := arithEq_of_rows h (row := 1368) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_18
  simp only [← c84039, ← c84040, ← c84041] at e_1368_18
  have e_1368_19 := arithEq_of_rows h (row := 1368) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1368_19
  simp only [← c84042, ← c84043, k2, ← c84044] at e_1368_19
  have hr := e_1368_19
  simp only [e_1368_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12928)) := by
  have c84045 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84046 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84047 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84048 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84049 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84050 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_0 := arithEq_of_rows h (row := 1369) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_0
  simp only [← c84045, ← c84046, ← c84047] at e_1369_0
  have e_1369_1 := arithEq_of_rows h (row := 1369) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_1
  simp only [← c84048, ← c84049, k2, ← c84050] at e_1369_1
  have hr := e_1369_1
  simp only [e_1369_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12929)) := by
  have c84051 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84052 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84053 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84054 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84055 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84056 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_2 := arithEq_of_rows h (row := 1369) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_2
  simp only [← c84051, ← c84052, ← c84053] at e_1369_2
  have e_1369_3 := arithEq_of_rows h (row := 1369) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_3
  simp only [← c84054, ← c84055, k2, ← c84056] at e_1369_3
  have hr := e_1369_3
  simp only [e_1369_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12930)) := by
  have c84057 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84058 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84059 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84060 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84061 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84062 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_4 := arithEq_of_rows h (row := 1369) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_4
  simp only [← c84057, ← c84058, ← c84059] at e_1369_4
  have e_1369_5 := arithEq_of_rows h (row := 1369) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_5
  simp only [← c84060, ← c84061, k2, ← c84062] at e_1369_5
  have hr := e_1369_5
  simp only [e_1369_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12931)) := by
  have c84063 := (publicBatchWrapper86_copies2626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84064 := (publicBatchWrapper86_copies2627 a h).1
  have c84065 := (publicBatchWrapper86_copies2627 a h).2.1
  have c84066 := (publicBatchWrapper86_copies2627 a h).2.2.1
  have c84067 := (publicBatchWrapper86_copies2627 a h).2.2.2.1
  have c84068 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_6 := arithEq_of_rows h (row := 1369) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_6
  simp only [← c84063, ← c84064, ← c84065] at e_1369_6
  have e_1369_7 := arithEq_of_rows h (row := 1369) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_7
  simp only [← c84066, ← c84067, k2, ← c84068] at e_1369_7
  have hr := e_1369_7
  simp only [e_1369_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12932)) := by
  have c84069 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.1
  have c84070 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.1
  have c84071 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.1
  have c84072 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.1
  have c84073 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.1
  have c84074 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_8 := arithEq_of_rows h (row := 1369) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_8
  simp only [← c84069, ← c84070, ← c84071] at e_1369_8
  have e_1369_9 := arithEq_of_rows h (row := 1369) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_9
  simp only [← c84072, ← c84073, k2, ← c84074] at e_1369_9
  have hr := e_1369_9
  simp only [e_1369_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12933)) := by
  have c84075 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84076 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84077 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84078 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84079 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84080 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_10 := arithEq_of_rows h (row := 1369) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_10
  simp only [← c84075, ← c84076, ← c84077] at e_1369_10
  have e_1369_11 := arithEq_of_rows h (row := 1369) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_11
  simp only [← c84078, ← c84079, k2, ← c84080] at e_1369_11
  have hr := e_1369_11
  simp only [e_1369_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12934)) := by
  have c84081 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84082 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84083 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84084 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84085 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84086 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_12 := arithEq_of_rows h (row := 1369) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_12
  simp only [← c84081, ← c84082, ← c84083] at e_1369_12
  have e_1369_13 := arithEq_of_rows h (row := 1369) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_13
  simp only [← c84084, ← c84085, k2, ← c84086] at e_1369_13
  have hr := e_1369_13
  simp only [e_1369_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12935)) := by
  have c84087 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84088 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84089 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84090 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84091 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84092 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_14 := arithEq_of_rows h (row := 1369) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_14
  simp only [← c84087, ← c84088, ← c84089] at e_1369_14
  have e_1369_15 := arithEq_of_rows h (row := 1369) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_15
  simp only [← c84090, ← c84091, k2, ← c84092] at e_1369_15
  have hr := e_1369_15
  simp only [e_1369_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12936)) := by
  have c84093 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84094 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84095 := (publicBatchWrapper86_copies2627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84096 := (publicBatchWrapper86_copies2628 a h).1
  have c84097 := (publicBatchWrapper86_copies2628 a h).2.1
  have c84098 := (publicBatchWrapper86_copies2628 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_16 := arithEq_of_rows h (row := 1369) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_16
  simp only [← c84093, ← c84094, ← c84095] at e_1369_16
  have e_1369_17 := arithEq_of_rows h (row := 1369) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_17
  simp only [← c84096, ← c84097, k2, ← c84098] at e_1369_17
  have hr := e_1369_17
  simp only [e_1369_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1369 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12937)) := by
  have c84099 := (publicBatchWrapper86_copies2628 a h).2.2.2.1
  have c84100 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.1
  have c84101 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.1
  have c84102 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.1
  have c84103 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.1
  have c84104 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1369_18 := arithEq_of_rows h (row := 1369) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_18
  simp only [← c84099, ← c84100, ← c84101] at e_1369_18
  have e_1369_19 := arithEq_of_rows h (row := 1369) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1369_19
  simp only [← c84102, ← c84103, k2, ← c84104] at e_1369_19
  have hr := e_1369_19
  simp only [e_1369_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12938)) := by
  have c84105 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.1
  have c84106 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84107 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84108 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84109 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84110 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_0 := arithEq_of_rows h (row := 1370) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_0
  simp only [← c84105, ← c84106, ← c84107] at e_1370_0
  have e_1370_1 := arithEq_of_rows h (row := 1370) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_1
  simp only [← c84108, ← c84109, k2, ← c84110] at e_1370_1
  have hr := e_1370_1
  simp only [e_1370_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12939)) := by
  have c84111 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84112 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84113 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84114 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84115 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84116 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_2 := arithEq_of_rows h (row := 1370) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_2
  simp only [← c84111, ← c84112, ← c84113] at e_1370_2
  have e_1370_3 := arithEq_of_rows h (row := 1370) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_3
  simp only [← c84114, ← c84115, k2, ← c84116] at e_1370_3
  have hr := e_1370_3
  simp only [e_1370_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12940)) := by
  have c84117 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84118 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84119 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84120 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84121 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84122 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_4 := arithEq_of_rows h (row := 1370) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_4
  simp only [← c84117, ← c84118, ← c84119] at e_1370_4
  have e_1370_5 := arithEq_of_rows h (row := 1370) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_5
  simp only [← c84120, ← c84121, k2, ← c84122] at e_1370_5
  have hr := e_1370_5
  simp only [e_1370_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12941)) := by
  have c84123 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84124 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84125 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84126 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84127 := (publicBatchWrapper86_copies2628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84128 := (publicBatchWrapper86_copies2629 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_6 := arithEq_of_rows h (row := 1370) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_6
  simp only [← c84123, ← c84124, ← c84125] at e_1370_6
  have e_1370_7 := arithEq_of_rows h (row := 1370) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_7
  simp only [← c84126, ← c84127, k2, ← c84128] at e_1370_7
  have hr := e_1370_7
  simp only [e_1370_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12942)) := by
  have c84129 := (publicBatchWrapper86_copies2629 a h).2.1
  have c84130 := (publicBatchWrapper86_copies2629 a h).2.2.1
  have c84131 := (publicBatchWrapper86_copies2629 a h).2.2.2.1
  have c84132 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.1
  have c84133 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.1
  have c84134 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_8 := arithEq_of_rows h (row := 1370) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_8
  simp only [← c84129, ← c84130, ← c84131] at e_1370_8
  have e_1370_9 := arithEq_of_rows h (row := 1370) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_9
  simp only [← c84132, ← c84133, k2, ← c84134] at e_1370_9
  have hr := e_1370_9
  simp only [e_1370_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12943)) := by
  have c84135 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.1
  have c84136 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.1
  have c84137 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.1
  have c84138 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84139 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84140 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_10 := arithEq_of_rows h (row := 1370) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_10
  simp only [← c84135, ← c84136, ← c84137] at e_1370_10
  have e_1370_11 := arithEq_of_rows h (row := 1370) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_11
  simp only [← c84138, ← c84139, k2, ← c84140] at e_1370_11
  have hr := e_1370_11
  simp only [e_1370_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12944)) := by
  have c84141 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84142 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84143 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84144 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84145 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84146 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_12 := arithEq_of_rows h (row := 1370) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_12
  simp only [← c84141, ← c84142, ← c84143] at e_1370_12
  have e_1370_13 := arithEq_of_rows h (row := 1370) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_13
  simp only [← c84144, ← c84145, k2, ← c84146] at e_1370_13
  have hr := e_1370_13
  simp only [e_1370_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12945)) := by
  have c84147 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84148 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84149 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84150 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84151 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84152 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_14 := arithEq_of_rows h (row := 1370) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_14
  simp only [← c84147, ← c84148, ← c84149] at e_1370_14
  have e_1370_15 := arithEq_of_rows h (row := 1370) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_15
  simp only [← c84150, ← c84151, k2, ← c84152] at e_1370_15
  have hr := e_1370_15
  simp only [e_1370_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12946)) := by
  have c84153 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84154 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84155 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84156 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84157 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84158 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_16 := arithEq_of_rows h (row := 1370) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_16
  simp only [← c84153, ← c84154, ← c84155] at e_1370_16
  have e_1370_17 := arithEq_of_rows h (row := 1370) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_17
  simp only [← c84156, ← c84157, k2, ← c84158] at e_1370_17
  have hr := e_1370_17
  simp only [e_1370_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1370 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12947)) := by
  have c84159 := (publicBatchWrapper86_copies2629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84160 := (publicBatchWrapper86_copies2630 a h).1
  have c84161 := (publicBatchWrapper86_copies2630 a h).2.1
  have c84162 := (publicBatchWrapper86_copies2630 a h).2.2.1
  have c84163 := (publicBatchWrapper86_copies2630 a h).2.2.2.1
  have c84164 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1370_18 := arithEq_of_rows h (row := 1370) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_18
  simp only [← c84159, ← c84160, ← c84161] at e_1370_18
  have e_1370_19 := arithEq_of_rows h (row := 1370) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1370_19
  simp only [← c84162, ← c84163, k2, ← c84164] at e_1370_19
  have hr := e_1370_19
  simp only [e_1370_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12948)) := by
  have c84165 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.1
  have c84166 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.1
  have c84167 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.1
  have c84168 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.1
  have c84169 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.1
  have c84170 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_0 := arithEq_of_rows h (row := 1371) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_0
  simp only [← c84165, ← c84166, ← c84167] at e_1371_0
  have e_1371_1 := arithEq_of_rows h (row := 1371) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_1
  simp only [← c84168, ← c84169, k2, ← c84170] at e_1371_1
  have hr := e_1371_1
  simp only [e_1371_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12949)) := by
  have c84171 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84172 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84173 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84174 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84175 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84176 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_2 := arithEq_of_rows h (row := 1371) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_2
  simp only [← c84171, ← c84172, ← c84173] at e_1371_2
  have e_1371_3 := arithEq_of_rows h (row := 1371) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_3
  simp only [← c84174, ← c84175, k2, ← c84176] at e_1371_3
  have hr := e_1371_3
  simp only [e_1371_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12950)) := by
  have c84177 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84178 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84179 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84180 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84181 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84182 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_4 := arithEq_of_rows h (row := 1371) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_4
  simp only [← c84177, ← c84178, ← c84179] at e_1371_4
  have e_1371_5 := arithEq_of_rows h (row := 1371) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_5
  simp only [← c84180, ← c84181, k2, ← c84182] at e_1371_5
  have hr := e_1371_5
  simp only [e_1371_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12951)) := by
  have c84183 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84184 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84185 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84186 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84187 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84188 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_6 := arithEq_of_rows h (row := 1371) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_6
  simp only [← c84183, ← c84184, ← c84185] at e_1371_6
  have e_1371_7 := arithEq_of_rows h (row := 1371) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_7
  simp only [← c84186, ← c84187, k2, ← c84188] at e_1371_7
  have hr := e_1371_7
  simp only [e_1371_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12952)) := by
  have c84189 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84190 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84191 := (publicBatchWrapper86_copies2630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84192 := (publicBatchWrapper86_copies2631 a h).1
  have c84193 := (publicBatchWrapper86_copies2631 a h).2.1
  have c84194 := (publicBatchWrapper86_copies2631 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_8 := arithEq_of_rows h (row := 1371) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_8
  simp only [← c84189, ← c84190, ← c84191] at e_1371_8
  have e_1371_9 := arithEq_of_rows h (row := 1371) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_9
  simp only [← c84192, ← c84193, k2, ← c84194] at e_1371_9
  have hr := e_1371_9
  simp only [e_1371_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12953)) := by
  have c84195 := (publicBatchWrapper86_copies2631 a h).2.2.2.1
  have c84196 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.1
  have c84197 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.1
  have c84198 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.1
  have c84199 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.1
  have c84200 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_10 := arithEq_of_rows h (row := 1371) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_10
  simp only [← c84195, ← c84196, ← c84197] at e_1371_10
  have e_1371_11 := arithEq_of_rows h (row := 1371) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_11
  simp only [← c84198, ← c84199, k2, ← c84200] at e_1371_11
  have hr := e_1371_11
  simp only [e_1371_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12954)) := by
  have c84201 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.1
  have c84202 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84203 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84204 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84205 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84206 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_12 := arithEq_of_rows h (row := 1371) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_12
  simp only [← c84201, ← c84202, ← c84203] at e_1371_12
  have e_1371_13 := arithEq_of_rows h (row := 1371) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_13
  simp only [← c84204, ← c84205, k2, ← c84206] at e_1371_13
  have hr := e_1371_13
  simp only [e_1371_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12955)) := by
  have c84207 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84208 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84209 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84210 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84211 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84212 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_14 := arithEq_of_rows h (row := 1371) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_14
  simp only [← c84207, ← c84208, ← c84209] at e_1371_14
  have e_1371_15 := arithEq_of_rows h (row := 1371) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_15
  simp only [← c84210, ← c84211, k2, ← c84212] at e_1371_15
  have hr := e_1371_15
  simp only [e_1371_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12956)) := by
  have c84213 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84214 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84215 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84216 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84217 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84218 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_16 := arithEq_of_rows h (row := 1371) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_16
  simp only [← c84213, ← c84214, ← c84215] at e_1371_16
  have e_1371_17 := arithEq_of_rows h (row := 1371) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_17
  simp only [← c84216, ← c84217, k2, ← c84218] at e_1371_17
  have hr := e_1371_17
  simp only [e_1371_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1371 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12957)) := by
  have c84219 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84220 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84221 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84222 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84223 := (publicBatchWrapper86_copies2631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84224 := (publicBatchWrapper86_copies2632 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1371_18 := arithEq_of_rows h (row := 1371) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_18
  simp only [← c84219, ← c84220, ← c84221] at e_1371_18
  have e_1371_19 := arithEq_of_rows h (row := 1371) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1371_19
  simp only [← c84222, ← c84223, k2, ← c84224] at e_1371_19
  have hr := e_1371_19
  simp only [e_1371_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12958)) := by
  have c84225 := (publicBatchWrapper86_copies2632 a h).2.1
  have c84226 := (publicBatchWrapper86_copies2632 a h).2.2.1
  have c84227 := (publicBatchWrapper86_copies2632 a h).2.2.2.1
  have c84228 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.1
  have c84229 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.1
  have c84230 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_0 := arithEq_of_rows h (row := 1372) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_0
  simp only [← c84225, ← c84226, ← c84227] at e_1372_0
  have e_1372_1 := arithEq_of_rows h (row := 1372) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_1
  simp only [← c84228, ← c84229, k2, ← c84230] at e_1372_1
  have hr := e_1372_1
  simp only [e_1372_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12959)) := by
  have c84231 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.1
  have c84232 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.1
  have c84233 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.1
  have c84234 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84235 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84236 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_2 := arithEq_of_rows h (row := 1372) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_2
  simp only [← c84231, ← c84232, ← c84233] at e_1372_2
  have e_1372_3 := arithEq_of_rows h (row := 1372) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_3
  simp only [← c84234, ← c84235, k2, ← c84236] at e_1372_3
  have hr := e_1372_3
  simp only [e_1372_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12960)) := by
  have c84237 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84238 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84239 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84240 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84241 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84242 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_4 := arithEq_of_rows h (row := 1372) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_4
  simp only [← c84237, ← c84238, ← c84239] at e_1372_4
  have e_1372_5 := arithEq_of_rows h (row := 1372) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_5
  simp only [← c84240, ← c84241, k2, ← c84242] at e_1372_5
  have hr := e_1372_5
  simp only [e_1372_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12961)) := by
  have c84243 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84244 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84245 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84246 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84247 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84248 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_6 := arithEq_of_rows h (row := 1372) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_6
  simp only [← c84243, ← c84244, ← c84245] at e_1372_6
  have e_1372_7 := arithEq_of_rows h (row := 1372) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_7
  simp only [← c84246, ← c84247, k2, ← c84248] at e_1372_7
  have hr := e_1372_7
  simp only [e_1372_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12962)) := by
  have c84249 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84250 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84251 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84252 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84253 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84254 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_8 := arithEq_of_rows h (row := 1372) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_8
  simp only [← c84249, ← c84250, ← c84251] at e_1372_8
  have e_1372_9 := arithEq_of_rows h (row := 1372) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_9
  simp only [← c84252, ← c84253, k2, ← c84254] at e_1372_9
  have hr := e_1372_9
  simp only [e_1372_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12963)) := by
  have c84255 := (publicBatchWrapper86_copies2632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84256 := (publicBatchWrapper86_copies2633 a h).1
  have c84257 := (publicBatchWrapper86_copies2633 a h).2.1
  have c84258 := (publicBatchWrapper86_copies2633 a h).2.2.1
  have c84259 := (publicBatchWrapper86_copies2633 a h).2.2.2.1
  have c84260 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_10 := arithEq_of_rows h (row := 1372) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_10
  simp only [← c84255, ← c84256, ← c84257] at e_1372_10
  have e_1372_11 := arithEq_of_rows h (row := 1372) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_11
  simp only [← c84258, ← c84259, k2, ← c84260] at e_1372_11
  have hr := e_1372_11
  simp only [e_1372_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12964)) := by
  have c84261 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.1
  have c84262 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.1
  have c84263 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.1
  have c84264 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.1
  have c84265 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.1
  have c84266 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_12 := arithEq_of_rows h (row := 1372) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_12
  simp only [← c84261, ← c84262, ← c84263] at e_1372_12
  have e_1372_13 := arithEq_of_rows h (row := 1372) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_13
  simp only [← c84264, ← c84265, k2, ← c84266] at e_1372_13
  have hr := e_1372_13
  simp only [e_1372_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12965)) := by
  have c84267 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84268 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84269 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84270 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84271 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84272 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_14 := arithEq_of_rows h (row := 1372) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_14
  simp only [← c84267, ← c84268, ← c84269] at e_1372_14
  have e_1372_15 := arithEq_of_rows h (row := 1372) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_15
  simp only [← c84270, ← c84271, k2, ← c84272] at e_1372_15
  have hr := e_1372_15
  simp only [e_1372_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12966)) := by
  have c84273 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84274 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84275 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84276 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84277 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84278 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_16 := arithEq_of_rows h (row := 1372) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_16
  simp only [← c84273, ← c84274, ← c84275] at e_1372_16
  have e_1372_17 := arithEq_of_rows h (row := 1372) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_17
  simp only [← c84276, ← c84277, k2, ← c84278] at e_1372_17
  have hr := e_1372_17
  simp only [e_1372_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1372 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12967)) := by
  have c84279 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84280 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84281 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84282 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84283 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84284 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1372_18 := arithEq_of_rows h (row := 1372) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_18
  simp only [← c84279, ← c84280, ← c84281] at e_1372_18
  have e_1372_19 := arithEq_of_rows h (row := 1372) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1372_19
  simp only [← c84282, ← c84283, k2, ← c84284] at e_1372_19
  have hr := e_1372_19
  simp only [e_1372_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12968)) := by
  have c84285 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84286 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84287 := (publicBatchWrapper86_copies2633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84288 := (publicBatchWrapper86_copies2634 a h).1
  have c84289 := (publicBatchWrapper86_copies2634 a h).2.1
  have c84290 := (publicBatchWrapper86_copies2634 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_0 := arithEq_of_rows h (row := 1373) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_0
  simp only [← c84285, ← c84286, ← c84287] at e_1373_0
  have e_1373_1 := arithEq_of_rows h (row := 1373) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_1
  simp only [← c84288, ← c84289, k2, ← c84290] at e_1373_1
  have hr := e_1373_1
  simp only [e_1373_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12969)) := by
  have c84291 := (publicBatchWrapper86_copies2634 a h).2.2.2.1
  have c84292 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.1
  have c84293 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.1
  have c84294 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.1
  have c84295 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.1
  have c84296 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_2 := arithEq_of_rows h (row := 1373) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_2
  simp only [← c84291, ← c84292, ← c84293] at e_1373_2
  have e_1373_3 := arithEq_of_rows h (row := 1373) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_3
  simp only [← c84294, ← c84295, k2, ← c84296] at e_1373_3
  have hr := e_1373_3
  simp only [e_1373_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13010)) := by
  have c84297 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.1
  have c84298 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84299 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84300 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84301 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84302 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_4 := arithEq_of_rows h (row := 1373) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_4
  simp only [← c84297, ← c84298, ← c84299] at e_1373_4
  have e_1373_5 := arithEq_of_rows h (row := 1373) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_5
  simp only [← c84300, ← c84301, k2, ← c84302] at e_1373_5
  have hr := e_1373_5
  simp only [e_1373_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13011)) := by
  have c84303 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84304 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84305 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84306 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84307 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84308 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_6 := arithEq_of_rows h (row := 1373) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_6
  simp only [← c84303, ← c84304, ← c84305] at e_1373_6
  have e_1373_7 := arithEq_of_rows h (row := 1373) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_7
  simp only [← c84306, ← c84307, k2, ← c84308] at e_1373_7
  have hr := e_1373_7
  simp only [e_1373_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13012)) := by
  have c84309 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84310 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84311 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84312 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84313 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84314 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_8 := arithEq_of_rows h (row := 1373) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_8
  simp only [← c84309, ← c84310, ← c84311] at e_1373_8
  have e_1373_9 := arithEq_of_rows h (row := 1373) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_9
  simp only [← c84312, ← c84313, k2, ← c84314] at e_1373_9
  have hr := e_1373_9
  simp only [e_1373_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13013)) := by
  have c84315 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84316 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84317 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84318 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84319 := (publicBatchWrapper86_copies2634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84320 := (publicBatchWrapper86_copies2635 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_10 := arithEq_of_rows h (row := 1373) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_10
  simp only [← c84315, ← c84316, ← c84317] at e_1373_10
  have e_1373_11 := arithEq_of_rows h (row := 1373) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_11
  simp only [← c84318, ← c84319, k2, ← c84320] at e_1373_11
  have hr := e_1373_11
  simp only [e_1373_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13014)) := by
  have c84321 := (publicBatchWrapper86_copies2635 a h).2.1
  have c84322 := (publicBatchWrapper86_copies2635 a h).2.2.1
  have c84323 := (publicBatchWrapper86_copies2635 a h).2.2.2.1
  have c84324 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.1
  have c84325 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.1
  have c84326 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_12 := arithEq_of_rows h (row := 1373) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_12
  simp only [← c84321, ← c84322, ← c84323] at e_1373_12
  have e_1373_13 := arithEq_of_rows h (row := 1373) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_13
  simp only [← c84324, ← c84325, k2, ← c84326] at e_1373_13
  have hr := e_1373_13
  simp only [e_1373_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13015)) := by
  have c84327 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.1
  have c84328 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.1
  have c84329 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.1
  have c84330 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84331 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84332 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_14 := arithEq_of_rows h (row := 1373) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_14
  simp only [← c84327, ← c84328, ← c84329] at e_1373_14
  have e_1373_15 := arithEq_of_rows h (row := 1373) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_15
  simp only [← c84330, ← c84331, k2, ← c84332] at e_1373_15
  have hr := e_1373_15
  simp only [e_1373_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13016)) := by
  have c84333 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84334 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84335 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84336 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84337 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84338 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_16 := arithEq_of_rows h (row := 1373) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_16
  simp only [← c84333, ← c84334, ← c84335] at e_1373_16
  have e_1373_17 := arithEq_of_rows h (row := 1373) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_17
  simp only [← c84336, ← c84337, k2, ← c84338] at e_1373_17
  have hr := e_1373_17
  simp only [e_1373_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1373 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13017)) := by
  have c84339 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84340 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84341 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84342 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84343 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84344 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1373_18 := arithEq_of_rows h (row := 1373) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_18
  simp only [← c84339, ← c84340, ← c84341] at e_1373_18
  have e_1373_19 := arithEq_of_rows h (row := 1373) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1373_19
  simp only [← c84342, ← c84343, k2, ← c84344] at e_1373_19
  have hr := e_1373_19
  simp only [e_1373_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13018)) := by
  have c84345 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84346 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84347 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84348 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84349 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84350 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_0 := arithEq_of_rows h (row := 1374) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_0
  simp only [← c84345, ← c84346, ← c84347] at e_1374_0
  have e_1374_1 := arithEq_of_rows h (row := 1374) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_1
  simp only [← c84348, ← c84349, k2, ← c84350] at e_1374_1
  have hr := e_1374_1
  simp only [e_1374_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13019)) := by
  have c84351 := (publicBatchWrapper86_copies2635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84352 := (publicBatchWrapper86_copies2636 a h).1
  have c84353 := (publicBatchWrapper86_copies2636 a h).2.1
  have c84354 := (publicBatchWrapper86_copies2636 a h).2.2.1
  have c84355 := (publicBatchWrapper86_copies2636 a h).2.2.2.1
  have c84356 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_2 := arithEq_of_rows h (row := 1374) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_2
  simp only [← c84351, ← c84352, ← c84353] at e_1374_2
  have e_1374_3 := arithEq_of_rows h (row := 1374) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_3
  simp only [← c84354, ← c84355, k2, ← c84356] at e_1374_3
  have hr := e_1374_3
  simp only [e_1374_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13020)) := by
  have c84357 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.1
  have c84358 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.1
  have c84359 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.1
  have c84360 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.1
  have c84361 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.1
  have c84362 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_4 := arithEq_of_rows h (row := 1374) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_4
  simp only [← c84357, ← c84358, ← c84359] at e_1374_4
  have e_1374_5 := arithEq_of_rows h (row := 1374) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_5
  simp only [← c84360, ← c84361, k2, ← c84362] at e_1374_5
  have hr := e_1374_5
  simp only [e_1374_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13021)) := by
  have c84363 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84364 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84365 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84366 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84367 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84368 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_6 := arithEq_of_rows h (row := 1374) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_6
  simp only [← c84363, ← c84364, ← c84365] at e_1374_6
  have e_1374_7 := arithEq_of_rows h (row := 1374) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_7
  simp only [← c84366, ← c84367, k2, ← c84368] at e_1374_7
  have hr := e_1374_7
  simp only [e_1374_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13022)) := by
  have c84369 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84370 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84371 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84372 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84373 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84374 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_8 := arithEq_of_rows h (row := 1374) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_8
  simp only [← c84369, ← c84370, ← c84371] at e_1374_8
  have e_1374_9 := arithEq_of_rows h (row := 1374) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_9
  simp only [← c84372, ← c84373, k2, ← c84374] at e_1374_9
  have hr := e_1374_9
  simp only [e_1374_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13023)) := by
  have c84375 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84376 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84377 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84378 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84379 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84380 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_10 := arithEq_of_rows h (row := 1374) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_10
  simp only [← c84375, ← c84376, ← c84377] at e_1374_10
  have e_1374_11 := arithEq_of_rows h (row := 1374) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_11
  simp only [← c84378, ← c84379, k2, ← c84380] at e_1374_11
  have hr := e_1374_11
  simp only [e_1374_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13024)) := by
  have c84381 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84382 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84383 := (publicBatchWrapper86_copies2636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84384 := (publicBatchWrapper86_copies2637 a h).1
  have c84385 := (publicBatchWrapper86_copies2637 a h).2.1
  have c84386 := (publicBatchWrapper86_copies2637 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_12 := arithEq_of_rows h (row := 1374) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_12
  simp only [← c84381, ← c84382, ← c84383] at e_1374_12
  have e_1374_13 := arithEq_of_rows h (row := 1374) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_13
  simp only [← c84384, ← c84385, k2, ← c84386] at e_1374_13
  have hr := e_1374_13
  simp only [e_1374_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13025)) := by
  have c84387 := (publicBatchWrapper86_copies2637 a h).2.2.2.1
  have c84388 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.1
  have c84389 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.1
  have c84390 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.1
  have c84391 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.1
  have c84392 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_14 := arithEq_of_rows h (row := 1374) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_14
  simp only [← c84387, ← c84388, ← c84389] at e_1374_14
  have e_1374_15 := arithEq_of_rows h (row := 1374) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_15
  simp only [← c84390, ← c84391, k2, ← c84392] at e_1374_15
  have hr := e_1374_15
  simp only [e_1374_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13026)) := by
  have c84393 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.1
  have c84394 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84395 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84396 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84397 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84398 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_16 := arithEq_of_rows h (row := 1374) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_16
  simp only [← c84393, ← c84394, ← c84395] at e_1374_16
  have e_1374_17 := arithEq_of_rows h (row := 1374) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_17
  simp only [← c84396, ← c84397, k2, ← c84398] at e_1374_17
  have hr := e_1374_17
  simp only [e_1374_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1374 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13027)) := by
  have c84399 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84400 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84401 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84402 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84403 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84404 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1374_18 := arithEq_of_rows h (row := 1374) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_18
  simp only [← c84399, ← c84400, ← c84401] at e_1374_18
  have e_1374_19 := arithEq_of_rows h (row := 1374) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1374_19
  simp only [← c84402, ← c84403, k2, ← c84404] at e_1374_19
  have hr := e_1374_19
  simp only [e_1374_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13028)) := by
  have c84405 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84406 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84407 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84408 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84409 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84410 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_0 := arithEq_of_rows h (row := 1375) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_0
  simp only [← c84405, ← c84406, ← c84407] at e_1375_0
  have e_1375_1 := arithEq_of_rows h (row := 1375) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_1
  simp only [← c84408, ← c84409, k2, ← c84410] at e_1375_1
  have hr := e_1375_1
  simp only [e_1375_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13029)) := by
  have c84411 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84412 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84413 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84414 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84415 := (publicBatchWrapper86_copies2637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84416 := (publicBatchWrapper86_copies2638 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_2 := arithEq_of_rows h (row := 1375) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_2
  simp only [← c84411, ← c84412, ← c84413] at e_1375_2
  have e_1375_3 := arithEq_of_rows h (row := 1375) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_3
  simp only [← c84414, ← c84415, k2, ← c84416] at e_1375_3
  have hr := e_1375_3
  simp only [e_1375_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13030)) := by
  have c84417 := (publicBatchWrapper86_copies2638 a h).2.1
  have c84418 := (publicBatchWrapper86_copies2638 a h).2.2.1
  have c84419 := (publicBatchWrapper86_copies2638 a h).2.2.2.1
  have c84420 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.1
  have c84421 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.1
  have c84422 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_4 := arithEq_of_rows h (row := 1375) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_4
  simp only [← c84417, ← c84418, ← c84419] at e_1375_4
  have e_1375_5 := arithEq_of_rows h (row := 1375) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_5
  simp only [← c84420, ← c84421, k2, ← c84422] at e_1375_5
  have hr := e_1375_5
  simp only [e_1375_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13031)) := by
  have c84423 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.1
  have c84424 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.1
  have c84425 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.1
  have c84426 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84427 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84428 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_6 := arithEq_of_rows h (row := 1375) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_6
  simp only [← c84423, ← c84424, ← c84425] at e_1375_6
  have e_1375_7 := arithEq_of_rows h (row := 1375) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_7
  simp only [← c84426, ← c84427, k2, ← c84428] at e_1375_7
  have hr := e_1375_7
  simp only [e_1375_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13032)) := by
  have c84429 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84430 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84431 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84432 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84433 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84434 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_8 := arithEq_of_rows h (row := 1375) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_8
  simp only [← c84429, ← c84430, ← c84431] at e_1375_8
  have e_1375_9 := arithEq_of_rows h (row := 1375) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_9
  simp only [← c84432, ← c84433, k2, ← c84434] at e_1375_9
  have hr := e_1375_9
  simp only [e_1375_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13033)) := by
  have c84435 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84436 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84437 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84438 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84439 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84440 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_10 := arithEq_of_rows h (row := 1375) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_10
  simp only [← c84435, ← c84436, ← c84437] at e_1375_10
  have e_1375_11 := arithEq_of_rows h (row := 1375) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_11
  simp only [← c84438, ← c84439, k2, ← c84440] at e_1375_11
  have hr := e_1375_11
  simp only [e_1375_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13034)) := by
  have c84441 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84442 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84443 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84444 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84445 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84446 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_12 := arithEq_of_rows h (row := 1375) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_12
  simp only [← c84441, ← c84442, ← c84443] at e_1375_12
  have e_1375_13 := arithEq_of_rows h (row := 1375) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_13
  simp only [← c84444, ← c84445, k2, ← c84446] at e_1375_13
  have hr := e_1375_13
  simp only [e_1375_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13035)) := by
  have c84447 := (publicBatchWrapper86_copies2638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84448 := (publicBatchWrapper86_copies2639 a h).1
  have c84449 := (publicBatchWrapper86_copies2639 a h).2.1
  have c84450 := (publicBatchWrapper86_copies2639 a h).2.2.1
  have c84451 := (publicBatchWrapper86_copies2639 a h).2.2.2.1
  have c84452 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_14 := arithEq_of_rows h (row := 1375) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_14
  simp only [← c84447, ← c84448, ← c84449] at e_1375_14
  have e_1375_15 := arithEq_of_rows h (row := 1375) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_15
  simp only [← c84450, ← c84451, k2, ← c84452] at e_1375_15
  have hr := e_1375_15
  simp only [e_1375_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13036)) := by
  have c84453 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.1
  have c84454 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.1
  have c84455 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.1
  have c84456 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.1
  have c84457 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.1
  have c84458 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_16 := arithEq_of_rows h (row := 1375) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_16
  simp only [← c84453, ← c84454, ← c84455] at e_1375_16
  have e_1375_17 := arithEq_of_rows h (row := 1375) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_17
  simp only [← c84456, ← c84457, k2, ← c84458] at e_1375_17
  have hr := e_1375_17
  simp only [e_1375_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1375 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13037)) := by
  have c84459 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84460 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84461 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84462 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84463 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84464 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1375_18 := arithEq_of_rows h (row := 1375) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_18
  simp only [← c84459, ← c84460, ← c84461] at e_1375_18
  have e_1375_19 := arithEq_of_rows h (row := 1375) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1375_19
  simp only [← c84462, ← c84463, k2, ← c84464] at e_1375_19
  have hr := e_1375_19
  simp only [e_1375_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13038)) := by
  have c84465 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84466 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84467 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84468 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84469 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84470 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_0 := arithEq_of_rows h (row := 1376) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_0
  simp only [← c84465, ← c84466, ← c84467] at e_1376_0
  have e_1376_1 := arithEq_of_rows h (row := 1376) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_1
  simp only [← c84468, ← c84469, k2, ← c84470] at e_1376_1
  have hr := e_1376_1
  simp only [e_1376_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13039)) := by
  have c84471 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84472 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84473 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84474 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84475 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84476 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_2 := arithEq_of_rows h (row := 1376) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_2
  simp only [← c84471, ← c84472, ← c84473] at e_1376_2
  have e_1376_3 := arithEq_of_rows h (row := 1376) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_3
  simp only [← c84474, ← c84475, k2, ← c84476] at e_1376_3
  have hr := e_1376_3
  simp only [e_1376_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13040)) := by
  have c84477 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84478 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84479 := (publicBatchWrapper86_copies2639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84480 := (publicBatchWrapper86_copies2640 a h).1
  have c84481 := (publicBatchWrapper86_copies2640 a h).2.1
  have c84482 := (publicBatchWrapper86_copies2640 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_4 := arithEq_of_rows h (row := 1376) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_4
  simp only [← c84477, ← c84478, ← c84479] at e_1376_4
  have e_1376_5 := arithEq_of_rows h (row := 1376) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_5
  simp only [← c84480, ← c84481, k2, ← c84482] at e_1376_5
  have hr := e_1376_5
  simp only [e_1376_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13041)) := by
  have c84483 := (publicBatchWrapper86_copies2640 a h).2.2.2.1
  have c84484 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.1
  have c84485 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.1
  have c84486 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.1
  have c84487 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.1
  have c84488 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_6 := arithEq_of_rows h (row := 1376) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_6
  simp only [← c84483, ← c84484, ← c84485] at e_1376_6
  have e_1376_7 := arithEq_of_rows h (row := 1376) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_7
  simp only [← c84486, ← c84487, k2, ← c84488] at e_1376_7
  have hr := e_1376_7
  simp only [e_1376_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13042)) := by
  have c84489 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.1
  have c84490 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84491 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84492 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84493 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84494 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_8 := arithEq_of_rows h (row := 1376) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_8
  simp only [← c84489, ← c84490, ← c84491] at e_1376_8
  have e_1376_9 := arithEq_of_rows h (row := 1376) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_9
  simp only [← c84492, ← c84493, k2, ← c84494] at e_1376_9
  have hr := e_1376_9
  simp only [e_1376_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13043)) := by
  have c84495 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84496 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84497 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84498 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84499 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84500 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_10 := arithEq_of_rows h (row := 1376) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_10
  simp only [← c84495, ← c84496, ← c84497] at e_1376_10
  have e_1376_11 := arithEq_of_rows h (row := 1376) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_11
  simp only [← c84498, ← c84499, k2, ← c84500] at e_1376_11
  have hr := e_1376_11
  simp only [e_1376_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13044)) := by
  have c84501 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84502 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84503 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84504 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84505 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84506 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_12 := arithEq_of_rows h (row := 1376) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_12
  simp only [← c84501, ← c84502, ← c84503] at e_1376_12
  have e_1376_13 := arithEq_of_rows h (row := 1376) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_13
  simp only [← c84504, ← c84505, k2, ← c84506] at e_1376_13
  have hr := e_1376_13
  simp only [e_1376_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13045)) := by
  have c84507 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84508 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84509 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84510 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84511 := (publicBatchWrapper86_copies2640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84512 := (publicBatchWrapper86_copies2641 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_14 := arithEq_of_rows h (row := 1376) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_14
  simp only [← c84507, ← c84508, ← c84509] at e_1376_14
  have e_1376_15 := arithEq_of_rows h (row := 1376) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_15
  simp only [← c84510, ← c84511, k2, ← c84512] at e_1376_15
  have hr := e_1376_15
  simp only [e_1376_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13046)) := by
  have c84513 := (publicBatchWrapper86_copies2641 a h).2.1
  have c84514 := (publicBatchWrapper86_copies2641 a h).2.2.1
  have c84515 := (publicBatchWrapper86_copies2641 a h).2.2.2.1
  have c84516 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.1
  have c84517 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.1
  have c84518 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_16 := arithEq_of_rows h (row := 1376) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_16
  simp only [← c84513, ← c84514, ← c84515] at e_1376_16
  have e_1376_17 := arithEq_of_rows h (row := 1376) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_17
  simp only [← c84516, ← c84517, k2, ← c84518] at e_1376_17
  have hr := e_1376_17
  simp only [e_1376_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1376 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13047)) := by
  have c84519 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.1
  have c84520 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.1
  have c84521 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.1
  have c84522 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84523 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84524 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1376_18 := arithEq_of_rows h (row := 1376) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_18
  simp only [← c84519, ← c84520, ← c84521] at e_1376_18
  have e_1376_19 := arithEq_of_rows h (row := 1376) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1376_19
  simp only [← c84522, ← c84523, k2, ← c84524] at e_1376_19
  have hr := e_1376_19
  simp only [e_1376_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13048)) := by
  have c84525 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84526 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84527 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84528 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84529 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84530 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_0 := arithEq_of_rows h (row := 1377) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_0
  simp only [← c84525, ← c84526, ← c84527] at e_1377_0
  have e_1377_1 := arithEq_of_rows h (row := 1377) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_1
  simp only [← c84528, ← c84529, k2, ← c84530] at e_1377_1
  have hr := e_1377_1
  simp only [e_1377_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13049)) := by
  have c84531 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84532 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84533 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84534 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84535 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84536 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_2 := arithEq_of_rows h (row := 1377) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_2
  simp only [← c84531, ← c84532, ← c84533] at e_1377_2
  have e_1377_3 := arithEq_of_rows h (row := 1377) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_3
  simp only [← c84534, ← c84535, k2, ← c84536] at e_1377_3
  have hr := e_1377_3
  simp only [e_1377_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13050)) := by
  have c84537 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84538 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84539 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84540 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84541 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84542 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_4 := arithEq_of_rows h (row := 1377) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_4
  simp only [← c84537, ← c84538, ← c84539] at e_1377_4
  have e_1377_5 := arithEq_of_rows h (row := 1377) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_5
  simp only [← c84540, ← c84541, k2, ← c84542] at e_1377_5
  have hr := e_1377_5
  simp only [e_1377_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13051)) := by
  have c84543 := (publicBatchWrapper86_copies2641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84544 := (publicBatchWrapper86_copies2642 a h).1
  have c84545 := (publicBatchWrapper86_copies2642 a h).2.1
  have c84546 := (publicBatchWrapper86_copies2642 a h).2.2.1
  have c84547 := (publicBatchWrapper86_copies2642 a h).2.2.2.1
  have c84548 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_6 := arithEq_of_rows h (row := 1377) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_6
  simp only [← c84543, ← c84544, ← c84545] at e_1377_6
  have e_1377_7 := arithEq_of_rows h (row := 1377) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_7
  simp only [← c84546, ← c84547, k2, ← c84548] at e_1377_7
  have hr := e_1377_7
  simp only [e_1377_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13052)) := by
  have c84549 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.1
  have c84550 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.1
  have c84551 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.1
  have c84552 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.1
  have c84553 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.1
  have c84554 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_8 := arithEq_of_rows h (row := 1377) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_8
  simp only [← c84549, ← c84550, ← c84551] at e_1377_8
  have e_1377_9 := arithEq_of_rows h (row := 1377) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_9
  simp only [← c84552, ← c84553, k2, ← c84554] at e_1377_9
  have hr := e_1377_9
  simp only [e_1377_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13053)) := by
  have c84555 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84556 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84557 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84558 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84559 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84560 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_10 := arithEq_of_rows h (row := 1377) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_10
  simp only [← c84555, ← c84556, ← c84557] at e_1377_10
  have e_1377_11 := arithEq_of_rows h (row := 1377) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_11
  simp only [← c84558, ← c84559, k2, ← c84560] at e_1377_11
  have hr := e_1377_11
  simp only [e_1377_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13054)) := by
  have c84561 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84562 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84563 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84564 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84565 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84566 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_12 := arithEq_of_rows h (row := 1377) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_12
  simp only [← c84561, ← c84562, ← c84563] at e_1377_12
  have e_1377_13 := arithEq_of_rows h (row := 1377) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_13
  simp only [← c84564, ← c84565, k2, ← c84566] at e_1377_13
  have hr := e_1377_13
  simp only [e_1377_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13055)) := by
  have c84567 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84568 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84569 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84570 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84571 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84572 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_14 := arithEq_of_rows h (row := 1377) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_14
  simp only [← c84567, ← c84568, ← c84569] at e_1377_14
  have e_1377_15 := arithEq_of_rows h (row := 1377) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_15
  simp only [← c84570, ← c84571, k2, ← c84572] at e_1377_15
  have hr := e_1377_15
  simp only [e_1377_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13056)) := by
  have c84573 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84574 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84575 := (publicBatchWrapper86_copies2642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84576 := (publicBatchWrapper86_copies2643 a h).1
  have c84577 := (publicBatchWrapper86_copies2643 a h).2.1
  have c84578 := (publicBatchWrapper86_copies2643 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_16 := arithEq_of_rows h (row := 1377) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_16
  simp only [← c84573, ← c84574, ← c84575] at e_1377_16
  have e_1377_17 := arithEq_of_rows h (row := 1377) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_17
  simp only [← c84576, ← c84577, k2, ← c84578] at e_1377_17
  have hr := e_1377_17
  simp only [e_1377_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1377 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13057)) := by
  have c84579 := (publicBatchWrapper86_copies2643 a h).2.2.2.1
  have c84580 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.1
  have c84581 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.1
  have c84582 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.1
  have c84583 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.1
  have c84584 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1377_18 := arithEq_of_rows h (row := 1377) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_18
  simp only [← c84579, ← c84580, ← c84581] at e_1377_18
  have e_1377_19 := arithEq_of_rows h (row := 1377) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1377_19
  simp only [← c84582, ← c84583, k2, ← c84584] at e_1377_19
  have hr := e_1377_19
  simp only [e_1377_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13058)) := by
  have c84585 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.1
  have c84586 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84587 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84588 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84589 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84590 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_0 := arithEq_of_rows h (row := 1378) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_0
  simp only [← c84585, ← c84586, ← c84587] at e_1378_0
  have e_1378_1 := arithEq_of_rows h (row := 1378) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_1
  simp only [← c84588, ← c84589, k2, ← c84590] at e_1378_1
  have hr := e_1378_1
  simp only [e_1378_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13059)) := by
  have c84591 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84592 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84593 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84594 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84595 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84596 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_2 := arithEq_of_rows h (row := 1378) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_2
  simp only [← c84591, ← c84592, ← c84593] at e_1378_2
  have e_1378_3 := arithEq_of_rows h (row := 1378) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_3
  simp only [← c84594, ← c84595, k2, ← c84596] at e_1378_3
  have hr := e_1378_3
  simp only [e_1378_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13060)) := by
  have c84597 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84598 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84599 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84600 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84601 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84602 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_4 := arithEq_of_rows h (row := 1378) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_4
  simp only [← c84597, ← c84598, ← c84599] at e_1378_4
  have e_1378_5 := arithEq_of_rows h (row := 1378) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_5
  simp only [← c84600, ← c84601, k2, ← c84602] at e_1378_5
  have hr := e_1378_5
  simp only [e_1378_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13061)) := by
  have c84603 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84604 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84605 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84606 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84607 := (publicBatchWrapper86_copies2643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84608 := (publicBatchWrapper86_copies2644 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_6 := arithEq_of_rows h (row := 1378) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_6
  simp only [← c84603, ← c84604, ← c84605] at e_1378_6
  have e_1378_7 := arithEq_of_rows h (row := 1378) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_7
  simp only [← c84606, ← c84607, k2, ← c84608] at e_1378_7
  have hr := e_1378_7
  simp only [e_1378_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13062)) := by
  have c84609 := (publicBatchWrapper86_copies2644 a h).2.1
  have c84610 := (publicBatchWrapper86_copies2644 a h).2.2.1
  have c84611 := (publicBatchWrapper86_copies2644 a h).2.2.2.1
  have c84612 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.1
  have c84613 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.1
  have c84614 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_8 := arithEq_of_rows h (row := 1378) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_8
  simp only [← c84609, ← c84610, ← c84611] at e_1378_8
  have e_1378_9 := arithEq_of_rows h (row := 1378) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_9
  simp only [← c84612, ← c84613, k2, ← c84614] at e_1378_9
  have hr := e_1378_9
  simp only [e_1378_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13063)) := by
  have c84615 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.1
  have c84616 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.1
  have c84617 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.1
  have c84618 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84619 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84620 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_10 := arithEq_of_rows h (row := 1378) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_10
  simp only [← c84615, ← c84616, ← c84617] at e_1378_10
  have e_1378_11 := arithEq_of_rows h (row := 1378) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_11
  simp only [← c84618, ← c84619, k2, ← c84620] at e_1378_11
  have hr := e_1378_11
  simp only [e_1378_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13064)) := by
  have c84621 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84622 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84623 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84624 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84625 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84626 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_12 := arithEq_of_rows h (row := 1378) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_12
  simp only [← c84621, ← c84622, ← c84623] at e_1378_12
  have e_1378_13 := arithEq_of_rows h (row := 1378) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_13
  simp only [← c84624, ← c84625, k2, ← c84626] at e_1378_13
  have hr := e_1378_13
  simp only [e_1378_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13065)) := by
  have c84627 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84628 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84629 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84630 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84631 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84632 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_14 := arithEq_of_rows h (row := 1378) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_14
  simp only [← c84627, ← c84628, ← c84629] at e_1378_14
  have e_1378_15 := arithEq_of_rows h (row := 1378) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_15
  simp only [← c84630, ← c84631, k2, ← c84632] at e_1378_15
  have hr := e_1378_15
  simp only [e_1378_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13066)) := by
  have c84633 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84634 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84635 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84636 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84637 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84638 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_16 := arithEq_of_rows h (row := 1378) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_16
  simp only [← c84633, ← c84634, ← c84635] at e_1378_16
  have e_1378_17 := arithEq_of_rows h (row := 1378) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_17
  simp only [← c84636, ← c84637, k2, ← c84638] at e_1378_17
  have hr := e_1378_17
  simp only [e_1378_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1378 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13067)) := by
  have c84639 := (publicBatchWrapper86_copies2644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84640 := (publicBatchWrapper86_copies2645 a h).1
  have c84641 := (publicBatchWrapper86_copies2645 a h).2.1
  have c84642 := (publicBatchWrapper86_copies2645 a h).2.2.1
  have c84643 := (publicBatchWrapper86_copies2645 a h).2.2.2.1
  have c84644 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1378_18 := arithEq_of_rows h (row := 1378) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_18
  simp only [← c84639, ← c84640, ← c84641] at e_1378_18
  have e_1378_19 := arithEq_of_rows h (row := 1378) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1378_19
  simp only [← c84642, ← c84643, k2, ← c84644] at e_1378_19
  have hr := e_1378_19
  simp only [e_1378_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13068)) := by
  have c84645 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.1
  have c84646 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.1
  have c84647 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.1
  have c84648 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.1
  have c84649 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.1
  have c84650 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_0 := arithEq_of_rows h (row := 1379) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_0
  simp only [← c84645, ← c84646, ← c84647] at e_1379_0
  have e_1379_1 := arithEq_of_rows h (row := 1379) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_1
  simp only [← c84648, ← c84649, k2, ← c84650] at e_1379_1
  have hr := e_1379_1
  simp only [e_1379_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13069)) := by
  have c84651 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84652 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84653 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84654 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84655 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84656 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_2 := arithEq_of_rows h (row := 1379) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_2
  simp only [← c84651, ← c84652, ← c84653] at e_1379_2
  have e_1379_3 := arithEq_of_rows h (row := 1379) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_3
  simp only [← c84654, ← c84655, k2, ← c84656] at e_1379_3
  have hr := e_1379_3
  simp only [e_1379_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13070)) := by
  have c84657 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84658 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84659 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84660 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84661 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84662 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_4 := arithEq_of_rows h (row := 1379) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_4
  simp only [← c84657, ← c84658, ← c84659] at e_1379_4
  have e_1379_5 := arithEq_of_rows h (row := 1379) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_5
  simp only [← c84660, ← c84661, k2, ← c84662] at e_1379_5
  have hr := e_1379_5
  simp only [e_1379_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13071)) := by
  have c84663 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84664 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84665 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84666 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84667 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84668 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_6 := arithEq_of_rows h (row := 1379) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_6
  simp only [← c84663, ← c84664, ← c84665] at e_1379_6
  have e_1379_7 := arithEq_of_rows h (row := 1379) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_7
  simp only [← c84666, ← c84667, k2, ← c84668] at e_1379_7
  have hr := e_1379_7
  simp only [e_1379_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13072)) := by
  have c84669 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84670 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84671 := (publicBatchWrapper86_copies2645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84672 := (publicBatchWrapper86_copies2646 a h).1
  have c84673 := (publicBatchWrapper86_copies2646 a h).2.1
  have c84674 := (publicBatchWrapper86_copies2646 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_8 := arithEq_of_rows h (row := 1379) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_8
  simp only [← c84669, ← c84670, ← c84671] at e_1379_8
  have e_1379_9 := arithEq_of_rows h (row := 1379) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_9
  simp only [← c84672, ← c84673, k2, ← c84674] at e_1379_9
  have hr := e_1379_9
  simp only [e_1379_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13073)) := by
  have c84675 := (publicBatchWrapper86_copies2646 a h).2.2.2.1
  have c84676 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.1
  have c84677 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.1
  have c84678 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.1
  have c84679 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.1
  have c84680 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_10 := arithEq_of_rows h (row := 1379) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_10
  simp only [← c84675, ← c84676, ← c84677] at e_1379_10
  have e_1379_11 := arithEq_of_rows h (row := 1379) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_11
  simp only [← c84678, ← c84679, k2, ← c84680] at e_1379_11
  have hr := e_1379_11
  simp only [e_1379_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13074)) := by
  have c84681 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.1
  have c84682 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84683 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84684 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84685 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84686 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_12 := arithEq_of_rows h (row := 1379) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_12
  simp only [← c84681, ← c84682, ← c84683] at e_1379_12
  have e_1379_13 := arithEq_of_rows h (row := 1379) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_13
  simp only [← c84684, ← c84685, k2, ← c84686] at e_1379_13
  have hr := e_1379_13
  simp only [e_1379_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13075)) := by
  have c84687 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84688 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84689 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84690 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84691 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84692 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_14 := arithEq_of_rows h (row := 1379) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_14
  simp only [← c84687, ← c84688, ← c84689] at e_1379_14
  have e_1379_15 := arithEq_of_rows h (row := 1379) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_15
  simp only [← c84690, ← c84691, k2, ← c84692] at e_1379_15
  have hr := e_1379_15
  simp only [e_1379_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13076)) := by
  have c84693 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84694 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84695 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84696 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84697 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84698 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_16 := arithEq_of_rows h (row := 1379) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_16
  simp only [← c84693, ← c84694, ← c84695] at e_1379_16
  have e_1379_17 := arithEq_of_rows h (row := 1379) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_17
  simp only [← c84696, ← c84697, k2, ← c84698] at e_1379_17
  have hr := e_1379_17
  simp only [e_1379_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1379 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13077)) := by
  have c84699 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84700 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84701 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84702 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84703 := (publicBatchWrapper86_copies2646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84704 := (publicBatchWrapper86_copies2647 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1379_18 := arithEq_of_rows h (row := 1379) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_18
  simp only [← c84699, ← c84700, ← c84701] at e_1379_18
  have e_1379_19 := arithEq_of_rows h (row := 1379) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1379_19
  simp only [← c84702, ← c84703, k2, ← c84704] at e_1379_19
  have hr := e_1379_19
  simp only [e_1379_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13078)) := by
  have c84705 := (publicBatchWrapper86_copies2647 a h).2.1
  have c84706 := (publicBatchWrapper86_copies2647 a h).2.2.1
  have c84707 := (publicBatchWrapper86_copies2647 a h).2.2.2.1
  have c84708 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.1
  have c84709 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.1
  have c84710 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_0 := arithEq_of_rows h (row := 1380) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_0
  simp only [← c84705, ← c84706, ← c84707] at e_1380_0
  have e_1380_1 := arithEq_of_rows h (row := 1380) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_1
  simp only [← c84708, ← c84709, k2, ← c84710] at e_1380_1
  have hr := e_1380_1
  simp only [e_1380_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13079)) := by
  have c84711 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.1
  have c84712 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.1
  have c84713 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.1
  have c84714 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84715 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84716 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_2 := arithEq_of_rows h (row := 1380) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_2
  simp only [← c84711, ← c84712, ← c84713] at e_1380_2
  have e_1380_3 := arithEq_of_rows h (row := 1380) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_3
  simp only [← c84714, ← c84715, k2, ← c84716] at e_1380_3
  have hr := e_1380_3
  simp only [e_1380_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13080)) := by
  have c84717 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84718 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84719 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84720 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84721 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84722 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_4 := arithEq_of_rows h (row := 1380) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_4
  simp only [← c84717, ← c84718, ← c84719] at e_1380_4
  have e_1380_5 := arithEq_of_rows h (row := 1380) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_5
  simp only [← c84720, ← c84721, k2, ← c84722] at e_1380_5
  have hr := e_1380_5
  simp only [e_1380_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13081)) := by
  have c84723 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84724 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84725 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84726 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84727 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84728 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_6 := arithEq_of_rows h (row := 1380) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_6
  simp only [← c84723, ← c84724, ← c84725] at e_1380_6
  have e_1380_7 := arithEq_of_rows h (row := 1380) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_7
  simp only [← c84726, ← c84727, k2, ← c84728] at e_1380_7
  have hr := e_1380_7
  simp only [e_1380_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13082)) := by
  have c84729 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84730 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84731 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84732 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84733 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84734 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_8 := arithEq_of_rows h (row := 1380) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_8
  simp only [← c84729, ← c84730, ← c84731] at e_1380_8
  have e_1380_9 := arithEq_of_rows h (row := 1380) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_9
  simp only [← c84732, ← c84733, k2, ← c84734] at e_1380_9
  have hr := e_1380_9
  simp only [e_1380_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13083)) := by
  have c84735 := (publicBatchWrapper86_copies2647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84736 := (publicBatchWrapper86_copies2648 a h).1
  have c84737 := (publicBatchWrapper86_copies2648 a h).2.1
  have c84738 := (publicBatchWrapper86_copies2648 a h).2.2.1
  have c84739 := (publicBatchWrapper86_copies2648 a h).2.2.2.1
  have c84740 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_10 := arithEq_of_rows h (row := 1380) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_10
  simp only [← c84735, ← c84736, ← c84737] at e_1380_10
  have e_1380_11 := arithEq_of_rows h (row := 1380) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_11
  simp only [← c84738, ← c84739, k2, ← c84740] at e_1380_11
  have hr := e_1380_11
  simp only [e_1380_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13084)) := by
  have c84741 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.1
  have c84742 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.1
  have c84743 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.1
  have c84744 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.1
  have c84745 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.1
  have c84746 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_12 := arithEq_of_rows h (row := 1380) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_12
  simp only [← c84741, ← c84742, ← c84743] at e_1380_12
  have e_1380_13 := arithEq_of_rows h (row := 1380) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_13
  simp only [← c84744, ← c84745, k2, ← c84746] at e_1380_13
  have hr := e_1380_13
  simp only [e_1380_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13085)) := by
  have c84747 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84748 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84749 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84750 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84751 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84752 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_14 := arithEq_of_rows h (row := 1380) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_14
  simp only [← c84747, ← c84748, ← c84749] at e_1380_14
  have e_1380_15 := arithEq_of_rows h (row := 1380) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_15
  simp only [← c84750, ← c84751, k2, ← c84752] at e_1380_15
  have hr := e_1380_15
  simp only [e_1380_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13086)) := by
  have c84753 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84754 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84755 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84756 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84757 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84758 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_16 := arithEq_of_rows h (row := 1380) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_16
  simp only [← c84753, ← c84754, ← c84755] at e_1380_16
  have e_1380_17 := arithEq_of_rows h (row := 1380) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_17
  simp only [← c84756, ← c84757, k2, ← c84758] at e_1380_17
  have hr := e_1380_17
  simp only [e_1380_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1380 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13087)) := by
  have c84759 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84760 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84761 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84762 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84763 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84764 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1380_18 := arithEq_of_rows h (row := 1380) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_18
  simp only [← c84759, ← c84760, ← c84761] at e_1380_18
  have e_1380_19 := arithEq_of_rows h (row := 1380) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1380_19
  simp only [← c84762, ← c84763, k2, ← c84764] at e_1380_19
  have hr := e_1380_19
  simp only [e_1380_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13088)) := by
  have c84765 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84766 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84767 := (publicBatchWrapper86_copies2648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84768 := (publicBatchWrapper86_copies2649 a h).1
  have c84769 := (publicBatchWrapper86_copies2649 a h).2.1
  have c84770 := (publicBatchWrapper86_copies2649 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_0 := arithEq_of_rows h (row := 1381) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_0
  simp only [← c84765, ← c84766, ← c84767] at e_1381_0
  have e_1381_1 := arithEq_of_rows h (row := 1381) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_1
  simp only [← c84768, ← c84769, k2, ← c84770] at e_1381_1
  have hr := e_1381_1
  simp only [e_1381_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13089)) := by
  have c84771 := (publicBatchWrapper86_copies2649 a h).2.2.2.1
  have c84772 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.1
  have c84773 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.1
  have c84774 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.1
  have c84775 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.1
  have c84776 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_2 := arithEq_of_rows h (row := 1381) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_2
  simp only [← c84771, ← c84772, ← c84773] at e_1381_2
  have e_1381_3 := arithEq_of_rows h (row := 1381) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_3
  simp only [← c84774, ← c84775, k2, ← c84776] at e_1381_3
  have hr := e_1381_3
  simp only [e_1381_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13090)) := by
  have c84777 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.1
  have c84778 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84779 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84780 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84781 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84782 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_4 := arithEq_of_rows h (row := 1381) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_4
  simp only [← c84777, ← c84778, ← c84779] at e_1381_4
  have e_1381_5 := arithEq_of_rows h (row := 1381) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_5
  simp only [← c84780, ← c84781, k2, ← c84782] at e_1381_5
  have hr := e_1381_5
  simp only [e_1381_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13091)) := by
  have c84783 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84784 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84785 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84786 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84787 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84788 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_6 := arithEq_of_rows h (row := 1381) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_6
  simp only [← c84783, ← c84784, ← c84785] at e_1381_6
  have e_1381_7 := arithEq_of_rows h (row := 1381) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_7
  simp only [← c84786, ← c84787, k2, ← c84788] at e_1381_7
  have hr := e_1381_7
  simp only [e_1381_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13092)) := by
  have c84789 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84790 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84791 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84792 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84793 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84794 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_8 := arithEq_of_rows h (row := 1381) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_8
  simp only [← c84789, ← c84790, ← c84791] at e_1381_8
  have e_1381_9 := arithEq_of_rows h (row := 1381) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_9
  simp only [← c84792, ← c84793, k2, ← c84794] at e_1381_9
  have hr := e_1381_9
  simp only [e_1381_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13093)) := by
  have c84795 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84796 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84797 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84798 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84799 := (publicBatchWrapper86_copies2649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84800 := (publicBatchWrapper86_copies2650 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_10 := arithEq_of_rows h (row := 1381) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_10
  simp only [← c84795, ← c84796, ← c84797] at e_1381_10
  have e_1381_11 := arithEq_of_rows h (row := 1381) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_11
  simp only [← c84798, ← c84799, k2, ← c84800] at e_1381_11
  have hr := e_1381_11
  simp only [e_1381_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13094)) := by
  have c84801 := (publicBatchWrapper86_copies2650 a h).2.1
  have c84802 := (publicBatchWrapper86_copies2650 a h).2.2.1
  have c84803 := (publicBatchWrapper86_copies2650 a h).2.2.2.1
  have c84804 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.1
  have c84805 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.1
  have c84806 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_12 := arithEq_of_rows h (row := 1381) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_12
  simp only [← c84801, ← c84802, ← c84803] at e_1381_12
  have e_1381_13 := arithEq_of_rows h (row := 1381) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_13
  simp only [← c84804, ← c84805, k2, ← c84806] at e_1381_13
  have hr := e_1381_13
  simp only [e_1381_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13095)) := by
  have c84807 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.1
  have c84808 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.1
  have c84809 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.1
  have c84810 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84811 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84812 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_14 := arithEq_of_rows h (row := 1381) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_14
  simp only [← c84807, ← c84808, ← c84809] at e_1381_14
  have e_1381_15 := arithEq_of_rows h (row := 1381) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_15
  simp only [← c84810, ← c84811, k2, ← c84812] at e_1381_15
  have hr := e_1381_15
  simp only [e_1381_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13096)) := by
  have c84813 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84814 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84815 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84816 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84817 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84818 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_16 := arithEq_of_rows h (row := 1381) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_16
  simp only [← c84813, ← c84814, ← c84815] at e_1381_16
  have e_1381_17 := arithEq_of_rows h (row := 1381) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_17
  simp only [← c84816, ← c84817, k2, ← c84818] at e_1381_17
  have hr := e_1381_17
  simp only [e_1381_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1381 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13097)) := by
  have c84819 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84820 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84821 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84822 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84823 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84824 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1381_18 := arithEq_of_rows h (row := 1381) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_18
  simp only [← c84819, ← c84820, ← c84821] at e_1381_18
  have e_1381_19 := arithEq_of_rows h (row := 1381) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1381_19
  simp only [← c84822, ← c84823, k2, ← c84824] at e_1381_19
  have hr := e_1381_19
  simp only [e_1381_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13098)) := by
  have c84825 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84826 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84827 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84828 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84829 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84830 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_0 := arithEq_of_rows h (row := 1382) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_0
  simp only [← c84825, ← c84826, ← c84827] at e_1382_0
  have e_1382_1 := arithEq_of_rows h (row := 1382) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_1
  simp only [← c84828, ← c84829, k2, ← c84830] at e_1382_1
  have hr := e_1382_1
  simp only [e_1382_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13099)) := by
  have c84831 := (publicBatchWrapper86_copies2650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84832 := (publicBatchWrapper86_copies2651 a h).1
  have c84833 := (publicBatchWrapper86_copies2651 a h).2.1
  have c84834 := (publicBatchWrapper86_copies2651 a h).2.2.1
  have c84835 := (publicBatchWrapper86_copies2651 a h).2.2.2.1
  have c84836 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_2 := arithEq_of_rows h (row := 1382) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_2
  simp only [← c84831, ← c84832, ← c84833] at e_1382_2
  have e_1382_3 := arithEq_of_rows h (row := 1382) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_3
  simp only [← c84834, ← c84835, k2, ← c84836] at e_1382_3
  have hr := e_1382_3
  simp only [e_1382_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13100)) := by
  have c84837 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.1
  have c84838 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.1
  have c84839 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.1
  have c84840 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.1
  have c84841 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.1
  have c84842 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_4 := arithEq_of_rows h (row := 1382) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_4
  simp only [← c84837, ← c84838, ← c84839] at e_1382_4
  have e_1382_5 := arithEq_of_rows h (row := 1382) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_5
  simp only [← c84840, ← c84841, k2, ← c84842] at e_1382_5
  have hr := e_1382_5
  simp only [e_1382_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13101)) := by
  have c84843 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84844 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84845 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84846 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84847 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84848 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_6 := arithEq_of_rows h (row := 1382) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_6
  simp only [← c84843, ← c84844, ← c84845] at e_1382_6
  have e_1382_7 := arithEq_of_rows h (row := 1382) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_7
  simp only [← c84846, ← c84847, k2, ← c84848] at e_1382_7
  have hr := e_1382_7
  simp only [e_1382_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13102)) := by
  have c84849 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84850 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84851 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84852 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84853 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84854 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_8 := arithEq_of_rows h (row := 1382) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_8
  simp only [← c84849, ← c84850, ← c84851] at e_1382_8
  have e_1382_9 := arithEq_of_rows h (row := 1382) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_9
  simp only [← c84852, ← c84853, k2, ← c84854] at e_1382_9
  have hr := e_1382_9
  simp only [e_1382_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13103)) := by
  have c84855 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84856 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84857 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84858 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84859 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84860 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_10 := arithEq_of_rows h (row := 1382) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_10
  simp only [← c84855, ← c84856, ← c84857] at e_1382_10
  have e_1382_11 := arithEq_of_rows h (row := 1382) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_11
  simp only [← c84858, ← c84859, k2, ← c84860] at e_1382_11
  have hr := e_1382_11
  simp only [e_1382_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13104)) := by
  have c84861 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84862 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84863 := (publicBatchWrapper86_copies2651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84864 := (publicBatchWrapper86_copies2652 a h).1
  have c84865 := (publicBatchWrapper86_copies2652 a h).2.1
  have c84866 := (publicBatchWrapper86_copies2652 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_12 := arithEq_of_rows h (row := 1382) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_12
  simp only [← c84861, ← c84862, ← c84863] at e_1382_12
  have e_1382_13 := arithEq_of_rows h (row := 1382) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_13
  simp only [← c84864, ← c84865, k2, ← c84866] at e_1382_13
  have hr := e_1382_13
  simp only [e_1382_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13105)) := by
  have c84867 := (publicBatchWrapper86_copies2652 a h).2.2.2.1
  have c84868 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.1
  have c84869 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.1
  have c84870 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.1
  have c84871 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.1
  have c84872 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_14 := arithEq_of_rows h (row := 1382) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_14
  simp only [← c84867, ← c84868, ← c84869] at e_1382_14
  have e_1382_15 := arithEq_of_rows h (row := 1382) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_15
  simp only [← c84870, ← c84871, k2, ← c84872] at e_1382_15
  have hr := e_1382_15
  simp only [e_1382_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13106)) := by
  have c84873 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.1
  have c84874 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84875 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84876 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84877 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84878 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_16 := arithEq_of_rows h (row := 1382) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_16
  simp only [← c84873, ← c84874, ← c84875] at e_1382_16
  have e_1382_17 := arithEq_of_rows h (row := 1382) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_17
  simp only [← c84876, ← c84877, k2, ← c84878] at e_1382_17
  have hr := e_1382_17
  simp only [e_1382_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1382 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13107)) := by
  have c84879 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84880 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84881 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84882 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84883 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84884 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1382_18 := arithEq_of_rows h (row := 1382) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_18
  simp only [← c84879, ← c84880, ← c84881] at e_1382_18
  have e_1382_19 := arithEq_of_rows h (row := 1382) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1382_19
  simp only [← c84882, ← c84883, k2, ← c84884] at e_1382_19
  have hr := e_1382_19
  simp only [e_1382_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13108)) := by
  have c84885 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84886 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84887 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84888 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84889 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84890 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_0 := arithEq_of_rows h (row := 1383) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_0
  simp only [← c84885, ← c84886, ← c84887] at e_1383_0
  have e_1383_1 := arithEq_of_rows h (row := 1383) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_1
  simp only [← c84888, ← c84889, k2, ← c84890] at e_1383_1
  have hr := e_1383_1
  simp only [e_1383_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13109)) := by
  have c84891 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84892 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84893 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84894 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84895 := (publicBatchWrapper86_copies2652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84896 := (publicBatchWrapper86_copies2653 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_2 := arithEq_of_rows h (row := 1383) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_2
  simp only [← c84891, ← c84892, ← c84893] at e_1383_2
  have e_1383_3 := arithEq_of_rows h (row := 1383) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_3
  simp only [← c84894, ← c84895, k2, ← c84896] at e_1383_3
  have hr := e_1383_3
  simp only [e_1383_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13110)) := by
  have c84897 := (publicBatchWrapper86_copies2653 a h).2.1
  have c84898 := (publicBatchWrapper86_copies2653 a h).2.2.1
  have c84899 := (publicBatchWrapper86_copies2653 a h).2.2.2.1
  have c84900 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.1
  have c84901 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.1
  have c84902 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_4 := arithEq_of_rows h (row := 1383) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_4
  simp only [← c84897, ← c84898, ← c84899] at e_1383_4
  have e_1383_5 := arithEq_of_rows h (row := 1383) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_5
  simp only [← c84900, ← c84901, k2, ← c84902] at e_1383_5
  have hr := e_1383_5
  simp only [e_1383_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13111)) := by
  have c84903 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.1
  have c84904 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.1
  have c84905 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.1
  have c84906 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84907 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84908 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_6 := arithEq_of_rows h (row := 1383) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_6
  simp only [← c84903, ← c84904, ← c84905] at e_1383_6
  have e_1383_7 := arithEq_of_rows h (row := 1383) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_7
  simp only [← c84906, ← c84907, k2, ← c84908] at e_1383_7
  have hr := e_1383_7
  simp only [e_1383_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13112)) := by
  have c84909 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84910 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84911 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84912 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84913 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84914 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_8 := arithEq_of_rows h (row := 1383) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_8
  simp only [← c84909, ← c84910, ← c84911] at e_1383_8
  have e_1383_9 := arithEq_of_rows h (row := 1383) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_9
  simp only [← c84912, ← c84913, k2, ← c84914] at e_1383_9
  have hr := e_1383_9
  simp only [e_1383_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13113)) := by
  have c84915 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84916 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84917 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84918 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84919 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84920 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_10 := arithEq_of_rows h (row := 1383) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_10
  simp only [← c84915, ← c84916, ← c84917] at e_1383_10
  have e_1383_11 := arithEq_of_rows h (row := 1383) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_11
  simp only [← c84918, ← c84919, k2, ← c84920] at e_1383_11
  have hr := e_1383_11
  simp only [e_1383_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13114)) := by
  have c84921 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84922 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84923 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84924 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84925 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84926 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_12 := arithEq_of_rows h (row := 1383) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_12
  simp only [← c84921, ← c84922, ← c84923] at e_1383_12
  have e_1383_13 := arithEq_of_rows h (row := 1383) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_13
  simp only [← c84924, ← c84925, k2, ← c84926] at e_1383_13
  have hr := e_1383_13
  simp only [e_1383_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13115)) := by
  have c84927 := (publicBatchWrapper86_copies2653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84928 := (publicBatchWrapper86_copies2654 a h).1
  have c84929 := (publicBatchWrapper86_copies2654 a h).2.1
  have c84930 := (publicBatchWrapper86_copies2654 a h).2.2.1
  have c84931 := (publicBatchWrapper86_copies2654 a h).2.2.2.1
  have c84932 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_14 := arithEq_of_rows h (row := 1383) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_14
  simp only [← c84927, ← c84928, ← c84929] at e_1383_14
  have e_1383_15 := arithEq_of_rows h (row := 1383) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_15
  simp only [← c84930, ← c84931, k2, ← c84932] at e_1383_15
  have hr := e_1383_15
  simp only [e_1383_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13116)) := by
  have c84933 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.1
  have c84934 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.1
  have c84935 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.1
  have c84936 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.1
  have c84937 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.1
  have c84938 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_16 := arithEq_of_rows h (row := 1383) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_16
  simp only [← c84933, ← c84934, ← c84935] at e_1383_16
  have e_1383_17 := arithEq_of_rows h (row := 1383) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_17
  simp only [← c84936, ← c84937, k2, ← c84938] at e_1383_17
  have hr := e_1383_17
  simp only [e_1383_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1383 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13117)) := by
  have c84939 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84940 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84941 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84942 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84943 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84944 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1383_18 := arithEq_of_rows h (row := 1383) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_18
  simp only [← c84939, ← c84940, ← c84941] at e_1383_18
  have e_1383_19 := arithEq_of_rows h (row := 1383) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1383_19
  simp only [← c84942, ← c84943, k2, ← c84944] at e_1383_19
  have hr := e_1383_19
  simp only [e_1383_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13118)) := by
  have c84945 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84946 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84947 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84948 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84949 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84950 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_0 := arithEq_of_rows h (row := 1384) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_0
  simp only [← c84945, ← c84946, ← c84947] at e_1384_0
  have e_1384_1 := arithEq_of_rows h (row := 1384) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_1
  simp only [← c84948, ← c84949, k2, ← c84950] at e_1384_1
  have hr := e_1384_1
  simp only [e_1384_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13119)) := by
  have c84951 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84952 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84953 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84954 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84955 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84956 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_2 := arithEq_of_rows h (row := 1384) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_2
  simp only [← c84951, ← c84952, ← c84953] at e_1384_2
  have e_1384_3 := arithEq_of_rows h (row := 1384) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_3
  simp only [← c84954, ← c84955, k2, ← c84956] at e_1384_3
  have hr := e_1384_3
  simp only [e_1384_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13120)) := by
  have c84957 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84958 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84959 := (publicBatchWrapper86_copies2654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84960 := (publicBatchWrapper86_copies2655 a h).1
  have c84961 := (publicBatchWrapper86_copies2655 a h).2.1
  have c84962 := (publicBatchWrapper86_copies2655 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_4 := arithEq_of_rows h (row := 1384) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_4
  simp only [← c84957, ← c84958, ← c84959] at e_1384_4
  have e_1384_5 := arithEq_of_rows h (row := 1384) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_5
  simp only [← c84960, ← c84961, k2, ← c84962] at e_1384_5
  have hr := e_1384_5
  simp only [e_1384_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13121)) := by
  have c84963 := (publicBatchWrapper86_copies2655 a h).2.2.2.1
  have c84964 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.1
  have c84965 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.1
  have c84966 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.1
  have c84967 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.1
  have c84968 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_6 := arithEq_of_rows h (row := 1384) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_6
  simp only [← c84963, ← c84964, ← c84965] at e_1384_6
  have e_1384_7 := arithEq_of_rows h (row := 1384) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_7
  simp only [← c84966, ← c84967, k2, ← c84968] at e_1384_7
  have hr := e_1384_7
  simp only [e_1384_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13122)) := by
  have c84969 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.1
  have c84970 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c84971 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c84972 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84973 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84974 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_8 := arithEq_of_rows h (row := 1384) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_8
  simp only [← c84969, ← c84970, ← c84971] at e_1384_8
  have e_1384_9 := arithEq_of_rows h (row := 1384) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_9
  simp only [← c84972, ← c84973, k2, ← c84974] at e_1384_9
  have hr := e_1384_9
  simp only [e_1384_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13123)) := by
  have c84975 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84976 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84977 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84978 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84979 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84980 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_10 := arithEq_of_rows h (row := 1384) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_10
  simp only [← c84975, ← c84976, ← c84977] at e_1384_10
  have e_1384_11 := arithEq_of_rows h (row := 1384) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_11
  simp only [← c84978, ← c84979, k2, ← c84980] at e_1384_11
  have hr := e_1384_11
  simp only [e_1384_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13124)) := by
  have c84981 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84982 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84983 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84984 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84985 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84986 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_12 := arithEq_of_rows h (row := 1384) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_12
  simp only [← c84981, ← c84982, ← c84983] at e_1384_12
  have e_1384_13 := arithEq_of_rows h (row := 1384) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_13
  simp only [← c84984, ← c84985, k2, ← c84986] at e_1384_13
  have hr := e_1384_13
  simp only [e_1384_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13125)) := by
  have c84987 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84988 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84989 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84990 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84991 := (publicBatchWrapper86_copies2655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c84992 := (publicBatchWrapper86_copies2656 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_14 := arithEq_of_rows h (row := 1384) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_14
  simp only [← c84987, ← c84988, ← c84989] at e_1384_14
  have e_1384_15 := arithEq_of_rows h (row := 1384) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_15
  simp only [← c84990, ← c84991, k2, ← c84992] at e_1384_15
  have hr := e_1384_15
  simp only [e_1384_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13126)) := by
  have c84993 := (publicBatchWrapper86_copies2656 a h).2.1
  have c84994 := (publicBatchWrapper86_copies2656 a h).2.2.1
  have c84995 := (publicBatchWrapper86_copies2656 a h).2.2.2.1
  have c84996 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.1
  have c84997 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.1
  have c84998 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_16 := arithEq_of_rows h (row := 1384) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_16
  simp only [← c84993, ← c84994, ← c84995] at e_1384_16
  have e_1384_17 := arithEq_of_rows h (row := 1384) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_17
  simp only [← c84996, ← c84997, k2, ← c84998] at e_1384_17
  have hr := e_1384_17
  simp only [e_1384_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1384 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13127)) := by
  have c84999 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.1
  have c85000 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.1
  have c85001 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.1
  have c85002 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85003 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85004 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1384_18 := arithEq_of_rows h (row := 1384) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_18
  simp only [← c84999, ← c85000, ← c85001] at e_1384_18
  have e_1384_19 := arithEq_of_rows h (row := 1384) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1384_19
  simp only [← c85002, ← c85003, k2, ← c85004] at e_1384_19
  have hr := e_1384_19
  simp only [e_1384_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13128)) := by
  have c85005 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85006 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85007 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85008 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85009 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85010 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_0 := arithEq_of_rows h (row := 1385) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_0
  simp only [← c85005, ← c85006, ← c85007] at e_1385_0
  have e_1385_1 := arithEq_of_rows h (row := 1385) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_1
  simp only [← c85008, ← c85009, k2, ← c85010] at e_1385_1
  have hr := e_1385_1
  simp only [e_1385_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13129)) := by
  have c85011 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85012 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85013 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85014 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85015 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85016 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_2 := arithEq_of_rows h (row := 1385) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_2
  simp only [← c85011, ← c85012, ← c85013] at e_1385_2
  have e_1385_3 := arithEq_of_rows h (row := 1385) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_3
  simp only [← c85014, ← c85015, k2, ← c85016] at e_1385_3
  have hr := e_1385_3
  simp only [e_1385_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13130)) := by
  have c85017 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85018 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85019 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85020 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85021 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85022 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_4 := arithEq_of_rows h (row := 1385) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_4
  simp only [← c85017, ← c85018, ← c85019] at e_1385_4
  have e_1385_5 := arithEq_of_rows h (row := 1385) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_5
  simp only [← c85020, ← c85021, k2, ← c85022] at e_1385_5
  have hr := e_1385_5
  simp only [e_1385_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13131)) := by
  have c85023 := (publicBatchWrapper86_copies2656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85024 := (publicBatchWrapper86_copies2657 a h).1
  have c85025 := (publicBatchWrapper86_copies2657 a h).2.1
  have c85026 := (publicBatchWrapper86_copies2657 a h).2.2.1
  have c85027 := (publicBatchWrapper86_copies2657 a h).2.2.2.1
  have c85028 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_6 := arithEq_of_rows h (row := 1385) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_6
  simp only [← c85023, ← c85024, ← c85025] at e_1385_6
  have e_1385_7 := arithEq_of_rows h (row := 1385) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_7
  simp only [← c85026, ← c85027, k2, ← c85028] at e_1385_7
  have hr := e_1385_7
  simp only [e_1385_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13132)) := by
  have c85029 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.1
  have c85030 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.1
  have c85031 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.1
  have c85032 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.1
  have c85033 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.1
  have c85034 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_8 := arithEq_of_rows h (row := 1385) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_8
  simp only [← c85029, ← c85030, ← c85031] at e_1385_8
  have e_1385_9 := arithEq_of_rows h (row := 1385) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_9
  simp only [← c85032, ← c85033, k2, ← c85034] at e_1385_9
  have hr := e_1385_9
  simp only [e_1385_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13133)) := by
  have c85035 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85036 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85037 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85038 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85039 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85040 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_10 := arithEq_of_rows h (row := 1385) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_10
  simp only [← c85035, ← c85036, ← c85037] at e_1385_10
  have e_1385_11 := arithEq_of_rows h (row := 1385) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_11
  simp only [← c85038, ← c85039, k2, ← c85040] at e_1385_11
  have hr := e_1385_11
  simp only [e_1385_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13134)) := by
  have c85041 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85042 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85043 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85044 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85045 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85046 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_12 := arithEq_of_rows h (row := 1385) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_12
  simp only [← c85041, ← c85042, ← c85043] at e_1385_12
  have e_1385_13 := arithEq_of_rows h (row := 1385) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_13
  simp only [← c85044, ← c85045, k2, ← c85046] at e_1385_13
  have hr := e_1385_13
  simp only [e_1385_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13135)) := by
  have c85047 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85048 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85049 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85050 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85051 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85052 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_14 := arithEq_of_rows h (row := 1385) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_14
  simp only [← c85047, ← c85048, ← c85049] at e_1385_14
  have e_1385_15 := arithEq_of_rows h (row := 1385) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_15
  simp only [← c85050, ← c85051, k2, ← c85052] at e_1385_15
  have hr := e_1385_15
  simp only [e_1385_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13136)) := by
  have c85053 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85054 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85055 := (publicBatchWrapper86_copies2657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85056 := (publicBatchWrapper86_copies2658 a h).1
  have c85057 := (publicBatchWrapper86_copies2658 a h).2.1
  have c85058 := (publicBatchWrapper86_copies2658 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_16 := arithEq_of_rows h (row := 1385) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_16
  simp only [← c85053, ← c85054, ← c85055] at e_1385_16
  have e_1385_17 := arithEq_of_rows h (row := 1385) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_17
  simp only [← c85056, ← c85057, k2, ← c85058] at e_1385_17
  have hr := e_1385_17
  simp only [e_1385_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1385 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13137)) := by
  have c85059 := (publicBatchWrapper86_copies2658 a h).2.2.2.1
  have c85060 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.1
  have c85061 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.1
  have c85062 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.1
  have c85063 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.1
  have c85064 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1385_18 := arithEq_of_rows h (row := 1385) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_18
  simp only [← c85059, ← c85060, ← c85061] at e_1385_18
  have e_1385_19 := arithEq_of_rows h (row := 1385) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1385_19
  simp only [← c85062, ← c85063, k2, ← c85064] at e_1385_19
  have hr := e_1385_19
  simp only [e_1385_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13138)) := by
  have c85065 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.1
  have c85066 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85067 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85068 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85069 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85070 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_0 := arithEq_of_rows h (row := 1386) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_0
  simp only [← c85065, ← c85066, ← c85067] at e_1386_0
  have e_1386_1 := arithEq_of_rows h (row := 1386) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_1
  simp only [← c85068, ← c85069, k2, ← c85070] at e_1386_1
  have hr := e_1386_1
  simp only [e_1386_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13139)) := by
  have c85071 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85072 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85073 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85074 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85075 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85076 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_2 := arithEq_of_rows h (row := 1386) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_2
  simp only [← c85071, ← c85072, ← c85073] at e_1386_2
  have e_1386_3 := arithEq_of_rows h (row := 1386) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_3
  simp only [← c85074, ← c85075, k2, ← c85076] at e_1386_3
  have hr := e_1386_3
  simp only [e_1386_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13140)) := by
  have c85077 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85078 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85079 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85080 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85081 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85082 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_4 := arithEq_of_rows h (row := 1386) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_4
  simp only [← c85077, ← c85078, ← c85079] at e_1386_4
  have e_1386_5 := arithEq_of_rows h (row := 1386) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_5
  simp only [← c85080, ← c85081, k2, ← c85082] at e_1386_5
  have hr := e_1386_5
  simp only [e_1386_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13141)) := by
  have c85083 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85084 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85085 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85086 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85087 := (publicBatchWrapper86_copies2658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85088 := (publicBatchWrapper86_copies2659 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_6 := arithEq_of_rows h (row := 1386) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_6
  simp only [← c85083, ← c85084, ← c85085] at e_1386_6
  have e_1386_7 := arithEq_of_rows h (row := 1386) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_7
  simp only [← c85086, ← c85087, k2, ← c85088] at e_1386_7
  have hr := e_1386_7
  simp only [e_1386_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13142)) := by
  have c85089 := (publicBatchWrapper86_copies2659 a h).2.1
  have c85090 := (publicBatchWrapper86_copies2659 a h).2.2.1
  have c85091 := (publicBatchWrapper86_copies2659 a h).2.2.2.1
  have c85092 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.1
  have c85093 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.1
  have c85094 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_8 := arithEq_of_rows h (row := 1386) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_8
  simp only [← c85089, ← c85090, ← c85091] at e_1386_8
  have e_1386_9 := arithEq_of_rows h (row := 1386) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_9
  simp only [← c85092, ← c85093, k2, ← c85094] at e_1386_9
  have hr := e_1386_9
  simp only [e_1386_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13143)) := by
  have c85095 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.1
  have c85096 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.1
  have c85097 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.1
  have c85098 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85099 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85100 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_10 := arithEq_of_rows h (row := 1386) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_10
  simp only [← c85095, ← c85096, ← c85097] at e_1386_10
  have e_1386_11 := arithEq_of_rows h (row := 1386) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_11
  simp only [← c85098, ← c85099, k2, ← c85100] at e_1386_11
  have hr := e_1386_11
  simp only [e_1386_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13144)) := by
  have c85101 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85102 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85103 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85104 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85105 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85106 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_12 := arithEq_of_rows h (row := 1386) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_12
  simp only [← c85101, ← c85102, ← c85103] at e_1386_12
  have e_1386_13 := arithEq_of_rows h (row := 1386) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_13
  simp only [← c85104, ← c85105, k2, ← c85106] at e_1386_13
  have hr := e_1386_13
  simp only [e_1386_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13145)) := by
  have c85107 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85108 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85109 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85110 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85111 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85112 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_14 := arithEq_of_rows h (row := 1386) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_14
  simp only [← c85107, ← c85108, ← c85109] at e_1386_14
  have e_1386_15 := arithEq_of_rows h (row := 1386) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_15
  simp only [← c85110, ← c85111, k2, ← c85112] at e_1386_15
  have hr := e_1386_15
  simp only [e_1386_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13146)) := by
  have c85113 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85114 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85115 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85116 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85117 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85118 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_16 := arithEq_of_rows h (row := 1386) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_16
  simp only [← c85113, ← c85114, ← c85115] at e_1386_16
  have e_1386_17 := arithEq_of_rows h (row := 1386) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_17
  simp only [← c85116, ← c85117, k2, ← c85118] at e_1386_17
  have hr := e_1386_17
  simp only [e_1386_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1386 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13147)) := by
  have c85119 := (publicBatchWrapper86_copies2659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85120 := (publicBatchWrapper86_copies2660 a h).1
  have c85121 := (publicBatchWrapper86_copies2660 a h).2.1
  have c85122 := (publicBatchWrapper86_copies2660 a h).2.2.1
  have c85123 := (publicBatchWrapper86_copies2660 a h).2.2.2.1
  have c85124 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1386_18 := arithEq_of_rows h (row := 1386) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_18
  simp only [← c85119, ← c85120, ← c85121] at e_1386_18
  have e_1386_19 := arithEq_of_rows h (row := 1386) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1386_19
  simp only [← c85122, ← c85123, k2, ← c85124] at e_1386_19
  have hr := e_1386_19
  simp only [e_1386_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13148)) := by
  have c85125 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.1
  have c85126 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.1
  have c85127 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.1
  have c85128 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.1
  have c85129 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.1
  have c85130 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_0 := arithEq_of_rows h (row := 1387) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_0
  simp only [← c85125, ← c85126, ← c85127] at e_1387_0
  have e_1387_1 := arithEq_of_rows h (row := 1387) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_1
  simp only [← c85128, ← c85129, k2, ← c85130] at e_1387_1
  have hr := e_1387_1
  simp only [e_1387_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13149)) := by
  have c85131 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85132 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85133 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85134 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85135 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85136 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_2 := arithEq_of_rows h (row := 1387) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_2
  simp only [← c85131, ← c85132, ← c85133] at e_1387_2
  have e_1387_3 := arithEq_of_rows h (row := 1387) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_3
  simp only [← c85134, ← c85135, k2, ← c85136] at e_1387_3
  have hr := e_1387_3
  simp only [e_1387_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13150)) := by
  have c85137 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85138 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85139 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85140 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85141 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85142 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_4 := arithEq_of_rows h (row := 1387) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_4
  simp only [← c85137, ← c85138, ← c85139] at e_1387_4
  have e_1387_5 := arithEq_of_rows h (row := 1387) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_5
  simp only [← c85140, ← c85141, k2, ← c85142] at e_1387_5
  have hr := e_1387_5
  simp only [e_1387_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13151)) := by
  have c85143 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85144 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85145 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85146 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85147 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85148 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_6 := arithEq_of_rows h (row := 1387) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_6
  simp only [← c85143, ← c85144, ← c85145] at e_1387_6
  have e_1387_7 := arithEq_of_rows h (row := 1387) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_7
  simp only [← c85146, ← c85147, k2, ← c85148] at e_1387_7
  have hr := e_1387_7
  simp only [e_1387_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13152)) := by
  have c85149 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85150 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85151 := (publicBatchWrapper86_copies2660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85152 := (publicBatchWrapper86_copies2661 a h).1
  have c85153 := (publicBatchWrapper86_copies2661 a h).2.1
  have c85154 := (publicBatchWrapper86_copies2661 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_8 := arithEq_of_rows h (row := 1387) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_8
  simp only [← c85149, ← c85150, ← c85151] at e_1387_8
  have e_1387_9 := arithEq_of_rows h (row := 1387) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_9
  simp only [← c85152, ← c85153, k2, ← c85154] at e_1387_9
  have hr := e_1387_9
  simp only [e_1387_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13153)) := by
  have c85155 := (publicBatchWrapper86_copies2661 a h).2.2.2.1
  have c85156 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.1
  have c85157 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.1
  have c85158 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.1
  have c85159 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.1
  have c85160 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_10 := arithEq_of_rows h (row := 1387) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_10
  simp only [← c85155, ← c85156, ← c85157] at e_1387_10
  have e_1387_11 := arithEq_of_rows h (row := 1387) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_11
  simp only [← c85158, ← c85159, k2, ← c85160] at e_1387_11
  have hr := e_1387_11
  simp only [e_1387_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13154)) := by
  have c85161 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.1
  have c85162 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85163 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85164 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85165 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85166 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_12 := arithEq_of_rows h (row := 1387) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_12
  simp only [← c85161, ← c85162, ← c85163] at e_1387_12
  have e_1387_13 := arithEq_of_rows h (row := 1387) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_13
  simp only [← c85164, ← c85165, k2, ← c85166] at e_1387_13
  have hr := e_1387_13
  simp only [e_1387_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13155)) := by
  have c85167 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85168 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85169 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85170 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85171 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85172 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_14 := arithEq_of_rows h (row := 1387) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_14
  simp only [← c85167, ← c85168, ← c85169] at e_1387_14
  have e_1387_15 := arithEq_of_rows h (row := 1387) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_15
  simp only [← c85170, ← c85171, k2, ← c85172] at e_1387_15
  have hr := e_1387_15
  simp only [e_1387_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13156)) := by
  have c85173 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85174 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85175 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85176 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85177 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85178 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_16 := arithEq_of_rows h (row := 1387) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_16
  simp only [← c85173, ← c85174, ← c85175] at e_1387_16
  have e_1387_17 := arithEq_of_rows h (row := 1387) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_17
  simp only [← c85176, ← c85177, k2, ← c85178] at e_1387_17
  have hr := e_1387_17
  simp only [e_1387_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1387 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13157)) := by
  have c85179 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85180 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85181 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85182 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85183 := (publicBatchWrapper86_copies2661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85184 := (publicBatchWrapper86_copies2662 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1387_18 := arithEq_of_rows h (row := 1387) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_18
  simp only [← c85179, ← c85180, ← c85181] at e_1387_18
  have e_1387_19 := arithEq_of_rows h (row := 1387) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1387_19
  simp only [← c85182, ← c85183, k2, ← c85184] at e_1387_19
  have hr := e_1387_19
  simp only [e_1387_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13158)) := by
  have c85185 := (publicBatchWrapper86_copies2662 a h).2.1
  have c85186 := (publicBatchWrapper86_copies2662 a h).2.2.1
  have c85187 := (publicBatchWrapper86_copies2662 a h).2.2.2.1
  have c85188 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.1
  have c85189 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.1
  have c85190 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_0 := arithEq_of_rows h (row := 1388) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_0
  simp only [← c85185, ← c85186, ← c85187] at e_1388_0
  have e_1388_1 := arithEq_of_rows h (row := 1388) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_1
  simp only [← c85188, ← c85189, k2, ← c85190] at e_1388_1
  have hr := e_1388_1
  simp only [e_1388_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13159)) := by
  have c85191 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.1
  have c85192 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.1
  have c85193 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.1
  have c85194 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85195 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85196 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_2 := arithEq_of_rows h (row := 1388) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_2
  simp only [← c85191, ← c85192, ← c85193] at e_1388_2
  have e_1388_3 := arithEq_of_rows h (row := 1388) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_3
  simp only [← c85194, ← c85195, k2, ← c85196] at e_1388_3
  have hr := e_1388_3
  simp only [e_1388_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13160)) := by
  have c85197 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85198 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85199 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85200 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85201 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85202 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_4 := arithEq_of_rows h (row := 1388) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_4
  simp only [← c85197, ← c85198, ← c85199] at e_1388_4
  have e_1388_5 := arithEq_of_rows h (row := 1388) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_5
  simp only [← c85200, ← c85201, k2, ← c85202] at e_1388_5
  have hr := e_1388_5
  simp only [e_1388_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13161)) := by
  have c85203 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85204 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85205 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85206 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85207 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85208 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_6 := arithEq_of_rows h (row := 1388) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_6
  simp only [← c85203, ← c85204, ← c85205] at e_1388_6
  have e_1388_7 := arithEq_of_rows h (row := 1388) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_7
  simp only [← c85206, ← c85207, k2, ← c85208] at e_1388_7
  have hr := e_1388_7
  simp only [e_1388_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13162)) := by
  have c85209 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85210 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85211 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85212 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85213 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85214 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_8 := arithEq_of_rows h (row := 1388) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_8
  simp only [← c85209, ← c85210, ← c85211] at e_1388_8
  have e_1388_9 := arithEq_of_rows h (row := 1388) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_9
  simp only [← c85212, ← c85213, k2, ← c85214] at e_1388_9
  have hr := e_1388_9
  simp only [e_1388_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13163)) := by
  have c85215 := (publicBatchWrapper86_copies2662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85216 := (publicBatchWrapper86_copies2663 a h).1
  have c85217 := (publicBatchWrapper86_copies2663 a h).2.1
  have c85218 := (publicBatchWrapper86_copies2663 a h).2.2.1
  have c85219 := (publicBatchWrapper86_copies2663 a h).2.2.2.1
  have c85220 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_10 := arithEq_of_rows h (row := 1388) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_10
  simp only [← c85215, ← c85216, ← c85217] at e_1388_10
  have e_1388_11 := arithEq_of_rows h (row := 1388) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_11
  simp only [← c85218, ← c85219, k2, ← c85220] at e_1388_11
  have hr := e_1388_11
  simp only [e_1388_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13164)) := by
  have c85221 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.1
  have c85222 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.1
  have c85223 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.1
  have c85224 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.1
  have c85225 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.1
  have c85226 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_12 := arithEq_of_rows h (row := 1388) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_12
  simp only [← c85221, ← c85222, ← c85223] at e_1388_12
  have e_1388_13 := arithEq_of_rows h (row := 1388) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_13
  simp only [← c85224, ← c85225, k2, ← c85226] at e_1388_13
  have hr := e_1388_13
  simp only [e_1388_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13165)) := by
  have c85227 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85228 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85229 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85230 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85231 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85232 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_14 := arithEq_of_rows h (row := 1388) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_14
  simp only [← c85227, ← c85228, ← c85229] at e_1388_14
  have e_1388_15 := arithEq_of_rows h (row := 1388) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_15
  simp only [← c85230, ← c85231, k2, ← c85232] at e_1388_15
  have hr := e_1388_15
  simp only [e_1388_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13166)) := by
  have c85233 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85234 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85235 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85236 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85237 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85238 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_16 := arithEq_of_rows h (row := 1388) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_16
  simp only [← c85233, ← c85234, ← c85235] at e_1388_16
  have e_1388_17 := arithEq_of_rows h (row := 1388) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_17
  simp only [← c85236, ← c85237, k2, ← c85238] at e_1388_17
  have hr := e_1388_17
  simp only [e_1388_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1388 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13167)) := by
  have c85239 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85240 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85241 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85242 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85243 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85244 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1388_18 := arithEq_of_rows h (row := 1388) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_18
  simp only [← c85239, ← c85240, ← c85241] at e_1388_18
  have e_1388_19 := arithEq_of_rows h (row := 1388) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1388_19
  simp only [← c85242, ← c85243, k2, ← c85244] at e_1388_19
  have hr := e_1388_19
  simp only [e_1388_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13168)) := by
  have c85245 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85246 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85247 := (publicBatchWrapper86_copies2663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85248 := (publicBatchWrapper86_copies2664 a h).1
  have c85249 := (publicBatchWrapper86_copies2664 a h).2.1
  have c85250 := (publicBatchWrapper86_copies2664 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_0 := arithEq_of_rows h (row := 1389) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_0
  simp only [← c85245, ← c85246, ← c85247] at e_1389_0
  have e_1389_1 := arithEq_of_rows h (row := 1389) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_1
  simp only [← c85248, ← c85249, k2, ← c85250] at e_1389_1
  have hr := e_1389_1
  simp only [e_1389_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13169)) := by
  have c85251 := (publicBatchWrapper86_copies2664 a h).2.2.2.1
  have c85252 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.1
  have c85253 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.1
  have c85254 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.1
  have c85255 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.1
  have c85256 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_2 := arithEq_of_rows h (row := 1389) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_2
  simp only [← c85251, ← c85252, ← c85253] at e_1389_2
  have e_1389_3 := arithEq_of_rows h (row := 1389) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_3
  simp only [← c85254, ← c85255, k2, ← c85256] at e_1389_3
  have hr := e_1389_3
  simp only [e_1389_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13210)) := by
  have c85257 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.1
  have c85258 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85259 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85260 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85261 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85262 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_4 := arithEq_of_rows h (row := 1389) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_4
  simp only [← c85257, ← c85258, ← c85259] at e_1389_4
  have e_1389_5 := arithEq_of_rows h (row := 1389) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_5
  simp only [← c85260, ← c85261, k2, ← c85262] at e_1389_5
  have hr := e_1389_5
  simp only [e_1389_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13211)) := by
  have c85263 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85264 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85265 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85266 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85267 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85268 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_6 := arithEq_of_rows h (row := 1389) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_6
  simp only [← c85263, ← c85264, ← c85265] at e_1389_6
  have e_1389_7 := arithEq_of_rows h (row := 1389) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_7
  simp only [← c85266, ← c85267, k2, ← c85268] at e_1389_7
  have hr := e_1389_7
  simp only [e_1389_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13212)) := by
  have c85269 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85270 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85271 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85272 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85273 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85274 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_8 := arithEq_of_rows h (row := 1389) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_8
  simp only [← c85269, ← c85270, ← c85271] at e_1389_8
  have e_1389_9 := arithEq_of_rows h (row := 1389) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_9
  simp only [← c85272, ← c85273, k2, ← c85274] at e_1389_9
  have hr := e_1389_9
  simp only [e_1389_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13213)) := by
  have c85275 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85276 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85277 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85278 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85279 := (publicBatchWrapper86_copies2664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85280 := (publicBatchWrapper86_copies2665 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_10 := arithEq_of_rows h (row := 1389) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_10
  simp only [← c85275, ← c85276, ← c85277] at e_1389_10
  have e_1389_11 := arithEq_of_rows h (row := 1389) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_11
  simp only [← c85278, ← c85279, k2, ← c85280] at e_1389_11
  have hr := e_1389_11
  simp only [e_1389_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13214)) := by
  have c85281 := (publicBatchWrapper86_copies2665 a h).2.1
  have c85282 := (publicBatchWrapper86_copies2665 a h).2.2.1
  have c85283 := (publicBatchWrapper86_copies2665 a h).2.2.2.1
  have c85284 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.1
  have c85285 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.1
  have c85286 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_12 := arithEq_of_rows h (row := 1389) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_12
  simp only [← c85281, ← c85282, ← c85283] at e_1389_12
  have e_1389_13 := arithEq_of_rows h (row := 1389) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_13
  simp only [← c85284, ← c85285, k2, ← c85286] at e_1389_13
  have hr := e_1389_13
  simp only [e_1389_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13215)) := by
  have c85287 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.1
  have c85288 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.1
  have c85289 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.1
  have c85290 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85291 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85292 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_14 := arithEq_of_rows h (row := 1389) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_14
  simp only [← c85287, ← c85288, ← c85289] at e_1389_14
  have e_1389_15 := arithEq_of_rows h (row := 1389) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_15
  simp only [← c85290, ← c85291, k2, ← c85292] at e_1389_15
  have hr := e_1389_15
  simp only [e_1389_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13216)) := by
  have c85293 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85294 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85295 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85296 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85297 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85298 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_16 := arithEq_of_rows h (row := 1389) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_16
  simp only [← c85293, ← c85294, ← c85295] at e_1389_16
  have e_1389_17 := arithEq_of_rows h (row := 1389) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_17
  simp only [← c85296, ← c85297, k2, ← c85298] at e_1389_17
  have hr := e_1389_17
  simp only [e_1389_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1389 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13217)) := by
  have c85299 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85300 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85301 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85302 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85303 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85304 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1389_18 := arithEq_of_rows h (row := 1389) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_18
  simp only [← c85299, ← c85300, ← c85301] at e_1389_18
  have e_1389_19 := arithEq_of_rows h (row := 1389) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1389_19
  simp only [← c85302, ← c85303, k2, ← c85304] at e_1389_19
  have hr := e_1389_19
  simp only [e_1389_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13218)) := by
  have c85305 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85306 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85307 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85308 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85309 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85310 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_0 := arithEq_of_rows h (row := 1390) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_0
  simp only [← c85305, ← c85306, ← c85307] at e_1390_0
  have e_1390_1 := arithEq_of_rows h (row := 1390) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_1
  simp only [← c85308, ← c85309, k2, ← c85310] at e_1390_1
  have hr := e_1390_1
  simp only [e_1390_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13219)) := by
  have c85311 := (publicBatchWrapper86_copies2665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85312 := (publicBatchWrapper86_copies2666 a h).1
  have c85313 := (publicBatchWrapper86_copies2666 a h).2.1
  have c85314 := (publicBatchWrapper86_copies2666 a h).2.2.1
  have c85315 := (publicBatchWrapper86_copies2666 a h).2.2.2.1
  have c85316 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_2 := arithEq_of_rows h (row := 1390) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_2
  simp only [← c85311, ← c85312, ← c85313] at e_1390_2
  have e_1390_3 := arithEq_of_rows h (row := 1390) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_3
  simp only [← c85314, ← c85315, k2, ← c85316] at e_1390_3
  have hr := e_1390_3
  simp only [e_1390_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13220)) := by
  have c85317 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.1
  have c85318 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.1
  have c85319 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.1
  have c85320 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.1
  have c85321 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.1
  have c85322 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_4 := arithEq_of_rows h (row := 1390) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_4
  simp only [← c85317, ← c85318, ← c85319] at e_1390_4
  have e_1390_5 := arithEq_of_rows h (row := 1390) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_5
  simp only [← c85320, ← c85321, k2, ← c85322] at e_1390_5
  have hr := e_1390_5
  simp only [e_1390_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13221)) := by
  have c85323 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85324 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85325 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85326 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85327 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85328 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_6 := arithEq_of_rows h (row := 1390) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_6
  simp only [← c85323, ← c85324, ← c85325] at e_1390_6
  have e_1390_7 := arithEq_of_rows h (row := 1390) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_7
  simp only [← c85326, ← c85327, k2, ← c85328] at e_1390_7
  have hr := e_1390_7
  simp only [e_1390_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13222)) := by
  have c85329 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85330 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85331 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85332 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85333 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85334 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_8 := arithEq_of_rows h (row := 1390) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_8
  simp only [← c85329, ← c85330, ← c85331] at e_1390_8
  have e_1390_9 := arithEq_of_rows h (row := 1390) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_9
  simp only [← c85332, ← c85333, k2, ← c85334] at e_1390_9
  have hr := e_1390_9
  simp only [e_1390_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13223)) := by
  have c85335 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85336 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85337 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85338 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85339 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85340 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_10 := arithEq_of_rows h (row := 1390) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_10
  simp only [← c85335, ← c85336, ← c85337] at e_1390_10
  have e_1390_11 := arithEq_of_rows h (row := 1390) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_11
  simp only [← c85338, ← c85339, k2, ← c85340] at e_1390_11
  have hr := e_1390_11
  simp only [e_1390_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13224)) := by
  have c85341 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85342 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85343 := (publicBatchWrapper86_copies2666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85344 := (publicBatchWrapper86_copies2667 a h).1
  have c85345 := (publicBatchWrapper86_copies2667 a h).2.1
  have c85346 := (publicBatchWrapper86_copies2667 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_12 := arithEq_of_rows h (row := 1390) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_12
  simp only [← c85341, ← c85342, ← c85343] at e_1390_12
  have e_1390_13 := arithEq_of_rows h (row := 1390) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_13
  simp only [← c85344, ← c85345, k2, ← c85346] at e_1390_13
  have hr := e_1390_13
  simp only [e_1390_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13225)) := by
  have c85347 := (publicBatchWrapper86_copies2667 a h).2.2.2.1
  have c85348 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.1
  have c85349 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.1
  have c85350 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.1
  have c85351 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.1
  have c85352 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_14 := arithEq_of_rows h (row := 1390) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_14
  simp only [← c85347, ← c85348, ← c85349] at e_1390_14
  have e_1390_15 := arithEq_of_rows h (row := 1390) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_15
  simp only [← c85350, ← c85351, k2, ← c85352] at e_1390_15
  have hr := e_1390_15
  simp only [e_1390_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13226)) := by
  have c85353 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.1
  have c85354 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85355 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85356 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85357 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85358 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_16 := arithEq_of_rows h (row := 1390) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_16
  simp only [← c85353, ← c85354, ← c85355] at e_1390_16
  have e_1390_17 := arithEq_of_rows h (row := 1390) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_17
  simp only [← c85356, ← c85357, k2, ← c85358] at e_1390_17
  have hr := e_1390_17
  simp only [e_1390_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1390 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13227)) := by
  have c85359 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85360 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85361 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85362 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85363 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85364 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1390_18 := arithEq_of_rows h (row := 1390) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_18
  simp only [← c85359, ← c85360, ← c85361] at e_1390_18
  have e_1390_19 := arithEq_of_rows h (row := 1390) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1390_19
  simp only [← c85362, ← c85363, k2, ← c85364] at e_1390_19
  have hr := e_1390_19
  simp only [e_1390_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13228)) := by
  have c85365 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85366 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85367 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85368 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85369 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85370 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_0 := arithEq_of_rows h (row := 1391) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_0
  simp only [← c85365, ← c85366, ← c85367] at e_1391_0
  have e_1391_1 := arithEq_of_rows h (row := 1391) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_1
  simp only [← c85368, ← c85369, k2, ← c85370] at e_1391_1
  have hr := e_1391_1
  simp only [e_1391_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13229)) := by
  have c85371 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85372 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85373 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85374 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85375 := (publicBatchWrapper86_copies2667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85376 := (publicBatchWrapper86_copies2668 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_2 := arithEq_of_rows h (row := 1391) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_2
  simp only [← c85371, ← c85372, ← c85373] at e_1391_2
  have e_1391_3 := arithEq_of_rows h (row := 1391) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_3
  simp only [← c85374, ← c85375, k2, ← c85376] at e_1391_3
  have hr := e_1391_3
  simp only [e_1391_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13230)) := by
  have c85377 := (publicBatchWrapper86_copies2668 a h).2.1
  have c85378 := (publicBatchWrapper86_copies2668 a h).2.2.1
  have c85379 := (publicBatchWrapper86_copies2668 a h).2.2.2.1
  have c85380 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.1
  have c85381 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.1
  have c85382 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_4 := arithEq_of_rows h (row := 1391) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_4
  simp only [← c85377, ← c85378, ← c85379] at e_1391_4
  have e_1391_5 := arithEq_of_rows h (row := 1391) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_5
  simp only [← c85380, ← c85381, k2, ← c85382] at e_1391_5
  have hr := e_1391_5
  simp only [e_1391_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13231)) := by
  have c85383 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.1
  have c85384 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.1
  have c85385 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.1
  have c85386 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85387 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85388 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_6 := arithEq_of_rows h (row := 1391) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_6
  simp only [← c85383, ← c85384, ← c85385] at e_1391_6
  have e_1391_7 := arithEq_of_rows h (row := 1391) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_7
  simp only [← c85386, ← c85387, k2, ← c85388] at e_1391_7
  have hr := e_1391_7
  simp only [e_1391_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13232)) := by
  have c85389 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85390 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85391 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85392 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85393 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85394 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_8 := arithEq_of_rows h (row := 1391) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_8
  simp only [← c85389, ← c85390, ← c85391] at e_1391_8
  have e_1391_9 := arithEq_of_rows h (row := 1391) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_9
  simp only [← c85392, ← c85393, k2, ← c85394] at e_1391_9
  have hr := e_1391_9
  simp only [e_1391_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13233)) := by
  have c85395 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85396 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85397 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85398 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85399 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85400 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_10 := arithEq_of_rows h (row := 1391) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_10
  simp only [← c85395, ← c85396, ← c85397] at e_1391_10
  have e_1391_11 := arithEq_of_rows h (row := 1391) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_11
  simp only [← c85398, ← c85399, k2, ← c85400] at e_1391_11
  have hr := e_1391_11
  simp only [e_1391_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13234)) := by
  have c85401 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85402 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85403 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85404 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85405 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85406 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_12 := arithEq_of_rows h (row := 1391) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_12
  simp only [← c85401, ← c85402, ← c85403] at e_1391_12
  have e_1391_13 := arithEq_of_rows h (row := 1391) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_13
  simp only [← c85404, ← c85405, k2, ← c85406] at e_1391_13
  have hr := e_1391_13
  simp only [e_1391_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13235)) := by
  have c85407 := (publicBatchWrapper86_copies2668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85408 := (publicBatchWrapper86_copies2669 a h).1
  have c85409 := (publicBatchWrapper86_copies2669 a h).2.1
  have c85410 := (publicBatchWrapper86_copies2669 a h).2.2.1
  have c85411 := (publicBatchWrapper86_copies2669 a h).2.2.2.1
  have c85412 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_14 := arithEq_of_rows h (row := 1391) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_14
  simp only [← c85407, ← c85408, ← c85409] at e_1391_14
  have e_1391_15 := arithEq_of_rows h (row := 1391) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_15
  simp only [← c85410, ← c85411, k2, ← c85412] at e_1391_15
  have hr := e_1391_15
  simp only [e_1391_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13236)) := by
  have c85413 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.1
  have c85414 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.1
  have c85415 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.1
  have c85416 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.1
  have c85417 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.1
  have c85418 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_16 := arithEq_of_rows h (row := 1391) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_16
  simp only [← c85413, ← c85414, ← c85415] at e_1391_16
  have e_1391_17 := arithEq_of_rows h (row := 1391) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_17
  simp only [← c85416, ← c85417, k2, ← c85418] at e_1391_17
  have hr := e_1391_17
  simp only [e_1391_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1391 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13237)) := by
  have c85419 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85420 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85421 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85422 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85423 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85424 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1391_18 := arithEq_of_rows h (row := 1391) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_18
  simp only [← c85419, ← c85420, ← c85421] at e_1391_18
  have e_1391_19 := arithEq_of_rows h (row := 1391) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1391_19
  simp only [← c85422, ← c85423, k2, ← c85424] at e_1391_19
  have hr := e_1391_19
  simp only [e_1391_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13238)) := by
  have c85425 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85426 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85427 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85428 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85429 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85430 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_0 := arithEq_of_rows h (row := 1392) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_0
  simp only [← c85425, ← c85426, ← c85427] at e_1392_0
  have e_1392_1 := arithEq_of_rows h (row := 1392) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_1
  simp only [← c85428, ← c85429, k2, ← c85430] at e_1392_1
  have hr := e_1392_1
  simp only [e_1392_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13239)) := by
  have c85431 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85432 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85433 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85434 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85435 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85436 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_2 := arithEq_of_rows h (row := 1392) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_2
  simp only [← c85431, ← c85432, ← c85433] at e_1392_2
  have e_1392_3 := arithEq_of_rows h (row := 1392) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_3
  simp only [← c85434, ← c85435, k2, ← c85436] at e_1392_3
  have hr := e_1392_3
  simp only [e_1392_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13240)) := by
  have c85437 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85438 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85439 := (publicBatchWrapper86_copies2669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85440 := (publicBatchWrapper86_copies2670 a h).1
  have c85441 := (publicBatchWrapper86_copies2670 a h).2.1
  have c85442 := (publicBatchWrapper86_copies2670 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_4 := arithEq_of_rows h (row := 1392) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_4
  simp only [← c85437, ← c85438, ← c85439] at e_1392_4
  have e_1392_5 := arithEq_of_rows h (row := 1392) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_5
  simp only [← c85440, ← c85441, k2, ← c85442] at e_1392_5
  have hr := e_1392_5
  simp only [e_1392_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13241)) := by
  have c85443 := (publicBatchWrapper86_copies2670 a h).2.2.2.1
  have c85444 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.1
  have c85445 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.1
  have c85446 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.1
  have c85447 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.1
  have c85448 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_6 := arithEq_of_rows h (row := 1392) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_6
  simp only [← c85443, ← c85444, ← c85445] at e_1392_6
  have e_1392_7 := arithEq_of_rows h (row := 1392) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_7
  simp only [← c85446, ← c85447, k2, ← c85448] at e_1392_7
  have hr := e_1392_7
  simp only [e_1392_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13242)) := by
  have c85449 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.1
  have c85450 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85451 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85452 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85453 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85454 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_8 := arithEq_of_rows h (row := 1392) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_8
  simp only [← c85449, ← c85450, ← c85451] at e_1392_8
  have e_1392_9 := arithEq_of_rows h (row := 1392) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_9
  simp only [← c85452, ← c85453, k2, ← c85454] at e_1392_9
  have hr := e_1392_9
  simp only [e_1392_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13243)) := by
  have c85455 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85456 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85457 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85458 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85459 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85460 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_10 := arithEq_of_rows h (row := 1392) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_10
  simp only [← c85455, ← c85456, ← c85457] at e_1392_10
  have e_1392_11 := arithEq_of_rows h (row := 1392) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_11
  simp only [← c85458, ← c85459, k2, ← c85460] at e_1392_11
  have hr := e_1392_11
  simp only [e_1392_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13244)) := by
  have c85461 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85462 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85463 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85464 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85465 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85466 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_12 := arithEq_of_rows h (row := 1392) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_12
  simp only [← c85461, ← c85462, ← c85463] at e_1392_12
  have e_1392_13 := arithEq_of_rows h (row := 1392) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_13
  simp only [← c85464, ← c85465, k2, ← c85466] at e_1392_13
  have hr := e_1392_13
  simp only [e_1392_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13245)) := by
  have c85467 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85468 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85469 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85470 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85471 := (publicBatchWrapper86_copies2670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85472 := (publicBatchWrapper86_copies2671 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_14 := arithEq_of_rows h (row := 1392) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_14
  simp only [← c85467, ← c85468, ← c85469] at e_1392_14
  have e_1392_15 := arithEq_of_rows h (row := 1392) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_15
  simp only [← c85470, ← c85471, k2, ← c85472] at e_1392_15
  have hr := e_1392_15
  simp only [e_1392_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13246)) := by
  have c85473 := (publicBatchWrapper86_copies2671 a h).2.1
  have c85474 := (publicBatchWrapper86_copies2671 a h).2.2.1
  have c85475 := (publicBatchWrapper86_copies2671 a h).2.2.2.1
  have c85476 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.1
  have c85477 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.1
  have c85478 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_16 := arithEq_of_rows h (row := 1392) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_16
  simp only [← c85473, ← c85474, ← c85475] at e_1392_16
  have e_1392_17 := arithEq_of_rows h (row := 1392) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_17
  simp only [← c85476, ← c85477, k2, ← c85478] at e_1392_17
  have hr := e_1392_17
  simp only [e_1392_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1392 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13247)) := by
  have c85479 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.1
  have c85480 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.1
  have c85481 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.1
  have c85482 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85483 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85484 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1392_18 := arithEq_of_rows h (row := 1392) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_18
  simp only [← c85479, ← c85480, ← c85481] at e_1392_18
  have e_1392_19 := arithEq_of_rows h (row := 1392) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1392_19
  simp only [← c85482, ← c85483, k2, ← c85484] at e_1392_19
  have hr := e_1392_19
  simp only [e_1392_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13248)) := by
  have c85485 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85486 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85487 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85488 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85489 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85490 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_0 := arithEq_of_rows h (row := 1393) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_0
  simp only [← c85485, ← c85486, ← c85487] at e_1393_0
  have e_1393_1 := arithEq_of_rows h (row := 1393) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_1
  simp only [← c85488, ← c85489, k2, ← c85490] at e_1393_1
  have hr := e_1393_1
  simp only [e_1393_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13249)) := by
  have c85491 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85492 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85493 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85494 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85495 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85496 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_2 := arithEq_of_rows h (row := 1393) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_2
  simp only [← c85491, ← c85492, ← c85493] at e_1393_2
  have e_1393_3 := arithEq_of_rows h (row := 1393) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_3
  simp only [← c85494, ← c85495, k2, ← c85496] at e_1393_3
  have hr := e_1393_3
  simp only [e_1393_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13250)) := by
  have c85497 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85498 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85499 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85500 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85501 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85502 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_4 := arithEq_of_rows h (row := 1393) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_4
  simp only [← c85497, ← c85498, ← c85499] at e_1393_4
  have e_1393_5 := arithEq_of_rows h (row := 1393) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_5
  simp only [← c85500, ← c85501, k2, ← c85502] at e_1393_5
  have hr := e_1393_5
  simp only [e_1393_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13251)) := by
  have c85503 := (publicBatchWrapper86_copies2671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85504 := (publicBatchWrapper86_copies2672 a h).1
  have c85505 := (publicBatchWrapper86_copies2672 a h).2.1
  have c85506 := (publicBatchWrapper86_copies2672 a h).2.2.1
  have c85507 := (publicBatchWrapper86_copies2672 a h).2.2.2.1
  have c85508 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_6 := arithEq_of_rows h (row := 1393) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_6
  simp only [← c85503, ← c85504, ← c85505] at e_1393_6
  have e_1393_7 := arithEq_of_rows h (row := 1393) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_7
  simp only [← c85506, ← c85507, k2, ← c85508] at e_1393_7
  have hr := e_1393_7
  simp only [e_1393_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13252)) := by
  have c85509 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.1
  have c85510 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.1
  have c85511 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.1
  have c85512 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.1
  have c85513 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.1
  have c85514 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_8 := arithEq_of_rows h (row := 1393) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_8
  simp only [← c85509, ← c85510, ← c85511] at e_1393_8
  have e_1393_9 := arithEq_of_rows h (row := 1393) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_9
  simp only [← c85512, ← c85513, k2, ← c85514] at e_1393_9
  have hr := e_1393_9
  simp only [e_1393_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13253)) := by
  have c85515 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85516 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85517 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85518 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85519 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85520 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_10 := arithEq_of_rows h (row := 1393) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_10
  simp only [← c85515, ← c85516, ← c85517] at e_1393_10
  have e_1393_11 := arithEq_of_rows h (row := 1393) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_11
  simp only [← c85518, ← c85519, k2, ← c85520] at e_1393_11
  have hr := e_1393_11
  simp only [e_1393_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13254)) := by
  have c85521 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85522 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85523 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85524 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85525 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85526 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_12 := arithEq_of_rows h (row := 1393) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_12
  simp only [← c85521, ← c85522, ← c85523] at e_1393_12
  have e_1393_13 := arithEq_of_rows h (row := 1393) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_13
  simp only [← c85524, ← c85525, k2, ← c85526] at e_1393_13
  have hr := e_1393_13
  simp only [e_1393_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13255)) := by
  have c85527 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85528 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85529 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85530 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85531 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85532 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_14 := arithEq_of_rows h (row := 1393) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_14
  simp only [← c85527, ← c85528, ← c85529] at e_1393_14
  have e_1393_15 := arithEq_of_rows h (row := 1393) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_15
  simp only [← c85530, ← c85531, k2, ← c85532] at e_1393_15
  have hr := e_1393_15
  simp only [e_1393_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13256)) := by
  have c85533 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85534 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85535 := (publicBatchWrapper86_copies2672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85536 := (publicBatchWrapper86_copies2673 a h).1
  have c85537 := (publicBatchWrapper86_copies2673 a h).2.1
  have c85538 := (publicBatchWrapper86_copies2673 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_16 := arithEq_of_rows h (row := 1393) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_16
  simp only [← c85533, ← c85534, ← c85535] at e_1393_16
  have e_1393_17 := arithEq_of_rows h (row := 1393) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_17
  simp only [← c85536, ← c85537, k2, ← c85538] at e_1393_17
  have hr := e_1393_17
  simp only [e_1393_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1393 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13257)) := by
  have c85539 := (publicBatchWrapper86_copies2673 a h).2.2.2.1
  have c85540 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.1
  have c85541 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.1
  have c85542 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.1
  have c85543 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.1
  have c85544 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1393_18 := arithEq_of_rows h (row := 1393) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_18
  simp only [← c85539, ← c85540, ← c85541] at e_1393_18
  have e_1393_19 := arithEq_of_rows h (row := 1393) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1393_19
  simp only [← c85542, ← c85543, k2, ← c85544] at e_1393_19
  have hr := e_1393_19
  simp only [e_1393_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13258)) := by
  have c85545 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.1
  have c85546 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85547 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85548 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85549 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85550 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_0 := arithEq_of_rows h (row := 1394) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_0
  simp only [← c85545, ← c85546, ← c85547] at e_1394_0
  have e_1394_1 := arithEq_of_rows h (row := 1394) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_1
  simp only [← c85548, ← c85549, k2, ← c85550] at e_1394_1
  have hr := e_1394_1
  simp only [e_1394_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13259)) := by
  have c85551 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85552 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85553 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85554 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85555 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85556 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_2 := arithEq_of_rows h (row := 1394) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_2
  simp only [← c85551, ← c85552, ← c85553] at e_1394_2
  have e_1394_3 := arithEq_of_rows h (row := 1394) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_3
  simp only [← c85554, ← c85555, k2, ← c85556] at e_1394_3
  have hr := e_1394_3
  simp only [e_1394_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13260)) := by
  have c85557 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85558 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85559 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85560 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85561 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85562 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_4 := arithEq_of_rows h (row := 1394) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_4
  simp only [← c85557, ← c85558, ← c85559] at e_1394_4
  have e_1394_5 := arithEq_of_rows h (row := 1394) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_5
  simp only [← c85560, ← c85561, k2, ← c85562] at e_1394_5
  have hr := e_1394_5
  simp only [e_1394_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13261)) := by
  have c85563 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85564 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85565 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85566 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85567 := (publicBatchWrapper86_copies2673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85568 := (publicBatchWrapper86_copies2674 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_6 := arithEq_of_rows h (row := 1394) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_6
  simp only [← c85563, ← c85564, ← c85565] at e_1394_6
  have e_1394_7 := arithEq_of_rows h (row := 1394) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_7
  simp only [← c85566, ← c85567, k2, ← c85568] at e_1394_7
  have hr := e_1394_7
  simp only [e_1394_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13262)) := by
  have c85569 := (publicBatchWrapper86_copies2674 a h).2.1
  have c85570 := (publicBatchWrapper86_copies2674 a h).2.2.1
  have c85571 := (publicBatchWrapper86_copies2674 a h).2.2.2.1
  have c85572 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.1
  have c85573 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.1
  have c85574 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_8 := arithEq_of_rows h (row := 1394) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_8
  simp only [← c85569, ← c85570, ← c85571] at e_1394_8
  have e_1394_9 := arithEq_of_rows h (row := 1394) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_9
  simp only [← c85572, ← c85573, k2, ← c85574] at e_1394_9
  have hr := e_1394_9
  simp only [e_1394_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13263)) := by
  have c85575 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.1
  have c85576 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.1
  have c85577 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.1
  have c85578 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85579 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85580 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_10 := arithEq_of_rows h (row := 1394) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_10
  simp only [← c85575, ← c85576, ← c85577] at e_1394_10
  have e_1394_11 := arithEq_of_rows h (row := 1394) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_11
  simp only [← c85578, ← c85579, k2, ← c85580] at e_1394_11
  have hr := e_1394_11
  simp only [e_1394_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13264)) := by
  have c85581 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85582 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85583 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85584 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85585 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85586 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_12 := arithEq_of_rows h (row := 1394) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_12
  simp only [← c85581, ← c85582, ← c85583] at e_1394_12
  have e_1394_13 := arithEq_of_rows h (row := 1394) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_13
  simp only [← c85584, ← c85585, k2, ← c85586] at e_1394_13
  have hr := e_1394_13
  simp only [e_1394_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13265)) := by
  have c85587 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85588 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85589 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85590 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85591 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85592 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_14 := arithEq_of_rows h (row := 1394) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_14
  simp only [← c85587, ← c85588, ← c85589] at e_1394_14
  have e_1394_15 := arithEq_of_rows h (row := 1394) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_15
  simp only [← c85590, ← c85591, k2, ← c85592] at e_1394_15
  have hr := e_1394_15
  simp only [e_1394_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13266)) := by
  have c85593 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85594 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85595 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85596 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85597 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85598 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_16 := arithEq_of_rows h (row := 1394) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_16
  simp only [← c85593, ← c85594, ← c85595] at e_1394_16
  have e_1394_17 := arithEq_of_rows h (row := 1394) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_17
  simp only [← c85596, ← c85597, k2, ← c85598] at e_1394_17
  have hr := e_1394_17
  simp only [e_1394_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1394 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13267)) := by
  have c85599 := (publicBatchWrapper86_copies2674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85600 := (publicBatchWrapper86_copies2675 a h).1
  have c85601 := (publicBatchWrapper86_copies2675 a h).2.1
  have c85602 := (publicBatchWrapper86_copies2675 a h).2.2.1
  have c85603 := (publicBatchWrapper86_copies2675 a h).2.2.2.1
  have c85604 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1394_18 := arithEq_of_rows h (row := 1394) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_18
  simp only [← c85599, ← c85600, ← c85601] at e_1394_18
  have e_1394_19 := arithEq_of_rows h (row := 1394) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1394_19
  simp only [← c85602, ← c85603, k2, ← c85604] at e_1394_19
  have hr := e_1394_19
  simp only [e_1394_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13268)) := by
  have c85605 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.1
  have c85606 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.1
  have c85607 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.1
  have c85608 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.1
  have c85609 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.1
  have c85610 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_0 := arithEq_of_rows h (row := 1395) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_0
  simp only [← c85605, ← c85606, ← c85607] at e_1395_0
  have e_1395_1 := arithEq_of_rows h (row := 1395) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_1
  simp only [← c85608, ← c85609, k2, ← c85610] at e_1395_1
  have hr := e_1395_1
  simp only [e_1395_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13269)) := by
  have c85611 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85612 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85613 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85614 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85615 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85616 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_2 := arithEq_of_rows h (row := 1395) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_2
  simp only [← c85611, ← c85612, ← c85613] at e_1395_2
  have e_1395_3 := arithEq_of_rows h (row := 1395) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_3
  simp only [← c85614, ← c85615, k2, ← c85616] at e_1395_3
  have hr := e_1395_3
  simp only [e_1395_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13270)) := by
  have c85617 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85618 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85619 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85620 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85621 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85622 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_4 := arithEq_of_rows h (row := 1395) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_4
  simp only [← c85617, ← c85618, ← c85619] at e_1395_4
  have e_1395_5 := arithEq_of_rows h (row := 1395) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_5
  simp only [← c85620, ← c85621, k2, ← c85622] at e_1395_5
  have hr := e_1395_5
  simp only [e_1395_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13271)) := by
  have c85623 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85624 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85625 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85626 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85627 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85628 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_6 := arithEq_of_rows h (row := 1395) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_6
  simp only [← c85623, ← c85624, ← c85625] at e_1395_6
  have e_1395_7 := arithEq_of_rows h (row := 1395) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_7
  simp only [← c85626, ← c85627, k2, ← c85628] at e_1395_7
  have hr := e_1395_7
  simp only [e_1395_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13272)) := by
  have c85629 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85630 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85631 := (publicBatchWrapper86_copies2675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85632 := (publicBatchWrapper86_copies2676 a h).1
  have c85633 := (publicBatchWrapper86_copies2676 a h).2.1
  have c85634 := (publicBatchWrapper86_copies2676 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_8 := arithEq_of_rows h (row := 1395) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_8
  simp only [← c85629, ← c85630, ← c85631] at e_1395_8
  have e_1395_9 := arithEq_of_rows h (row := 1395) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_9
  simp only [← c85632, ← c85633, k2, ← c85634] at e_1395_9
  have hr := e_1395_9
  simp only [e_1395_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13273)) := by
  have c85635 := (publicBatchWrapper86_copies2676 a h).2.2.2.1
  have c85636 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.1
  have c85637 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.1
  have c85638 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.1
  have c85639 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.1
  have c85640 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_10 := arithEq_of_rows h (row := 1395) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_10
  simp only [← c85635, ← c85636, ← c85637] at e_1395_10
  have e_1395_11 := arithEq_of_rows h (row := 1395) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_11
  simp only [← c85638, ← c85639, k2, ← c85640] at e_1395_11
  have hr := e_1395_11
  simp only [e_1395_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13274)) := by
  have c85641 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.1
  have c85642 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85643 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85644 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85645 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85646 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_12 := arithEq_of_rows h (row := 1395) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_12
  simp only [← c85641, ← c85642, ← c85643] at e_1395_12
  have e_1395_13 := arithEq_of_rows h (row := 1395) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_13
  simp only [← c85644, ← c85645, k2, ← c85646] at e_1395_13
  have hr := e_1395_13
  simp only [e_1395_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13275)) := by
  have c85647 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85648 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85649 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85650 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85651 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85652 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_14 := arithEq_of_rows h (row := 1395) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_14
  simp only [← c85647, ← c85648, ← c85649] at e_1395_14
  have e_1395_15 := arithEq_of_rows h (row := 1395) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_15
  simp only [← c85650, ← c85651, k2, ← c85652] at e_1395_15
  have hr := e_1395_15
  simp only [e_1395_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13276)) := by
  have c85653 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85654 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85655 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85656 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85657 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85658 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_16 := arithEq_of_rows h (row := 1395) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_16
  simp only [← c85653, ← c85654, ← c85655] at e_1395_16
  have e_1395_17 := arithEq_of_rows h (row := 1395) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_17
  simp only [← c85656, ← c85657, k2, ← c85658] at e_1395_17
  have hr := e_1395_17
  simp only [e_1395_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1395 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13277)) := by
  have c85659 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85660 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85661 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85662 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85663 := (publicBatchWrapper86_copies2676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85664 := (publicBatchWrapper86_copies2677 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1395_18 := arithEq_of_rows h (row := 1395) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_18
  simp only [← c85659, ← c85660, ← c85661] at e_1395_18
  have e_1395_19 := arithEq_of_rows h (row := 1395) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1395_19
  simp only [← c85662, ← c85663, k2, ← c85664] at e_1395_19
  have hr := e_1395_19
  simp only [e_1395_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13278)) := by
  have c85665 := (publicBatchWrapper86_copies2677 a h).2.1
  have c85666 := (publicBatchWrapper86_copies2677 a h).2.2.1
  have c85667 := (publicBatchWrapper86_copies2677 a h).2.2.2.1
  have c85668 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.1
  have c85669 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.1
  have c85670 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_0 := arithEq_of_rows h (row := 1396) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_0
  simp only [← c85665, ← c85666, ← c85667] at e_1396_0
  have e_1396_1 := arithEq_of_rows h (row := 1396) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_1
  simp only [← c85668, ← c85669, k2, ← c85670] at e_1396_1
  have hr := e_1396_1
  simp only [e_1396_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13279)) := by
  have c85671 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.1
  have c85672 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.1
  have c85673 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.1
  have c85674 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85675 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85676 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_2 := arithEq_of_rows h (row := 1396) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_2
  simp only [← c85671, ← c85672, ← c85673] at e_1396_2
  have e_1396_3 := arithEq_of_rows h (row := 1396) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_3
  simp only [← c85674, ← c85675, k2, ← c85676] at e_1396_3
  have hr := e_1396_3
  simp only [e_1396_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13280)) := by
  have c85677 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85678 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85679 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85680 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85681 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85682 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_4 := arithEq_of_rows h (row := 1396) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_4
  simp only [← c85677, ← c85678, ← c85679] at e_1396_4
  have e_1396_5 := arithEq_of_rows h (row := 1396) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_5
  simp only [← c85680, ← c85681, k2, ← c85682] at e_1396_5
  have hr := e_1396_5
  simp only [e_1396_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13281)) := by
  have c85683 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85684 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85685 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85686 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85687 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85688 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_6 := arithEq_of_rows h (row := 1396) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_6
  simp only [← c85683, ← c85684, ← c85685] at e_1396_6
  have e_1396_7 := arithEq_of_rows h (row := 1396) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_7
  simp only [← c85686, ← c85687, k2, ← c85688] at e_1396_7
  have hr := e_1396_7
  simp only [e_1396_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13282)) := by
  have c85689 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85690 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85691 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85692 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85693 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85694 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_8 := arithEq_of_rows h (row := 1396) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_8
  simp only [← c85689, ← c85690, ← c85691] at e_1396_8
  have e_1396_9 := arithEq_of_rows h (row := 1396) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_9
  simp only [← c85692, ← c85693, k2, ← c85694] at e_1396_9
  have hr := e_1396_9
  simp only [e_1396_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13283)) := by
  have c85695 := (publicBatchWrapper86_copies2677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85696 := (publicBatchWrapper86_copies2678 a h).1
  have c85697 := (publicBatchWrapper86_copies2678 a h).2.1
  have c85698 := (publicBatchWrapper86_copies2678 a h).2.2.1
  have c85699 := (publicBatchWrapper86_copies2678 a h).2.2.2.1
  have c85700 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_10 := arithEq_of_rows h (row := 1396) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_10
  simp only [← c85695, ← c85696, ← c85697] at e_1396_10
  have e_1396_11 := arithEq_of_rows h (row := 1396) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_11
  simp only [← c85698, ← c85699, k2, ← c85700] at e_1396_11
  have hr := e_1396_11
  simp only [e_1396_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13284)) := by
  have c85701 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.1
  have c85702 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.1
  have c85703 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.1
  have c85704 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.1
  have c85705 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.1
  have c85706 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_12 := arithEq_of_rows h (row := 1396) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_12
  simp only [← c85701, ← c85702, ← c85703] at e_1396_12
  have e_1396_13 := arithEq_of_rows h (row := 1396) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_13
  simp only [← c85704, ← c85705, k2, ← c85706] at e_1396_13
  have hr := e_1396_13
  simp only [e_1396_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13285)) := by
  have c85707 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85708 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85709 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85710 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85711 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85712 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_14 := arithEq_of_rows h (row := 1396) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_14
  simp only [← c85707, ← c85708, ← c85709] at e_1396_14
  have e_1396_15 := arithEq_of_rows h (row := 1396) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_15
  simp only [← c85710, ← c85711, k2, ← c85712] at e_1396_15
  have hr := e_1396_15
  simp only [e_1396_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13286)) := by
  have c85713 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85714 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85715 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85716 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85717 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85718 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_16 := arithEq_of_rows h (row := 1396) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_16
  simp only [← c85713, ← c85714, ← c85715] at e_1396_16
  have e_1396_17 := arithEq_of_rows h (row := 1396) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_17
  simp only [← c85716, ← c85717, k2, ← c85718] at e_1396_17
  have hr := e_1396_17
  simp only [e_1396_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1396 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13287)) := by
  have c85719 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85720 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85721 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85722 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85723 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85724 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1396_18 := arithEq_of_rows h (row := 1396) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_18
  simp only [← c85719, ← c85720, ← c85721] at e_1396_18
  have e_1396_19 := arithEq_of_rows h (row := 1396) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1396_19
  simp only [← c85722, ← c85723, k2, ← c85724] at e_1396_19
  have hr := e_1396_19
  simp only [e_1396_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13288)) := by
  have c85725 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85726 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85727 := (publicBatchWrapper86_copies2678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85728 := (publicBatchWrapper86_copies2679 a h).1
  have c85729 := (publicBatchWrapper86_copies2679 a h).2.1
  have c85730 := (publicBatchWrapper86_copies2679 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_0 := arithEq_of_rows h (row := 1397) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_0
  simp only [← c85725, ← c85726, ← c85727] at e_1397_0
  have e_1397_1 := arithEq_of_rows h (row := 1397) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_1
  simp only [← c85728, ← c85729, k2, ← c85730] at e_1397_1
  have hr := e_1397_1
  simp only [e_1397_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13289)) := by
  have c85731 := (publicBatchWrapper86_copies2679 a h).2.2.2.1
  have c85732 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.1
  have c85733 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.1
  have c85734 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.1
  have c85735 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.1
  have c85736 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_2 := arithEq_of_rows h (row := 1397) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_2
  simp only [← c85731, ← c85732, ← c85733] at e_1397_2
  have e_1397_3 := arithEq_of_rows h (row := 1397) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_3
  simp only [← c85734, ← c85735, k2, ← c85736] at e_1397_3
  have hr := e_1397_3
  simp only [e_1397_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13290)) := by
  have c85737 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.1
  have c85738 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85739 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85740 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85741 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85742 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_4 := arithEq_of_rows h (row := 1397) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_4
  simp only [← c85737, ← c85738, ← c85739] at e_1397_4
  have e_1397_5 := arithEq_of_rows h (row := 1397) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_5
  simp only [← c85740, ← c85741, k2, ← c85742] at e_1397_5
  have hr := e_1397_5
  simp only [e_1397_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13291)) := by
  have c85743 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85744 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85745 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85746 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85747 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85748 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_6 := arithEq_of_rows h (row := 1397) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_6
  simp only [← c85743, ← c85744, ← c85745] at e_1397_6
  have e_1397_7 := arithEq_of_rows h (row := 1397) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_7
  simp only [← c85746, ← c85747, k2, ← c85748] at e_1397_7
  have hr := e_1397_7
  simp only [e_1397_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13292)) := by
  have c85749 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85750 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85751 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85752 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85753 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85754 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_8 := arithEq_of_rows h (row := 1397) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_8
  simp only [← c85749, ← c85750, ← c85751] at e_1397_8
  have e_1397_9 := arithEq_of_rows h (row := 1397) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_9
  simp only [← c85752, ← c85753, k2, ← c85754] at e_1397_9
  have hr := e_1397_9
  simp only [e_1397_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13293)) := by
  have c85755 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85756 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85757 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85758 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85759 := (publicBatchWrapper86_copies2679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85760 := (publicBatchWrapper86_copies2680 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_10 := arithEq_of_rows h (row := 1397) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_10
  simp only [← c85755, ← c85756, ← c85757] at e_1397_10
  have e_1397_11 := arithEq_of_rows h (row := 1397) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_11
  simp only [← c85758, ← c85759, k2, ← c85760] at e_1397_11
  have hr := e_1397_11
  simp only [e_1397_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13294)) := by
  have c85761 := (publicBatchWrapper86_copies2680 a h).2.1
  have c85762 := (publicBatchWrapper86_copies2680 a h).2.2.1
  have c85763 := (publicBatchWrapper86_copies2680 a h).2.2.2.1
  have c85764 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.1
  have c85765 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.1
  have c85766 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_12 := arithEq_of_rows h (row := 1397) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_12
  simp only [← c85761, ← c85762, ← c85763] at e_1397_12
  have e_1397_13 := arithEq_of_rows h (row := 1397) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_13
  simp only [← c85764, ← c85765, k2, ← c85766] at e_1397_13
  have hr := e_1397_13
  simp only [e_1397_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13295)) := by
  have c85767 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.1
  have c85768 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.1
  have c85769 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.1
  have c85770 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85771 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85772 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_14 := arithEq_of_rows h (row := 1397) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_14
  simp only [← c85767, ← c85768, ← c85769] at e_1397_14
  have e_1397_15 := arithEq_of_rows h (row := 1397) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_15
  simp only [← c85770, ← c85771, k2, ← c85772] at e_1397_15
  have hr := e_1397_15
  simp only [e_1397_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13296)) := by
  have c85773 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85774 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85775 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85776 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85777 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85778 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_16 := arithEq_of_rows h (row := 1397) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_16
  simp only [← c85773, ← c85774, ← c85775] at e_1397_16
  have e_1397_17 := arithEq_of_rows h (row := 1397) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_17
  simp only [← c85776, ← c85777, k2, ← c85778] at e_1397_17
  have hr := e_1397_17
  simp only [e_1397_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1397 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13297)) := by
  have c85779 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85780 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85781 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85782 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85783 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85784 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1397_18 := arithEq_of_rows h (row := 1397) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_18
  simp only [← c85779, ← c85780, ← c85781] at e_1397_18
  have e_1397_19 := arithEq_of_rows h (row := 1397) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1397_19
  simp only [← c85782, ← c85783, k2, ← c85784] at e_1397_19
  have hr := e_1397_19
  simp only [e_1397_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13298)) := by
  have c85785 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85786 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85787 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85788 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85789 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85790 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_0 := arithEq_of_rows h (row := 1398) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_0
  simp only [← c85785, ← c85786, ← c85787] at e_1398_0
  have e_1398_1 := arithEq_of_rows h (row := 1398) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_1
  simp only [← c85788, ← c85789, k2, ← c85790] at e_1398_1
  have hr := e_1398_1
  simp only [e_1398_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13299)) := by
  have c85791 := (publicBatchWrapper86_copies2680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85792 := (publicBatchWrapper86_copies2681 a h).1
  have c85793 := (publicBatchWrapper86_copies2681 a h).2.1
  have c85794 := (publicBatchWrapper86_copies2681 a h).2.2.1
  have c85795 := (publicBatchWrapper86_copies2681 a h).2.2.2.1
  have c85796 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_2 := arithEq_of_rows h (row := 1398) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_2
  simp only [← c85791, ← c85792, ← c85793] at e_1398_2
  have e_1398_3 := arithEq_of_rows h (row := 1398) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_3
  simp only [← c85794, ← c85795, k2, ← c85796] at e_1398_3
  have hr := e_1398_3
  simp only [e_1398_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13300)) := by
  have c85797 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.1
  have c85798 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.1
  have c85799 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.1
  have c85800 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.1
  have c85801 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.1
  have c85802 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_4 := arithEq_of_rows h (row := 1398) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_4
  simp only [← c85797, ← c85798, ← c85799] at e_1398_4
  have e_1398_5 := arithEq_of_rows h (row := 1398) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_5
  simp only [← c85800, ← c85801, k2, ← c85802] at e_1398_5
  have hr := e_1398_5
  simp only [e_1398_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13301)) := by
  have c85803 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85804 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85805 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85806 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85807 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85808 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_6 := arithEq_of_rows h (row := 1398) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_6
  simp only [← c85803, ← c85804, ← c85805] at e_1398_6
  have e_1398_7 := arithEq_of_rows h (row := 1398) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_7
  simp only [← c85806, ← c85807, k2, ← c85808] at e_1398_7
  have hr := e_1398_7
  simp only [e_1398_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13302)) := by
  have c85809 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85810 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85811 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85812 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85813 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85814 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_8 := arithEq_of_rows h (row := 1398) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_8
  simp only [← c85809, ← c85810, ← c85811] at e_1398_8
  have e_1398_9 := arithEq_of_rows h (row := 1398) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_9
  simp only [← c85812, ← c85813, k2, ← c85814] at e_1398_9
  have hr := e_1398_9
  simp only [e_1398_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13303)) := by
  have c85815 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85816 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85817 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85818 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85819 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85820 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_10 := arithEq_of_rows h (row := 1398) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_10
  simp only [← c85815, ← c85816, ← c85817] at e_1398_10
  have e_1398_11 := arithEq_of_rows h (row := 1398) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_11
  simp only [← c85818, ← c85819, k2, ← c85820] at e_1398_11
  have hr := e_1398_11
  simp only [e_1398_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13304)) := by
  have c85821 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85822 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85823 := (publicBatchWrapper86_copies2681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85824 := (publicBatchWrapper86_copies2682 a h).1
  have c85825 := (publicBatchWrapper86_copies2682 a h).2.1
  have c85826 := (publicBatchWrapper86_copies2682 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_12 := arithEq_of_rows h (row := 1398) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_12
  simp only [← c85821, ← c85822, ← c85823] at e_1398_12
  have e_1398_13 := arithEq_of_rows h (row := 1398) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_13
  simp only [← c85824, ← c85825, k2, ← c85826] at e_1398_13
  have hr := e_1398_13
  simp only [e_1398_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13305)) := by
  have c85827 := (publicBatchWrapper86_copies2682 a h).2.2.2.1
  have c85828 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.1
  have c85829 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.1
  have c85830 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.1
  have c85831 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.1
  have c85832 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_14 := arithEq_of_rows h (row := 1398) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_14
  simp only [← c85827, ← c85828, ← c85829] at e_1398_14
  have e_1398_15 := arithEq_of_rows h (row := 1398) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_15
  simp only [← c85830, ← c85831, k2, ← c85832] at e_1398_15
  have hr := e_1398_15
  simp only [e_1398_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13306)) := by
  have c85833 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.1
  have c85834 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85835 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85836 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85837 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85838 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_16 := arithEq_of_rows h (row := 1398) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_16
  simp only [← c85833, ← c85834, ← c85835] at e_1398_16
  have e_1398_17 := arithEq_of_rows h (row := 1398) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_17
  simp only [← c85836, ← c85837, k2, ← c85838] at e_1398_17
  have hr := e_1398_17
  simp only [e_1398_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1398 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13307)) := by
  have c85839 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85840 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85841 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85842 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85843 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85844 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1398_18 := arithEq_of_rows h (row := 1398) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_18
  simp only [← c85839, ← c85840, ← c85841] at e_1398_18
  have e_1398_19 := arithEq_of_rows h (row := 1398) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1398_19
  simp only [← c85842, ← c85843, k2, ← c85844] at e_1398_19
  have hr := e_1398_19
  simp only [e_1398_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13308)) := by
  have c85845 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85846 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85847 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85848 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85849 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85850 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_0 := arithEq_of_rows h (row := 1399) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_0
  simp only [← c85845, ← c85846, ← c85847] at e_1399_0
  have e_1399_1 := arithEq_of_rows h (row := 1399) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_1
  simp only [← c85848, ← c85849, k2, ← c85850] at e_1399_1
  have hr := e_1399_1
  simp only [e_1399_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13309)) := by
  have c85851 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85852 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85853 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85854 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85855 := (publicBatchWrapper86_copies2682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85856 := (publicBatchWrapper86_copies2683 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_2 := arithEq_of_rows h (row := 1399) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_2
  simp only [← c85851, ← c85852, ← c85853] at e_1399_2
  have e_1399_3 := arithEq_of_rows h (row := 1399) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_3
  simp only [← c85854, ← c85855, k2, ← c85856] at e_1399_3
  have hr := e_1399_3
  simp only [e_1399_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13310)) := by
  have c85857 := (publicBatchWrapper86_copies2683 a h).2.1
  have c85858 := (publicBatchWrapper86_copies2683 a h).2.2.1
  have c85859 := (publicBatchWrapper86_copies2683 a h).2.2.2.1
  have c85860 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.1
  have c85861 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.1
  have c85862 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_4 := arithEq_of_rows h (row := 1399) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_4
  simp only [← c85857, ← c85858, ← c85859] at e_1399_4
  have e_1399_5 := arithEq_of_rows h (row := 1399) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_5
  simp only [← c85860, ← c85861, k2, ← c85862] at e_1399_5
  have hr := e_1399_5
  simp only [e_1399_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13311)) := by
  have c85863 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.1
  have c85864 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.1
  have c85865 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.1
  have c85866 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85867 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85868 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_6 := arithEq_of_rows h (row := 1399) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_6
  simp only [← c85863, ← c85864, ← c85865] at e_1399_6
  have e_1399_7 := arithEq_of_rows h (row := 1399) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_7
  simp only [← c85866, ← c85867, k2, ← c85868] at e_1399_7
  have hr := e_1399_7
  simp only [e_1399_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13312)) := by
  have c85869 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85870 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85871 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85872 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85873 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85874 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_8 := arithEq_of_rows h (row := 1399) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_8
  simp only [← c85869, ← c85870, ← c85871] at e_1399_8
  have e_1399_9 := arithEq_of_rows h (row := 1399) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_9
  simp only [← c85872, ← c85873, k2, ← c85874] at e_1399_9
  have hr := e_1399_9
  simp only [e_1399_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13313)) := by
  have c85875 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85876 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85877 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85878 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85879 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85880 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_10 := arithEq_of_rows h (row := 1399) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_10
  simp only [← c85875, ← c85876, ← c85877] at e_1399_10
  have e_1399_11 := arithEq_of_rows h (row := 1399) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_11
  simp only [← c85878, ← c85879, k2, ← c85880] at e_1399_11
  have hr := e_1399_11
  simp only [e_1399_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13314)) := by
  have c85881 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85882 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85883 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85884 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85885 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85886 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_12 := arithEq_of_rows h (row := 1399) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_12
  simp only [← c85881, ← c85882, ← c85883] at e_1399_12
  have e_1399_13 := arithEq_of_rows h (row := 1399) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_13
  simp only [← c85884, ← c85885, k2, ← c85886] at e_1399_13
  have hr := e_1399_13
  simp only [e_1399_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13315)) := by
  have c85887 := (publicBatchWrapper86_copies2683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85888 := (publicBatchWrapper86_copies2684 a h).1
  have c85889 := (publicBatchWrapper86_copies2684 a h).2.1
  have c85890 := (publicBatchWrapper86_copies2684 a h).2.2.1
  have c85891 := (publicBatchWrapper86_copies2684 a h).2.2.2.1
  have c85892 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_14 := arithEq_of_rows h (row := 1399) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_14
  simp only [← c85887, ← c85888, ← c85889] at e_1399_14
  have e_1399_15 := arithEq_of_rows h (row := 1399) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_15
  simp only [← c85890, ← c85891, k2, ← c85892] at e_1399_15
  have hr := e_1399_15
  simp only [e_1399_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13316)) := by
  have c85893 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.1
  have c85894 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.1
  have c85895 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.1
  have c85896 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.1
  have c85897 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.1
  have c85898 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_16 := arithEq_of_rows h (row := 1399) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_16
  simp only [← c85893, ← c85894, ← c85895] at e_1399_16
  have e_1399_17 := arithEq_of_rows h (row := 1399) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_17
  simp only [← c85896, ← c85897, k2, ← c85898] at e_1399_17
  have hr := e_1399_17
  simp only [e_1399_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1399 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13317)) := by
  have c85899 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85900 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85901 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85902 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85903 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85904 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1399_18 := arithEq_of_rows h (row := 1399) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_18
  simp only [← c85899, ← c85900, ← c85901] at e_1399_18
  have e_1399_19 := arithEq_of_rows h (row := 1399) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1399_19
  simp only [← c85902, ← c85903, k2, ← c85904] at e_1399_19
  have hr := e_1399_19
  simp only [e_1399_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13318)) := by
  have c85905 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85906 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85907 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85908 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85909 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85910 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_0 := arithEq_of_rows h (row := 1400) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_0
  simp only [← c85905, ← c85906, ← c85907] at e_1400_0
  have e_1400_1 := arithEq_of_rows h (row := 1400) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_1
  simp only [← c85908, ← c85909, k2, ← c85910] at e_1400_1
  have hr := e_1400_1
  simp only [e_1400_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13319)) := by
  have c85911 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85912 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85913 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85914 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85915 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85916 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_2 := arithEq_of_rows h (row := 1400) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_2
  simp only [← c85911, ← c85912, ← c85913] at e_1400_2
  have e_1400_3 := arithEq_of_rows h (row := 1400) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_3
  simp only [← c85914, ← c85915, k2, ← c85916] at e_1400_3
  have hr := e_1400_3
  simp only [e_1400_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13320)) := by
  have c85917 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85918 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85919 := (publicBatchWrapper86_copies2684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85920 := (publicBatchWrapper86_copies2685 a h).1
  have c85921 := (publicBatchWrapper86_copies2685 a h).2.1
  have c85922 := (publicBatchWrapper86_copies2685 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_4 := arithEq_of_rows h (row := 1400) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_4
  simp only [← c85917, ← c85918, ← c85919] at e_1400_4
  have e_1400_5 := arithEq_of_rows h (row := 1400) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_5
  simp only [← c85920, ← c85921, k2, ← c85922] at e_1400_5
  have hr := e_1400_5
  simp only [e_1400_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13321)) := by
  have c85923 := (publicBatchWrapper86_copies2685 a h).2.2.2.1
  have c85924 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.1
  have c85925 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.1
  have c85926 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.1
  have c85927 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.1
  have c85928 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_6 := arithEq_of_rows h (row := 1400) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_6
  simp only [← c85923, ← c85924, ← c85925] at e_1400_6
  have e_1400_7 := arithEq_of_rows h (row := 1400) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_7
  simp only [← c85926, ← c85927, k2, ← c85928] at e_1400_7
  have hr := e_1400_7
  simp only [e_1400_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13322)) := by
  have c85929 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.1
  have c85930 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85931 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85932 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85933 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85934 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_8 := arithEq_of_rows h (row := 1400) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_8
  simp only [← c85929, ← c85930, ← c85931] at e_1400_8
  have e_1400_9 := arithEq_of_rows h (row := 1400) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_9
  simp only [← c85932, ← c85933, k2, ← c85934] at e_1400_9
  have hr := e_1400_9
  simp only [e_1400_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13323)) := by
  have c85935 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85936 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85937 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85938 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85939 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85940 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_10 := arithEq_of_rows h (row := 1400) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_10
  simp only [← c85935, ← c85936, ← c85937] at e_1400_10
  have e_1400_11 := arithEq_of_rows h (row := 1400) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_11
  simp only [← c85938, ← c85939, k2, ← c85940] at e_1400_11
  have hr := e_1400_11
  simp only [e_1400_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13324)) := by
  have c85941 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85942 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85943 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85944 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85945 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85946 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_12 := arithEq_of_rows h (row := 1400) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_12
  simp only [← c85941, ← c85942, ← c85943] at e_1400_12
  have e_1400_13 := arithEq_of_rows h (row := 1400) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_13
  simp only [← c85944, ← c85945, k2, ← c85946] at e_1400_13
  have hr := e_1400_13
  simp only [e_1400_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13325)) := by
  have c85947 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85948 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85949 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85950 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85951 := (publicBatchWrapper86_copies2685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85952 := (publicBatchWrapper86_copies2686 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_14 := arithEq_of_rows h (row := 1400) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_14
  simp only [← c85947, ← c85948, ← c85949] at e_1400_14
  have e_1400_15 := arithEq_of_rows h (row := 1400) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_15
  simp only [← c85950, ← c85951, k2, ← c85952] at e_1400_15
  have hr := e_1400_15
  simp only [e_1400_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13326)) := by
  have c85953 := (publicBatchWrapper86_copies2686 a h).2.1
  have c85954 := (publicBatchWrapper86_copies2686 a h).2.2.1
  have c85955 := (publicBatchWrapper86_copies2686 a h).2.2.2.1
  have c85956 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.1
  have c85957 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.1
  have c85958 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_16 := arithEq_of_rows h (row := 1400) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_16
  simp only [← c85953, ← c85954, ← c85955] at e_1400_16
  have e_1400_17 := arithEq_of_rows h (row := 1400) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_17
  simp only [← c85956, ← c85957, k2, ← c85958] at e_1400_17
  have hr := e_1400_17
  simp only [e_1400_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1400 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13327)) := by
  have c85959 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.1
  have c85960 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.1
  have c85961 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.1
  have c85962 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c85963 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85964 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1400_18 := arithEq_of_rows h (row := 1400) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_18
  simp only [← c85959, ← c85960, ← c85961] at e_1400_18
  have e_1400_19 := arithEq_of_rows h (row := 1400) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1400_19
  simp only [← c85962, ← c85963, k2, ← c85964] at e_1400_19
  have hr := e_1400_19
  simp only [e_1400_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13328)) := by
  have c85965 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85966 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85967 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85968 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85969 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85970 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_0 := arithEq_of_rows h (row := 1401) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_0
  simp only [← c85965, ← c85966, ← c85967] at e_1401_0
  have e_1401_1 := arithEq_of_rows h (row := 1401) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_1
  simp only [← c85968, ← c85969, k2, ← c85970] at e_1401_1
  have hr := e_1401_1
  simp only [e_1401_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13329)) := by
  have c85971 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85972 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85973 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85974 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85975 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85976 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_2 := arithEq_of_rows h (row := 1401) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_2
  simp only [← c85971, ← c85972, ← c85973] at e_1401_2
  have e_1401_3 := arithEq_of_rows h (row := 1401) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_3
  simp only [← c85974, ← c85975, k2, ← c85976] at e_1401_3
  have hr := e_1401_3
  simp only [e_1401_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13330)) := by
  have c85977 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85978 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85979 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85980 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85981 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85982 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_4 := arithEq_of_rows h (row := 1401) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_4
  simp only [← c85977, ← c85978, ← c85979] at e_1401_4
  have e_1401_5 := arithEq_of_rows h (row := 1401) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_5
  simp only [← c85980, ← c85981, k2, ← c85982] at e_1401_5
  have hr := e_1401_5
  simp only [e_1401_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13331)) := by
  have c85983 := (publicBatchWrapper86_copies2686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c85984 := (publicBatchWrapper86_copies2687 a h).1
  have c85985 := (publicBatchWrapper86_copies2687 a h).2.1
  have c85986 := (publicBatchWrapper86_copies2687 a h).2.2.1
  have c85987 := (publicBatchWrapper86_copies2687 a h).2.2.2.1
  have c85988 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_6 := arithEq_of_rows h (row := 1401) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_6
  simp only [← c85983, ← c85984, ← c85985] at e_1401_6
  have e_1401_7 := arithEq_of_rows h (row := 1401) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_7
  simp only [← c85986, ← c85987, k2, ← c85988] at e_1401_7
  have hr := e_1401_7
  simp only [e_1401_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13332)) := by
  have c85989 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.1
  have c85990 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.1
  have c85991 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.1
  have c85992 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.1
  have c85993 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.1
  have c85994 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_8 := arithEq_of_rows h (row := 1401) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_8
  simp only [← c85989, ← c85990, ← c85991] at e_1401_8
  have e_1401_9 := arithEq_of_rows h (row := 1401) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_9
  simp only [← c85992, ← c85993, k2, ← c85994] at e_1401_9
  have hr := e_1401_9
  simp only [e_1401_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13333)) := by
  have c85995 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c85996 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85997 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85998 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85999 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86000 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_10 := arithEq_of_rows h (row := 1401) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_10
  simp only [← c85995, ← c85996, ← c85997] at e_1401_10
  have e_1401_11 := arithEq_of_rows h (row := 1401) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_11
  simp only [← c85998, ← c85999, k2, ← c86000] at e_1401_11
  have hr := e_1401_11
  simp only [e_1401_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13334)) := by
  have c86001 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86002 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86003 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86004 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86005 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86006 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_12 := arithEq_of_rows h (row := 1401) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_12
  simp only [← c86001, ← c86002, ← c86003] at e_1401_12
  have e_1401_13 := arithEq_of_rows h (row := 1401) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_13
  simp only [← c86004, ← c86005, k2, ← c86006] at e_1401_13
  have hr := e_1401_13
  simp only [e_1401_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13335)) := by
  have c86007 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86008 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86009 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86010 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86011 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86012 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_14 := arithEq_of_rows h (row := 1401) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_14
  simp only [← c86007, ← c86008, ← c86009] at e_1401_14
  have e_1401_15 := arithEq_of_rows h (row := 1401) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_15
  simp only [← c86010, ← c86011, k2, ← c86012] at e_1401_15
  have hr := e_1401_15
  simp only [e_1401_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13336)) := by
  have c86013 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86014 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86015 := (publicBatchWrapper86_copies2687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86016 := (publicBatchWrapper86_copies2688 a h).1
  have c86017 := (publicBatchWrapper86_copies2688 a h).2.1
  have c86018 := (publicBatchWrapper86_copies2688 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_16 := arithEq_of_rows h (row := 1401) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_16
  simp only [← c86013, ← c86014, ← c86015] at e_1401_16
  have e_1401_17 := arithEq_of_rows h (row := 1401) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_17
  simp only [← c86016, ← c86017, k2, ← c86018] at e_1401_17
  have hr := e_1401_17
  simp only [e_1401_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1401 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13337)) := by
  have c86019 := (publicBatchWrapper86_copies2688 a h).2.2.2.1
  have c86020 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.1
  have c86021 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.1
  have c86022 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.1
  have c86023 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.1
  have c86024 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1401_18 := arithEq_of_rows h (row := 1401) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_18
  simp only [← c86019, ← c86020, ← c86021] at e_1401_18
  have e_1401_19 := arithEq_of_rows h (row := 1401) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1401_19
  simp only [← c86022, ← c86023, k2, ← c86024] at e_1401_19
  have hr := e_1401_19
  simp only [e_1401_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13338)) := by
  have c86025 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.1
  have c86026 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86027 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86028 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86029 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86030 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_0 := arithEq_of_rows h (row := 1402) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_0
  simp only [← c86025, ← c86026, ← c86027] at e_1402_0
  have e_1402_1 := arithEq_of_rows h (row := 1402) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_1
  simp only [← c86028, ← c86029, k2, ← c86030] at e_1402_1
  have hr := e_1402_1
  simp only [e_1402_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13339)) := by
  have c86031 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86032 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86033 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86034 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86035 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86036 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_2 := arithEq_of_rows h (row := 1402) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_2
  simp only [← c86031, ← c86032, ← c86033] at e_1402_2
  have e_1402_3 := arithEq_of_rows h (row := 1402) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_3
  simp only [← c86034, ← c86035, k2, ← c86036] at e_1402_3
  have hr := e_1402_3
  simp only [e_1402_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13340)) := by
  have c86037 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86038 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86039 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86040 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86041 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86042 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_4 := arithEq_of_rows h (row := 1402) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_4
  simp only [← c86037, ← c86038, ← c86039] at e_1402_4
  have e_1402_5 := arithEq_of_rows h (row := 1402) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_5
  simp only [← c86040, ← c86041, k2, ← c86042] at e_1402_5
  have hr := e_1402_5
  simp only [e_1402_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13341)) := by
  have c86043 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86044 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86045 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86046 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86047 := (publicBatchWrapper86_copies2688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86048 := (publicBatchWrapper86_copies2689 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_6 := arithEq_of_rows h (row := 1402) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_6
  simp only [← c86043, ← c86044, ← c86045] at e_1402_6
  have e_1402_7 := arithEq_of_rows h (row := 1402) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_7
  simp only [← c86046, ← c86047, k2, ← c86048] at e_1402_7
  have hr := e_1402_7
  simp only [e_1402_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13342)) := by
  have c86049 := (publicBatchWrapper86_copies2689 a h).2.1
  have c86050 := (publicBatchWrapper86_copies2689 a h).2.2.1
  have c86051 := (publicBatchWrapper86_copies2689 a h).2.2.2.1
  have c86052 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.1
  have c86053 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.1
  have c86054 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_8 := arithEq_of_rows h (row := 1402) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_8
  simp only [← c86049, ← c86050, ← c86051] at e_1402_8
  have e_1402_9 := arithEq_of_rows h (row := 1402) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_9
  simp only [← c86052, ← c86053, k2, ← c86054] at e_1402_9
  have hr := e_1402_9
  simp only [e_1402_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13343)) := by
  have c86055 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.1
  have c86056 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.1
  have c86057 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.1
  have c86058 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86059 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86060 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_10 := arithEq_of_rows h (row := 1402) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_10
  simp only [← c86055, ← c86056, ← c86057] at e_1402_10
  have e_1402_11 := arithEq_of_rows h (row := 1402) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_11
  simp only [← c86058, ← c86059, k2, ← c86060] at e_1402_11
  have hr := e_1402_11
  simp only [e_1402_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13344)) := by
  have c86061 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86062 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86063 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86064 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86065 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86066 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_12 := arithEq_of_rows h (row := 1402) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_12
  simp only [← c86061, ← c86062, ← c86063] at e_1402_12
  have e_1402_13 := arithEq_of_rows h (row := 1402) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_13
  simp only [← c86064, ← c86065, k2, ← c86066] at e_1402_13
  have hr := e_1402_13
  simp only [e_1402_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13345)) := by
  have c86067 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86068 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86069 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86070 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86071 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86072 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_14 := arithEq_of_rows h (row := 1402) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_14
  simp only [← c86067, ← c86068, ← c86069] at e_1402_14
  have e_1402_15 := arithEq_of_rows h (row := 1402) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_15
  simp only [← c86070, ← c86071, k2, ← c86072] at e_1402_15
  have hr := e_1402_15
  simp only [e_1402_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13346)) := by
  have c86073 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86074 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86075 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86076 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86077 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86078 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_16 := arithEq_of_rows h (row := 1402) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_16
  simp only [← c86073, ← c86074, ← c86075] at e_1402_16
  have e_1402_17 := arithEq_of_rows h (row := 1402) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_17
  simp only [← c86076, ← c86077, k2, ← c86078] at e_1402_17
  have hr := e_1402_17
  simp only [e_1402_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1402 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13347)) := by
  have c86079 := (publicBatchWrapper86_copies2689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86080 := (publicBatchWrapper86_copies2690 a h).1
  have c86081 := (publicBatchWrapper86_copies2690 a h).2.1
  have c86082 := (publicBatchWrapper86_copies2690 a h).2.2.1
  have c86083 := (publicBatchWrapper86_copies2690 a h).2.2.2.1
  have c86084 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1402_18 := arithEq_of_rows h (row := 1402) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_18
  simp only [← c86079, ← c86080, ← c86081] at e_1402_18
  have e_1402_19 := arithEq_of_rows h (row := 1402) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1402_19
  simp only [← c86082, ← c86083, k2, ← c86084] at e_1402_19
  have hr := e_1402_19
  simp only [e_1402_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13348)) := by
  have c86085 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.1
  have c86086 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.1
  have c86087 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.1
  have c86088 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.1
  have c86089 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.1
  have c86090 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_0 := arithEq_of_rows h (row := 1403) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_0
  simp only [← c86085, ← c86086, ← c86087] at e_1403_0
  have e_1403_1 := arithEq_of_rows h (row := 1403) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_1
  simp only [← c86088, ← c86089, k2, ← c86090] at e_1403_1
  have hr := e_1403_1
  simp only [e_1403_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13349)) := by
  have c86091 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86092 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86093 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86094 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86095 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86096 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_2 := arithEq_of_rows h (row := 1403) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_2
  simp only [← c86091, ← c86092, ← c86093] at e_1403_2
  have e_1403_3 := arithEq_of_rows h (row := 1403) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_3
  simp only [← c86094, ← c86095, k2, ← c86096] at e_1403_3
  have hr := e_1403_3
  simp only [e_1403_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13350)) := by
  have c86097 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86098 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86099 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86100 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86101 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86102 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_4 := arithEq_of_rows h (row := 1403) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_4
  simp only [← c86097, ← c86098, ← c86099] at e_1403_4
  have e_1403_5 := arithEq_of_rows h (row := 1403) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_5
  simp only [← c86100, ← c86101, k2, ← c86102] at e_1403_5
  have hr := e_1403_5
  simp only [e_1403_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13351)) := by
  have c86103 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86104 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86105 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86106 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86107 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86108 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_6 := arithEq_of_rows h (row := 1403) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_6
  simp only [← c86103, ← c86104, ← c86105] at e_1403_6
  have e_1403_7 := arithEq_of_rows h (row := 1403) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_7
  simp only [← c86106, ← c86107, k2, ← c86108] at e_1403_7
  have hr := e_1403_7
  simp only [e_1403_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13352)) := by
  have c86109 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86110 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86111 := (publicBatchWrapper86_copies2690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86112 := (publicBatchWrapper86_copies2691 a h).1
  have c86113 := (publicBatchWrapper86_copies2691 a h).2.1
  have c86114 := (publicBatchWrapper86_copies2691 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_8 := arithEq_of_rows h (row := 1403) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_8
  simp only [← c86109, ← c86110, ← c86111] at e_1403_8
  have e_1403_9 := arithEq_of_rows h (row := 1403) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_9
  simp only [← c86112, ← c86113, k2, ← c86114] at e_1403_9
  have hr := e_1403_9
  simp only [e_1403_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13353)) := by
  have c86115 := (publicBatchWrapper86_copies2691 a h).2.2.2.1
  have c86116 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.1
  have c86117 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.1
  have c86118 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.1
  have c86119 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.1
  have c86120 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_10 := arithEq_of_rows h (row := 1403) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_10
  simp only [← c86115, ← c86116, ← c86117] at e_1403_10
  have e_1403_11 := arithEq_of_rows h (row := 1403) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_11
  simp only [← c86118, ← c86119, k2, ← c86120] at e_1403_11
  have hr := e_1403_11
  simp only [e_1403_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13354)) := by
  have c86121 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.1
  have c86122 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86123 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86124 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86125 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86126 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_12 := arithEq_of_rows h (row := 1403) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_12
  simp only [← c86121, ← c86122, ← c86123] at e_1403_12
  have e_1403_13 := arithEq_of_rows h (row := 1403) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_13
  simp only [← c86124, ← c86125, k2, ← c86126] at e_1403_13
  have hr := e_1403_13
  simp only [e_1403_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13355)) := by
  have c86127 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86128 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86129 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86130 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86131 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86132 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_14 := arithEq_of_rows h (row := 1403) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_14
  simp only [← c86127, ← c86128, ← c86129] at e_1403_14
  have e_1403_15 := arithEq_of_rows h (row := 1403) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_15
  simp only [← c86130, ← c86131, k2, ← c86132] at e_1403_15
  have hr := e_1403_15
  simp only [e_1403_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13356)) := by
  have c86133 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86134 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86135 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86136 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86137 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86138 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_16 := arithEq_of_rows h (row := 1403) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_16
  simp only [← c86133, ← c86134, ← c86135] at e_1403_16
  have e_1403_17 := arithEq_of_rows h (row := 1403) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_17
  simp only [← c86136, ← c86137, k2, ← c86138] at e_1403_17
  have hr := e_1403_17
  simp only [e_1403_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1403 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13357)) := by
  have c86139 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86140 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86141 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86142 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86143 := (publicBatchWrapper86_copies2691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86144 := (publicBatchWrapper86_copies2692 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1403_18 := arithEq_of_rows h (row := 1403) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_18
  simp only [← c86139, ← c86140, ← c86141] at e_1403_18
  have e_1403_19 := arithEq_of_rows h (row := 1403) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1403_19
  simp only [← c86142, ← c86143, k2, ← c86144] at e_1403_19
  have hr := e_1403_19
  simp only [e_1403_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13358)) := by
  have c86145 := (publicBatchWrapper86_copies2692 a h).2.1
  have c86146 := (publicBatchWrapper86_copies2692 a h).2.2.1
  have c86147 := (publicBatchWrapper86_copies2692 a h).2.2.2.1
  have c86148 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.1
  have c86149 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.1
  have c86150 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_0 := arithEq_of_rows h (row := 1404) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_0
  simp only [← c86145, ← c86146, ← c86147] at e_1404_0
  have e_1404_1 := arithEq_of_rows h (row := 1404) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_1
  simp only [← c86148, ← c86149, k2, ← c86150] at e_1404_1
  have hr := e_1404_1
  simp only [e_1404_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13359)) := by
  have c86151 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.1
  have c86152 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.1
  have c86153 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.1
  have c86154 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86155 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86156 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_2 := arithEq_of_rows h (row := 1404) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_2
  simp only [← c86151, ← c86152, ← c86153] at e_1404_2
  have e_1404_3 := arithEq_of_rows h (row := 1404) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_3
  simp only [← c86154, ← c86155, k2, ← c86156] at e_1404_3
  have hr := e_1404_3
  simp only [e_1404_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13360)) := by
  have c86157 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86158 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86159 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86160 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86161 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86162 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_4 := arithEq_of_rows h (row := 1404) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_4
  simp only [← c86157, ← c86158, ← c86159] at e_1404_4
  have e_1404_5 := arithEq_of_rows h (row := 1404) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_5
  simp only [← c86160, ← c86161, k2, ← c86162] at e_1404_5
  have hr := e_1404_5
  simp only [e_1404_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13361)) := by
  have c86163 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86164 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86165 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86166 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86167 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86168 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_6 := arithEq_of_rows h (row := 1404) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_6
  simp only [← c86163, ← c86164, ← c86165] at e_1404_6
  have e_1404_7 := arithEq_of_rows h (row := 1404) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_7
  simp only [← c86166, ← c86167, k2, ← c86168] at e_1404_7
  have hr := e_1404_7
  simp only [e_1404_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13362)) := by
  have c86169 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86170 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86171 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86172 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86173 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86174 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_8 := arithEq_of_rows h (row := 1404) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_8
  simp only [← c86169, ← c86170, ← c86171] at e_1404_8
  have e_1404_9 := arithEq_of_rows h (row := 1404) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_9
  simp only [← c86172, ← c86173, k2, ← c86174] at e_1404_9
  have hr := e_1404_9
  simp only [e_1404_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13363)) := by
  have c86175 := (publicBatchWrapper86_copies2692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86176 := (publicBatchWrapper86_copies2693 a h).1
  have c86177 := (publicBatchWrapper86_copies2693 a h).2.1
  have c86178 := (publicBatchWrapper86_copies2693 a h).2.2.1
  have c86179 := (publicBatchWrapper86_copies2693 a h).2.2.2.1
  have c86180 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_10 := arithEq_of_rows h (row := 1404) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_10
  simp only [← c86175, ← c86176, ← c86177] at e_1404_10
  have e_1404_11 := arithEq_of_rows h (row := 1404) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_11
  simp only [← c86178, ← c86179, k2, ← c86180] at e_1404_11
  have hr := e_1404_11
  simp only [e_1404_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13364)) := by
  have c86181 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.1
  have c86182 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.1
  have c86183 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.1
  have c86184 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.1
  have c86185 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.1
  have c86186 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_12 := arithEq_of_rows h (row := 1404) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_12
  simp only [← c86181, ← c86182, ← c86183] at e_1404_12
  have e_1404_13 := arithEq_of_rows h (row := 1404) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_13
  simp only [← c86184, ← c86185, k2, ← c86186] at e_1404_13
  have hr := e_1404_13
  simp only [e_1404_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13365)) := by
  have c86187 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86188 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86189 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86190 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86191 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86192 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_14 := arithEq_of_rows h (row := 1404) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_14
  simp only [← c86187, ← c86188, ← c86189] at e_1404_14
  have e_1404_15 := arithEq_of_rows h (row := 1404) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_15
  simp only [← c86190, ← c86191, k2, ← c86192] at e_1404_15
  have hr := e_1404_15
  simp only [e_1404_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13366)) := by
  have c86193 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86194 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86195 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86196 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86197 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86198 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_16 := arithEq_of_rows h (row := 1404) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_16
  simp only [← c86193, ← c86194, ← c86195] at e_1404_16
  have e_1404_17 := arithEq_of_rows h (row := 1404) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_17
  simp only [← c86196, ← c86197, k2, ← c86198] at e_1404_17
  have hr := e_1404_17
  simp only [e_1404_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1404 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13367)) := by
  have c86199 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86200 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86201 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86202 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86203 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86204 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1404_18 := arithEq_of_rows h (row := 1404) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_18
  simp only [← c86199, ← c86200, ← c86201] at e_1404_18
  have e_1404_19 := arithEq_of_rows h (row := 1404) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1404_19
  simp only [← c86202, ← c86203, k2, ← c86204] at e_1404_19
  have hr := e_1404_19
  simp only [e_1404_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13368)) := by
  have c86205 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86206 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86207 := (publicBatchWrapper86_copies2693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86208 := (publicBatchWrapper86_copies2694 a h).1
  have c86209 := (publicBatchWrapper86_copies2694 a h).2.1
  have c86210 := (publicBatchWrapper86_copies2694 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_0 := arithEq_of_rows h (row := 1405) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_0
  simp only [← c86205, ← c86206, ← c86207] at e_1405_0
  have e_1405_1 := arithEq_of_rows h (row := 1405) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_1
  simp only [← c86208, ← c86209, k2, ← c86210] at e_1405_1
  have hr := e_1405_1
  simp only [e_1405_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13369)) := by
  have c86211 := (publicBatchWrapper86_copies2694 a h).2.2.2.1
  have c86212 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.1
  have c86213 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.1
  have c86214 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.1
  have c86215 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.1
  have c86216 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_2 := arithEq_of_rows h (row := 1405) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_2
  simp only [← c86211, ← c86212, ← c86213] at e_1405_2
  have e_1405_3 := arithEq_of_rows h (row := 1405) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_3
  simp only [← c86214, ← c86215, k2, ← c86216] at e_1405_3
  have hr := e_1405_3
  simp only [e_1405_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13410)) := by
  have c86217 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.1
  have c86218 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86219 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86220 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86221 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86222 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_4 := arithEq_of_rows h (row := 1405) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_4
  simp only [← c86217, ← c86218, ← c86219] at e_1405_4
  have e_1405_5 := arithEq_of_rows h (row := 1405) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_5
  simp only [← c86220, ← c86221, k2, ← c86222] at e_1405_5
  have hr := e_1405_5
  simp only [e_1405_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13411)) := by
  have c86223 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86224 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86225 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86226 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86227 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86228 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_6 := arithEq_of_rows h (row := 1405) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_6
  simp only [← c86223, ← c86224, ← c86225] at e_1405_6
  have e_1405_7 := arithEq_of_rows h (row := 1405) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_7
  simp only [← c86226, ← c86227, k2, ← c86228] at e_1405_7
  have hr := e_1405_7
  simp only [e_1405_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13412)) := by
  have c86229 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86230 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86231 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86232 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86233 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86234 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_8 := arithEq_of_rows h (row := 1405) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_8
  simp only [← c86229, ← c86230, ← c86231] at e_1405_8
  have e_1405_9 := arithEq_of_rows h (row := 1405) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_9
  simp only [← c86232, ← c86233, k2, ← c86234] at e_1405_9
  have hr := e_1405_9
  simp only [e_1405_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13413)) := by
  have c86235 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86236 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86237 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86238 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86239 := (publicBatchWrapper86_copies2694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86240 := (publicBatchWrapper86_copies2695 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_10 := arithEq_of_rows h (row := 1405) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_10
  simp only [← c86235, ← c86236, ← c86237] at e_1405_10
  have e_1405_11 := arithEq_of_rows h (row := 1405) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_11
  simp only [← c86238, ← c86239, k2, ← c86240] at e_1405_11
  have hr := e_1405_11
  simp only [e_1405_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13414)) := by
  have c86241 := (publicBatchWrapper86_copies2695 a h).2.1
  have c86242 := (publicBatchWrapper86_copies2695 a h).2.2.1
  have c86243 := (publicBatchWrapper86_copies2695 a h).2.2.2.1
  have c86244 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.1
  have c86245 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.1
  have c86246 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_12 := arithEq_of_rows h (row := 1405) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_12
  simp only [← c86241, ← c86242, ← c86243] at e_1405_12
  have e_1405_13 := arithEq_of_rows h (row := 1405) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_13
  simp only [← c86244, ← c86245, k2, ← c86246] at e_1405_13
  have hr := e_1405_13
  simp only [e_1405_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13415)) := by
  have c86247 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.1
  have c86248 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.1
  have c86249 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.1
  have c86250 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86251 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86252 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_14 := arithEq_of_rows h (row := 1405) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_14
  simp only [← c86247, ← c86248, ← c86249] at e_1405_14
  have e_1405_15 := arithEq_of_rows h (row := 1405) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_15
  simp only [← c86250, ← c86251, k2, ← c86252] at e_1405_15
  have hr := e_1405_15
  simp only [e_1405_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13416)) := by
  have c86253 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86254 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86255 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86256 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86257 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86258 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_16 := arithEq_of_rows h (row := 1405) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_16
  simp only [← c86253, ← c86254, ← c86255] at e_1405_16
  have e_1405_17 := arithEq_of_rows h (row := 1405) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_17
  simp only [← c86256, ← c86257, k2, ← c86258] at e_1405_17
  have hr := e_1405_17
  simp only [e_1405_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1405 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13417)) := by
  have c86259 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86260 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86261 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86262 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86263 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86264 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1405_18 := arithEq_of_rows h (row := 1405) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_18
  simp only [← c86259, ← c86260, ← c86261] at e_1405_18
  have e_1405_19 := arithEq_of_rows h (row := 1405) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_19
  simp only [← c86262, ← c86263, k2, ← c86264] at e_1405_19
  have hr := e_1405_19
  simp only [e_1405_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13418)) := by
  have c86265 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86266 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86267 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86268 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86269 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86270 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_0 := arithEq_of_rows h (row := 1406) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_0
  simp only [← c86265, ← c86266, ← c86267] at e_1406_0
  have e_1406_1 := arithEq_of_rows h (row := 1406) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_1
  simp only [← c86268, ← c86269, k2, ← c86270] at e_1406_1
  have hr := e_1406_1
  simp only [e_1406_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13419)) := by
  have c86271 := (publicBatchWrapper86_copies2695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86272 := (publicBatchWrapper86_copies2696 a h).1
  have c86273 := (publicBatchWrapper86_copies2696 a h).2.1
  have c86274 := (publicBatchWrapper86_copies2696 a h).2.2.1
  have c86275 := (publicBatchWrapper86_copies2696 a h).2.2.2.1
  have c86276 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_2 := arithEq_of_rows h (row := 1406) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_2
  simp only [← c86271, ← c86272, ← c86273] at e_1406_2
  have e_1406_3 := arithEq_of_rows h (row := 1406) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_3
  simp only [← c86274, ← c86275, k2, ← c86276] at e_1406_3
  have hr := e_1406_3
  simp only [e_1406_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13420)) := by
  have c86277 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.1
  have c86278 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.1
  have c86279 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.1
  have c86280 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.1
  have c86281 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.1
  have c86282 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_4 := arithEq_of_rows h (row := 1406) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_4
  simp only [← c86277, ← c86278, ← c86279] at e_1406_4
  have e_1406_5 := arithEq_of_rows h (row := 1406) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_5
  simp only [← c86280, ← c86281, k2, ← c86282] at e_1406_5
  have hr := e_1406_5
  simp only [e_1406_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13421)) := by
  have c86283 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86284 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86285 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86286 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86287 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86288 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_6 := arithEq_of_rows h (row := 1406) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_6
  simp only [← c86283, ← c86284, ← c86285] at e_1406_6
  have e_1406_7 := arithEq_of_rows h (row := 1406) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_7
  simp only [← c86286, ← c86287, k2, ← c86288] at e_1406_7
  have hr := e_1406_7
  simp only [e_1406_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13422)) := by
  have c86289 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86290 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86291 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86292 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86293 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86294 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_8 := arithEq_of_rows h (row := 1406) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_8
  simp only [← c86289, ← c86290, ← c86291] at e_1406_8
  have e_1406_9 := arithEq_of_rows h (row := 1406) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_9
  simp only [← c86292, ← c86293, k2, ← c86294] at e_1406_9
  have hr := e_1406_9
  simp only [e_1406_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13423)) := by
  have c86295 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86296 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86297 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86298 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86299 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86300 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_10 := arithEq_of_rows h (row := 1406) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_10
  simp only [← c86295, ← c86296, ← c86297] at e_1406_10
  have e_1406_11 := arithEq_of_rows h (row := 1406) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_11
  simp only [← c86298, ← c86299, k2, ← c86300] at e_1406_11
  have hr := e_1406_11
  simp only [e_1406_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13424)) := by
  have c86301 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86302 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86303 := (publicBatchWrapper86_copies2696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86304 := (publicBatchWrapper86_copies2697 a h).1
  have c86305 := (publicBatchWrapper86_copies2697 a h).2.1
  have c86306 := (publicBatchWrapper86_copies2697 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_12 := arithEq_of_rows h (row := 1406) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_12
  simp only [← c86301, ← c86302, ← c86303] at e_1406_12
  have e_1406_13 := arithEq_of_rows h (row := 1406) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_13
  simp only [← c86304, ← c86305, k2, ← c86306] at e_1406_13
  have hr := e_1406_13
  simp only [e_1406_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13425)) := by
  have c86307 := (publicBatchWrapper86_copies2697 a h).2.2.2.1
  have c86308 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.1
  have c86309 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.1
  have c86310 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.1
  have c86311 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.1
  have c86312 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_14 := arithEq_of_rows h (row := 1406) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_14
  simp only [← c86307, ← c86308, ← c86309] at e_1406_14
  have e_1406_15 := arithEq_of_rows h (row := 1406) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_15
  simp only [← c86310, ← c86311, k2, ← c86312] at e_1406_15
  have hr := e_1406_15
  simp only [e_1406_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13426)) := by
  have c86313 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.1
  have c86314 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86315 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86316 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86317 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86318 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_16 := arithEq_of_rows h (row := 1406) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_16
  simp only [← c86313, ← c86314, ← c86315] at e_1406_16
  have e_1406_17 := arithEq_of_rows h (row := 1406) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_17
  simp only [← c86316, ← c86317, k2, ← c86318] at e_1406_17
  have hr := e_1406_17
  simp only [e_1406_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1406 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13427)) := by
  have c86319 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86320 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86321 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86322 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86323 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86324 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1406_18 := arithEq_of_rows h (row := 1406) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_18
  simp only [← c86319, ← c86320, ← c86321] at e_1406_18
  have e_1406_19 := arithEq_of_rows h (row := 1406) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_19
  simp only [← c86322, ← c86323, k2, ← c86324] at e_1406_19
  have hr := e_1406_19
  simp only [e_1406_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13428)) := by
  have c86325 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86326 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86327 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86328 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86329 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86330 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_0 := arithEq_of_rows h (row := 1407) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_0
  simp only [← c86325, ← c86326, ← c86327] at e_1407_0
  have e_1407_1 := arithEq_of_rows h (row := 1407) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_1
  simp only [← c86328, ← c86329, k2, ← c86330] at e_1407_1
  have hr := e_1407_1
  simp only [e_1407_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13429)) := by
  have c86331 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86332 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86333 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86334 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86335 := (publicBatchWrapper86_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86336 := (publicBatchWrapper86_copies2698 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_2 := arithEq_of_rows h (row := 1407) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_2
  simp only [← c86331, ← c86332, ← c86333] at e_1407_2
  have e_1407_3 := arithEq_of_rows h (row := 1407) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_3
  simp only [← c86334, ← c86335, k2, ← c86336] at e_1407_3
  have hr := e_1407_3
  simp only [e_1407_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13430)) := by
  have c86337 := (publicBatchWrapper86_copies2698 a h).2.1
  have c86338 := (publicBatchWrapper86_copies2698 a h).2.2.1
  have c86339 := (publicBatchWrapper86_copies2698 a h).2.2.2.1
  have c86340 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.1
  have c86341 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.1
  have c86342 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_4 := arithEq_of_rows h (row := 1407) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_4
  simp only [← c86337, ← c86338, ← c86339] at e_1407_4
  have e_1407_5 := arithEq_of_rows h (row := 1407) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_5
  simp only [← c86340, ← c86341, k2, ← c86342] at e_1407_5
  have hr := e_1407_5
  simp only [e_1407_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13431)) := by
  have c86343 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.1
  have c86344 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.1
  have c86345 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.1
  have c86346 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86347 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86348 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_6 := arithEq_of_rows h (row := 1407) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_6
  simp only [← c86343, ← c86344, ← c86345] at e_1407_6
  have e_1407_7 := arithEq_of_rows h (row := 1407) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_7
  simp only [← c86346, ← c86347, k2, ← c86348] at e_1407_7
  have hr := e_1407_7
  simp only [e_1407_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13432)) := by
  have c86349 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86350 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86351 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86352 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86353 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86354 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_8 := arithEq_of_rows h (row := 1407) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_8
  simp only [← c86349, ← c86350, ← c86351] at e_1407_8
  have e_1407_9 := arithEq_of_rows h (row := 1407) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_9
  simp only [← c86352, ← c86353, k2, ← c86354] at e_1407_9
  have hr := e_1407_9
  simp only [e_1407_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13433)) := by
  have c86355 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86356 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86357 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86358 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86359 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86360 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_10 := arithEq_of_rows h (row := 1407) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_10
  simp only [← c86355, ← c86356, ← c86357] at e_1407_10
  have e_1407_11 := arithEq_of_rows h (row := 1407) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_11
  simp only [← c86358, ← c86359, k2, ← c86360] at e_1407_11
  have hr := e_1407_11
  simp only [e_1407_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13434)) := by
  have c86361 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86362 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86363 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86364 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86365 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86366 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_12 := arithEq_of_rows h (row := 1407) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_12
  simp only [← c86361, ← c86362, ← c86363] at e_1407_12
  have e_1407_13 := arithEq_of_rows h (row := 1407) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_13
  simp only [← c86364, ← c86365, k2, ← c86366] at e_1407_13
  have hr := e_1407_13
  simp only [e_1407_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13435)) := by
  have c86367 := (publicBatchWrapper86_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86368 := (publicBatchWrapper86_copies2699 a h).1
  have c86369 := (publicBatchWrapper86_copies2699 a h).2.1
  have c86370 := (publicBatchWrapper86_copies2699 a h).2.2.1
  have c86371 := (publicBatchWrapper86_copies2699 a h).2.2.2.1
  have c86372 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_14 := arithEq_of_rows h (row := 1407) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_14
  simp only [← c86367, ← c86368, ← c86369] at e_1407_14
  have e_1407_15 := arithEq_of_rows h (row := 1407) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_15
  simp only [← c86370, ← c86371, k2, ← c86372] at e_1407_15
  have hr := e_1407_15
  simp only [e_1407_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13436)) := by
  have c86373 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.1
  have c86374 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.1
  have c86375 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.1
  have c86376 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.1
  have c86377 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.1
  have c86378 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_16 := arithEq_of_rows h (row := 1407) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_16
  simp only [← c86373, ← c86374, ← c86375] at e_1407_16
  have e_1407_17 := arithEq_of_rows h (row := 1407) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_17
  simp only [← c86376, ← c86377, k2, ← c86378] at e_1407_17
  have hr := e_1407_17
  simp only [e_1407_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1407 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13437)) := by
  have c86379 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86380 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86381 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86382 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86383 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86384 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1407_18 := arithEq_of_rows h (row := 1407) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_18
  simp only [← c86379, ← c86380, ← c86381] at e_1407_18
  have e_1407_19 := arithEq_of_rows h (row := 1407) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_19
  simp only [← c86382, ← c86383, k2, ← c86384] at e_1407_19
  have hr := e_1407_19
  simp only [e_1407_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13438)) := by
  have c86385 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86386 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86387 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86388 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86389 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86390 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_0 := arithEq_of_rows h (row := 1408) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_0
  simp only [← c86385, ← c86386, ← c86387] at e_1408_0
  have e_1408_1 := arithEq_of_rows h (row := 1408) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_1
  simp only [← c86388, ← c86389, k2, ← c86390] at e_1408_1
  have hr := e_1408_1
  simp only [e_1408_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13439)) := by
  have c86391 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86392 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86393 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86394 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86395 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86396 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_2 := arithEq_of_rows h (row := 1408) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_2
  simp only [← c86391, ← c86392, ← c86393] at e_1408_2
  have e_1408_3 := arithEq_of_rows h (row := 1408) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_3
  simp only [← c86394, ← c86395, k2, ← c86396] at e_1408_3
  have hr := e_1408_3
  simp only [e_1408_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13440)) := by
  have c86397 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86398 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86399 := (publicBatchWrapper86_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86400 := (publicBatchWrapper86_copies2700 a h).1
  have c86401 := (publicBatchWrapper86_copies2700 a h).2.1
  have c86402 := (publicBatchWrapper86_copies2700 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_4 := arithEq_of_rows h (row := 1408) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_4
  simp only [← c86397, ← c86398, ← c86399] at e_1408_4
  have e_1408_5 := arithEq_of_rows h (row := 1408) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_5
  simp only [← c86400, ← c86401, k2, ← c86402] at e_1408_5
  have hr := e_1408_5
  simp only [e_1408_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13441)) := by
  have c86403 := (publicBatchWrapper86_copies2700 a h).2.2.2.1
  have c86404 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.1
  have c86405 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.1
  have c86406 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.1
  have c86407 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.1
  have c86408 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_6 := arithEq_of_rows h (row := 1408) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_6
  simp only [← c86403, ← c86404, ← c86405] at e_1408_6
  have e_1408_7 := arithEq_of_rows h (row := 1408) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_7
  simp only [← c86406, ← c86407, k2, ← c86408] at e_1408_7
  have hr := e_1408_7
  simp only [e_1408_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13442)) := by
  have c86409 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.1
  have c86410 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86411 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86412 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86413 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86414 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_8 := arithEq_of_rows h (row := 1408) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_8
  simp only [← c86409, ← c86410, ← c86411] at e_1408_8
  have e_1408_9 := arithEq_of_rows h (row := 1408) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_9
  simp only [← c86412, ← c86413, k2, ← c86414] at e_1408_9
  have hr := e_1408_9
  simp only [e_1408_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13443)) := by
  have c86415 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86416 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86417 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86418 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86419 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86420 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_10 := arithEq_of_rows h (row := 1408) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_10
  simp only [← c86415, ← c86416, ← c86417] at e_1408_10
  have e_1408_11 := arithEq_of_rows h (row := 1408) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_11
  simp only [← c86418, ← c86419, k2, ← c86420] at e_1408_11
  have hr := e_1408_11
  simp only [e_1408_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13444)) := by
  have c86421 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86422 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86423 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86424 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86425 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86426 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_12 := arithEq_of_rows h (row := 1408) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_12
  simp only [← c86421, ← c86422, ← c86423] at e_1408_12
  have e_1408_13 := arithEq_of_rows h (row := 1408) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_13
  simp only [← c86424, ← c86425, k2, ← c86426] at e_1408_13
  have hr := e_1408_13
  simp only [e_1408_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13445)) := by
  have c86427 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86428 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86429 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86430 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86431 := (publicBatchWrapper86_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86432 := (publicBatchWrapper86_copies2701 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_14 := arithEq_of_rows h (row := 1408) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_14
  simp only [← c86427, ← c86428, ← c86429] at e_1408_14
  have e_1408_15 := arithEq_of_rows h (row := 1408) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_15
  simp only [← c86430, ← c86431, k2, ← c86432] at e_1408_15
  have hr := e_1408_15
  simp only [e_1408_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13446)) := by
  have c86433 := (publicBatchWrapper86_copies2701 a h).2.1
  have c86434 := (publicBatchWrapper86_copies2701 a h).2.2.1
  have c86435 := (publicBatchWrapper86_copies2701 a h).2.2.2.1
  have c86436 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.1
  have c86437 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.1
  have c86438 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_16 := arithEq_of_rows h (row := 1408) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_16
  simp only [← c86433, ← c86434, ← c86435] at e_1408_16
  have e_1408_17 := arithEq_of_rows h (row := 1408) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_17
  simp only [← c86436, ← c86437, k2, ← c86438] at e_1408_17
  have hr := e_1408_17
  simp only [e_1408_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1408 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13447)) := by
  have c86439 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.1
  have c86440 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.1
  have c86441 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.1
  have c86442 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86443 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86444 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1408_18 := arithEq_of_rows h (row := 1408) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_18
  simp only [← c86439, ← c86440, ← c86441] at e_1408_18
  have e_1408_19 := arithEq_of_rows h (row := 1408) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_19
  simp only [← c86442, ← c86443, k2, ← c86444] at e_1408_19
  have hr := e_1408_19
  simp only [e_1408_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13448)) := by
  have c86445 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86446 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86447 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86448 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86449 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86450 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_0 := arithEq_of_rows h (row := 1409) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_0
  simp only [← c86445, ← c86446, ← c86447] at e_1409_0
  have e_1409_1 := arithEq_of_rows h (row := 1409) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_1
  simp only [← c86448, ← c86449, k2, ← c86450] at e_1409_1
  have hr := e_1409_1
  simp only [e_1409_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13449)) := by
  have c86451 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86452 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86453 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86454 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86455 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86456 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_2 := arithEq_of_rows h (row := 1409) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_2
  simp only [← c86451, ← c86452, ← c86453] at e_1409_2
  have e_1409_3 := arithEq_of_rows h (row := 1409) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_3
  simp only [← c86454, ← c86455, k2, ← c86456] at e_1409_3
  have hr := e_1409_3
  simp only [e_1409_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13450)) := by
  have c86457 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86458 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86459 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86460 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86461 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86462 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_4 := arithEq_of_rows h (row := 1409) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_4
  simp only [← c86457, ← c86458, ← c86459] at e_1409_4
  have e_1409_5 := arithEq_of_rows h (row := 1409) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_5
  simp only [← c86460, ← c86461, k2, ← c86462] at e_1409_5
  have hr := e_1409_5
  simp only [e_1409_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13451)) := by
  have c86463 := (publicBatchWrapper86_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86464 := (publicBatchWrapper86_copies2702 a h).1
  have c86465 := (publicBatchWrapper86_copies2702 a h).2.1
  have c86466 := (publicBatchWrapper86_copies2702 a h).2.2.1
  have c86467 := (publicBatchWrapper86_copies2702 a h).2.2.2.1
  have c86468 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_6 := arithEq_of_rows h (row := 1409) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_6
  simp only [← c86463, ← c86464, ← c86465] at e_1409_6
  have e_1409_7 := arithEq_of_rows h (row := 1409) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_7
  simp only [← c86466, ← c86467, k2, ← c86468] at e_1409_7
  have hr := e_1409_7
  simp only [e_1409_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13452)) := by
  have c86469 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.1
  have c86470 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.1
  have c86471 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.1
  have c86472 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.1
  have c86473 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.1
  have c86474 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_8 := arithEq_of_rows h (row := 1409) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_8
  simp only [← c86469, ← c86470, ← c86471] at e_1409_8
  have e_1409_9 := arithEq_of_rows h (row := 1409) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_9
  simp only [← c86472, ← c86473, k2, ← c86474] at e_1409_9
  have hr := e_1409_9
  simp only [e_1409_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13453)) := by
  have c86475 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86476 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86477 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86478 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86479 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86480 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_10 := arithEq_of_rows h (row := 1409) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_10
  simp only [← c86475, ← c86476, ← c86477] at e_1409_10
  have e_1409_11 := arithEq_of_rows h (row := 1409) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_11
  simp only [← c86478, ← c86479, k2, ← c86480] at e_1409_11
  have hr := e_1409_11
  simp only [e_1409_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13454)) := by
  have c86481 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86482 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86483 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86484 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86485 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86486 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_12 := arithEq_of_rows h (row := 1409) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_12
  simp only [← c86481, ← c86482, ← c86483] at e_1409_12
  have e_1409_13 := arithEq_of_rows h (row := 1409) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_13
  simp only [← c86484, ← c86485, k2, ← c86486] at e_1409_13
  have hr := e_1409_13
  simp only [e_1409_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13455)) := by
  have c86487 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86488 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86489 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86490 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86491 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86492 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_14 := arithEq_of_rows h (row := 1409) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_14
  simp only [← c86487, ← c86488, ← c86489] at e_1409_14
  have e_1409_15 := arithEq_of_rows h (row := 1409) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_15
  simp only [← c86490, ← c86491, k2, ← c86492] at e_1409_15
  have hr := e_1409_15
  simp only [e_1409_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13456)) := by
  have c86493 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86494 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86495 := (publicBatchWrapper86_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86496 := (publicBatchWrapper86_copies2703 a h).1
  have c86497 := (publicBatchWrapper86_copies2703 a h).2.1
  have c86498 := (publicBatchWrapper86_copies2703 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_16 := arithEq_of_rows h (row := 1409) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_16
  simp only [← c86493, ← c86494, ← c86495] at e_1409_16
  have e_1409_17 := arithEq_of_rows h (row := 1409) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_17
  simp only [← c86496, ← c86497, k2, ← c86498] at e_1409_17
  have hr := e_1409_17
  simp only [e_1409_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1409 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13457)) := by
  have c86499 := (publicBatchWrapper86_copies2703 a h).2.2.2.1
  have c86500 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.1
  have c86501 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.1
  have c86502 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.1
  have c86503 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.1
  have c86504 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1409_18 := arithEq_of_rows h (row := 1409) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_18
  simp only [← c86499, ← c86500, ← c86501] at e_1409_18
  have e_1409_19 := arithEq_of_rows h (row := 1409) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_19
  simp only [← c86502, ← c86503, k2, ← c86504] at e_1409_19
  have hr := e_1409_19
  simp only [e_1409_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13458)) := by
  have c86505 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.1
  have c86506 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86507 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86508 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86509 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86510 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_0 := arithEq_of_rows h (row := 1410) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_0
  simp only [← c86505, ← c86506, ← c86507] at e_1410_0
  have e_1410_1 := arithEq_of_rows h (row := 1410) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_1
  simp only [← c86508, ← c86509, k2, ← c86510] at e_1410_1
  have hr := e_1410_1
  simp only [e_1410_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13459)) := by
  have c86511 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86512 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86513 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86514 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86515 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86516 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_2 := arithEq_of_rows h (row := 1410) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_2
  simp only [← c86511, ← c86512, ← c86513] at e_1410_2
  have e_1410_3 := arithEq_of_rows h (row := 1410) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_3
  simp only [← c86514, ← c86515, k2, ← c86516] at e_1410_3
  have hr := e_1410_3
  simp only [e_1410_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13460)) := by
  have c86517 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86518 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86519 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86520 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86521 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86522 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_4 := arithEq_of_rows h (row := 1410) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_4
  simp only [← c86517, ← c86518, ← c86519] at e_1410_4
  have e_1410_5 := arithEq_of_rows h (row := 1410) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_5
  simp only [← c86520, ← c86521, k2, ← c86522] at e_1410_5
  have hr := e_1410_5
  simp only [e_1410_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13461)) := by
  have c86523 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86524 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86525 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86526 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86527 := (publicBatchWrapper86_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86528 := (publicBatchWrapper86_copies2704 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_6 := arithEq_of_rows h (row := 1410) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_6
  simp only [← c86523, ← c86524, ← c86525] at e_1410_6
  have e_1410_7 := arithEq_of_rows h (row := 1410) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_7
  simp only [← c86526, ← c86527, k2, ← c86528] at e_1410_7
  have hr := e_1410_7
  simp only [e_1410_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13462)) := by
  have c86529 := (publicBatchWrapper86_copies2704 a h).2.1
  have c86530 := (publicBatchWrapper86_copies2704 a h).2.2.1
  have c86531 := (publicBatchWrapper86_copies2704 a h).2.2.2.1
  have c86532 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.1
  have c86533 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.1
  have c86534 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_8 := arithEq_of_rows h (row := 1410) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_8
  simp only [← c86529, ← c86530, ← c86531] at e_1410_8
  have e_1410_9 := arithEq_of_rows h (row := 1410) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_9
  simp only [← c86532, ← c86533, k2, ← c86534] at e_1410_9
  have hr := e_1410_9
  simp only [e_1410_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13463)) := by
  have c86535 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.1
  have c86536 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.1
  have c86537 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.1
  have c86538 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86539 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86540 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_10 := arithEq_of_rows h (row := 1410) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_10
  simp only [← c86535, ← c86536, ← c86537] at e_1410_10
  have e_1410_11 := arithEq_of_rows h (row := 1410) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_11
  simp only [← c86538, ← c86539, k2, ← c86540] at e_1410_11
  have hr := e_1410_11
  simp only [e_1410_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13464)) := by
  have c86541 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86542 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86543 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86544 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86545 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86546 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_12 := arithEq_of_rows h (row := 1410) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_12
  simp only [← c86541, ← c86542, ← c86543] at e_1410_12
  have e_1410_13 := arithEq_of_rows h (row := 1410) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_13
  simp only [← c86544, ← c86545, k2, ← c86546] at e_1410_13
  have hr := e_1410_13
  simp only [e_1410_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13465)) := by
  have c86547 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86548 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86549 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86550 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86551 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86552 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_14 := arithEq_of_rows h (row := 1410) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_14
  simp only [← c86547, ← c86548, ← c86549] at e_1410_14
  have e_1410_15 := arithEq_of_rows h (row := 1410) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_15
  simp only [← c86550, ← c86551, k2, ← c86552] at e_1410_15
  have hr := e_1410_15
  simp only [e_1410_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13466)) := by
  have c86553 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86554 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86555 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86556 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86557 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86558 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_16 := arithEq_of_rows h (row := 1410) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_16
  simp only [← c86553, ← c86554, ← c86555] at e_1410_16
  have e_1410_17 := arithEq_of_rows h (row := 1410) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_17
  simp only [← c86556, ← c86557, k2, ← c86558] at e_1410_17
  have hr := e_1410_17
  simp only [e_1410_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1410 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13467)) := by
  have c86559 := (publicBatchWrapper86_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86560 := (publicBatchWrapper86_copies2705 a h).1
  have c86561 := (publicBatchWrapper86_copies2705 a h).2.1
  have c86562 := (publicBatchWrapper86_copies2705 a h).2.2.1
  have c86563 := (publicBatchWrapper86_copies2705 a h).2.2.2.1
  have c86564 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1410_18 := arithEq_of_rows h (row := 1410) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_18
  simp only [← c86559, ← c86560, ← c86561] at e_1410_18
  have e_1410_19 := arithEq_of_rows h (row := 1410) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_19
  simp only [← c86562, ← c86563, k2, ← c86564] at e_1410_19
  have hr := e_1410_19
  simp only [e_1410_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13468)) := by
  have c86565 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.1
  have c86566 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.1
  have c86567 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.1
  have c86568 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.1
  have c86569 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.1
  have c86570 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_0 := arithEq_of_rows h (row := 1411) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_0
  simp only [← c86565, ← c86566, ← c86567] at e_1411_0
  have e_1411_1 := arithEq_of_rows h (row := 1411) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_1
  simp only [← c86568, ← c86569, k2, ← c86570] at e_1411_1
  have hr := e_1411_1
  simp only [e_1411_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13469)) := by
  have c86571 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86572 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86573 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86574 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86575 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86576 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_2 := arithEq_of_rows h (row := 1411) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_2
  simp only [← c86571, ← c86572, ← c86573] at e_1411_2
  have e_1411_3 := arithEq_of_rows h (row := 1411) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_3
  simp only [← c86574, ← c86575, k2, ← c86576] at e_1411_3
  have hr := e_1411_3
  simp only [e_1411_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13470)) := by
  have c86577 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86578 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86579 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86580 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86581 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86582 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_4 := arithEq_of_rows h (row := 1411) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_4
  simp only [← c86577, ← c86578, ← c86579] at e_1411_4
  have e_1411_5 := arithEq_of_rows h (row := 1411) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_5
  simp only [← c86580, ← c86581, k2, ← c86582] at e_1411_5
  have hr := e_1411_5
  simp only [e_1411_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13471)) := by
  have c86583 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86584 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86585 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86586 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86587 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86588 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_6 := arithEq_of_rows h (row := 1411) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_6
  simp only [← c86583, ← c86584, ← c86585] at e_1411_6
  have e_1411_7 := arithEq_of_rows h (row := 1411) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_7
  simp only [← c86586, ← c86587, k2, ← c86588] at e_1411_7
  have hr := e_1411_7
  simp only [e_1411_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13472)) := by
  have c86589 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86590 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86591 := (publicBatchWrapper86_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86592 := (publicBatchWrapper86_copies2706 a h).1
  have c86593 := (publicBatchWrapper86_copies2706 a h).2.1
  have c86594 := (publicBatchWrapper86_copies2706 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_8 := arithEq_of_rows h (row := 1411) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_8
  simp only [← c86589, ← c86590, ← c86591] at e_1411_8
  have e_1411_9 := arithEq_of_rows h (row := 1411) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_9
  simp only [← c86592, ← c86593, k2, ← c86594] at e_1411_9
  have hr := e_1411_9
  simp only [e_1411_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13473)) := by
  have c86595 := (publicBatchWrapper86_copies2706 a h).2.2.2.1
  have c86596 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.1
  have c86597 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.1
  have c86598 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.1
  have c86599 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.1
  have c86600 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_10 := arithEq_of_rows h (row := 1411) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_10
  simp only [← c86595, ← c86596, ← c86597] at e_1411_10
  have e_1411_11 := arithEq_of_rows h (row := 1411) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_11
  simp only [← c86598, ← c86599, k2, ← c86600] at e_1411_11
  have hr := e_1411_11
  simp only [e_1411_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13474)) := by
  have c86601 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.1
  have c86602 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86603 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86604 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86605 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86606 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_12 := arithEq_of_rows h (row := 1411) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_12
  simp only [← c86601, ← c86602, ← c86603] at e_1411_12
  have e_1411_13 := arithEq_of_rows h (row := 1411) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_13
  simp only [← c86604, ← c86605, k2, ← c86606] at e_1411_13
  have hr := e_1411_13
  simp only [e_1411_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13475)) := by
  have c86607 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86608 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86609 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86610 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86611 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86612 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_14 := arithEq_of_rows h (row := 1411) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_14
  simp only [← c86607, ← c86608, ← c86609] at e_1411_14
  have e_1411_15 := arithEq_of_rows h (row := 1411) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_15
  simp only [← c86610, ← c86611, k2, ← c86612] at e_1411_15
  have hr := e_1411_15
  simp only [e_1411_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13476)) := by
  have c86613 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86614 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86615 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86616 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86617 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86618 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_16 := arithEq_of_rows h (row := 1411) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_16
  simp only [← c86613, ← c86614, ← c86615] at e_1411_16
  have e_1411_17 := arithEq_of_rows h (row := 1411) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_17
  simp only [← c86616, ← c86617, k2, ← c86618] at e_1411_17
  have hr := e_1411_17
  simp only [e_1411_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1411 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13477)) := by
  have c86619 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86620 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86621 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86622 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86623 := (publicBatchWrapper86_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86624 := (publicBatchWrapper86_copies2707 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1411_18 := arithEq_of_rows h (row := 1411) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_18
  simp only [← c86619, ← c86620, ← c86621] at e_1411_18
  have e_1411_19 := arithEq_of_rows h (row := 1411) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_19
  simp only [← c86622, ← c86623, k2, ← c86624] at e_1411_19
  have hr := e_1411_19
  simp only [e_1411_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13478)) := by
  have c86625 := (publicBatchWrapper86_copies2707 a h).2.1
  have c86626 := (publicBatchWrapper86_copies2707 a h).2.2.1
  have c86627 := (publicBatchWrapper86_copies2707 a h).2.2.2.1
  have c86628 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.1
  have c86629 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.1
  have c86630 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_0 := arithEq_of_rows h (row := 1412) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_0
  simp only [← c86625, ← c86626, ← c86627] at e_1412_0
  have e_1412_1 := arithEq_of_rows h (row := 1412) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_1
  simp only [← c86628, ← c86629, k2, ← c86630] at e_1412_1
  have hr := e_1412_1
  simp only [e_1412_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13479)) := by
  have c86631 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.1
  have c86632 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.1
  have c86633 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.1
  have c86634 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86635 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86636 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_2 := arithEq_of_rows h (row := 1412) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_2
  simp only [← c86631, ← c86632, ← c86633] at e_1412_2
  have e_1412_3 := arithEq_of_rows h (row := 1412) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_3
  simp only [← c86634, ← c86635, k2, ← c86636] at e_1412_3
  have hr := e_1412_3
  simp only [e_1412_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13480)) := by
  have c86637 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86638 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86639 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86640 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86641 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86642 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_4 := arithEq_of_rows h (row := 1412) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_4
  simp only [← c86637, ← c86638, ← c86639] at e_1412_4
  have e_1412_5 := arithEq_of_rows h (row := 1412) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_5
  simp only [← c86640, ← c86641, k2, ← c86642] at e_1412_5
  have hr := e_1412_5
  simp only [e_1412_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13481)) := by
  have c86643 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86644 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86645 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86646 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86647 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86648 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_6 := arithEq_of_rows h (row := 1412) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_6
  simp only [← c86643, ← c86644, ← c86645] at e_1412_6
  have e_1412_7 := arithEq_of_rows h (row := 1412) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_7
  simp only [← c86646, ← c86647, k2, ← c86648] at e_1412_7
  have hr := e_1412_7
  simp only [e_1412_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13482)) := by
  have c86649 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86650 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86651 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86652 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86653 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86654 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_8 := arithEq_of_rows h (row := 1412) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_8
  simp only [← c86649, ← c86650, ← c86651] at e_1412_8
  have e_1412_9 := arithEq_of_rows h (row := 1412) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_9
  simp only [← c86652, ← c86653, k2, ← c86654] at e_1412_9
  have hr := e_1412_9
  simp only [e_1412_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13483)) := by
  have c86655 := (publicBatchWrapper86_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86656 := (publicBatchWrapper86_copies2708 a h).1
  have c86657 := (publicBatchWrapper86_copies2708 a h).2.1
  have c86658 := (publicBatchWrapper86_copies2708 a h).2.2.1
  have c86659 := (publicBatchWrapper86_copies2708 a h).2.2.2.1
  have c86660 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_10 := arithEq_of_rows h (row := 1412) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_10
  simp only [← c86655, ← c86656, ← c86657] at e_1412_10
  have e_1412_11 := arithEq_of_rows h (row := 1412) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_11
  simp only [← c86658, ← c86659, k2, ← c86660] at e_1412_11
  have hr := e_1412_11
  simp only [e_1412_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13484)) := by
  have c86661 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.1
  have c86662 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.1
  have c86663 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.1
  have c86664 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.1
  have c86665 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.1
  have c86666 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_12 := arithEq_of_rows h (row := 1412) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_12
  simp only [← c86661, ← c86662, ← c86663] at e_1412_12
  have e_1412_13 := arithEq_of_rows h (row := 1412) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_13
  simp only [← c86664, ← c86665, k2, ← c86666] at e_1412_13
  have hr := e_1412_13
  simp only [e_1412_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13485)) := by
  have c86667 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86668 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86669 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86670 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86671 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86672 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_14 := arithEq_of_rows h (row := 1412) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_14
  simp only [← c86667, ← c86668, ← c86669] at e_1412_14
  have e_1412_15 := arithEq_of_rows h (row := 1412) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_15
  simp only [← c86670, ← c86671, k2, ← c86672] at e_1412_15
  have hr := e_1412_15
  simp only [e_1412_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13486)) := by
  have c86673 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86674 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86675 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86676 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86677 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86678 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_16 := arithEq_of_rows h (row := 1412) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_16
  simp only [← c86673, ← c86674, ← c86675] at e_1412_16
  have e_1412_17 := arithEq_of_rows h (row := 1412) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_17
  simp only [← c86676, ← c86677, k2, ← c86678] at e_1412_17
  have hr := e_1412_17
  simp only [e_1412_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1412 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13487)) := by
  have c86679 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86680 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86681 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86682 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86683 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86684 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1412_18 := arithEq_of_rows h (row := 1412) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_18
  simp only [← c86679, ← c86680, ← c86681] at e_1412_18
  have e_1412_19 := arithEq_of_rows h (row := 1412) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_19
  simp only [← c86682, ← c86683, k2, ← c86684] at e_1412_19
  have hr := e_1412_19
  simp only [e_1412_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13488)) := by
  have c86685 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86686 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86687 := (publicBatchWrapper86_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86688 := (publicBatchWrapper86_copies2709 a h).1
  have c86689 := (publicBatchWrapper86_copies2709 a h).2.1
  have c86690 := (publicBatchWrapper86_copies2709 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_0 := arithEq_of_rows h (row := 1413) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_0
  simp only [← c86685, ← c86686, ← c86687] at e_1413_0
  have e_1413_1 := arithEq_of_rows h (row := 1413) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_1
  simp only [← c86688, ← c86689, k2, ← c86690] at e_1413_1
  have hr := e_1413_1
  simp only [e_1413_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13489)) := by
  have c86691 := (publicBatchWrapper86_copies2709 a h).2.2.2.1
  have c86692 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.1
  have c86693 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.1
  have c86694 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.1
  have c86695 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.1
  have c86696 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_2 := arithEq_of_rows h (row := 1413) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_2
  simp only [← c86691, ← c86692, ← c86693] at e_1413_2
  have e_1413_3 := arithEq_of_rows h (row := 1413) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_3
  simp only [← c86694, ← c86695, k2, ← c86696] at e_1413_3
  have hr := e_1413_3
  simp only [e_1413_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13490)) := by
  have c86697 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.1
  have c86698 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86699 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86700 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86701 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86702 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_4 := arithEq_of_rows h (row := 1413) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_4
  simp only [← c86697, ← c86698, ← c86699] at e_1413_4
  have e_1413_5 := arithEq_of_rows h (row := 1413) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_5
  simp only [← c86700, ← c86701, k2, ← c86702] at e_1413_5
  have hr := e_1413_5
  simp only [e_1413_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13491)) := by
  have c86703 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86704 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86705 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86706 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86707 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86708 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_6 := arithEq_of_rows h (row := 1413) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_6
  simp only [← c86703, ← c86704, ← c86705] at e_1413_6
  have e_1413_7 := arithEq_of_rows h (row := 1413) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_7
  simp only [← c86706, ← c86707, k2, ← c86708] at e_1413_7
  have hr := e_1413_7
  simp only [e_1413_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13492)) := by
  have c86709 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86710 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86711 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86712 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86713 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86714 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_8 := arithEq_of_rows h (row := 1413) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_8
  simp only [← c86709, ← c86710, ← c86711] at e_1413_8
  have e_1413_9 := arithEq_of_rows h (row := 1413) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_9
  simp only [← c86712, ← c86713, k2, ← c86714] at e_1413_9
  have hr := e_1413_9
  simp only [e_1413_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13493)) := by
  have c86715 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86716 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86717 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86718 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86719 := (publicBatchWrapper86_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86720 := (publicBatchWrapper86_copies2710 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_10 := arithEq_of_rows h (row := 1413) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_10
  simp only [← c86715, ← c86716, ← c86717] at e_1413_10
  have e_1413_11 := arithEq_of_rows h (row := 1413) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_11
  simp only [← c86718, ← c86719, k2, ← c86720] at e_1413_11
  have hr := e_1413_11
  simp only [e_1413_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13494)) := by
  have c86721 := (publicBatchWrapper86_copies2710 a h).2.1
  have c86722 := (publicBatchWrapper86_copies2710 a h).2.2.1
  have c86723 := (publicBatchWrapper86_copies2710 a h).2.2.2.1
  have c86724 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.1
  have c86725 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.1
  have c86726 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_12 := arithEq_of_rows h (row := 1413) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_12
  simp only [← c86721, ← c86722, ← c86723] at e_1413_12
  have e_1413_13 := arithEq_of_rows h (row := 1413) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_13
  simp only [← c86724, ← c86725, k2, ← c86726] at e_1413_13
  have hr := e_1413_13
  simp only [e_1413_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13495)) := by
  have c86727 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.1
  have c86728 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.1
  have c86729 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.1
  have c86730 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86731 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86732 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_14 := arithEq_of_rows h (row := 1413) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_14
  simp only [← c86727, ← c86728, ← c86729] at e_1413_14
  have e_1413_15 := arithEq_of_rows h (row := 1413) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_15
  simp only [← c86730, ← c86731, k2, ← c86732] at e_1413_15
  have hr := e_1413_15
  simp only [e_1413_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13496)) := by
  have c86733 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86734 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86735 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86736 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86737 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86738 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_16 := arithEq_of_rows h (row := 1413) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_16
  simp only [← c86733, ← c86734, ← c86735] at e_1413_16
  have e_1413_17 := arithEq_of_rows h (row := 1413) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_17
  simp only [← c86736, ← c86737, k2, ← c86738] at e_1413_17
  have hr := e_1413_17
  simp only [e_1413_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1413 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13497)) := by
  have c86739 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86740 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86741 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86742 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86743 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86744 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1413_18 := arithEq_of_rows h (row := 1413) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_18
  simp only [← c86739, ← c86740, ← c86741] at e_1413_18
  have e_1413_19 := arithEq_of_rows h (row := 1413) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_19
  simp only [← c86742, ← c86743, k2, ← c86744] at e_1413_19
  have hr := e_1413_19
  simp only [e_1413_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13498)) := by
  have c86745 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86746 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86747 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86748 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86749 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86750 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_0 := arithEq_of_rows h (row := 1414) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_0
  simp only [← c86745, ← c86746, ← c86747] at e_1414_0
  have e_1414_1 := arithEq_of_rows h (row := 1414) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_1
  simp only [← c86748, ← c86749, k2, ← c86750] at e_1414_1
  have hr := e_1414_1
  simp only [e_1414_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13499)) := by
  have c86751 := (publicBatchWrapper86_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86752 := (publicBatchWrapper86_copies2711 a h).1
  have c86753 := (publicBatchWrapper86_copies2711 a h).2.1
  have c86754 := (publicBatchWrapper86_copies2711 a h).2.2.1
  have c86755 := (publicBatchWrapper86_copies2711 a h).2.2.2.1
  have c86756 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_2 := arithEq_of_rows h (row := 1414) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_2
  simp only [← c86751, ← c86752, ← c86753] at e_1414_2
  have e_1414_3 := arithEq_of_rows h (row := 1414) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_3
  simp only [← c86754, ← c86755, k2, ← c86756] at e_1414_3
  have hr := e_1414_3
  simp only [e_1414_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13500)) := by
  have c86757 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.1
  have c86758 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.1
  have c86759 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.1
  have c86760 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.1
  have c86761 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.1
  have c86762 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_4 := arithEq_of_rows h (row := 1414) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_4
  simp only [← c86757, ← c86758, ← c86759] at e_1414_4
  have e_1414_5 := arithEq_of_rows h (row := 1414) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_5
  simp only [← c86760, ← c86761, k2, ← c86762] at e_1414_5
  have hr := e_1414_5
  simp only [e_1414_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13501)) := by
  have c86763 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86764 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86765 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86766 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86767 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86768 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_6 := arithEq_of_rows h (row := 1414) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_6
  simp only [← c86763, ← c86764, ← c86765] at e_1414_6
  have e_1414_7 := arithEq_of_rows h (row := 1414) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_7
  simp only [← c86766, ← c86767, k2, ← c86768] at e_1414_7
  have hr := e_1414_7
  simp only [e_1414_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13502)) := by
  have c86769 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86770 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86771 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86772 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86773 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86774 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_8 := arithEq_of_rows h (row := 1414) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_8
  simp only [← c86769, ← c86770, ← c86771] at e_1414_8
  have e_1414_9 := arithEq_of_rows h (row := 1414) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_9
  simp only [← c86772, ← c86773, k2, ← c86774] at e_1414_9
  have hr := e_1414_9
  simp only [e_1414_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13503)) := by
  have c86775 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86776 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86777 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86778 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86779 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86780 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_10 := arithEq_of_rows h (row := 1414) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_10
  simp only [← c86775, ← c86776, ← c86777] at e_1414_10
  have e_1414_11 := arithEq_of_rows h (row := 1414) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_11
  simp only [← c86778, ← c86779, k2, ← c86780] at e_1414_11
  have hr := e_1414_11
  simp only [e_1414_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13504)) := by
  have c86781 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86782 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86783 := (publicBatchWrapper86_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86784 := (publicBatchWrapper86_copies2712 a h).1
  have c86785 := (publicBatchWrapper86_copies2712 a h).2.1
  have c86786 := (publicBatchWrapper86_copies2712 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_12 := arithEq_of_rows h (row := 1414) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_12
  simp only [← c86781, ← c86782, ← c86783] at e_1414_12
  have e_1414_13 := arithEq_of_rows h (row := 1414) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_13
  simp only [← c86784, ← c86785, k2, ← c86786] at e_1414_13
  have hr := e_1414_13
  simp only [e_1414_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13505)) := by
  have c86787 := (publicBatchWrapper86_copies2712 a h).2.2.2.1
  have c86788 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.1
  have c86789 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.1
  have c86790 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.1
  have c86791 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.1
  have c86792 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_14 := arithEq_of_rows h (row := 1414) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_14
  simp only [← c86787, ← c86788, ← c86789] at e_1414_14
  have e_1414_15 := arithEq_of_rows h (row := 1414) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_15
  simp only [← c86790, ← c86791, k2, ← c86792] at e_1414_15
  have hr := e_1414_15
  simp only [e_1414_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13506)) := by
  have c86793 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.1
  have c86794 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86795 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86796 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86797 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86798 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_16 := arithEq_of_rows h (row := 1414) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_16
  simp only [← c86793, ← c86794, ← c86795] at e_1414_16
  have e_1414_17 := arithEq_of_rows h (row := 1414) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_17
  simp only [← c86796, ← c86797, k2, ← c86798] at e_1414_17
  have hr := e_1414_17
  simp only [e_1414_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1414 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13507)) := by
  have c86799 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86800 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86801 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86802 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86803 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86804 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1414_18 := arithEq_of_rows h (row := 1414) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_18
  simp only [← c86799, ← c86800, ← c86801] at e_1414_18
  have e_1414_19 := arithEq_of_rows h (row := 1414) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_19
  simp only [← c86802, ← c86803, k2, ← c86804] at e_1414_19
  have hr := e_1414_19
  simp only [e_1414_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13508)) := by
  have c86805 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86806 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86807 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86808 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86809 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86810 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_0 := arithEq_of_rows h (row := 1415) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_0
  simp only [← c86805, ← c86806, ← c86807] at e_1415_0
  have e_1415_1 := arithEq_of_rows h (row := 1415) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_1
  simp only [← c86808, ← c86809, k2, ← c86810] at e_1415_1
  have hr := e_1415_1
  simp only [e_1415_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13509)) := by
  have c86811 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86812 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86813 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86814 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86815 := (publicBatchWrapper86_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86816 := (publicBatchWrapper86_copies2713 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_2 := arithEq_of_rows h (row := 1415) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_2
  simp only [← c86811, ← c86812, ← c86813] at e_1415_2
  have e_1415_3 := arithEq_of_rows h (row := 1415) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_3
  simp only [← c86814, ← c86815, k2, ← c86816] at e_1415_3
  have hr := e_1415_3
  simp only [e_1415_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13510)) := by
  have c86817 := (publicBatchWrapper86_copies2713 a h).2.1
  have c86818 := (publicBatchWrapper86_copies2713 a h).2.2.1
  have c86819 := (publicBatchWrapper86_copies2713 a h).2.2.2.1
  have c86820 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.1
  have c86821 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.1
  have c86822 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_4 := arithEq_of_rows h (row := 1415) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_4
  simp only [← c86817, ← c86818, ← c86819] at e_1415_4
  have e_1415_5 := arithEq_of_rows h (row := 1415) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_5
  simp only [← c86820, ← c86821, k2, ← c86822] at e_1415_5
  have hr := e_1415_5
  simp only [e_1415_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13511)) := by
  have c86823 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.1
  have c86824 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.1
  have c86825 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.1
  have c86826 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86827 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86828 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_6 := arithEq_of_rows h (row := 1415) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_6
  simp only [← c86823, ← c86824, ← c86825] at e_1415_6
  have e_1415_7 := arithEq_of_rows h (row := 1415) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_7
  simp only [← c86826, ← c86827, k2, ← c86828] at e_1415_7
  have hr := e_1415_7
  simp only [e_1415_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13512)) := by
  have c86829 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86830 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86831 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86832 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86833 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86834 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_8 := arithEq_of_rows h (row := 1415) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_8
  simp only [← c86829, ← c86830, ← c86831] at e_1415_8
  have e_1415_9 := arithEq_of_rows h (row := 1415) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_9
  simp only [← c86832, ← c86833, k2, ← c86834] at e_1415_9
  have hr := e_1415_9
  simp only [e_1415_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13513)) := by
  have c86835 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86836 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86837 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86838 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86839 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86840 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_10 := arithEq_of_rows h (row := 1415) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_10
  simp only [← c86835, ← c86836, ← c86837] at e_1415_10
  have e_1415_11 := arithEq_of_rows h (row := 1415) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_11
  simp only [← c86838, ← c86839, k2, ← c86840] at e_1415_11
  have hr := e_1415_11
  simp only [e_1415_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13514)) := by
  have c86841 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86842 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86843 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86844 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86845 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86846 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_12 := arithEq_of_rows h (row := 1415) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_12
  simp only [← c86841, ← c86842, ← c86843] at e_1415_12
  have e_1415_13 := arithEq_of_rows h (row := 1415) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_13
  simp only [← c86844, ← c86845, k2, ← c86846] at e_1415_13
  have hr := e_1415_13
  simp only [e_1415_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13515)) := by
  have c86847 := (publicBatchWrapper86_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86848 := (publicBatchWrapper86_copies2714 a h).1
  have c86849 := (publicBatchWrapper86_copies2714 a h).2.1
  have c86850 := (publicBatchWrapper86_copies2714 a h).2.2.1
  have c86851 := (publicBatchWrapper86_copies2714 a h).2.2.2.1
  have c86852 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_14 := arithEq_of_rows h (row := 1415) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_14
  simp only [← c86847, ← c86848, ← c86849] at e_1415_14
  have e_1415_15 := arithEq_of_rows h (row := 1415) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_15
  simp only [← c86850, ← c86851, k2, ← c86852] at e_1415_15
  have hr := e_1415_15
  simp only [e_1415_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13516)) := by
  have c86853 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.1
  have c86854 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.1
  have c86855 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.1
  have c86856 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.1
  have c86857 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.1
  have c86858 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_16 := arithEq_of_rows h (row := 1415) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_16
  simp only [← c86853, ← c86854, ← c86855] at e_1415_16
  have e_1415_17 := arithEq_of_rows h (row := 1415) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_17
  simp only [← c86856, ← c86857, k2, ← c86858] at e_1415_17
  have hr := e_1415_17
  simp only [e_1415_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1415 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13517)) := by
  have c86859 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86860 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86861 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86862 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86863 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86864 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1415_18 := arithEq_of_rows h (row := 1415) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_18
  simp only [← c86859, ← c86860, ← c86861] at e_1415_18
  have e_1415_19 := arithEq_of_rows h (row := 1415) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_19
  simp only [← c86862, ← c86863, k2, ← c86864] at e_1415_19
  have hr := e_1415_19
  simp only [e_1415_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13518)) := by
  have c86865 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86866 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86867 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86868 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86869 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86870 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_0 := arithEq_of_rows h (row := 1416) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_0
  simp only [← c86865, ← c86866, ← c86867] at e_1416_0
  have e_1416_1 := arithEq_of_rows h (row := 1416) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_1
  simp only [← c86868, ← c86869, k2, ← c86870] at e_1416_1
  have hr := e_1416_1
  simp only [e_1416_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13519)) := by
  have c86871 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86872 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86873 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86874 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86875 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86876 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_2 := arithEq_of_rows h (row := 1416) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_2
  simp only [← c86871, ← c86872, ← c86873] at e_1416_2
  have e_1416_3 := arithEq_of_rows h (row := 1416) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_3
  simp only [← c86874, ← c86875, k2, ← c86876] at e_1416_3
  have hr := e_1416_3
  simp only [e_1416_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13520)) := by
  have c86877 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86878 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86879 := (publicBatchWrapper86_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86880 := (publicBatchWrapper86_copies2715 a h).1
  have c86881 := (publicBatchWrapper86_copies2715 a h).2.1
  have c86882 := (publicBatchWrapper86_copies2715 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_4 := arithEq_of_rows h (row := 1416) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_4
  simp only [← c86877, ← c86878, ← c86879] at e_1416_4
  have e_1416_5 := arithEq_of_rows h (row := 1416) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_5
  simp only [← c86880, ← c86881, k2, ← c86882] at e_1416_5
  have hr := e_1416_5
  simp only [e_1416_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13521)) := by
  have c86883 := (publicBatchWrapper86_copies2715 a h).2.2.2.1
  have c86884 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.1
  have c86885 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.1
  have c86886 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.1
  have c86887 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.1
  have c86888 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_6 := arithEq_of_rows h (row := 1416) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_6
  simp only [← c86883, ← c86884, ← c86885] at e_1416_6
  have e_1416_7 := arithEq_of_rows h (row := 1416) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_7
  simp only [← c86886, ← c86887, k2, ← c86888] at e_1416_7
  have hr := e_1416_7
  simp only [e_1416_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13522)) := by
  have c86889 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.1
  have c86890 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86891 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86892 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86893 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86894 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_8 := arithEq_of_rows h (row := 1416) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_8
  simp only [← c86889, ← c86890, ← c86891] at e_1416_8
  have e_1416_9 := arithEq_of_rows h (row := 1416) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_9
  simp only [← c86892, ← c86893, k2, ← c86894] at e_1416_9
  have hr := e_1416_9
  simp only [e_1416_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13523)) := by
  have c86895 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86896 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86897 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86898 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86899 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86900 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_10 := arithEq_of_rows h (row := 1416) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_10
  simp only [← c86895, ← c86896, ← c86897] at e_1416_10
  have e_1416_11 := arithEq_of_rows h (row := 1416) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_11
  simp only [← c86898, ← c86899, k2, ← c86900] at e_1416_11
  have hr := e_1416_11
  simp only [e_1416_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13524)) := by
  have c86901 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86902 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86903 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86904 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86905 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86906 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_12 := arithEq_of_rows h (row := 1416) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_12
  simp only [← c86901, ← c86902, ← c86903] at e_1416_12
  have e_1416_13 := arithEq_of_rows h (row := 1416) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_13
  simp only [← c86904, ← c86905, k2, ← c86906] at e_1416_13
  have hr := e_1416_13
  simp only [e_1416_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13525)) := by
  have c86907 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86908 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86909 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86910 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86911 := (publicBatchWrapper86_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86912 := (publicBatchWrapper86_copies2716 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_14 := arithEq_of_rows h (row := 1416) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_14
  simp only [← c86907, ← c86908, ← c86909] at e_1416_14
  have e_1416_15 := arithEq_of_rows h (row := 1416) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_15
  simp only [← c86910, ← c86911, k2, ← c86912] at e_1416_15
  have hr := e_1416_15
  simp only [e_1416_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13526)) := by
  have c86913 := (publicBatchWrapper86_copies2716 a h).2.1
  have c86914 := (publicBatchWrapper86_copies2716 a h).2.2.1
  have c86915 := (publicBatchWrapper86_copies2716 a h).2.2.2.1
  have c86916 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.1
  have c86917 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.1
  have c86918 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_16 := arithEq_of_rows h (row := 1416) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_16
  simp only [← c86913, ← c86914, ← c86915] at e_1416_16
  have e_1416_17 := arithEq_of_rows h (row := 1416) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_17
  simp only [← c86916, ← c86917, k2, ← c86918] at e_1416_17
  have hr := e_1416_17
  simp only [e_1416_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1416 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13527)) := by
  have c86919 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.1
  have c86920 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.1
  have c86921 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.1
  have c86922 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86923 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86924 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1416_18 := arithEq_of_rows h (row := 1416) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_18
  simp only [← c86919, ← c86920, ← c86921] at e_1416_18
  have e_1416_19 := arithEq_of_rows h (row := 1416) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_19
  simp only [← c86922, ← c86923, k2, ← c86924] at e_1416_19
  have hr := e_1416_19
  simp only [e_1416_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13528)) := by
  have c86925 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86926 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86927 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86928 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86929 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86930 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_0 := arithEq_of_rows h (row := 1417) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_0
  simp only [← c86925, ← c86926, ← c86927] at e_1417_0
  have e_1417_1 := arithEq_of_rows h (row := 1417) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_1
  simp only [← c86928, ← c86929, k2, ← c86930] at e_1417_1
  have hr := e_1417_1
  simp only [e_1417_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13529)) := by
  have c86931 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86932 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86933 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86934 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86935 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86936 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_2 := arithEq_of_rows h (row := 1417) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_2
  simp only [← c86931, ← c86932, ← c86933] at e_1417_2
  have e_1417_3 := arithEq_of_rows h (row := 1417) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_3
  simp only [← c86934, ← c86935, k2, ← c86936] at e_1417_3
  have hr := e_1417_3
  simp only [e_1417_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13530)) := by
  have c86937 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86938 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86939 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86940 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86941 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86942 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_4 := arithEq_of_rows h (row := 1417) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_4
  simp only [← c86937, ← c86938, ← c86939] at e_1417_4
  have e_1417_5 := arithEq_of_rows h (row := 1417) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_5
  simp only [← c86940, ← c86941, k2, ← c86942] at e_1417_5
  have hr := e_1417_5
  simp only [e_1417_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13531)) := by
  have c86943 := (publicBatchWrapper86_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86944 := (publicBatchWrapper86_copies2717 a h).1
  have c86945 := (publicBatchWrapper86_copies2717 a h).2.1
  have c86946 := (publicBatchWrapper86_copies2717 a h).2.2.1
  have c86947 := (publicBatchWrapper86_copies2717 a h).2.2.2.1
  have c86948 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_6 := arithEq_of_rows h (row := 1417) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_6
  simp only [← c86943, ← c86944, ← c86945] at e_1417_6
  have e_1417_7 := arithEq_of_rows h (row := 1417) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_7
  simp only [← c86946, ← c86947, k2, ← c86948] at e_1417_7
  have hr := e_1417_7
  simp only [e_1417_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13532)) := by
  have c86949 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.1
  have c86950 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.1
  have c86951 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.1
  have c86952 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.1
  have c86953 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.1
  have c86954 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_8 := arithEq_of_rows h (row := 1417) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_8
  simp only [← c86949, ← c86950, ← c86951] at e_1417_8
  have e_1417_9 := arithEq_of_rows h (row := 1417) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_9
  simp only [← c86952, ← c86953, k2, ← c86954] at e_1417_9
  have hr := e_1417_9
  simp only [e_1417_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13533)) := by
  have c86955 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86956 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86957 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86958 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86959 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86960 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_10 := arithEq_of_rows h (row := 1417) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_10
  simp only [← c86955, ← c86956, ← c86957] at e_1417_10
  have e_1417_11 := arithEq_of_rows h (row := 1417) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_11
  simp only [← c86958, ← c86959, k2, ← c86960] at e_1417_11
  have hr := e_1417_11
  simp only [e_1417_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13534)) := by
  have c86961 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86962 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86963 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86964 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86965 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86966 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_12 := arithEq_of_rows h (row := 1417) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_12
  simp only [← c86961, ← c86962, ← c86963] at e_1417_12
  have e_1417_13 := arithEq_of_rows h (row := 1417) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_13
  simp only [← c86964, ← c86965, k2, ← c86966] at e_1417_13
  have hr := e_1417_13
  simp only [e_1417_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13535)) := by
  have c86967 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86968 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86969 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86970 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86971 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86972 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_14 := arithEq_of_rows h (row := 1417) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_14
  simp only [← c86967, ← c86968, ← c86969] at e_1417_14
  have e_1417_15 := arithEq_of_rows h (row := 1417) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_15
  simp only [← c86970, ← c86971, k2, ← c86972] at e_1417_15
  have hr := e_1417_15
  simp only [e_1417_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13536)) := by
  have c86973 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86974 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86975 := (publicBatchWrapper86_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86976 := (publicBatchWrapper86_copies2718 a h).1
  have c86977 := (publicBatchWrapper86_copies2718 a h).2.1
  have c86978 := (publicBatchWrapper86_copies2718 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_16 := arithEq_of_rows h (row := 1417) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_16
  simp only [← c86973, ← c86974, ← c86975] at e_1417_16
  have e_1417_17 := arithEq_of_rows h (row := 1417) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_17
  simp only [← c86976, ← c86977, k2, ← c86978] at e_1417_17
  have hr := e_1417_17
  simp only [e_1417_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1417 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13537)) := by
  have c86979 := (publicBatchWrapper86_copies2718 a h).2.2.2.1
  have c86980 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.1
  have c86981 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.1
  have c86982 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.1
  have c86983 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.1
  have c86984 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1417_18 := arithEq_of_rows h (row := 1417) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_18
  simp only [← c86979, ← c86980, ← c86981] at e_1417_18
  have e_1417_19 := arithEq_of_rows h (row := 1417) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_19
  simp only [← c86982, ← c86983, k2, ← c86984] at e_1417_19
  have hr := e_1417_19
  simp only [e_1417_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13538)) := by
  have c86985 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.1
  have c86986 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86987 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86988 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86989 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86990 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_0 := arithEq_of_rows h (row := 1418) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_0
  simp only [← c86985, ← c86986, ← c86987] at e_1418_0
  have e_1418_1 := arithEq_of_rows h (row := 1418) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_1
  simp only [← c86988, ← c86989, k2, ← c86990] at e_1418_1
  have hr := e_1418_1
  simp only [e_1418_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13539)) := by
  have c86991 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86992 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86993 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86994 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86995 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86996 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_2 := arithEq_of_rows h (row := 1418) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_2
  simp only [← c86991, ← c86992, ← c86993] at e_1418_2
  have e_1418_3 := arithEq_of_rows h (row := 1418) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_3
  simp only [← c86994, ← c86995, k2, ← c86996] at e_1418_3
  have hr := e_1418_3
  simp only [e_1418_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13540)) := by
  have c86997 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86998 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86999 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87000 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87001 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87002 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_4 := arithEq_of_rows h (row := 1418) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_4
  simp only [← c86997, ← c86998, ← c86999] at e_1418_4
  have e_1418_5 := arithEq_of_rows h (row := 1418) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_5
  simp only [← c87000, ← c87001, k2, ← c87002] at e_1418_5
  have hr := e_1418_5
  simp only [e_1418_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13541)) := by
  have c87003 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87004 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87005 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87006 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87007 := (publicBatchWrapper86_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87008 := (publicBatchWrapper86_copies2719 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_6 := arithEq_of_rows h (row := 1418) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_6
  simp only [← c87003, ← c87004, ← c87005] at e_1418_6
  have e_1418_7 := arithEq_of_rows h (row := 1418) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_7
  simp only [← c87006, ← c87007, k2, ← c87008] at e_1418_7
  have hr := e_1418_7
  simp only [e_1418_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13542)) := by
  have c87009 := (publicBatchWrapper86_copies2719 a h).2.1
  have c87010 := (publicBatchWrapper86_copies2719 a h).2.2.1
  have c87011 := (publicBatchWrapper86_copies2719 a h).2.2.2.1
  have c87012 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.1
  have c87013 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.1
  have c87014 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_8 := arithEq_of_rows h (row := 1418) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_8
  simp only [← c87009, ← c87010, ← c87011] at e_1418_8
  have e_1418_9 := arithEq_of_rows h (row := 1418) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_9
  simp only [← c87012, ← c87013, k2, ← c87014] at e_1418_9
  have hr := e_1418_9
  simp only [e_1418_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13543)) := by
  have c87015 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.1
  have c87016 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.1
  have c87017 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.1
  have c87018 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87019 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87020 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_10 := arithEq_of_rows h (row := 1418) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_10
  simp only [← c87015, ← c87016, ← c87017] at e_1418_10
  have e_1418_11 := arithEq_of_rows h (row := 1418) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_11
  simp only [← c87018, ← c87019, k2, ← c87020] at e_1418_11
  have hr := e_1418_11
  simp only [e_1418_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13544)) := by
  have c87021 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87022 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87023 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87024 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87025 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87026 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_12 := arithEq_of_rows h (row := 1418) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_12
  simp only [← c87021, ← c87022, ← c87023] at e_1418_12
  have e_1418_13 := arithEq_of_rows h (row := 1418) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_13
  simp only [← c87024, ← c87025, k2, ← c87026] at e_1418_13
  have hr := e_1418_13
  simp only [e_1418_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13545)) := by
  have c87027 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87028 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87029 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87030 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87031 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87032 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_14 := arithEq_of_rows h (row := 1418) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_14
  simp only [← c87027, ← c87028, ← c87029] at e_1418_14
  have e_1418_15 := arithEq_of_rows h (row := 1418) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_15
  simp only [← c87030, ← c87031, k2, ← c87032] at e_1418_15
  have hr := e_1418_15
  simp only [e_1418_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13546)) := by
  have c87033 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87034 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87035 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87036 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87037 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87038 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_16 := arithEq_of_rows h (row := 1418) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_16
  simp only [← c87033, ← c87034, ← c87035] at e_1418_16
  have e_1418_17 := arithEq_of_rows h (row := 1418) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_17
  simp only [← c87036, ← c87037, k2, ← c87038] at e_1418_17
  have hr := e_1418_17
  simp only [e_1418_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1418 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13547)) := by
  have c87039 := (publicBatchWrapper86_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87040 := (publicBatchWrapper86_copies2720 a h).1
  have c87041 := (publicBatchWrapper86_copies2720 a h).2.1
  have c87042 := (publicBatchWrapper86_copies2720 a h).2.2.1
  have c87043 := (publicBatchWrapper86_copies2720 a h).2.2.2.1
  have c87044 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1418_18 := arithEq_of_rows h (row := 1418) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_18
  simp only [← c87039, ← c87040, ← c87041] at e_1418_18
  have e_1418_19 := arithEq_of_rows h (row := 1418) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_19
  simp only [← c87042, ← c87043, k2, ← c87044] at e_1418_19
  have hr := e_1418_19
  simp only [e_1418_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13548)) := by
  have c87045 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.1
  have c87046 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.1
  have c87047 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.1
  have c87048 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.1
  have c87049 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.1
  have c87050 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_0 := arithEq_of_rows h (row := 1419) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_0
  simp only [← c87045, ← c87046, ← c87047] at e_1419_0
  have e_1419_1 := arithEq_of_rows h (row := 1419) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_1
  simp only [← c87048, ← c87049, k2, ← c87050] at e_1419_1
  have hr := e_1419_1
  simp only [e_1419_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13549)) := by
  have c87051 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87052 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87053 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87054 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87055 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87056 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_2 := arithEq_of_rows h (row := 1419) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_2
  simp only [← c87051, ← c87052, ← c87053] at e_1419_2
  have e_1419_3 := arithEq_of_rows h (row := 1419) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_3
  simp only [← c87054, ← c87055, k2, ← c87056] at e_1419_3
  have hr := e_1419_3
  simp only [e_1419_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13550)) := by
  have c87057 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87058 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87059 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87060 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87061 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87062 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_4 := arithEq_of_rows h (row := 1419) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_4
  simp only [← c87057, ← c87058, ← c87059] at e_1419_4
  have e_1419_5 := arithEq_of_rows h (row := 1419) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_5
  simp only [← c87060, ← c87061, k2, ← c87062] at e_1419_5
  have hr := e_1419_5
  simp only [e_1419_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13551)) := by
  have c87063 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87064 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87065 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87066 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87067 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87068 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_6 := arithEq_of_rows h (row := 1419) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_6
  simp only [← c87063, ← c87064, ← c87065] at e_1419_6
  have e_1419_7 := arithEq_of_rows h (row := 1419) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_7
  simp only [← c87066, ← c87067, k2, ← c87068] at e_1419_7
  have hr := e_1419_7
  simp only [e_1419_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13552)) := by
  have c87069 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87070 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87071 := (publicBatchWrapper86_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87072 := (publicBatchWrapper86_copies2721 a h).1
  have c87073 := (publicBatchWrapper86_copies2721 a h).2.1
  have c87074 := (publicBatchWrapper86_copies2721 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_8 := arithEq_of_rows h (row := 1419) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_8
  simp only [← c87069, ← c87070, ← c87071] at e_1419_8
  have e_1419_9 := arithEq_of_rows h (row := 1419) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_9
  simp only [← c87072, ← c87073, k2, ← c87074] at e_1419_9
  have hr := e_1419_9
  simp only [e_1419_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13553)) := by
  have c87075 := (publicBatchWrapper86_copies2721 a h).2.2.2.1
  have c87076 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.1
  have c87077 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.1
  have c87078 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.1
  have c87079 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.1
  have c87080 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_10 := arithEq_of_rows h (row := 1419) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_10
  simp only [← c87075, ← c87076, ← c87077] at e_1419_10
  have e_1419_11 := arithEq_of_rows h (row := 1419) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_11
  simp only [← c87078, ← c87079, k2, ← c87080] at e_1419_11
  have hr := e_1419_11
  simp only [e_1419_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13554)) := by
  have c87081 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.1
  have c87082 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87083 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87084 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87085 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87086 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_12 := arithEq_of_rows h (row := 1419) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_12
  simp only [← c87081, ← c87082, ← c87083] at e_1419_12
  have e_1419_13 := arithEq_of_rows h (row := 1419) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_13
  simp only [← c87084, ← c87085, k2, ← c87086] at e_1419_13
  have hr := e_1419_13
  simp only [e_1419_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13555)) := by
  have c87087 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87088 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87089 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87090 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87091 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87092 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_14 := arithEq_of_rows h (row := 1419) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_14
  simp only [← c87087, ← c87088, ← c87089] at e_1419_14
  have e_1419_15 := arithEq_of_rows h (row := 1419) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_15
  simp only [← c87090, ← c87091, k2, ← c87092] at e_1419_15
  have hr := e_1419_15
  simp only [e_1419_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13556)) := by
  have c87093 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87094 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87095 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87096 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87097 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87098 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_16 := arithEq_of_rows h (row := 1419) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_16
  simp only [← c87093, ← c87094, ← c87095] at e_1419_16
  have e_1419_17 := arithEq_of_rows h (row := 1419) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_17
  simp only [← c87096, ← c87097, k2, ← c87098] at e_1419_17
  have hr := e_1419_17
  simp only [e_1419_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1419 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13557)) := by
  have c87099 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87100 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87101 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87102 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87103 := (publicBatchWrapper86_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87104 := (publicBatchWrapper86_copies2722 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1419_18 := arithEq_of_rows h (row := 1419) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_18
  simp only [← c87099, ← c87100, ← c87101] at e_1419_18
  have e_1419_19 := arithEq_of_rows h (row := 1419) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_19
  simp only [← c87102, ← c87103, k2, ← c87104] at e_1419_19
  have hr := e_1419_19
  simp only [e_1419_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13558)) := by
  have c87105 := (publicBatchWrapper86_copies2722 a h).2.1
  have c87106 := (publicBatchWrapper86_copies2722 a h).2.2.1
  have c87107 := (publicBatchWrapper86_copies2722 a h).2.2.2.1
  have c87108 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.1
  have c87109 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.1
  have c87110 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_0 := arithEq_of_rows h (row := 1420) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_0
  simp only [← c87105, ← c87106, ← c87107] at e_1420_0
  have e_1420_1 := arithEq_of_rows h (row := 1420) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_1
  simp only [← c87108, ← c87109, k2, ← c87110] at e_1420_1
  have hr := e_1420_1
  simp only [e_1420_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13559)) := by
  have c87111 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.1
  have c87112 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.1
  have c87113 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.1
  have c87114 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87115 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87116 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_2 := arithEq_of_rows h (row := 1420) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_2
  simp only [← c87111, ← c87112, ← c87113] at e_1420_2
  have e_1420_3 := arithEq_of_rows h (row := 1420) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_3
  simp only [← c87114, ← c87115, k2, ← c87116] at e_1420_3
  have hr := e_1420_3
  simp only [e_1420_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13560)) := by
  have c87117 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87118 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87119 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87120 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87121 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87122 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_4 := arithEq_of_rows h (row := 1420) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_4
  simp only [← c87117, ← c87118, ← c87119] at e_1420_4
  have e_1420_5 := arithEq_of_rows h (row := 1420) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_5
  simp only [← c87120, ← c87121, k2, ← c87122] at e_1420_5
  have hr := e_1420_5
  simp only [e_1420_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13561)) := by
  have c87123 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87124 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87125 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87126 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87127 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87128 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_6 := arithEq_of_rows h (row := 1420) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_6
  simp only [← c87123, ← c87124, ← c87125] at e_1420_6
  have e_1420_7 := arithEq_of_rows h (row := 1420) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_7
  simp only [← c87126, ← c87127, k2, ← c87128] at e_1420_7
  have hr := e_1420_7
  simp only [e_1420_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13562)) := by
  have c87129 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87130 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87131 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87132 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87133 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87134 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_8 := arithEq_of_rows h (row := 1420) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_8
  simp only [← c87129, ← c87130, ← c87131] at e_1420_8
  have e_1420_9 := arithEq_of_rows h (row := 1420) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_9
  simp only [← c87132, ← c87133, k2, ← c87134] at e_1420_9
  have hr := e_1420_9
  simp only [e_1420_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13563)) := by
  have c87135 := (publicBatchWrapper86_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87136 := (publicBatchWrapper86_copies2723 a h).1
  have c87137 := (publicBatchWrapper86_copies2723 a h).2.1
  have c87138 := (publicBatchWrapper86_copies2723 a h).2.2.1
  have c87139 := (publicBatchWrapper86_copies2723 a h).2.2.2.1
  have c87140 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_10 := arithEq_of_rows h (row := 1420) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_10
  simp only [← c87135, ← c87136, ← c87137] at e_1420_10
  have e_1420_11 := arithEq_of_rows h (row := 1420) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_11
  simp only [← c87138, ← c87139, k2, ← c87140] at e_1420_11
  have hr := e_1420_11
  simp only [e_1420_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13564)) := by
  have c87141 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.1
  have c87142 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.1
  have c87143 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.1
  have c87144 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.1
  have c87145 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.1
  have c87146 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_12 := arithEq_of_rows h (row := 1420) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_12
  simp only [← c87141, ← c87142, ← c87143] at e_1420_12
  have e_1420_13 := arithEq_of_rows h (row := 1420) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_13
  simp only [← c87144, ← c87145, k2, ← c87146] at e_1420_13
  have hr := e_1420_13
  simp only [e_1420_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13565)) := by
  have c87147 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87148 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87149 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87150 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87151 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87152 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_14 := arithEq_of_rows h (row := 1420) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_14
  simp only [← c87147, ← c87148, ← c87149] at e_1420_14
  have e_1420_15 := arithEq_of_rows h (row := 1420) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_15
  simp only [← c87150, ← c87151, k2, ← c87152] at e_1420_15
  have hr := e_1420_15
  simp only [e_1420_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13566)) := by
  have c87153 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87154 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87155 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87156 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87157 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87158 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_16 := arithEq_of_rows h (row := 1420) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_16
  simp only [← c87153, ← c87154, ← c87155] at e_1420_16
  have e_1420_17 := arithEq_of_rows h (row := 1420) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_17
  simp only [← c87156, ← c87157, k2, ← c87158] at e_1420_17
  have hr := e_1420_17
  simp only [e_1420_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1420 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13567)) := by
  have c87159 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87160 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87161 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87162 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87163 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87164 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1420_18 := arithEq_of_rows h (row := 1420) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_18
  simp only [← c87159, ← c87160, ← c87161] at e_1420_18
  have e_1420_19 := arithEq_of_rows h (row := 1420) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_19
  simp only [← c87162, ← c87163, k2, ← c87164] at e_1420_19
  have hr := e_1420_19
  simp only [e_1420_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13568)) := by
  have c87165 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87166 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87167 := (publicBatchWrapper86_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87168 := (publicBatchWrapper86_copies2724 a h).1
  have c87169 := (publicBatchWrapper86_copies2724 a h).2.1
  have c87170 := (publicBatchWrapper86_copies2724 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_0 := arithEq_of_rows h (row := 1421) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_0
  simp only [← c87165, ← c87166, ← c87167] at e_1421_0
  have e_1421_1 := arithEq_of_rows h (row := 1421) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_1
  simp only [← c87168, ← c87169, k2, ← c87170] at e_1421_1
  have hr := e_1421_1
  simp only [e_1421_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13569)) := by
  have c87171 := (publicBatchWrapper86_copies2724 a h).2.2.2.1
  have c87172 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.1
  have c87173 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.1
  have c87174 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.1
  have c87175 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.1
  have c87176 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_2 := arithEq_of_rows h (row := 1421) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_2
  simp only [← c87171, ← c87172, ← c87173] at e_1421_2
  have e_1421_3 := arithEq_of_rows h (row := 1421) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_3
  simp only [← c87174, ← c87175, k2, ← c87176] at e_1421_3
  have hr := e_1421_3
  simp only [e_1421_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13610)) := by
  have c87177 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.1
  have c87178 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87179 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87180 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87181 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87182 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_4 := arithEq_of_rows h (row := 1421) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_4
  simp only [← c87177, ← c87178, ← c87179] at e_1421_4
  have e_1421_5 := arithEq_of_rows h (row := 1421) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_5
  simp only [← c87180, ← c87181, k2, ← c87182] at e_1421_5
  have hr := e_1421_5
  simp only [e_1421_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13611)) := by
  have c87183 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87184 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87185 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87186 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87187 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87188 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_6 := arithEq_of_rows h (row := 1421) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_6
  simp only [← c87183, ← c87184, ← c87185] at e_1421_6
  have e_1421_7 := arithEq_of_rows h (row := 1421) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_7
  simp only [← c87186, ← c87187, k2, ← c87188] at e_1421_7
  have hr := e_1421_7
  simp only [e_1421_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13612)) := by
  have c87189 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87190 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87191 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87192 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87193 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87194 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_8 := arithEq_of_rows h (row := 1421) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_8
  simp only [← c87189, ← c87190, ← c87191] at e_1421_8
  have e_1421_9 := arithEq_of_rows h (row := 1421) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_9
  simp only [← c87192, ← c87193, k2, ← c87194] at e_1421_9
  have hr := e_1421_9
  simp only [e_1421_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13613)) := by
  have c87195 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87196 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87197 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87198 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87199 := (publicBatchWrapper86_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87200 := (publicBatchWrapper86_copies2725 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_10 := arithEq_of_rows h (row := 1421) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_10
  simp only [← c87195, ← c87196, ← c87197] at e_1421_10
  have e_1421_11 := arithEq_of_rows h (row := 1421) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_11
  simp only [← c87198, ← c87199, k2, ← c87200] at e_1421_11
  have hr := e_1421_11
  simp only [e_1421_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13614)) := by
  have c87201 := (publicBatchWrapper86_copies2725 a h).2.1
  have c87202 := (publicBatchWrapper86_copies2725 a h).2.2.1
  have c87203 := (publicBatchWrapper86_copies2725 a h).2.2.2.1
  have c87204 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.1
  have c87205 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.1
  have c87206 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_12 := arithEq_of_rows h (row := 1421) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_12
  simp only [← c87201, ← c87202, ← c87203] at e_1421_12
  have e_1421_13 := arithEq_of_rows h (row := 1421) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_13
  simp only [← c87204, ← c87205, k2, ← c87206] at e_1421_13
  have hr := e_1421_13
  simp only [e_1421_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13615)) := by
  have c87207 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.1
  have c87208 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.1
  have c87209 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.1
  have c87210 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87211 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87212 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_14 := arithEq_of_rows h (row := 1421) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_14
  simp only [← c87207, ← c87208, ← c87209] at e_1421_14
  have e_1421_15 := arithEq_of_rows h (row := 1421) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_15
  simp only [← c87210, ← c87211, k2, ← c87212] at e_1421_15
  have hr := e_1421_15
  simp only [e_1421_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13616)) := by
  have c87213 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87214 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87215 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87216 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87217 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87218 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_16 := arithEq_of_rows h (row := 1421) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_16
  simp only [← c87213, ← c87214, ← c87215] at e_1421_16
  have e_1421_17 := arithEq_of_rows h (row := 1421) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_17
  simp only [← c87216, ← c87217, k2, ← c87218] at e_1421_17
  have hr := e_1421_17
  simp only [e_1421_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1421 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13617)) := by
  have c87219 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87220 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87221 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87222 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87223 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87224 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1421_18 := arithEq_of_rows h (row := 1421) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_18
  simp only [← c87219, ← c87220, ← c87221] at e_1421_18
  have e_1421_19 := arithEq_of_rows h (row := 1421) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_19
  simp only [← c87222, ← c87223, k2, ← c87224] at e_1421_19
  have hr := e_1421_19
  simp only [e_1421_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13618)) := by
  have c87225 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87226 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87227 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87228 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87229 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87230 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_0 := arithEq_of_rows h (row := 1422) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_0
  simp only [← c87225, ← c87226, ← c87227] at e_1422_0
  have e_1422_1 := arithEq_of_rows h (row := 1422) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_1
  simp only [← c87228, ← c87229, k2, ← c87230] at e_1422_1
  have hr := e_1422_1
  simp only [e_1422_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13619)) := by
  have c87231 := (publicBatchWrapper86_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87232 := (publicBatchWrapper86_copies2726 a h).1
  have c87233 := (publicBatchWrapper86_copies2726 a h).2.1
  have c87234 := (publicBatchWrapper86_copies2726 a h).2.2.1
  have c87235 := (publicBatchWrapper86_copies2726 a h).2.2.2.1
  have c87236 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_2 := arithEq_of_rows h (row := 1422) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_2
  simp only [← c87231, ← c87232, ← c87233] at e_1422_2
  have e_1422_3 := arithEq_of_rows h (row := 1422) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_3
  simp only [← c87234, ← c87235, k2, ← c87236] at e_1422_3
  have hr := e_1422_3
  simp only [e_1422_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13620)) := by
  have c87237 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.1
  have c87238 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.1
  have c87239 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.1
  have c87240 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.1
  have c87241 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.1
  have c87242 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_4 := arithEq_of_rows h (row := 1422) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_4
  simp only [← c87237, ← c87238, ← c87239] at e_1422_4
  have e_1422_5 := arithEq_of_rows h (row := 1422) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_5
  simp only [← c87240, ← c87241, k2, ← c87242] at e_1422_5
  have hr := e_1422_5
  simp only [e_1422_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13621)) := by
  have c87243 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87244 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87245 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87246 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87247 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87248 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_6 := arithEq_of_rows h (row := 1422) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_6
  simp only [← c87243, ← c87244, ← c87245] at e_1422_6
  have e_1422_7 := arithEq_of_rows h (row := 1422) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_7
  simp only [← c87246, ← c87247, k2, ← c87248] at e_1422_7
  have hr := e_1422_7
  simp only [e_1422_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13622)) := by
  have c87249 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87250 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87251 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87252 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87253 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87254 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_8 := arithEq_of_rows h (row := 1422) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_8
  simp only [← c87249, ← c87250, ← c87251] at e_1422_8
  have e_1422_9 := arithEq_of_rows h (row := 1422) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_9
  simp only [← c87252, ← c87253, k2, ← c87254] at e_1422_9
  have hr := e_1422_9
  simp only [e_1422_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13623)) := by
  have c87255 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87256 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87257 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87258 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87259 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87260 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_10 := arithEq_of_rows h (row := 1422) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_10
  simp only [← c87255, ← c87256, ← c87257] at e_1422_10
  have e_1422_11 := arithEq_of_rows h (row := 1422) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_11
  simp only [← c87258, ← c87259, k2, ← c87260] at e_1422_11
  have hr := e_1422_11
  simp only [e_1422_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13624)) := by
  have c87261 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87262 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87263 := (publicBatchWrapper86_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87264 := (publicBatchWrapper86_copies2727 a h).1
  have c87265 := (publicBatchWrapper86_copies2727 a h).2.1
  have c87266 := (publicBatchWrapper86_copies2727 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_12 := arithEq_of_rows h (row := 1422) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_12
  simp only [← c87261, ← c87262, ← c87263] at e_1422_12
  have e_1422_13 := arithEq_of_rows h (row := 1422) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_13
  simp only [← c87264, ← c87265, k2, ← c87266] at e_1422_13
  have hr := e_1422_13
  simp only [e_1422_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13625)) := by
  have c87267 := (publicBatchWrapper86_copies2727 a h).2.2.2.1
  have c87268 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.1
  have c87269 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.1
  have c87270 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.1
  have c87271 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.1
  have c87272 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_14 := arithEq_of_rows h (row := 1422) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_14
  simp only [← c87267, ← c87268, ← c87269] at e_1422_14
  have e_1422_15 := arithEq_of_rows h (row := 1422) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_15
  simp only [← c87270, ← c87271, k2, ← c87272] at e_1422_15
  have hr := e_1422_15
  simp only [e_1422_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13626)) := by
  have c87273 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.1
  have c87274 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87275 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87276 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87277 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87278 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_16 := arithEq_of_rows h (row := 1422) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_16
  simp only [← c87273, ← c87274, ← c87275] at e_1422_16
  have e_1422_17 := arithEq_of_rows h (row := 1422) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_17
  simp only [← c87276, ← c87277, k2, ← c87278] at e_1422_17
  have hr := e_1422_17
  simp only [e_1422_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1422 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13627)) := by
  have c87279 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87280 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87281 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87282 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87283 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87284 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1422_18 := arithEq_of_rows h (row := 1422) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_18
  simp only [← c87279, ← c87280, ← c87281] at e_1422_18
  have e_1422_19 := arithEq_of_rows h (row := 1422) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_19
  simp only [← c87282, ← c87283, k2, ← c87284] at e_1422_19
  have hr := e_1422_19
  simp only [e_1422_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13628)) := by
  have c87285 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87286 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87287 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87288 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87289 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87290 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_0 := arithEq_of_rows h (row := 1423) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_0
  simp only [← c87285, ← c87286, ← c87287] at e_1423_0
  have e_1423_1 := arithEq_of_rows h (row := 1423) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_1
  simp only [← c87288, ← c87289, k2, ← c87290] at e_1423_1
  have hr := e_1423_1
  simp only [e_1423_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13629)) := by
  have c87291 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87292 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87293 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87294 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87295 := (publicBatchWrapper86_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87296 := (publicBatchWrapper86_copies2728 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_2 := arithEq_of_rows h (row := 1423) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_2
  simp only [← c87291, ← c87292, ← c87293] at e_1423_2
  have e_1423_3 := arithEq_of_rows h (row := 1423) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_3
  simp only [← c87294, ← c87295, k2, ← c87296] at e_1423_3
  have hr := e_1423_3
  simp only [e_1423_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13630)) := by
  have c87297 := (publicBatchWrapper86_copies2728 a h).2.1
  have c87298 := (publicBatchWrapper86_copies2728 a h).2.2.1
  have c87299 := (publicBatchWrapper86_copies2728 a h).2.2.2.1
  have c87300 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.1
  have c87301 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.1
  have c87302 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_4 := arithEq_of_rows h (row := 1423) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_4
  simp only [← c87297, ← c87298, ← c87299] at e_1423_4
  have e_1423_5 := arithEq_of_rows h (row := 1423) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_5
  simp only [← c87300, ← c87301, k2, ← c87302] at e_1423_5
  have hr := e_1423_5
  simp only [e_1423_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13631)) := by
  have c87303 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.1
  have c87304 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.1
  have c87305 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.1
  have c87306 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87307 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87308 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_6 := arithEq_of_rows h (row := 1423) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_6
  simp only [← c87303, ← c87304, ← c87305] at e_1423_6
  have e_1423_7 := arithEq_of_rows h (row := 1423) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_7
  simp only [← c87306, ← c87307, k2, ← c87308] at e_1423_7
  have hr := e_1423_7
  simp only [e_1423_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13632)) := by
  have c87309 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87310 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87311 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87312 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87313 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87314 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_8 := arithEq_of_rows h (row := 1423) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_8
  simp only [← c87309, ← c87310, ← c87311] at e_1423_8
  have e_1423_9 := arithEq_of_rows h (row := 1423) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_9
  simp only [← c87312, ← c87313, k2, ← c87314] at e_1423_9
  have hr := e_1423_9
  simp only [e_1423_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13633)) := by
  have c87315 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87316 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87317 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87318 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87319 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87320 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_10 := arithEq_of_rows h (row := 1423) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_10
  simp only [← c87315, ← c87316, ← c87317] at e_1423_10
  have e_1423_11 := arithEq_of_rows h (row := 1423) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_11
  simp only [← c87318, ← c87319, k2, ← c87320] at e_1423_11
  have hr := e_1423_11
  simp only [e_1423_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13634)) := by
  have c87321 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87322 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87323 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87324 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87325 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87326 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_12 := arithEq_of_rows h (row := 1423) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_12
  simp only [← c87321, ← c87322, ← c87323] at e_1423_12
  have e_1423_13 := arithEq_of_rows h (row := 1423) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_13
  simp only [← c87324, ← c87325, k2, ← c87326] at e_1423_13
  have hr := e_1423_13
  simp only [e_1423_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13635)) := by
  have c87327 := (publicBatchWrapper86_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87328 := (publicBatchWrapper86_copies2729 a h).1
  have c87329 := (publicBatchWrapper86_copies2729 a h).2.1
  have c87330 := (publicBatchWrapper86_copies2729 a h).2.2.1
  have c87331 := (publicBatchWrapper86_copies2729 a h).2.2.2.1
  have c87332 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_14 := arithEq_of_rows h (row := 1423) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_14
  simp only [← c87327, ← c87328, ← c87329] at e_1423_14
  have e_1423_15 := arithEq_of_rows h (row := 1423) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_15
  simp only [← c87330, ← c87331, k2, ← c87332] at e_1423_15
  have hr := e_1423_15
  simp only [e_1423_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13636)) := by
  have c87333 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.1
  have c87334 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.1
  have c87335 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.1
  have c87336 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.1
  have c87337 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.1
  have c87338 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_16 := arithEq_of_rows h (row := 1423) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_16
  simp only [← c87333, ← c87334, ← c87335] at e_1423_16
  have e_1423_17 := arithEq_of_rows h (row := 1423) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_17
  simp only [← c87336, ← c87337, k2, ← c87338] at e_1423_17
  have hr := e_1423_17
  simp only [e_1423_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1423 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13637)) := by
  have c87339 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87340 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87341 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87342 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87343 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87344 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1423_18 := arithEq_of_rows h (row := 1423) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_18
  simp only [← c87339, ← c87340, ← c87341] at e_1423_18
  have e_1423_19 := arithEq_of_rows h (row := 1423) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_19
  simp only [← c87342, ← c87343, k2, ← c87344] at e_1423_19
  have hr := e_1423_19
  simp only [e_1423_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13638)) := by
  have c87345 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87346 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87347 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87348 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87349 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87350 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_0 := arithEq_of_rows h (row := 1424) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_0
  simp only [← c87345, ← c87346, ← c87347] at e_1424_0
  have e_1424_1 := arithEq_of_rows h (row := 1424) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_1
  simp only [← c87348, ← c87349, k2, ← c87350] at e_1424_1
  have hr := e_1424_1
  simp only [e_1424_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13639)) := by
  have c87351 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87352 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87353 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87354 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87355 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87356 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_2 := arithEq_of_rows h (row := 1424) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_2
  simp only [← c87351, ← c87352, ← c87353] at e_1424_2
  have e_1424_3 := arithEq_of_rows h (row := 1424) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_3
  simp only [← c87354, ← c87355, k2, ← c87356] at e_1424_3
  have hr := e_1424_3
  simp only [e_1424_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13640)) := by
  have c87357 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87358 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87359 := (publicBatchWrapper86_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87360 := (publicBatchWrapper86_copies2730 a h).1
  have c87361 := (publicBatchWrapper86_copies2730 a h).2.1
  have c87362 := (publicBatchWrapper86_copies2730 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_4 := arithEq_of_rows h (row := 1424) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_4
  simp only [← c87357, ← c87358, ← c87359] at e_1424_4
  have e_1424_5 := arithEq_of_rows h (row := 1424) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_5
  simp only [← c87360, ← c87361, k2, ← c87362] at e_1424_5
  have hr := e_1424_5
  simp only [e_1424_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13641)) := by
  have c87363 := (publicBatchWrapper86_copies2730 a h).2.2.2.1
  have c87364 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.1
  have c87365 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.1
  have c87366 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.1
  have c87367 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.1
  have c87368 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_6 := arithEq_of_rows h (row := 1424) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_6
  simp only [← c87363, ← c87364, ← c87365] at e_1424_6
  have e_1424_7 := arithEq_of_rows h (row := 1424) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_7
  simp only [← c87366, ← c87367, k2, ← c87368] at e_1424_7
  have hr := e_1424_7
  simp only [e_1424_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13642)) := by
  have c87369 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.1
  have c87370 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87371 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87372 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87373 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87374 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_8 := arithEq_of_rows h (row := 1424) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_8
  simp only [← c87369, ← c87370, ← c87371] at e_1424_8
  have e_1424_9 := arithEq_of_rows h (row := 1424) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_9
  simp only [← c87372, ← c87373, k2, ← c87374] at e_1424_9
  have hr := e_1424_9
  simp only [e_1424_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13643)) := by
  have c87375 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87376 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87377 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87378 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87379 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87380 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_10 := arithEq_of_rows h (row := 1424) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_10
  simp only [← c87375, ← c87376, ← c87377] at e_1424_10
  have e_1424_11 := arithEq_of_rows h (row := 1424) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_11
  simp only [← c87378, ← c87379, k2, ← c87380] at e_1424_11
  have hr := e_1424_11
  simp only [e_1424_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13644)) := by
  have c87381 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87382 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87383 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87384 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87385 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87386 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_12 := arithEq_of_rows h (row := 1424) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_12
  simp only [← c87381, ← c87382, ← c87383] at e_1424_12
  have e_1424_13 := arithEq_of_rows h (row := 1424) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_13
  simp only [← c87384, ← c87385, k2, ← c87386] at e_1424_13
  have hr := e_1424_13
  simp only [e_1424_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13645)) := by
  have c87387 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87388 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87389 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87390 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87391 := (publicBatchWrapper86_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87392 := (publicBatchWrapper86_copies2731 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_14 := arithEq_of_rows h (row := 1424) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_14
  simp only [← c87387, ← c87388, ← c87389] at e_1424_14
  have e_1424_15 := arithEq_of_rows h (row := 1424) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_15
  simp only [← c87390, ← c87391, k2, ← c87392] at e_1424_15
  have hr := e_1424_15
  simp only [e_1424_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13646)) := by
  have c87393 := (publicBatchWrapper86_copies2731 a h).2.1
  have c87394 := (publicBatchWrapper86_copies2731 a h).2.2.1
  have c87395 := (publicBatchWrapper86_copies2731 a h).2.2.2.1
  have c87396 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.1
  have c87397 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.1
  have c87398 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_16 := arithEq_of_rows h (row := 1424) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_16
  simp only [← c87393, ← c87394, ← c87395] at e_1424_16
  have e_1424_17 := arithEq_of_rows h (row := 1424) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_17
  simp only [← c87396, ← c87397, k2, ← c87398] at e_1424_17
  have hr := e_1424_17
  simp only [e_1424_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1424 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13647)) := by
  have c87399 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.1
  have c87400 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.1
  have c87401 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.1
  have c87402 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87403 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87404 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1424_18 := arithEq_of_rows h (row := 1424) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_18
  simp only [← c87399, ← c87400, ← c87401] at e_1424_18
  have e_1424_19 := arithEq_of_rows h (row := 1424) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_19
  simp only [← c87402, ← c87403, k2, ← c87404] at e_1424_19
  have hr := e_1424_19
  simp only [e_1424_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13648)) := by
  have c87405 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87406 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87407 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87408 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87409 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87410 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_0 := arithEq_of_rows h (row := 1425) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_0
  simp only [← c87405, ← c87406, ← c87407] at e_1425_0
  have e_1425_1 := arithEq_of_rows h (row := 1425) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_1
  simp only [← c87408, ← c87409, k2, ← c87410] at e_1425_1
  have hr := e_1425_1
  simp only [e_1425_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13649)) := by
  have c87411 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87412 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87413 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87414 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87415 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87416 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_2 := arithEq_of_rows h (row := 1425) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_2
  simp only [← c87411, ← c87412, ← c87413] at e_1425_2
  have e_1425_3 := arithEq_of_rows h (row := 1425) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_3
  simp only [← c87414, ← c87415, k2, ← c87416] at e_1425_3
  have hr := e_1425_3
  simp only [e_1425_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13650)) := by
  have c87417 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87418 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87419 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87420 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87421 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87422 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_4 := arithEq_of_rows h (row := 1425) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_4
  simp only [← c87417, ← c87418, ← c87419] at e_1425_4
  have e_1425_5 := arithEq_of_rows h (row := 1425) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_5
  simp only [← c87420, ← c87421, k2, ← c87422] at e_1425_5
  have hr := e_1425_5
  simp only [e_1425_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13651)) := by
  have c87423 := (publicBatchWrapper86_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87424 := (publicBatchWrapper86_copies2732 a h).1
  have c87425 := (publicBatchWrapper86_copies2732 a h).2.1
  have c87426 := (publicBatchWrapper86_copies2732 a h).2.2.1
  have c87427 := (publicBatchWrapper86_copies2732 a h).2.2.2.1
  have c87428 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_6 := arithEq_of_rows h (row := 1425) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_6
  simp only [← c87423, ← c87424, ← c87425] at e_1425_6
  have e_1425_7 := arithEq_of_rows h (row := 1425) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_7
  simp only [← c87426, ← c87427, k2, ← c87428] at e_1425_7
  have hr := e_1425_7
  simp only [e_1425_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13652)) := by
  have c87429 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.1
  have c87430 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.1
  have c87431 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.1
  have c87432 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.1
  have c87433 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.1
  have c87434 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_8 := arithEq_of_rows h (row := 1425) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_8
  simp only [← c87429, ← c87430, ← c87431] at e_1425_8
  have e_1425_9 := arithEq_of_rows h (row := 1425) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_9
  simp only [← c87432, ← c87433, k2, ← c87434] at e_1425_9
  have hr := e_1425_9
  simp only [e_1425_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13653)) := by
  have c87435 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87436 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87437 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87438 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87439 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87440 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_10 := arithEq_of_rows h (row := 1425) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_10
  simp only [← c87435, ← c87436, ← c87437] at e_1425_10
  have e_1425_11 := arithEq_of_rows h (row := 1425) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_11
  simp only [← c87438, ← c87439, k2, ← c87440] at e_1425_11
  have hr := e_1425_11
  simp only [e_1425_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13654)) := by
  have c87441 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87442 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87443 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87444 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87445 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87446 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_12 := arithEq_of_rows h (row := 1425) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_12
  simp only [← c87441, ← c87442, ← c87443] at e_1425_12
  have e_1425_13 := arithEq_of_rows h (row := 1425) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_13
  simp only [← c87444, ← c87445, k2, ← c87446] at e_1425_13
  have hr := e_1425_13
  simp only [e_1425_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13655)) := by
  have c87447 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87448 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87449 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87450 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87451 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87452 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_14 := arithEq_of_rows h (row := 1425) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_14
  simp only [← c87447, ← c87448, ← c87449] at e_1425_14
  have e_1425_15 := arithEq_of_rows h (row := 1425) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_15
  simp only [← c87450, ← c87451, k2, ← c87452] at e_1425_15
  have hr := e_1425_15
  simp only [e_1425_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13656)) := by
  have c87453 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87454 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87455 := (publicBatchWrapper86_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87456 := (publicBatchWrapper86_copies2733 a h).1
  have c87457 := (publicBatchWrapper86_copies2733 a h).2.1
  have c87458 := (publicBatchWrapper86_copies2733 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_16 := arithEq_of_rows h (row := 1425) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_16
  simp only [← c87453, ← c87454, ← c87455] at e_1425_16
  have e_1425_17 := arithEq_of_rows h (row := 1425) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_17
  simp only [← c87456, ← c87457, k2, ← c87458] at e_1425_17
  have hr := e_1425_17
  simp only [e_1425_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1425 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13657)) := by
  have c87459 := (publicBatchWrapper86_copies2733 a h).2.2.2.1
  have c87460 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.1
  have c87461 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.1
  have c87462 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.1
  have c87463 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.1
  have c87464 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1425_18 := arithEq_of_rows h (row := 1425) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_18
  simp only [← c87459, ← c87460, ← c87461] at e_1425_18
  have e_1425_19 := arithEq_of_rows h (row := 1425) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_19
  simp only [← c87462, ← c87463, k2, ← c87464] at e_1425_19
  have hr := e_1425_19
  simp only [e_1425_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13658)) := by
  have c87465 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.1
  have c87466 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87467 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87468 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87469 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87470 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_0 := arithEq_of_rows h (row := 1426) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_0
  simp only [← c87465, ← c87466, ← c87467] at e_1426_0
  have e_1426_1 := arithEq_of_rows h (row := 1426) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_1
  simp only [← c87468, ← c87469, k2, ← c87470] at e_1426_1
  have hr := e_1426_1
  simp only [e_1426_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13659)) := by
  have c87471 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87472 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87473 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87474 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87475 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87476 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_2 := arithEq_of_rows h (row := 1426) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_2
  simp only [← c87471, ← c87472, ← c87473] at e_1426_2
  have e_1426_3 := arithEq_of_rows h (row := 1426) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_3
  simp only [← c87474, ← c87475, k2, ← c87476] at e_1426_3
  have hr := e_1426_3
  simp only [e_1426_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13660)) := by
  have c87477 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87478 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87479 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87480 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87481 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87482 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_4 := arithEq_of_rows h (row := 1426) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_4
  simp only [← c87477, ← c87478, ← c87479] at e_1426_4
  have e_1426_5 := arithEq_of_rows h (row := 1426) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_5
  simp only [← c87480, ← c87481, k2, ← c87482] at e_1426_5
  have hr := e_1426_5
  simp only [e_1426_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13661)) := by
  have c87483 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87484 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87485 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87486 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87487 := (publicBatchWrapper86_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87488 := (publicBatchWrapper86_copies2734 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_6 := arithEq_of_rows h (row := 1426) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_6
  simp only [← c87483, ← c87484, ← c87485] at e_1426_6
  have e_1426_7 := arithEq_of_rows h (row := 1426) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_7
  simp only [← c87486, ← c87487, k2, ← c87488] at e_1426_7
  have hr := e_1426_7
  simp only [e_1426_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13662)) := by
  have c87489 := (publicBatchWrapper86_copies2734 a h).2.1
  have c87490 := (publicBatchWrapper86_copies2734 a h).2.2.1
  have c87491 := (publicBatchWrapper86_copies2734 a h).2.2.2.1
  have c87492 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.1
  have c87493 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.1
  have c87494 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_8 := arithEq_of_rows h (row := 1426) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_8
  simp only [← c87489, ← c87490, ← c87491] at e_1426_8
  have e_1426_9 := arithEq_of_rows h (row := 1426) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_9
  simp only [← c87492, ← c87493, k2, ← c87494] at e_1426_9
  have hr := e_1426_9
  simp only [e_1426_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13663)) := by
  have c87495 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.1
  have c87496 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.1
  have c87497 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.1
  have c87498 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87499 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87500 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_10 := arithEq_of_rows h (row := 1426) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_10
  simp only [← c87495, ← c87496, ← c87497] at e_1426_10
  have e_1426_11 := arithEq_of_rows h (row := 1426) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_11
  simp only [← c87498, ← c87499, k2, ← c87500] at e_1426_11
  have hr := e_1426_11
  simp only [e_1426_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13664)) := by
  have c87501 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87502 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87503 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87504 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87505 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87506 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_12 := arithEq_of_rows h (row := 1426) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_12
  simp only [← c87501, ← c87502, ← c87503] at e_1426_12
  have e_1426_13 := arithEq_of_rows h (row := 1426) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_13
  simp only [← c87504, ← c87505, k2, ← c87506] at e_1426_13
  have hr := e_1426_13
  simp only [e_1426_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13665)) := by
  have c87507 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87508 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87509 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87510 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87511 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87512 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_14 := arithEq_of_rows h (row := 1426) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_14
  simp only [← c87507, ← c87508, ← c87509] at e_1426_14
  have e_1426_15 := arithEq_of_rows h (row := 1426) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_15
  simp only [← c87510, ← c87511, k2, ← c87512] at e_1426_15
  have hr := e_1426_15
  simp only [e_1426_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13666)) := by
  have c87513 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87514 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87515 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87516 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87517 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87518 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_16 := arithEq_of_rows h (row := 1426) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_16
  simp only [← c87513, ← c87514, ← c87515] at e_1426_16
  have e_1426_17 := arithEq_of_rows h (row := 1426) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_17
  simp only [← c87516, ← c87517, k2, ← c87518] at e_1426_17
  have hr := e_1426_17
  simp only [e_1426_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1426 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13667)) := by
  have c87519 := (publicBatchWrapper86_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87520 := (publicBatchWrapper86_copies2735 a h).1
  have c87521 := (publicBatchWrapper86_copies2735 a h).2.1
  have c87522 := (publicBatchWrapper86_copies2735 a h).2.2.1
  have c87523 := (publicBatchWrapper86_copies2735 a h).2.2.2.1
  have c87524 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1426_18 := arithEq_of_rows h (row := 1426) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_18
  simp only [← c87519, ← c87520, ← c87521] at e_1426_18
  have e_1426_19 := arithEq_of_rows h (row := 1426) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_19
  simp only [← c87522, ← c87523, k2, ← c87524] at e_1426_19
  have hr := e_1426_19
  simp only [e_1426_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13668)) := by
  have c87525 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.1
  have c87526 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.1
  have c87527 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.1
  have c87528 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.1
  have c87529 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.1
  have c87530 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_0 := arithEq_of_rows h (row := 1427) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_0
  simp only [← c87525, ← c87526, ← c87527] at e_1427_0
  have e_1427_1 := arithEq_of_rows h (row := 1427) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_1
  simp only [← c87528, ← c87529, k2, ← c87530] at e_1427_1
  have hr := e_1427_1
  simp only [e_1427_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13669)) := by
  have c87531 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87532 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87533 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87534 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87535 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87536 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_2 := arithEq_of_rows h (row := 1427) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_2
  simp only [← c87531, ← c87532, ← c87533] at e_1427_2
  have e_1427_3 := arithEq_of_rows h (row := 1427) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_3
  simp only [← c87534, ← c87535, k2, ← c87536] at e_1427_3
  have hr := e_1427_3
  simp only [e_1427_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13670)) := by
  have c87537 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87538 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87539 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87540 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87541 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87542 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_4 := arithEq_of_rows h (row := 1427) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_4
  simp only [← c87537, ← c87538, ← c87539] at e_1427_4
  have e_1427_5 := arithEq_of_rows h (row := 1427) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_5
  simp only [← c87540, ← c87541, k2, ← c87542] at e_1427_5
  have hr := e_1427_5
  simp only [e_1427_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13671)) := by
  have c87543 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87544 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87545 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87546 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87547 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87548 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_6 := arithEq_of_rows h (row := 1427) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_6
  simp only [← c87543, ← c87544, ← c87545] at e_1427_6
  have e_1427_7 := arithEq_of_rows h (row := 1427) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_7
  simp only [← c87546, ← c87547, k2, ← c87548] at e_1427_7
  have hr := e_1427_7
  simp only [e_1427_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13672)) := by
  have c87549 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87550 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87551 := (publicBatchWrapper86_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87552 := (publicBatchWrapper86_copies2736 a h).1
  have c87553 := (publicBatchWrapper86_copies2736 a h).2.1
  have c87554 := (publicBatchWrapper86_copies2736 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_8 := arithEq_of_rows h (row := 1427) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_8
  simp only [← c87549, ← c87550, ← c87551] at e_1427_8
  have e_1427_9 := arithEq_of_rows h (row := 1427) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_9
  simp only [← c87552, ← c87553, k2, ← c87554] at e_1427_9
  have hr := e_1427_9
  simp only [e_1427_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13673)) := by
  have c87555 := (publicBatchWrapper86_copies2736 a h).2.2.2.1
  have c87556 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.1
  have c87557 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.1
  have c87558 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.1
  have c87559 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.1
  have c87560 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_10 := arithEq_of_rows h (row := 1427) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_10
  simp only [← c87555, ← c87556, ← c87557] at e_1427_10
  have e_1427_11 := arithEq_of_rows h (row := 1427) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_11
  simp only [← c87558, ← c87559, k2, ← c87560] at e_1427_11
  have hr := e_1427_11
  simp only [e_1427_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13674)) := by
  have c87561 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.1
  have c87562 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87563 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87564 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87565 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87566 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_12 := arithEq_of_rows h (row := 1427) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_12
  simp only [← c87561, ← c87562, ← c87563] at e_1427_12
  have e_1427_13 := arithEq_of_rows h (row := 1427) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_13
  simp only [← c87564, ← c87565, k2, ← c87566] at e_1427_13
  have hr := e_1427_13
  simp only [e_1427_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13675)) := by
  have c87567 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87568 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87569 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87570 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87571 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87572 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_14 := arithEq_of_rows h (row := 1427) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_14
  simp only [← c87567, ← c87568, ← c87569] at e_1427_14
  have e_1427_15 := arithEq_of_rows h (row := 1427) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_15
  simp only [← c87570, ← c87571, k2, ← c87572] at e_1427_15
  have hr := e_1427_15
  simp only [e_1427_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13676)) := by
  have c87573 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87574 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87575 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87576 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87577 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87578 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_16 := arithEq_of_rows h (row := 1427) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_16
  simp only [← c87573, ← c87574, ← c87575] at e_1427_16
  have e_1427_17 := arithEq_of_rows h (row := 1427) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_17
  simp only [← c87576, ← c87577, k2, ← c87578] at e_1427_17
  have hr := e_1427_17
  simp only [e_1427_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1427 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13677)) := by
  have c87579 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87580 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87581 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87582 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87583 := (publicBatchWrapper86_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87584 := (publicBatchWrapper86_copies2737 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1427_18 := arithEq_of_rows h (row := 1427) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_18
  simp only [← c87579, ← c87580, ← c87581] at e_1427_18
  have e_1427_19 := arithEq_of_rows h (row := 1427) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_19
  simp only [← c87582, ← c87583, k2, ← c87584] at e_1427_19
  have hr := e_1427_19
  simp only [e_1427_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13678)) := by
  have c87585 := (publicBatchWrapper86_copies2737 a h).2.1
  have c87586 := (publicBatchWrapper86_copies2737 a h).2.2.1
  have c87587 := (publicBatchWrapper86_copies2737 a h).2.2.2.1
  have c87588 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.1
  have c87589 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.1
  have c87590 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_0 := arithEq_of_rows h (row := 1428) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_0
  simp only [← c87585, ← c87586, ← c87587] at e_1428_0
  have e_1428_1 := arithEq_of_rows h (row := 1428) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_1
  simp only [← c87588, ← c87589, k2, ← c87590] at e_1428_1
  have hr := e_1428_1
  simp only [e_1428_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13679)) := by
  have c87591 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.1
  have c87592 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.1
  have c87593 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.1
  have c87594 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87595 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87596 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_2 := arithEq_of_rows h (row := 1428) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_2
  simp only [← c87591, ← c87592, ← c87593] at e_1428_2
  have e_1428_3 := arithEq_of_rows h (row := 1428) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_3
  simp only [← c87594, ← c87595, k2, ← c87596] at e_1428_3
  have hr := e_1428_3
  simp only [e_1428_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13680)) := by
  have c87597 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87598 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87599 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87600 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87601 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87602 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_4 := arithEq_of_rows h (row := 1428) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_4
  simp only [← c87597, ← c87598, ← c87599] at e_1428_4
  have e_1428_5 := arithEq_of_rows h (row := 1428) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_5
  simp only [← c87600, ← c87601, k2, ← c87602] at e_1428_5
  have hr := e_1428_5
  simp only [e_1428_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13681)) := by
  have c87603 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87604 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87605 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87606 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87607 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87608 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_6 := arithEq_of_rows h (row := 1428) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_6
  simp only [← c87603, ← c87604, ← c87605] at e_1428_6
  have e_1428_7 := arithEq_of_rows h (row := 1428) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_7
  simp only [← c87606, ← c87607, k2, ← c87608] at e_1428_7
  have hr := e_1428_7
  simp only [e_1428_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13682)) := by
  have c87609 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87610 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87611 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87612 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87613 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87614 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_8 := arithEq_of_rows h (row := 1428) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_8
  simp only [← c87609, ← c87610, ← c87611] at e_1428_8
  have e_1428_9 := arithEq_of_rows h (row := 1428) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_9
  simp only [← c87612, ← c87613, k2, ← c87614] at e_1428_9
  have hr := e_1428_9
  simp only [e_1428_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13683)) := by
  have c87615 := (publicBatchWrapper86_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87616 := (publicBatchWrapper86_copies2738 a h).1
  have c87617 := (publicBatchWrapper86_copies2738 a h).2.1
  have c87618 := (publicBatchWrapper86_copies2738 a h).2.2.1
  have c87619 := (publicBatchWrapper86_copies2738 a h).2.2.2.1
  have c87620 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_10 := arithEq_of_rows h (row := 1428) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_10
  simp only [← c87615, ← c87616, ← c87617] at e_1428_10
  have e_1428_11 := arithEq_of_rows h (row := 1428) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_11
  simp only [← c87618, ← c87619, k2, ← c87620] at e_1428_11
  have hr := e_1428_11
  simp only [e_1428_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13684)) := by
  have c87621 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.1
  have c87622 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.1
  have c87623 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.1
  have c87624 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.1
  have c87625 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.1
  have c87626 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_12 := arithEq_of_rows h (row := 1428) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_12
  simp only [← c87621, ← c87622, ← c87623] at e_1428_12
  have e_1428_13 := arithEq_of_rows h (row := 1428) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_13
  simp only [← c87624, ← c87625, k2, ← c87626] at e_1428_13
  have hr := e_1428_13
  simp only [e_1428_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13685)) := by
  have c87627 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87628 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87629 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87630 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87631 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87632 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_14 := arithEq_of_rows h (row := 1428) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_14
  simp only [← c87627, ← c87628, ← c87629] at e_1428_14
  have e_1428_15 := arithEq_of_rows h (row := 1428) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_15
  simp only [← c87630, ← c87631, k2, ← c87632] at e_1428_15
  have hr := e_1428_15
  simp only [e_1428_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13686)) := by
  have c87633 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87634 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87635 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87636 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87637 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87638 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_16 := arithEq_of_rows h (row := 1428) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_16
  simp only [← c87633, ← c87634, ← c87635] at e_1428_16
  have e_1428_17 := arithEq_of_rows h (row := 1428) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_17
  simp only [← c87636, ← c87637, k2, ← c87638] at e_1428_17
  have hr := e_1428_17
  simp only [e_1428_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1428 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13687)) := by
  have c87639 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87640 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87641 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87642 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87643 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87644 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1428_18 := arithEq_of_rows h (row := 1428) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_18
  simp only [← c87639, ← c87640, ← c87641] at e_1428_18
  have e_1428_19 := arithEq_of_rows h (row := 1428) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_19
  simp only [← c87642, ← c87643, k2, ← c87644] at e_1428_19
  have hr := e_1428_19
  simp only [e_1428_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13688)) := by
  have c87645 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87646 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87647 := (publicBatchWrapper86_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87648 := (publicBatchWrapper86_copies2739 a h).1
  have c87649 := (publicBatchWrapper86_copies2739 a h).2.1
  have c87650 := (publicBatchWrapper86_copies2739 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_0 := arithEq_of_rows h (row := 1429) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_0
  simp only [← c87645, ← c87646, ← c87647] at e_1429_0
  have e_1429_1 := arithEq_of_rows h (row := 1429) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_1
  simp only [← c87648, ← c87649, k2, ← c87650] at e_1429_1
  have hr := e_1429_1
  simp only [e_1429_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13689)) := by
  have c87651 := (publicBatchWrapper86_copies2739 a h).2.2.2.1
  have c87652 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.1
  have c87653 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.1
  have c87654 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.1
  have c87655 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.1
  have c87656 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_2 := arithEq_of_rows h (row := 1429) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_2
  simp only [← c87651, ← c87652, ← c87653] at e_1429_2
  have e_1429_3 := arithEq_of_rows h (row := 1429) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_3
  simp only [← c87654, ← c87655, k2, ← c87656] at e_1429_3
  have hr := e_1429_3
  simp only [e_1429_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13690)) := by
  have c87657 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.1
  have c87658 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87659 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87660 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87661 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87662 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_4 := arithEq_of_rows h (row := 1429) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_4
  simp only [← c87657, ← c87658, ← c87659] at e_1429_4
  have e_1429_5 := arithEq_of_rows h (row := 1429) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_5
  simp only [← c87660, ← c87661, k2, ← c87662] at e_1429_5
  have hr := e_1429_5
  simp only [e_1429_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
