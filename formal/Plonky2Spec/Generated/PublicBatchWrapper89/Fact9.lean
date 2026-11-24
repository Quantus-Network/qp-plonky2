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

theorem publicBatchWrapper89_f6803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4707)) := by
  have c45324 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45325 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45326 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45327 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45328 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45329 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_4 := arithEq_of_rows h (row := 722) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_4
  simp only [← c45324, ← c45325, ← c45326] at e_722_4
  have e_722_5 := arithEq_of_rows h (row := 722) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_5
  simp only [← c45327, ← c45328, k2, ← c45329] at e_722_5
  have hr := e_722_5
  simp only [e_722_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4708)) := by
  have c45330 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45331 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45332 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45333 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45334 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45335 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_6 := arithEq_of_rows h (row := 722) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_6
  simp only [← c45330, ← c45331, ← c45332] at e_722_6
  have e_722_7 := arithEq_of_rows h (row := 722) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_7
  simp only [← c45333, ← c45334, k2, ← c45335] at e_722_7
  have hr := e_722_7
  simp only [e_722_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4709)) := by
  have c45336 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45337 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45338 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45339 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45340 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45341 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_8 := arithEq_of_rows h (row := 722) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_8
  simp only [← c45336, ← c45337, ← c45338] at e_722_8
  have e_722_9 := arithEq_of_rows h (row := 722) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_9
  simp only [← c45339, ← c45340, k2, ← c45341] at e_722_9
  have hr := e_722_9
  simp only [e_722_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4710)) := by
  have c45342 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45343 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45344 := (publicBatchWrapper89_copies1417 a h).1
  have c45345 := (publicBatchWrapper89_copies1417 a h).2.1
  have c45346 := (publicBatchWrapper89_copies1417 a h).2.2.1
  have c45347 := (publicBatchWrapper89_copies1417 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_10 := arithEq_of_rows h (row := 722) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_10
  simp only [← c45342, ← c45343, ← c45344] at e_722_10
  have e_722_11 := arithEq_of_rows h (row := 722) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_11
  simp only [← c45345, ← c45346, k2, ← c45347] at e_722_11
  have hr := e_722_11
  simp only [e_722_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4711)) := by
  have c45348 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.1
  have c45349 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.1
  have c45350 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.1
  have c45351 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.1
  have c45352 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.1
  have c45353 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_12 := arithEq_of_rows h (row := 722) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_12
  simp only [← c45348, ← c45349, ← c45350] at e_722_12
  have e_722_13 := arithEq_of_rows h (row := 722) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_13
  simp only [← c45351, ← c45352, k2, ← c45353] at e_722_13
  have hr := e_722_13
  simp only [e_722_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4712)) := by
  have c45354 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45355 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45356 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45357 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45358 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45359 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_14 := arithEq_of_rows h (row := 722) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_14
  simp only [← c45354, ← c45355, ← c45356] at e_722_14
  have e_722_15 := arithEq_of_rows h (row := 722) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_15
  simp only [← c45357, ← c45358, k2, ← c45359] at e_722_15
  have hr := e_722_15
  simp only [e_722_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4713)) := by
  have c45360 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45361 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45362 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45363 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45364 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45365 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_16 := arithEq_of_rows h (row := 722) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_16
  simp only [← c45360, ← c45361, ← c45362] at e_722_16
  have e_722_17 := arithEq_of_rows h (row := 722) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_17
  simp only [← c45363, ← c45364, k2, ← c45365] at e_722_17
  have hr := e_722_17
  simp only [e_722_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4714)) := by
  have c45366 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45367 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45368 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45369 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45370 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45371 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_18 := arithEq_of_rows h (row := 722) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_18
  simp only [← c45366, ← c45367, ← c45368] at e_722_18
  have e_722_19 := arithEq_of_rows h (row := 722) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_19
  simp only [← c45369, ← c45370, k2, ← c45371] at e_722_19
  have hr := e_722_19
  simp only [e_722_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4715)) := by
  have c45372 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45373 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45374 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45375 := (publicBatchWrapper89_copies1417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45376 := (publicBatchWrapper89_copies1418 a h).1
  have c45377 := (publicBatchWrapper89_copies1418 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_0 := arithEq_of_rows h (row := 723) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_0
  simp only [← c45372, ← c45373, ← c45374] at e_723_0
  have e_723_1 := arithEq_of_rows h (row := 723) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_1
  simp only [← c45375, ← c45376, k2, ← c45377] at e_723_1
  have hr := e_723_1
  simp only [e_723_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4716)) := by
  have c45378 := (publicBatchWrapper89_copies1418 a h).2.2.1
  have c45379 := (publicBatchWrapper89_copies1418 a h).2.2.2.1
  have c45380 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.1
  have c45381 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.1
  have c45382 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.1
  have c45383 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_2 := arithEq_of_rows h (row := 723) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_2
  simp only [← c45378, ← c45379, ← c45380] at e_723_2
  have e_723_3 := arithEq_of_rows h (row := 723) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_3
  simp only [← c45381, ← c45382, k2, ← c45383] at e_723_3
  have hr := e_723_3
  simp only [e_723_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4717)) := by
  have c45384 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.1
  have c45385 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.1
  have c45386 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45387 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45388 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45389 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_4 := arithEq_of_rows h (row := 723) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_4
  simp only [← c45384, ← c45385, ← c45386] at e_723_4
  have e_723_5 := arithEq_of_rows h (row := 723) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_5
  simp only [← c45387, ← c45388, k2, ← c45389] at e_723_5
  have hr := e_723_5
  simp only [e_723_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4718)) := by
  have c45390 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45391 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45392 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45393 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45394 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45395 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_6 := arithEq_of_rows h (row := 723) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_6
  simp only [← c45390, ← c45391, ← c45392] at e_723_6
  have e_723_7 := arithEq_of_rows h (row := 723) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_7
  simp only [← c45393, ← c45394, k2, ← c45395] at e_723_7
  have hr := e_723_7
  simp only [e_723_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4719)) := by
  have c45396 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45397 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45398 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45399 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45400 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45401 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_8 := arithEq_of_rows h (row := 723) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_8
  simp only [← c45396, ← c45397, ← c45398] at e_723_8
  have e_723_9 := arithEq_of_rows h (row := 723) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_9
  simp only [← c45399, ← c45400, k2, ← c45401] at e_723_9
  have hr := e_723_9
  simp only [e_723_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4720)) := by
  have c45402 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45403 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45404 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45405 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45406 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45407 := (publicBatchWrapper89_copies1418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_10 := arithEq_of_rows h (row := 723) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_10
  simp only [← c45402, ← c45403, ← c45404] at e_723_10
  have e_723_11 := arithEq_of_rows h (row := 723) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_11
  simp only [← c45405, ← c45406, k2, ← c45407] at e_723_11
  have hr := e_723_11
  simp only [e_723_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4721)) := by
  have c45408 := (publicBatchWrapper89_copies1419 a h).1
  have c45409 := (publicBatchWrapper89_copies1419 a h).2.1
  have c45410 := (publicBatchWrapper89_copies1419 a h).2.2.1
  have c45411 := (publicBatchWrapper89_copies1419 a h).2.2.2.1
  have c45412 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.1
  have c45413 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_12 := arithEq_of_rows h (row := 723) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_12
  simp only [← c45408, ← c45409, ← c45410] at e_723_12
  have e_723_13 := arithEq_of_rows h (row := 723) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_13
  simp only [← c45411, ← c45412, k2, ← c45413] at e_723_13
  have hr := e_723_13
  simp only [e_723_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4722)) := by
  have c45414 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.1
  have c45415 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.1
  have c45416 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.1
  have c45417 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.1
  have c45418 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45419 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_14 := arithEq_of_rows h (row := 723) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_14
  simp only [← c45414, ← c45415, ← c45416] at e_723_14
  have e_723_15 := arithEq_of_rows h (row := 723) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_15
  simp only [← c45417, ← c45418, k2, ← c45419] at e_723_15
  have hr := e_723_15
  simp only [e_723_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4723)) := by
  have c45420 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45421 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45422 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45423 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45424 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45425 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_16 := arithEq_of_rows h (row := 723) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_16
  simp only [← c45420, ← c45421, ← c45422] at e_723_16
  have e_723_17 := arithEq_of_rows h (row := 723) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_17
  simp only [← c45423, ← c45424, k2, ← c45425] at e_723_17
  have hr := e_723_17
  simp only [e_723_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 723 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4724)) := by
  have c45426 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45427 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45428 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45429 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45430 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45431 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_723_18 := arithEq_of_rows h (row := 723) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_18
  simp only [← c45426, ← c45427, ← c45428] at e_723_18
  have e_723_19 := arithEq_of_rows h (row := 723) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_723_19
  simp only [← c45429, ← c45430, k2, ← c45431] at e_723_19
  have hr := e_723_19
  simp only [e_723_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4725)) := by
  have c45432 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45433 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45434 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45435 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45436 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45437 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_0 := arithEq_of_rows h (row := 724) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_0
  simp only [← c45432, ← c45433, ← c45434] at e_724_0
  have e_724_1 := arithEq_of_rows h (row := 724) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_1
  simp only [← c45435, ← c45436, k2, ← c45437] at e_724_1
  have hr := e_724_1
  simp only [e_724_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4726)) := by
  have c45438 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45439 := (publicBatchWrapper89_copies1419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45440 := (publicBatchWrapper89_copies1420 a h).1
  have c45441 := (publicBatchWrapper89_copies1420 a h).2.1
  have c45442 := (publicBatchWrapper89_copies1420 a h).2.2.1
  have c45443 := (publicBatchWrapper89_copies1420 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_2 := arithEq_of_rows h (row := 724) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_2
  simp only [← c45438, ← c45439, ← c45440] at e_724_2
  have e_724_3 := arithEq_of_rows h (row := 724) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_3
  simp only [← c45441, ← c45442, k2, ← c45443] at e_724_3
  have hr := e_724_3
  simp only [e_724_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4727)) := by
  have c45444 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.1
  have c45445 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.1
  have c45446 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.1
  have c45447 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.1
  have c45448 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.1
  have c45449 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_4 := arithEq_of_rows h (row := 724) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_4
  simp only [← c45444, ← c45445, ← c45446] at e_724_4
  have e_724_5 := arithEq_of_rows h (row := 724) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_5
  simp only [← c45447, ← c45448, k2, ← c45449] at e_724_5
  have hr := e_724_5
  simp only [e_724_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4728)) := by
  have c45450 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45451 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45452 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45453 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45454 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45455 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_6 := arithEq_of_rows h (row := 724) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_6
  simp only [← c45450, ← c45451, ← c45452] at e_724_6
  have e_724_7 := arithEq_of_rows h (row := 724) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_7
  simp only [← c45453, ← c45454, k2, ← c45455] at e_724_7
  have hr := e_724_7
  simp only [e_724_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4729)) := by
  have c45456 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45457 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45458 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45459 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45460 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45461 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_8 := arithEq_of_rows h (row := 724) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_8
  simp only [← c45456, ← c45457, ← c45458] at e_724_8
  have e_724_9 := arithEq_of_rows h (row := 724) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_9
  simp only [← c45459, ← c45460, k2, ← c45461] at e_724_9
  have hr := e_724_9
  simp only [e_724_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4730)) := by
  have c45462 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45463 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45464 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45465 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45466 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45467 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_10 := arithEq_of_rows h (row := 724) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_10
  simp only [← c45462, ← c45463, ← c45464] at e_724_10
  have e_724_11 := arithEq_of_rows h (row := 724) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_11
  simp only [← c45465, ← c45466, k2, ← c45467] at e_724_11
  have hr := e_724_11
  simp only [e_724_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4731)) := by
  have c45468 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45469 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45470 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45471 := (publicBatchWrapper89_copies1420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45472 := (publicBatchWrapper89_copies1421 a h).1
  have c45473 := (publicBatchWrapper89_copies1421 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_12 := arithEq_of_rows h (row := 724) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_12
  simp only [← c45468, ← c45469, ← c45470] at e_724_12
  have e_724_13 := arithEq_of_rows h (row := 724) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_13
  simp only [← c45471, ← c45472, k2, ← c45473] at e_724_13
  have hr := e_724_13
  simp only [e_724_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4732)) := by
  have c45474 := (publicBatchWrapper89_copies1421 a h).2.2.1
  have c45475 := (publicBatchWrapper89_copies1421 a h).2.2.2.1
  have c45476 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.1
  have c45477 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.1
  have c45478 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.1
  have c45479 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_14 := arithEq_of_rows h (row := 724) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_14
  simp only [← c45474, ← c45475, ← c45476] at e_724_14
  have e_724_15 := arithEq_of_rows h (row := 724) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_15
  simp only [← c45477, ← c45478, k2, ← c45479] at e_724_15
  have hr := e_724_15
  simp only [e_724_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4733)) := by
  have c45480 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.1
  have c45481 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.1
  have c45482 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45483 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45484 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45485 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_16 := arithEq_of_rows h (row := 724) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_16
  simp only [← c45480, ← c45481, ← c45482] at e_724_16
  have e_724_17 := arithEq_of_rows h (row := 724) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_17
  simp only [← c45483, ← c45484, k2, ← c45485] at e_724_17
  have hr := e_724_17
  simp only [e_724_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 724 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4734)) := by
  have c45486 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45487 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45488 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45489 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45490 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45491 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_724_18 := arithEq_of_rows h (row := 724) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_18
  simp only [← c45486, ← c45487, ← c45488] at e_724_18
  have e_724_19 := arithEq_of_rows h (row := 724) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_724_19
  simp only [← c45489, ← c45490, k2, ← c45491] at e_724_19
  have hr := e_724_19
  simp only [e_724_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4735)) := by
  have c45492 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45493 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45494 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45495 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45496 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45497 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_0 := arithEq_of_rows h (row := 725) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_0
  simp only [← c45492, ← c45493, ← c45494] at e_725_0
  have e_725_1 := arithEq_of_rows h (row := 725) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_1
  simp only [← c45495, ← c45496, k2, ← c45497] at e_725_1
  have hr := e_725_1
  simp only [e_725_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4736)) := by
  have c45498 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45499 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45500 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45501 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45502 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45503 := (publicBatchWrapper89_copies1421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_2 := arithEq_of_rows h (row := 725) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_2
  simp only [← c45498, ← c45499, ← c45500] at e_725_2
  have e_725_3 := arithEq_of_rows h (row := 725) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_3
  simp only [← c45501, ← c45502, k2, ← c45503] at e_725_3
  have hr := e_725_3
  simp only [e_725_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4737)) := by
  have c45504 := (publicBatchWrapper89_copies1422 a h).1
  have c45505 := (publicBatchWrapper89_copies1422 a h).2.1
  have c45506 := (publicBatchWrapper89_copies1422 a h).2.2.1
  have c45507 := (publicBatchWrapper89_copies1422 a h).2.2.2.1
  have c45508 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.1
  have c45509 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_4 := arithEq_of_rows h (row := 725) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_4
  simp only [← c45504, ← c45505, ← c45506] at e_725_4
  have e_725_5 := arithEq_of_rows h (row := 725) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_5
  simp only [← c45507, ← c45508, k2, ← c45509] at e_725_5
  have hr := e_725_5
  simp only [e_725_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4738)) := by
  have c45510 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.1
  have c45511 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.1
  have c45512 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.1
  have c45513 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.1
  have c45514 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45515 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_6 := arithEq_of_rows h (row := 725) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_6
  simp only [← c45510, ← c45511, ← c45512] at e_725_6
  have e_725_7 := arithEq_of_rows h (row := 725) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_7
  simp only [← c45513, ← c45514, k2, ← c45515] at e_725_7
  have hr := e_725_7
  simp only [e_725_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4739)) := by
  have c45516 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45517 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45518 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45519 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45520 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45521 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_8 := arithEq_of_rows h (row := 725) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_8
  simp only [← c45516, ← c45517, ← c45518] at e_725_8
  have e_725_9 := arithEq_of_rows h (row := 725) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_9
  simp only [← c45519, ← c45520, k2, ← c45521] at e_725_9
  have hr := e_725_9
  simp only [e_725_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4740)) := by
  have c45522 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45523 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45524 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45525 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45526 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45527 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_10 := arithEq_of_rows h (row := 725) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_10
  simp only [← c45522, ← c45523, ← c45524] at e_725_10
  have e_725_11 := arithEq_of_rows h (row := 725) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_11
  simp only [← c45525, ← c45526, k2, ← c45527] at e_725_11
  have hr := e_725_11
  simp only [e_725_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4741)) := by
  have c45528 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45529 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45530 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45531 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45532 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45533 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_12 := arithEq_of_rows h (row := 725) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_12
  simp only [← c45528, ← c45529, ← c45530] at e_725_12
  have e_725_13 := arithEq_of_rows h (row := 725) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_13
  simp only [← c45531, ← c45532, k2, ← c45533] at e_725_13
  have hr := e_725_13
  simp only [e_725_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4742)) := by
  have c45534 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45535 := (publicBatchWrapper89_copies1422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45536 := (publicBatchWrapper89_copies1423 a h).1
  have c45537 := (publicBatchWrapper89_copies1423 a h).2.1
  have c45538 := (publicBatchWrapper89_copies1423 a h).2.2.1
  have c45539 := (publicBatchWrapper89_copies1423 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_14 := arithEq_of_rows h (row := 725) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_14
  simp only [← c45534, ← c45535, ← c45536] at e_725_14
  have e_725_15 := arithEq_of_rows h (row := 725) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_15
  simp only [← c45537, ← c45538, k2, ← c45539] at e_725_15
  have hr := e_725_15
  simp only [e_725_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4743)) := by
  have c45540 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.1
  have c45541 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.1
  have c45542 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.1
  have c45543 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.1
  have c45544 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.1
  have c45545 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_16 := arithEq_of_rows h (row := 725) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_16
  simp only [← c45540, ← c45541, ← c45542] at e_725_16
  have e_725_17 := arithEq_of_rows h (row := 725) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_17
  simp only [← c45543, ← c45544, k2, ← c45545] at e_725_17
  have hr := e_725_17
  simp only [e_725_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 725 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4744)) := by
  have c45546 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45547 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45548 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45549 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45550 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45551 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_725_18 := arithEq_of_rows h (row := 725) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_18
  simp only [← c45546, ← c45547, ← c45548] at e_725_18
  have e_725_19 := arithEq_of_rows h (row := 725) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_725_19
  simp only [← c45549, ← c45550, k2, ← c45551] at e_725_19
  have hr := e_725_19
  simp only [e_725_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4745)) := by
  have c45552 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45553 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45554 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45555 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45556 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45557 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_0 := arithEq_of_rows h (row := 726) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_0
  simp only [← c45552, ← c45553, ← c45554] at e_726_0
  have e_726_1 := arithEq_of_rows h (row := 726) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_1
  simp only [← c45555, ← c45556, k2, ← c45557] at e_726_1
  have hr := e_726_1
  simp only [e_726_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4746)) := by
  have c45558 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45559 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45560 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45561 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45562 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45563 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_2 := arithEq_of_rows h (row := 726) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_2
  simp only [← c45558, ← c45559, ← c45560] at e_726_2
  have e_726_3 := arithEq_of_rows h (row := 726) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_3
  simp only [← c45561, ← c45562, k2, ← c45563] at e_726_3
  have hr := e_726_3
  simp only [e_726_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4747)) := by
  have c45564 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45565 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45566 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45567 := (publicBatchWrapper89_copies1423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45568 := (publicBatchWrapper89_copies1424 a h).1
  have c45569 := (publicBatchWrapper89_copies1424 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_4 := arithEq_of_rows h (row := 726) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_4
  simp only [← c45564, ← c45565, ← c45566] at e_726_4
  have e_726_5 := arithEq_of_rows h (row := 726) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_5
  simp only [← c45567, ← c45568, k2, ← c45569] at e_726_5
  have hr := e_726_5
  simp only [e_726_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4748)) := by
  have c45570 := (publicBatchWrapper89_copies1424 a h).2.2.1
  have c45571 := (publicBatchWrapper89_copies1424 a h).2.2.2.1
  have c45572 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.1
  have c45573 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.1
  have c45574 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.1
  have c45575 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_6 := arithEq_of_rows h (row := 726) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_6
  simp only [← c45570, ← c45571, ← c45572] at e_726_6
  have e_726_7 := arithEq_of_rows h (row := 726) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_7
  simp only [← c45573, ← c45574, k2, ← c45575] at e_726_7
  have hr := e_726_7
  simp only [e_726_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4749)) := by
  have c45576 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.1
  have c45577 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.1
  have c45578 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45579 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45580 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45581 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_8 := arithEq_of_rows h (row := 726) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_8
  simp only [← c45576, ← c45577, ← c45578] at e_726_8
  have e_726_9 := arithEq_of_rows h (row := 726) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_9
  simp only [← c45579, ← c45580, k2, ← c45581] at e_726_9
  have hr := e_726_9
  simp only [e_726_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4750)) := by
  have c45582 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45583 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45584 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45585 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45586 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45587 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_10 := arithEq_of_rows h (row := 726) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_10
  simp only [← c45582, ← c45583, ← c45584] at e_726_10
  have e_726_11 := arithEq_of_rows h (row := 726) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_11
  simp only [← c45585, ← c45586, k2, ← c45587] at e_726_11
  have hr := e_726_11
  simp only [e_726_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4751)) := by
  have c45588 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45589 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45590 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45591 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45592 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45593 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_12 := arithEq_of_rows h (row := 726) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_12
  simp only [← c45588, ← c45589, ← c45590] at e_726_12
  have e_726_13 := arithEq_of_rows h (row := 726) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_13
  simp only [← c45591, ← c45592, k2, ← c45593] at e_726_13
  have hr := e_726_13
  simp only [e_726_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4752)) := by
  have c45594 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45595 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45596 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45597 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45598 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45599 := (publicBatchWrapper89_copies1424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_14 := arithEq_of_rows h (row := 726) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_14
  simp only [← c45594, ← c45595, ← c45596] at e_726_14
  have e_726_15 := arithEq_of_rows h (row := 726) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_15
  simp only [← c45597, ← c45598, k2, ← c45599] at e_726_15
  have hr := e_726_15
  simp only [e_726_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4753)) := by
  have c45600 := (publicBatchWrapper89_copies1425 a h).1
  have c45601 := (publicBatchWrapper89_copies1425 a h).2.1
  have c45602 := (publicBatchWrapper89_copies1425 a h).2.2.1
  have c45603 := (publicBatchWrapper89_copies1425 a h).2.2.2.1
  have c45604 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.1
  have c45605 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_16 := arithEq_of_rows h (row := 726) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_16
  simp only [← c45600, ← c45601, ← c45602] at e_726_16
  have e_726_17 := arithEq_of_rows h (row := 726) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_17
  simp only [← c45603, ← c45604, k2, ← c45605] at e_726_17
  have hr := e_726_17
  simp only [e_726_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 726 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4754)) := by
  have c45606 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.1
  have c45607 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.1
  have c45608 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.1
  have c45609 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.1
  have c45610 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45611 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_726_18 := arithEq_of_rows h (row := 726) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_18
  simp only [← c45606, ← c45607, ← c45608] at e_726_18
  have e_726_19 := arithEq_of_rows h (row := 726) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_726_19
  simp only [← c45609, ← c45610, k2, ← c45611] at e_726_19
  have hr := e_726_19
  simp only [e_726_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4755)) := by
  have c45612 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45613 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45614 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45615 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45616 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45617 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_0 := arithEq_of_rows h (row := 727) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_0
  simp only [← c45612, ← c45613, ← c45614] at e_727_0
  have e_727_1 := arithEq_of_rows h (row := 727) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_1
  simp only [← c45615, ← c45616, k2, ← c45617] at e_727_1
  have hr := e_727_1
  simp only [e_727_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4756)) := by
  have c45618 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45619 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45620 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45621 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45622 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45623 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_2 := arithEq_of_rows h (row := 727) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_2
  simp only [← c45618, ← c45619, ← c45620] at e_727_2
  have e_727_3 := arithEq_of_rows h (row := 727) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_3
  simp only [← c45621, ← c45622, k2, ← c45623] at e_727_3
  have hr := e_727_3
  simp only [e_727_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4757)) := by
  have c45624 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45625 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45626 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45627 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45628 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45629 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_4 := arithEq_of_rows h (row := 727) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_4
  simp only [← c45624, ← c45625, ← c45626] at e_727_4
  have e_727_5 := arithEq_of_rows h (row := 727) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_5
  simp only [← c45627, ← c45628, k2, ← c45629] at e_727_5
  have hr := e_727_5
  simp only [e_727_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4758)) := by
  have c45630 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45631 := (publicBatchWrapper89_copies1425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45632 := (publicBatchWrapper89_copies1426 a h).1
  have c45633 := (publicBatchWrapper89_copies1426 a h).2.1
  have c45634 := (publicBatchWrapper89_copies1426 a h).2.2.1
  have c45635 := (publicBatchWrapper89_copies1426 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_6 := arithEq_of_rows h (row := 727) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_6
  simp only [← c45630, ← c45631, ← c45632] at e_727_6
  have e_727_7 := arithEq_of_rows h (row := 727) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_7
  simp only [← c45633, ← c45634, k2, ← c45635] at e_727_7
  have hr := e_727_7
  simp only [e_727_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4759)) := by
  have c45636 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.1
  have c45637 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.1
  have c45638 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.1
  have c45639 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.1
  have c45640 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.1
  have c45641 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_8 := arithEq_of_rows h (row := 727) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_8
  simp only [← c45636, ← c45637, ← c45638] at e_727_8
  have e_727_9 := arithEq_of_rows h (row := 727) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_9
  simp only [← c45639, ← c45640, k2, ← c45641] at e_727_9
  have hr := e_727_9
  simp only [e_727_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4760)) := by
  have c45642 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45643 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45644 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45645 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45646 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45647 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_10 := arithEq_of_rows h (row := 727) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_10
  simp only [← c45642, ← c45643, ← c45644] at e_727_10
  have e_727_11 := arithEq_of_rows h (row := 727) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_11
  simp only [← c45645, ← c45646, k2, ← c45647] at e_727_11
  have hr := e_727_11
  simp only [e_727_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4761)) := by
  have c45648 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45649 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45650 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45651 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45652 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45653 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_12 := arithEq_of_rows h (row := 727) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_12
  simp only [← c45648, ← c45649, ← c45650] at e_727_12
  have e_727_13 := arithEq_of_rows h (row := 727) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_13
  simp only [← c45651, ← c45652, k2, ← c45653] at e_727_13
  have hr := e_727_13
  simp only [e_727_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4762)) := by
  have c45654 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45655 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45656 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45657 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45658 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45659 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_14 := arithEq_of_rows h (row := 727) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_14
  simp only [← c45654, ← c45655, ← c45656] at e_727_14
  have e_727_15 := arithEq_of_rows h (row := 727) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_15
  simp only [← c45657, ← c45658, k2, ← c45659] at e_727_15
  have hr := e_727_15
  simp only [e_727_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4763)) := by
  have c45660 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45661 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45662 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45663 := (publicBatchWrapper89_copies1426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45664 := (publicBatchWrapper89_copies1427 a h).1
  have c45665 := (publicBatchWrapper89_copies1427 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_16 := arithEq_of_rows h (row := 727) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_16
  simp only [← c45660, ← c45661, ← c45662] at e_727_16
  have e_727_17 := arithEq_of_rows h (row := 727) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_17
  simp only [← c45663, ← c45664, k2, ← c45665] at e_727_17
  have hr := e_727_17
  simp only [e_727_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 727 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4764)) := by
  have c45666 := (publicBatchWrapper89_copies1427 a h).2.2.1
  have c45667 := (publicBatchWrapper89_copies1427 a h).2.2.2.1
  have c45668 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.1
  have c45669 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.1
  have c45670 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.1
  have c45671 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_727_18 := arithEq_of_rows h (row := 727) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_18
  simp only [← c45666, ← c45667, ← c45668] at e_727_18
  have e_727_19 := arithEq_of_rows h (row := 727) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_727_19
  simp only [← c45669, ← c45670, k2, ← c45671] at e_727_19
  have hr := e_727_19
  simp only [e_727_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4765)) := by
  have c45672 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.1
  have c45673 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.1
  have c45674 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45675 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45676 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45677 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_0 := arithEq_of_rows h (row := 728) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_0
  simp only [← c45672, ← c45673, ← c45674] at e_728_0
  have e_728_1 := arithEq_of_rows h (row := 728) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_1
  simp only [← c45675, ← c45676, k2, ← c45677] at e_728_1
  have hr := e_728_1
  simp only [e_728_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4766)) := by
  have c45678 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45679 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45680 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45681 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45682 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45683 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_2 := arithEq_of_rows h (row := 728) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_2
  simp only [← c45678, ← c45679, ← c45680] at e_728_2
  have e_728_3 := arithEq_of_rows h (row := 728) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_3
  simp only [← c45681, ← c45682, k2, ← c45683] at e_728_3
  have hr := e_728_3
  simp only [e_728_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4767)) := by
  have c45684 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45685 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45686 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45687 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45688 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45689 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_4 := arithEq_of_rows h (row := 728) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_4
  simp only [← c45684, ← c45685, ← c45686] at e_728_4
  have e_728_5 := arithEq_of_rows h (row := 728) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_5
  simp only [← c45687, ← c45688, k2, ← c45689] at e_728_5
  have hr := e_728_5
  simp only [e_728_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4768)) := by
  have c45690 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45691 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45692 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45693 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45694 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45695 := (publicBatchWrapper89_copies1427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_6 := arithEq_of_rows h (row := 728) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_6
  simp only [← c45690, ← c45691, ← c45692] at e_728_6
  have e_728_7 := arithEq_of_rows h (row := 728) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_7
  simp only [← c45693, ← c45694, k2, ← c45695] at e_728_7
  have hr := e_728_7
  simp only [e_728_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4769)) := by
  have c45696 := (publicBatchWrapper89_copies1428 a h).1
  have c45697 := (publicBatchWrapper89_copies1428 a h).2.1
  have c45698 := (publicBatchWrapper89_copies1428 a h).2.2.1
  have c45699 := (publicBatchWrapper89_copies1428 a h).2.2.2.1
  have c45700 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.1
  have c45701 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_8 := arithEq_of_rows h (row := 728) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_8
  simp only [← c45696, ← c45697, ← c45698] at e_728_8
  have e_728_9 := arithEq_of_rows h (row := 728) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_9
  simp only [← c45699, ← c45700, k2, ← c45701] at e_728_9
  have hr := e_728_9
  simp only [e_728_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4810)) := by
  have c45702 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.1
  have c45703 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.1
  have c45704 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.1
  have c45705 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.1
  have c45706 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45707 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_10 := arithEq_of_rows h (row := 728) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_10
  simp only [← c45702, ← c45703, ← c45704] at e_728_10
  have e_728_11 := arithEq_of_rows h (row := 728) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_11
  simp only [← c45705, ← c45706, k2, ← c45707] at e_728_11
  have hr := e_728_11
  simp only [e_728_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4811)) := by
  have c45708 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45709 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45710 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45711 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45712 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45713 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_12 := arithEq_of_rows h (row := 728) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_12
  simp only [← c45708, ← c45709, ← c45710] at e_728_12
  have e_728_13 := arithEq_of_rows h (row := 728) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_13
  simp only [← c45711, ← c45712, k2, ← c45713] at e_728_13
  have hr := e_728_13
  simp only [e_728_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4812)) := by
  have c45714 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45715 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45716 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45717 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45718 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45719 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_14 := arithEq_of_rows h (row := 728) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_14
  simp only [← c45714, ← c45715, ← c45716] at e_728_14
  have e_728_15 := arithEq_of_rows h (row := 728) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_15
  simp only [← c45717, ← c45718, k2, ← c45719] at e_728_15
  have hr := e_728_15
  simp only [e_728_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4813)) := by
  have c45720 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45721 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45722 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45723 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45724 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45725 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_16 := arithEq_of_rows h (row := 728) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_16
  simp only [← c45720, ← c45721, ← c45722] at e_728_16
  have e_728_17 := arithEq_of_rows h (row := 728) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_17
  simp only [← c45723, ← c45724, k2, ← c45725] at e_728_17
  have hr := e_728_17
  simp only [e_728_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 728 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4814)) := by
  have c45726 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45727 := (publicBatchWrapper89_copies1428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45728 := (publicBatchWrapper89_copies1429 a h).1
  have c45729 := (publicBatchWrapper89_copies1429 a h).2.1
  have c45730 := (publicBatchWrapper89_copies1429 a h).2.2.1
  have c45731 := (publicBatchWrapper89_copies1429 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_728_18 := arithEq_of_rows h (row := 728) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_18
  simp only [← c45726, ← c45727, ← c45728] at e_728_18
  have e_728_19 := arithEq_of_rows h (row := 728) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_728_19
  simp only [← c45729, ← c45730, k2, ← c45731] at e_728_19
  have hr := e_728_19
  simp only [e_728_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4815)) := by
  have c45732 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.1
  have c45733 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.1
  have c45734 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.1
  have c45735 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.1
  have c45736 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.1
  have c45737 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_0 := arithEq_of_rows h (row := 729) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_0
  simp only [← c45732, ← c45733, ← c45734] at e_729_0
  have e_729_1 := arithEq_of_rows h (row := 729) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_1
  simp only [← c45735, ← c45736, k2, ← c45737] at e_729_1
  have hr := e_729_1
  simp only [e_729_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4816)) := by
  have c45738 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45739 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45740 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45741 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45742 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45743 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_2 := arithEq_of_rows h (row := 729) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_2
  simp only [← c45738, ← c45739, ← c45740] at e_729_2
  have e_729_3 := arithEq_of_rows h (row := 729) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_3
  simp only [← c45741, ← c45742, k2, ← c45743] at e_729_3
  have hr := e_729_3
  simp only [e_729_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4817)) := by
  have c45744 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45745 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45746 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45747 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45748 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45749 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_4 := arithEq_of_rows h (row := 729) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_4
  simp only [← c45744, ← c45745, ← c45746] at e_729_4
  have e_729_5 := arithEq_of_rows h (row := 729) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_5
  simp only [← c45747, ← c45748, k2, ← c45749] at e_729_5
  have hr := e_729_5
  simp only [e_729_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4818)) := by
  have c45750 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45751 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45752 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45753 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45754 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45755 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_6 := arithEq_of_rows h (row := 729) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_6
  simp only [← c45750, ← c45751, ← c45752] at e_729_6
  have e_729_7 := arithEq_of_rows h (row := 729) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_7
  simp only [← c45753, ← c45754, k2, ← c45755] at e_729_7
  have hr := e_729_7
  simp only [e_729_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4819)) := by
  have c45756 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45757 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45758 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45759 := (publicBatchWrapper89_copies1429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45760 := (publicBatchWrapper89_copies1430 a h).1
  have c45761 := (publicBatchWrapper89_copies1430 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_8 := arithEq_of_rows h (row := 729) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_8
  simp only [← c45756, ← c45757, ← c45758] at e_729_8
  have e_729_9 := arithEq_of_rows h (row := 729) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_9
  simp only [← c45759, ← c45760, k2, ← c45761] at e_729_9
  have hr := e_729_9
  simp only [e_729_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4820)) := by
  have c45762 := (publicBatchWrapper89_copies1430 a h).2.2.1
  have c45763 := (publicBatchWrapper89_copies1430 a h).2.2.2.1
  have c45764 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.1
  have c45765 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.1
  have c45766 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.1
  have c45767 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_10 := arithEq_of_rows h (row := 729) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_10
  simp only [← c45762, ← c45763, ← c45764] at e_729_10
  have e_729_11 := arithEq_of_rows h (row := 729) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_11
  simp only [← c45765, ← c45766, k2, ← c45767] at e_729_11
  have hr := e_729_11
  simp only [e_729_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4821)) := by
  have c45768 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.1
  have c45769 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.1
  have c45770 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45771 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45772 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45773 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_12 := arithEq_of_rows h (row := 729) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_12
  simp only [← c45768, ← c45769, ← c45770] at e_729_12
  have e_729_13 := arithEq_of_rows h (row := 729) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_13
  simp only [← c45771, ← c45772, k2, ← c45773] at e_729_13
  have hr := e_729_13
  simp only [e_729_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4822)) := by
  have c45774 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45775 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45776 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45777 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45778 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45779 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_14 := arithEq_of_rows h (row := 729) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_14
  simp only [← c45774, ← c45775, ← c45776] at e_729_14
  have e_729_15 := arithEq_of_rows h (row := 729) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_15
  simp only [← c45777, ← c45778, k2, ← c45779] at e_729_15
  have hr := e_729_15
  simp only [e_729_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4823)) := by
  have c45780 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45781 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45782 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45783 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45784 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45785 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_16 := arithEq_of_rows h (row := 729) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_16
  simp only [← c45780, ← c45781, ← c45782] at e_729_16
  have e_729_17 := arithEq_of_rows h (row := 729) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_17
  simp only [← c45783, ← c45784, k2, ← c45785] at e_729_17
  have hr := e_729_17
  simp only [e_729_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 729 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4824)) := by
  have c45786 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45787 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45788 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45789 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45790 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45791 := (publicBatchWrapper89_copies1430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_729_18 := arithEq_of_rows h (row := 729) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_18
  simp only [← c45786, ← c45787, ← c45788] at e_729_18
  have e_729_19 := arithEq_of_rows h (row := 729) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_729_19
  simp only [← c45789, ← c45790, k2, ← c45791] at e_729_19
  have hr := e_729_19
  simp only [e_729_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4825)) := by
  have c45792 := (publicBatchWrapper89_copies1431 a h).1
  have c45793 := (publicBatchWrapper89_copies1431 a h).2.1
  have c45794 := (publicBatchWrapper89_copies1431 a h).2.2.1
  have c45795 := (publicBatchWrapper89_copies1431 a h).2.2.2.1
  have c45796 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.1
  have c45797 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_0 := arithEq_of_rows h (row := 730) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_0
  simp only [← c45792, ← c45793, ← c45794] at e_730_0
  have e_730_1 := arithEq_of_rows h (row := 730) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_1
  simp only [← c45795, ← c45796, k2, ← c45797] at e_730_1
  have hr := e_730_1
  simp only [e_730_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4826)) := by
  have c45798 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.1
  have c45799 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.1
  have c45800 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.1
  have c45801 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.1
  have c45802 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45803 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_2 := arithEq_of_rows h (row := 730) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_2
  simp only [← c45798, ← c45799, ← c45800] at e_730_2
  have e_730_3 := arithEq_of_rows h (row := 730) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_3
  simp only [← c45801, ← c45802, k2, ← c45803] at e_730_3
  have hr := e_730_3
  simp only [e_730_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4827)) := by
  have c45804 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45805 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45806 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45807 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45808 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45809 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_4 := arithEq_of_rows h (row := 730) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_4
  simp only [← c45804, ← c45805, ← c45806] at e_730_4
  have e_730_5 := arithEq_of_rows h (row := 730) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_5
  simp only [← c45807, ← c45808, k2, ← c45809] at e_730_5
  have hr := e_730_5
  simp only [e_730_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4828)) := by
  have c45810 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45811 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45812 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45813 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45814 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45815 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_6 := arithEq_of_rows h (row := 730) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_6
  simp only [← c45810, ← c45811, ← c45812] at e_730_6
  have e_730_7 := arithEq_of_rows h (row := 730) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_7
  simp only [← c45813, ← c45814, k2, ← c45815] at e_730_7
  have hr := e_730_7
  simp only [e_730_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4829)) := by
  have c45816 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45817 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45818 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45819 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45820 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45821 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_8 := arithEq_of_rows h (row := 730) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_8
  simp only [← c45816, ← c45817, ← c45818] at e_730_8
  have e_730_9 := arithEq_of_rows h (row := 730) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_9
  simp only [← c45819, ← c45820, k2, ← c45821] at e_730_9
  have hr := e_730_9
  simp only [e_730_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4830)) := by
  have c45822 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45823 := (publicBatchWrapper89_copies1431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45824 := (publicBatchWrapper89_copies1432 a h).1
  have c45825 := (publicBatchWrapper89_copies1432 a h).2.1
  have c45826 := (publicBatchWrapper89_copies1432 a h).2.2.1
  have c45827 := (publicBatchWrapper89_copies1432 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_10 := arithEq_of_rows h (row := 730) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_10
  simp only [← c45822, ← c45823, ← c45824] at e_730_10
  have e_730_11 := arithEq_of_rows h (row := 730) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_11
  simp only [← c45825, ← c45826, k2, ← c45827] at e_730_11
  have hr := e_730_11
  simp only [e_730_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4831)) := by
  have c45828 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.1
  have c45829 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.1
  have c45830 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.1
  have c45831 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.1
  have c45832 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.1
  have c45833 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_12 := arithEq_of_rows h (row := 730) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_12
  simp only [← c45828, ← c45829, ← c45830] at e_730_12
  have e_730_13 := arithEq_of_rows h (row := 730) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_13
  simp only [← c45831, ← c45832, k2, ← c45833] at e_730_13
  have hr := e_730_13
  simp only [e_730_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4832)) := by
  have c45834 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45835 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45836 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45837 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45838 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45839 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_14 := arithEq_of_rows h (row := 730) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_14
  simp only [← c45834, ← c45835, ← c45836] at e_730_14
  have e_730_15 := arithEq_of_rows h (row := 730) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_15
  simp only [← c45837, ← c45838, k2, ← c45839] at e_730_15
  have hr := e_730_15
  simp only [e_730_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4833)) := by
  have c45840 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45841 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45842 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45843 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45844 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45845 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_16 := arithEq_of_rows h (row := 730) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_16
  simp only [← c45840, ← c45841, ← c45842] at e_730_16
  have e_730_17 := arithEq_of_rows h (row := 730) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_17
  simp only [← c45843, ← c45844, k2, ← c45845] at e_730_17
  have hr := e_730_17
  simp only [e_730_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 730 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4834)) := by
  have c45846 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45847 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45848 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45849 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45850 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45851 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_730_18 := arithEq_of_rows h (row := 730) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_18
  simp only [← c45846, ← c45847, ← c45848] at e_730_18
  have e_730_19 := arithEq_of_rows h (row := 730) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_730_19
  simp only [← c45849, ← c45850, k2, ← c45851] at e_730_19
  have hr := e_730_19
  simp only [e_730_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4835)) := by
  have c45852 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45853 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45854 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45855 := (publicBatchWrapper89_copies1432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45856 := (publicBatchWrapper89_copies1433 a h).1
  have c45857 := (publicBatchWrapper89_copies1433 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_0 := arithEq_of_rows h (row := 731) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_0
  simp only [← c45852, ← c45853, ← c45854] at e_731_0
  have e_731_1 := arithEq_of_rows h (row := 731) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_1
  simp only [← c45855, ← c45856, k2, ← c45857] at e_731_1
  have hr := e_731_1
  simp only [e_731_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4836)) := by
  have c45858 := (publicBatchWrapper89_copies1433 a h).2.2.1
  have c45859 := (publicBatchWrapper89_copies1433 a h).2.2.2.1
  have c45860 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.1
  have c45861 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.1
  have c45862 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.1
  have c45863 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_2 := arithEq_of_rows h (row := 731) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_2
  simp only [← c45858, ← c45859, ← c45860] at e_731_2
  have e_731_3 := arithEq_of_rows h (row := 731) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_3
  simp only [← c45861, ← c45862, k2, ← c45863] at e_731_3
  have hr := e_731_3
  simp only [e_731_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4837)) := by
  have c45864 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.1
  have c45865 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.1
  have c45866 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45867 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45868 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45869 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_4 := arithEq_of_rows h (row := 731) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_4
  simp only [← c45864, ← c45865, ← c45866] at e_731_4
  have e_731_5 := arithEq_of_rows h (row := 731) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_5
  simp only [← c45867, ← c45868, k2, ← c45869] at e_731_5
  have hr := e_731_5
  simp only [e_731_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4838)) := by
  have c45870 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45871 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45872 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45873 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45874 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45875 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_6 := arithEq_of_rows h (row := 731) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_6
  simp only [← c45870, ← c45871, ← c45872] at e_731_6
  have e_731_7 := arithEq_of_rows h (row := 731) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_7
  simp only [← c45873, ← c45874, k2, ← c45875] at e_731_7
  have hr := e_731_7
  simp only [e_731_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4839)) := by
  have c45876 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45877 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45878 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45879 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45880 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45881 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_8 := arithEq_of_rows h (row := 731) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_8
  simp only [← c45876, ← c45877, ← c45878] at e_731_8
  have e_731_9 := arithEq_of_rows h (row := 731) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_9
  simp only [← c45879, ← c45880, k2, ← c45881] at e_731_9
  have hr := e_731_9
  simp only [e_731_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4840)) := by
  have c45882 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45883 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45884 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45885 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45886 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45887 := (publicBatchWrapper89_copies1433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_10 := arithEq_of_rows h (row := 731) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_10
  simp only [← c45882, ← c45883, ← c45884] at e_731_10
  have e_731_11 := arithEq_of_rows h (row := 731) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_11
  simp only [← c45885, ← c45886, k2, ← c45887] at e_731_11
  have hr := e_731_11
  simp only [e_731_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4841)) := by
  have c45888 := (publicBatchWrapper89_copies1434 a h).1
  have c45889 := (publicBatchWrapper89_copies1434 a h).2.1
  have c45890 := (publicBatchWrapper89_copies1434 a h).2.2.1
  have c45891 := (publicBatchWrapper89_copies1434 a h).2.2.2.1
  have c45892 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.1
  have c45893 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_12 := arithEq_of_rows h (row := 731) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_12
  simp only [← c45888, ← c45889, ← c45890] at e_731_12
  have e_731_13 := arithEq_of_rows h (row := 731) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_13
  simp only [← c45891, ← c45892, k2, ← c45893] at e_731_13
  have hr := e_731_13
  simp only [e_731_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4842)) := by
  have c45894 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.1
  have c45895 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.1
  have c45896 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.1
  have c45897 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.1
  have c45898 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45899 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_14 := arithEq_of_rows h (row := 731) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_14
  simp only [← c45894, ← c45895, ← c45896] at e_731_14
  have e_731_15 := arithEq_of_rows h (row := 731) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_15
  simp only [← c45897, ← c45898, k2, ← c45899] at e_731_15
  have hr := e_731_15
  simp only [e_731_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4843)) := by
  have c45900 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45901 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45902 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45903 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45904 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45905 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_16 := arithEq_of_rows h (row := 731) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_16
  simp only [← c45900, ← c45901, ← c45902] at e_731_16
  have e_731_17 := arithEq_of_rows h (row := 731) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_17
  simp only [← c45903, ← c45904, k2, ← c45905] at e_731_17
  have hr := e_731_17
  simp only [e_731_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 731 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4844)) := by
  have c45906 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45907 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45908 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45909 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45910 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45911 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_731_18 := arithEq_of_rows h (row := 731) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_18
  simp only [← c45906, ← c45907, ← c45908] at e_731_18
  have e_731_19 := arithEq_of_rows h (row := 731) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_731_19
  simp only [← c45909, ← c45910, k2, ← c45911] at e_731_19
  have hr := e_731_19
  simp only [e_731_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4845)) := by
  have c45912 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45913 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45914 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45915 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45916 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45917 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_0 := arithEq_of_rows h (row := 732) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_0
  simp only [← c45912, ← c45913, ← c45914] at e_732_0
  have e_732_1 := arithEq_of_rows h (row := 732) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_1
  simp only [← c45915, ← c45916, k2, ← c45917] at e_732_1
  have hr := e_732_1
  simp only [e_732_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4846)) := by
  have c45918 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45919 := (publicBatchWrapper89_copies1434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45920 := (publicBatchWrapper89_copies1435 a h).1
  have c45921 := (publicBatchWrapper89_copies1435 a h).2.1
  have c45922 := (publicBatchWrapper89_copies1435 a h).2.2.1
  have c45923 := (publicBatchWrapper89_copies1435 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_2 := arithEq_of_rows h (row := 732) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_2
  simp only [← c45918, ← c45919, ← c45920] at e_732_2
  have e_732_3 := arithEq_of_rows h (row := 732) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_3
  simp only [← c45921, ← c45922, k2, ← c45923] at e_732_3
  have hr := e_732_3
  simp only [e_732_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4847)) := by
  have c45924 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.1
  have c45925 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.1
  have c45926 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.1
  have c45927 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.1
  have c45928 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.1
  have c45929 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_4 := arithEq_of_rows h (row := 732) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_4
  simp only [← c45924, ← c45925, ← c45926] at e_732_4
  have e_732_5 := arithEq_of_rows h (row := 732) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_5
  simp only [← c45927, ← c45928, k2, ← c45929] at e_732_5
  have hr := e_732_5
  simp only [e_732_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4848)) := by
  have c45930 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45931 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45932 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45933 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45934 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45935 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_6 := arithEq_of_rows h (row := 732) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_6
  simp only [← c45930, ← c45931, ← c45932] at e_732_6
  have e_732_7 := arithEq_of_rows h (row := 732) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_7
  simp only [← c45933, ← c45934, k2, ← c45935] at e_732_7
  have hr := e_732_7
  simp only [e_732_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4849)) := by
  have c45936 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45937 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45938 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45939 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45940 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45941 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_8 := arithEq_of_rows h (row := 732) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_8
  simp only [← c45936, ← c45937, ← c45938] at e_732_8
  have e_732_9 := arithEq_of_rows h (row := 732) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_9
  simp only [← c45939, ← c45940, k2, ← c45941] at e_732_9
  have hr := e_732_9
  simp only [e_732_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4850)) := by
  have c45942 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45943 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45944 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45945 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45946 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45947 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_10 := arithEq_of_rows h (row := 732) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_10
  simp only [← c45942, ← c45943, ← c45944] at e_732_10
  have e_732_11 := arithEq_of_rows h (row := 732) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_11
  simp only [← c45945, ← c45946, k2, ← c45947] at e_732_11
  have hr := e_732_11
  simp only [e_732_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4851)) := by
  have c45948 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45949 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45950 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45951 := (publicBatchWrapper89_copies1435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45952 := (publicBatchWrapper89_copies1436 a h).1
  have c45953 := (publicBatchWrapper89_copies1436 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_12 := arithEq_of_rows h (row := 732) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_12
  simp only [← c45948, ← c45949, ← c45950] at e_732_12
  have e_732_13 := arithEq_of_rows h (row := 732) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_13
  simp only [← c45951, ← c45952, k2, ← c45953] at e_732_13
  have hr := e_732_13
  simp only [e_732_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4852)) := by
  have c45954 := (publicBatchWrapper89_copies1436 a h).2.2.1
  have c45955 := (publicBatchWrapper89_copies1436 a h).2.2.2.1
  have c45956 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.1
  have c45957 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.1
  have c45958 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.1
  have c45959 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_14 := arithEq_of_rows h (row := 732) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_14
  simp only [← c45954, ← c45955, ← c45956] at e_732_14
  have e_732_15 := arithEq_of_rows h (row := 732) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_15
  simp only [← c45957, ← c45958, k2, ← c45959] at e_732_15
  have hr := e_732_15
  simp only [e_732_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4853)) := by
  have c45960 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.1
  have c45961 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.1
  have c45962 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45963 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45964 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45965 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_16 := arithEq_of_rows h (row := 732) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_16
  simp only [← c45960, ← c45961, ← c45962] at e_732_16
  have e_732_17 := arithEq_of_rows h (row := 732) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_17
  simp only [← c45963, ← c45964, k2, ← c45965] at e_732_17
  have hr := e_732_17
  simp only [e_732_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 732 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4854)) := by
  have c45966 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45967 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45968 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45969 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45970 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45971 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_732_18 := arithEq_of_rows h (row := 732) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_18
  simp only [← c45966, ← c45967, ← c45968] at e_732_18
  have e_732_19 := arithEq_of_rows h (row := 732) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_732_19
  simp only [← c45969, ← c45970, k2, ← c45971] at e_732_19
  have hr := e_732_19
  simp only [e_732_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4855)) := by
  have c45972 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45973 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45974 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45975 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45976 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45977 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_0 := arithEq_of_rows h (row := 733) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_0
  simp only [← c45972, ← c45973, ← c45974] at e_733_0
  have e_733_1 := arithEq_of_rows h (row := 733) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_1
  simp only [← c45975, ← c45976, k2, ← c45977] at e_733_1
  have hr := e_733_1
  simp only [e_733_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4856)) := by
  have c45978 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45979 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45980 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45981 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45982 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45983 := (publicBatchWrapper89_copies1436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_2 := arithEq_of_rows h (row := 733) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_2
  simp only [← c45978, ← c45979, ← c45980] at e_733_2
  have e_733_3 := arithEq_of_rows h (row := 733) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_3
  simp only [← c45981, ← c45982, k2, ← c45983] at e_733_3
  have hr := e_733_3
  simp only [e_733_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4857)) := by
  have c45984 := (publicBatchWrapper89_copies1437 a h).1
  have c45985 := (publicBatchWrapper89_copies1437 a h).2.1
  have c45986 := (publicBatchWrapper89_copies1437 a h).2.2.1
  have c45987 := (publicBatchWrapper89_copies1437 a h).2.2.2.1
  have c45988 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.1
  have c45989 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_4 := arithEq_of_rows h (row := 733) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_4
  simp only [← c45984, ← c45985, ← c45986] at e_733_4
  have e_733_5 := arithEq_of_rows h (row := 733) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_5
  simp only [← c45987, ← c45988, k2, ← c45989] at e_733_5
  have hr := e_733_5
  simp only [e_733_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4858)) := by
  have c45990 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.1
  have c45991 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.1
  have c45992 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.1
  have c45993 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.1
  have c45994 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45995 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_6 := arithEq_of_rows h (row := 733) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_6
  simp only [← c45990, ← c45991, ← c45992] at e_733_6
  have e_733_7 := arithEq_of_rows h (row := 733) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_7
  simp only [← c45993, ← c45994, k2, ← c45995] at e_733_7
  have hr := e_733_7
  simp only [e_733_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4859)) := by
  have c45996 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45997 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45998 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45999 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46000 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46001 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_8 := arithEq_of_rows h (row := 733) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_8
  simp only [← c45996, ← c45997, ← c45998] at e_733_8
  have e_733_9 := arithEq_of_rows h (row := 733) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_9
  simp only [← c45999, ← c46000, k2, ← c46001] at e_733_9
  have hr := e_733_9
  simp only [e_733_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4860)) := by
  have c46002 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46003 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46004 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46005 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46006 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46007 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_10 := arithEq_of_rows h (row := 733) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_10
  simp only [← c46002, ← c46003, ← c46004] at e_733_10
  have e_733_11 := arithEq_of_rows h (row := 733) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_11
  simp only [← c46005, ← c46006, k2, ← c46007] at e_733_11
  have hr := e_733_11
  simp only [e_733_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4861)) := by
  have c46008 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46009 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46010 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46011 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46012 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46013 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_12 := arithEq_of_rows h (row := 733) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_12
  simp only [← c46008, ← c46009, ← c46010] at e_733_12
  have e_733_13 := arithEq_of_rows h (row := 733) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_13
  simp only [← c46011, ← c46012, k2, ← c46013] at e_733_13
  have hr := e_733_13
  simp only [e_733_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4862)) := by
  have c46014 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46015 := (publicBatchWrapper89_copies1437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46016 := (publicBatchWrapper89_copies1438 a h).1
  have c46017 := (publicBatchWrapper89_copies1438 a h).2.1
  have c46018 := (publicBatchWrapper89_copies1438 a h).2.2.1
  have c46019 := (publicBatchWrapper89_copies1438 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_14 := arithEq_of_rows h (row := 733) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_14
  simp only [← c46014, ← c46015, ← c46016] at e_733_14
  have e_733_15 := arithEq_of_rows h (row := 733) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_15
  simp only [← c46017, ← c46018, k2, ← c46019] at e_733_15
  have hr := e_733_15
  simp only [e_733_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4863)) := by
  have c46020 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.1
  have c46021 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.1
  have c46022 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.1
  have c46023 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.1
  have c46024 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.1
  have c46025 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_16 := arithEq_of_rows h (row := 733) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_16
  simp only [← c46020, ← c46021, ← c46022] at e_733_16
  have e_733_17 := arithEq_of_rows h (row := 733) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_17
  simp only [← c46023, ← c46024, k2, ← c46025] at e_733_17
  have hr := e_733_17
  simp only [e_733_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 733 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4864)) := by
  have c46026 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46027 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46028 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46029 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46030 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46031 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_733_18 := arithEq_of_rows h (row := 733) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_18
  simp only [← c46026, ← c46027, ← c46028] at e_733_18
  have e_733_19 := arithEq_of_rows h (row := 733) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_733_19
  simp only [← c46029, ← c46030, k2, ← c46031] at e_733_19
  have hr := e_733_19
  simp only [e_733_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4865)) := by
  have c46032 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46033 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46034 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46035 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46036 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46037 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_0 := arithEq_of_rows h (row := 734) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_0
  simp only [← c46032, ← c46033, ← c46034] at e_734_0
  have e_734_1 := arithEq_of_rows h (row := 734) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_1
  simp only [← c46035, ← c46036, k2, ← c46037] at e_734_1
  have hr := e_734_1
  simp only [e_734_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4866)) := by
  have c46038 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46039 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46040 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46041 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46042 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46043 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_2 := arithEq_of_rows h (row := 734) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_2
  simp only [← c46038, ← c46039, ← c46040] at e_734_2
  have e_734_3 := arithEq_of_rows h (row := 734) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_3
  simp only [← c46041, ← c46042, k2, ← c46043] at e_734_3
  have hr := e_734_3
  simp only [e_734_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4867)) := by
  have c46044 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46045 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46046 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46047 := (publicBatchWrapper89_copies1438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46048 := (publicBatchWrapper89_copies1439 a h).1
  have c46049 := (publicBatchWrapper89_copies1439 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_4 := arithEq_of_rows h (row := 734) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_4
  simp only [← c46044, ← c46045, ← c46046] at e_734_4
  have e_734_5 := arithEq_of_rows h (row := 734) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_5
  simp only [← c46047, ← c46048, k2, ← c46049] at e_734_5
  have hr := e_734_5
  simp only [e_734_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4868)) := by
  have c46050 := (publicBatchWrapper89_copies1439 a h).2.2.1
  have c46051 := (publicBatchWrapper89_copies1439 a h).2.2.2.1
  have c46052 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.1
  have c46053 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.1
  have c46054 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.1
  have c46055 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_6 := arithEq_of_rows h (row := 734) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_6
  simp only [← c46050, ← c46051, ← c46052] at e_734_6
  have e_734_7 := arithEq_of_rows h (row := 734) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_7
  simp only [← c46053, ← c46054, k2, ← c46055] at e_734_7
  have hr := e_734_7
  simp only [e_734_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4869)) := by
  have c46056 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.1
  have c46057 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.1
  have c46058 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46059 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46060 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46061 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_8 := arithEq_of_rows h (row := 734) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_8
  simp only [← c46056, ← c46057, ← c46058] at e_734_8
  have e_734_9 := arithEq_of_rows h (row := 734) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_9
  simp only [← c46059, ← c46060, k2, ← c46061] at e_734_9
  have hr := e_734_9
  simp only [e_734_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4870)) := by
  have c46062 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46063 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46064 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46065 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46066 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46067 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_10 := arithEq_of_rows h (row := 734) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_10
  simp only [← c46062, ← c46063, ← c46064] at e_734_10
  have e_734_11 := arithEq_of_rows h (row := 734) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_11
  simp only [← c46065, ← c46066, k2, ← c46067] at e_734_11
  have hr := e_734_11
  simp only [e_734_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4871)) := by
  have c46068 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46069 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46070 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46071 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46072 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46073 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_12 := arithEq_of_rows h (row := 734) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_12
  simp only [← c46068, ← c46069, ← c46070] at e_734_12
  have e_734_13 := arithEq_of_rows h (row := 734) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_13
  simp only [← c46071, ← c46072, k2, ← c46073] at e_734_13
  have hr := e_734_13
  simp only [e_734_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4872)) := by
  have c46074 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46075 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46076 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46077 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46078 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46079 := (publicBatchWrapper89_copies1439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_14 := arithEq_of_rows h (row := 734) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_14
  simp only [← c46074, ← c46075, ← c46076] at e_734_14
  have e_734_15 := arithEq_of_rows h (row := 734) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_15
  simp only [← c46077, ← c46078, k2, ← c46079] at e_734_15
  have hr := e_734_15
  simp only [e_734_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4873)) := by
  have c46080 := (publicBatchWrapper89_copies1440 a h).1
  have c46081 := (publicBatchWrapper89_copies1440 a h).2.1
  have c46082 := (publicBatchWrapper89_copies1440 a h).2.2.1
  have c46083 := (publicBatchWrapper89_copies1440 a h).2.2.2.1
  have c46084 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.1
  have c46085 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_16 := arithEq_of_rows h (row := 734) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_16
  simp only [← c46080, ← c46081, ← c46082] at e_734_16
  have e_734_17 := arithEq_of_rows h (row := 734) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_17
  simp only [← c46083, ← c46084, k2, ← c46085] at e_734_17
  have hr := e_734_17
  simp only [e_734_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 734 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4874)) := by
  have c46086 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.1
  have c46087 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.1
  have c46088 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.1
  have c46089 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.1
  have c46090 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46091 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_734_18 := arithEq_of_rows h (row := 734) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_18
  simp only [← c46086, ← c46087, ← c46088] at e_734_18
  have e_734_19 := arithEq_of_rows h (row := 734) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_734_19
  simp only [← c46089, ← c46090, k2, ← c46091] at e_734_19
  have hr := e_734_19
  simp only [e_734_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4875)) := by
  have c46092 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46093 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46094 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46095 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46096 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46097 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_0 := arithEq_of_rows h (row := 735) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_0
  simp only [← c46092, ← c46093, ← c46094] at e_735_0
  have e_735_1 := arithEq_of_rows h (row := 735) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_1
  simp only [← c46095, ← c46096, k2, ← c46097] at e_735_1
  have hr := e_735_1
  simp only [e_735_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4876)) := by
  have c46098 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46099 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46100 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46101 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46102 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46103 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_2 := arithEq_of_rows h (row := 735) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_2
  simp only [← c46098, ← c46099, ← c46100] at e_735_2
  have e_735_3 := arithEq_of_rows h (row := 735) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_3
  simp only [← c46101, ← c46102, k2, ← c46103] at e_735_3
  have hr := e_735_3
  simp only [e_735_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4877)) := by
  have c46104 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46105 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46106 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46107 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46108 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46109 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_4 := arithEq_of_rows h (row := 735) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_4
  simp only [← c46104, ← c46105, ← c46106] at e_735_4
  have e_735_5 := arithEq_of_rows h (row := 735) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_5
  simp only [← c46107, ← c46108, k2, ← c46109] at e_735_5
  have hr := e_735_5
  simp only [e_735_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4878)) := by
  have c46110 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46111 := (publicBatchWrapper89_copies1440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46112 := (publicBatchWrapper89_copies1441 a h).1
  have c46113 := (publicBatchWrapper89_copies1441 a h).2.1
  have c46114 := (publicBatchWrapper89_copies1441 a h).2.2.1
  have c46115 := (publicBatchWrapper89_copies1441 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_6 := arithEq_of_rows h (row := 735) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_6
  simp only [← c46110, ← c46111, ← c46112] at e_735_6
  have e_735_7 := arithEq_of_rows h (row := 735) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_7
  simp only [← c46113, ← c46114, k2, ← c46115] at e_735_7
  have hr := e_735_7
  simp only [e_735_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4879)) := by
  have c46116 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.1
  have c46117 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.1
  have c46118 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.1
  have c46119 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.1
  have c46120 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.1
  have c46121 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_8 := arithEq_of_rows h (row := 735) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_8
  simp only [← c46116, ← c46117, ← c46118] at e_735_8
  have e_735_9 := arithEq_of_rows h (row := 735) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_9
  simp only [← c46119, ← c46120, k2, ← c46121] at e_735_9
  have hr := e_735_9
  simp only [e_735_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4880)) := by
  have c46122 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46123 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46124 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46125 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46126 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46127 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_10 := arithEq_of_rows h (row := 735) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_10
  simp only [← c46122, ← c46123, ← c46124] at e_735_10
  have e_735_11 := arithEq_of_rows h (row := 735) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_11
  simp only [← c46125, ← c46126, k2, ← c46127] at e_735_11
  have hr := e_735_11
  simp only [e_735_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4881)) := by
  have c46128 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46129 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46130 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46131 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46132 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46133 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_12 := arithEq_of_rows h (row := 735) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_12
  simp only [← c46128, ← c46129, ← c46130] at e_735_12
  have e_735_13 := arithEq_of_rows h (row := 735) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_13
  simp only [← c46131, ← c46132, k2, ← c46133] at e_735_13
  have hr := e_735_13
  simp only [e_735_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4882)) := by
  have c46134 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46135 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46136 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46137 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46138 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46139 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_14 := arithEq_of_rows h (row := 735) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_14
  simp only [← c46134, ← c46135, ← c46136] at e_735_14
  have e_735_15 := arithEq_of_rows h (row := 735) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_15
  simp only [← c46137, ← c46138, k2, ← c46139] at e_735_15
  have hr := e_735_15
  simp only [e_735_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4883)) := by
  have c46140 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46141 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46142 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46143 := (publicBatchWrapper89_copies1441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46144 := (publicBatchWrapper89_copies1442 a h).1
  have c46145 := (publicBatchWrapper89_copies1442 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_16 := arithEq_of_rows h (row := 735) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_16
  simp only [← c46140, ← c46141, ← c46142] at e_735_16
  have e_735_17 := arithEq_of_rows h (row := 735) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_17
  simp only [← c46143, ← c46144, k2, ← c46145] at e_735_17
  have hr := e_735_17
  simp only [e_735_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 735 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4884)) := by
  have c46146 := (publicBatchWrapper89_copies1442 a h).2.2.1
  have c46147 := (publicBatchWrapper89_copies1442 a h).2.2.2.1
  have c46148 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.1
  have c46149 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.1
  have c46150 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.1
  have c46151 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_735_18 := arithEq_of_rows h (row := 735) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_18
  simp only [← c46146, ← c46147, ← c46148] at e_735_18
  have e_735_19 := arithEq_of_rows h (row := 735) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_735_19
  simp only [← c46149, ← c46150, k2, ← c46151] at e_735_19
  have hr := e_735_19
  simp only [e_735_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4885)) := by
  have c46152 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.1
  have c46153 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.1
  have c46154 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46155 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46156 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46157 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_0 := arithEq_of_rows h (row := 736) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_0
  simp only [← c46152, ← c46153, ← c46154] at e_736_0
  have e_736_1 := arithEq_of_rows h (row := 736) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_1
  simp only [← c46155, ← c46156, k2, ← c46157] at e_736_1
  have hr := e_736_1
  simp only [e_736_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4886)) := by
  have c46158 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46159 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46160 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46161 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46162 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46163 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_2 := arithEq_of_rows h (row := 736) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_2
  simp only [← c46158, ← c46159, ← c46160] at e_736_2
  have e_736_3 := arithEq_of_rows h (row := 736) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_3
  simp only [← c46161, ← c46162, k2, ← c46163] at e_736_3
  have hr := e_736_3
  simp only [e_736_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4887)) := by
  have c46164 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46165 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46166 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46167 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46168 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46169 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_4 := arithEq_of_rows h (row := 736) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_4
  simp only [← c46164, ← c46165, ← c46166] at e_736_4
  have e_736_5 := arithEq_of_rows h (row := 736) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_5
  simp only [← c46167, ← c46168, k2, ← c46169] at e_736_5
  have hr := e_736_5
  simp only [e_736_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4888)) := by
  have c46170 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46171 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46172 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46173 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46174 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46175 := (publicBatchWrapper89_copies1442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_6 := arithEq_of_rows h (row := 736) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_6
  simp only [← c46170, ← c46171, ← c46172] at e_736_6
  have e_736_7 := arithEq_of_rows h (row := 736) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_7
  simp only [← c46173, ← c46174, k2, ← c46175] at e_736_7
  have hr := e_736_7
  simp only [e_736_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4889)) := by
  have c46176 := (publicBatchWrapper89_copies1443 a h).1
  have c46177 := (publicBatchWrapper89_copies1443 a h).2.1
  have c46178 := (publicBatchWrapper89_copies1443 a h).2.2.1
  have c46179 := (publicBatchWrapper89_copies1443 a h).2.2.2.1
  have c46180 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.1
  have c46181 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_8 := arithEq_of_rows h (row := 736) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_8
  simp only [← c46176, ← c46177, ← c46178] at e_736_8
  have e_736_9 := arithEq_of_rows h (row := 736) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_9
  simp only [← c46179, ← c46180, k2, ← c46181] at e_736_9
  have hr := e_736_9
  simp only [e_736_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4890)) := by
  have c46182 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.1
  have c46183 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.1
  have c46184 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.1
  have c46185 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.1
  have c46186 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46187 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_10 := arithEq_of_rows h (row := 736) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_10
  simp only [← c46182, ← c46183, ← c46184] at e_736_10
  have e_736_11 := arithEq_of_rows h (row := 736) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_11
  simp only [← c46185, ← c46186, k2, ← c46187] at e_736_11
  have hr := e_736_11
  simp only [e_736_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4891)) := by
  have c46188 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46189 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46190 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46191 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46192 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46193 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_12 := arithEq_of_rows h (row := 736) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_12
  simp only [← c46188, ← c46189, ← c46190] at e_736_12
  have e_736_13 := arithEq_of_rows h (row := 736) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_13
  simp only [← c46191, ← c46192, k2, ← c46193] at e_736_13
  have hr := e_736_13
  simp only [e_736_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4892)) := by
  have c46194 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46195 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46196 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46197 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46198 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46199 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_14 := arithEq_of_rows h (row := 736) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_14
  simp only [← c46194, ← c46195, ← c46196] at e_736_14
  have e_736_15 := arithEq_of_rows h (row := 736) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_15
  simp only [← c46197, ← c46198, k2, ← c46199] at e_736_15
  have hr := e_736_15
  simp only [e_736_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4893)) := by
  have c46200 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46201 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46202 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46203 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46204 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46205 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_16 := arithEq_of_rows h (row := 736) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_16
  simp only [← c46200, ← c46201, ← c46202] at e_736_16
  have e_736_17 := arithEq_of_rows h (row := 736) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_17
  simp only [← c46203, ← c46204, k2, ← c46205] at e_736_17
  have hr := e_736_17
  simp only [e_736_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 736 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4894)) := by
  have c46206 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46207 := (publicBatchWrapper89_copies1443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46208 := (publicBatchWrapper89_copies1444 a h).1
  have c46209 := (publicBatchWrapper89_copies1444 a h).2.1
  have c46210 := (publicBatchWrapper89_copies1444 a h).2.2.1
  have c46211 := (publicBatchWrapper89_copies1444 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_736_18 := arithEq_of_rows h (row := 736) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_18
  simp only [← c46206, ← c46207, ← c46208] at e_736_18
  have e_736_19 := arithEq_of_rows h (row := 736) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_736_19
  simp only [← c46209, ← c46210, k2, ← c46211] at e_736_19
  have hr := e_736_19
  simp only [e_736_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4895)) := by
  have c46212 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.1
  have c46213 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.1
  have c46214 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.1
  have c46215 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.1
  have c46216 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.1
  have c46217 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_0 := arithEq_of_rows h (row := 737) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_0
  simp only [← c46212, ← c46213, ← c46214] at e_737_0
  have e_737_1 := arithEq_of_rows h (row := 737) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_1
  simp only [← c46215, ← c46216, k2, ← c46217] at e_737_1
  have hr := e_737_1
  simp only [e_737_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4896)) := by
  have c46218 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46219 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46220 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46221 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46222 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46223 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_2 := arithEq_of_rows h (row := 737) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_2
  simp only [← c46218, ← c46219, ← c46220] at e_737_2
  have e_737_3 := arithEq_of_rows h (row := 737) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_3
  simp only [← c46221, ← c46222, k2, ← c46223] at e_737_3
  have hr := e_737_3
  simp only [e_737_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4897)) := by
  have c46224 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46225 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46226 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46227 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46228 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46229 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_4 := arithEq_of_rows h (row := 737) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_4
  simp only [← c46224, ← c46225, ← c46226] at e_737_4
  have e_737_5 := arithEq_of_rows h (row := 737) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_5
  simp only [← c46227, ← c46228, k2, ← c46229] at e_737_5
  have hr := e_737_5
  simp only [e_737_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4898)) := by
  have c46230 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46231 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46232 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46233 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46234 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46235 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_6 := arithEq_of_rows h (row := 737) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_6
  simp only [← c46230, ← c46231, ← c46232] at e_737_6
  have e_737_7 := arithEq_of_rows h (row := 737) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_7
  simp only [← c46233, ← c46234, k2, ← c46235] at e_737_7
  have hr := e_737_7
  simp only [e_737_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4899)) := by
  have c46236 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46237 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46238 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46239 := (publicBatchWrapper89_copies1444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46240 := (publicBatchWrapper89_copies1445 a h).1
  have c46241 := (publicBatchWrapper89_copies1445 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_8 := arithEq_of_rows h (row := 737) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_8
  simp only [← c46236, ← c46237, ← c46238] at e_737_8
  have e_737_9 := arithEq_of_rows h (row := 737) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_9
  simp only [← c46239, ← c46240, k2, ← c46241] at e_737_9
  have hr := e_737_9
  simp only [e_737_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4900)) := by
  have c46242 := (publicBatchWrapper89_copies1445 a h).2.2.1
  have c46243 := (publicBatchWrapper89_copies1445 a h).2.2.2.1
  have c46244 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.1
  have c46245 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.1
  have c46246 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.1
  have c46247 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_10 := arithEq_of_rows h (row := 737) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_10
  simp only [← c46242, ← c46243, ← c46244] at e_737_10
  have e_737_11 := arithEq_of_rows h (row := 737) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_11
  simp only [← c46245, ← c46246, k2, ← c46247] at e_737_11
  have hr := e_737_11
  simp only [e_737_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4901)) := by
  have c46248 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.1
  have c46249 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.1
  have c46250 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46251 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46252 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46253 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_12 := arithEq_of_rows h (row := 737) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_12
  simp only [← c46248, ← c46249, ← c46250] at e_737_12
  have e_737_13 := arithEq_of_rows h (row := 737) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_13
  simp only [← c46251, ← c46252, k2, ← c46253] at e_737_13
  have hr := e_737_13
  simp only [e_737_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4902)) := by
  have c46254 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46255 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46256 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46257 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46258 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46259 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_14 := arithEq_of_rows h (row := 737) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_14
  simp only [← c46254, ← c46255, ← c46256] at e_737_14
  have e_737_15 := arithEq_of_rows h (row := 737) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_15
  simp only [← c46257, ← c46258, k2, ← c46259] at e_737_15
  have hr := e_737_15
  simp only [e_737_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4903)) := by
  have c46260 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46261 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46262 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46263 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46264 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46265 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_16 := arithEq_of_rows h (row := 737) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_16
  simp only [← c46260, ← c46261, ← c46262] at e_737_16
  have e_737_17 := arithEq_of_rows h (row := 737) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_17
  simp only [← c46263, ← c46264, k2, ← c46265] at e_737_17
  have hr := e_737_17
  simp only [e_737_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 737 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4904)) := by
  have c46266 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46267 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46268 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46269 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46270 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46271 := (publicBatchWrapper89_copies1445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_737_18 := arithEq_of_rows h (row := 737) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_18
  simp only [← c46266, ← c46267, ← c46268] at e_737_18
  have e_737_19 := arithEq_of_rows h (row := 737) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_737_19
  simp only [← c46269, ← c46270, k2, ← c46271] at e_737_19
  have hr := e_737_19
  simp only [e_737_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4905)) := by
  have c46272 := (publicBatchWrapper89_copies1446 a h).1
  have c46273 := (publicBatchWrapper89_copies1446 a h).2.1
  have c46274 := (publicBatchWrapper89_copies1446 a h).2.2.1
  have c46275 := (publicBatchWrapper89_copies1446 a h).2.2.2.1
  have c46276 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.1
  have c46277 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_0 := arithEq_of_rows h (row := 738) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_0
  simp only [← c46272, ← c46273, ← c46274] at e_738_0
  have e_738_1 := arithEq_of_rows h (row := 738) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_1
  simp only [← c46275, ← c46276, k2, ← c46277] at e_738_1
  have hr := e_738_1
  simp only [e_738_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4906)) := by
  have c46278 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.1
  have c46279 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.1
  have c46280 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.1
  have c46281 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.1
  have c46282 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46283 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_2 := arithEq_of_rows h (row := 738) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_2
  simp only [← c46278, ← c46279, ← c46280] at e_738_2
  have e_738_3 := arithEq_of_rows h (row := 738) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_3
  simp only [← c46281, ← c46282, k2, ← c46283] at e_738_3
  have hr := e_738_3
  simp only [e_738_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4907)) := by
  have c46284 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46285 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46286 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46287 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46288 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46289 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_4 := arithEq_of_rows h (row := 738) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_4
  simp only [← c46284, ← c46285, ← c46286] at e_738_4
  have e_738_5 := arithEq_of_rows h (row := 738) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_5
  simp only [← c46287, ← c46288, k2, ← c46289] at e_738_5
  have hr := e_738_5
  simp only [e_738_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4908)) := by
  have c46290 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46291 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46292 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46293 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46294 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46295 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_6 := arithEq_of_rows h (row := 738) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_6
  simp only [← c46290, ← c46291, ← c46292] at e_738_6
  have e_738_7 := arithEq_of_rows h (row := 738) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_7
  simp only [← c46293, ← c46294, k2, ← c46295] at e_738_7
  have hr := e_738_7
  simp only [e_738_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4909)) := by
  have c46296 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46297 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46298 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46299 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46300 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46301 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_8 := arithEq_of_rows h (row := 738) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_8
  simp only [← c46296, ← c46297, ← c46298] at e_738_8
  have e_738_9 := arithEq_of_rows h (row := 738) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_9
  simp only [← c46299, ← c46300, k2, ← c46301] at e_738_9
  have hr := e_738_9
  simp only [e_738_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4910)) := by
  have c46302 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46303 := (publicBatchWrapper89_copies1446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46304 := (publicBatchWrapper89_copies1447 a h).1
  have c46305 := (publicBatchWrapper89_copies1447 a h).2.1
  have c46306 := (publicBatchWrapper89_copies1447 a h).2.2.1
  have c46307 := (publicBatchWrapper89_copies1447 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_10 := arithEq_of_rows h (row := 738) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_10
  simp only [← c46302, ← c46303, ← c46304] at e_738_10
  have e_738_11 := arithEq_of_rows h (row := 738) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_11
  simp only [← c46305, ← c46306, k2, ← c46307] at e_738_11
  have hr := e_738_11
  simp only [e_738_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4911)) := by
  have c46308 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.1
  have c46309 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.1
  have c46310 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.1
  have c46311 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.1
  have c46312 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.1
  have c46313 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_12 := arithEq_of_rows h (row := 738) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_12
  simp only [← c46308, ← c46309, ← c46310] at e_738_12
  have e_738_13 := arithEq_of_rows h (row := 738) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_13
  simp only [← c46311, ← c46312, k2, ← c46313] at e_738_13
  have hr := e_738_13
  simp only [e_738_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4912)) := by
  have c46314 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46315 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46316 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46317 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46318 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46319 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_14 := arithEq_of_rows h (row := 738) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_14
  simp only [← c46314, ← c46315, ← c46316] at e_738_14
  have e_738_15 := arithEq_of_rows h (row := 738) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_15
  simp only [← c46317, ← c46318, k2, ← c46319] at e_738_15
  have hr := e_738_15
  simp only [e_738_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4913)) := by
  have c46320 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46321 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46322 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46323 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46324 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46325 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_16 := arithEq_of_rows h (row := 738) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_16
  simp only [← c46320, ← c46321, ← c46322] at e_738_16
  have e_738_17 := arithEq_of_rows h (row := 738) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_17
  simp only [← c46323, ← c46324, k2, ← c46325] at e_738_17
  have hr := e_738_17
  simp only [e_738_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 738 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4914)) := by
  have c46326 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46327 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46328 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46329 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46330 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46331 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_738_18 := arithEq_of_rows h (row := 738) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_18
  simp only [← c46326, ← c46327, ← c46328] at e_738_18
  have e_738_19 := arithEq_of_rows h (row := 738) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_738_19
  simp only [← c46329, ← c46330, k2, ← c46331] at e_738_19
  have hr := e_738_19
  simp only [e_738_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4915)) := by
  have c46332 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46333 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46334 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46335 := (publicBatchWrapper89_copies1447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46336 := (publicBatchWrapper89_copies1448 a h).1
  have c46337 := (publicBatchWrapper89_copies1448 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_0 := arithEq_of_rows h (row := 739) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_0
  simp only [← c46332, ← c46333, ← c46334] at e_739_0
  have e_739_1 := arithEq_of_rows h (row := 739) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_1
  simp only [← c46335, ← c46336, k2, ← c46337] at e_739_1
  have hr := e_739_1
  simp only [e_739_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4916)) := by
  have c46338 := (publicBatchWrapper89_copies1448 a h).2.2.1
  have c46339 := (publicBatchWrapper89_copies1448 a h).2.2.2.1
  have c46340 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.1
  have c46341 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.1
  have c46342 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.1
  have c46343 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_2 := arithEq_of_rows h (row := 739) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_2
  simp only [← c46338, ← c46339, ← c46340] at e_739_2
  have e_739_3 := arithEq_of_rows h (row := 739) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_3
  simp only [← c46341, ← c46342, k2, ← c46343] at e_739_3
  have hr := e_739_3
  simp only [e_739_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4917)) := by
  have c46344 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.1
  have c46345 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.1
  have c46346 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46347 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46348 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46349 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_4 := arithEq_of_rows h (row := 739) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_4
  simp only [← c46344, ← c46345, ← c46346] at e_739_4
  have e_739_5 := arithEq_of_rows h (row := 739) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_5
  simp only [← c46347, ← c46348, k2, ← c46349] at e_739_5
  have hr := e_739_5
  simp only [e_739_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4918)) := by
  have c46350 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46351 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46352 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46353 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46354 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46355 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_6 := arithEq_of_rows h (row := 739) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_6
  simp only [← c46350, ← c46351, ← c46352] at e_739_6
  have e_739_7 := arithEq_of_rows h (row := 739) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_7
  simp only [← c46353, ← c46354, k2, ← c46355] at e_739_7
  have hr := e_739_7
  simp only [e_739_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4919)) := by
  have c46356 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46357 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46358 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46359 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46360 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46361 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_8 := arithEq_of_rows h (row := 739) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_8
  simp only [← c46356, ← c46357, ← c46358] at e_739_8
  have e_739_9 := arithEq_of_rows h (row := 739) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_9
  simp only [← c46359, ← c46360, k2, ← c46361] at e_739_9
  have hr := e_739_9
  simp only [e_739_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4920)) := by
  have c46362 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46363 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46364 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46365 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46366 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46367 := (publicBatchWrapper89_copies1448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_10 := arithEq_of_rows h (row := 739) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_10
  simp only [← c46362, ← c46363, ← c46364] at e_739_10
  have e_739_11 := arithEq_of_rows h (row := 739) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_11
  simp only [← c46365, ← c46366, k2, ← c46367] at e_739_11
  have hr := e_739_11
  simp only [e_739_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4921)) := by
  have c46368 := (publicBatchWrapper89_copies1449 a h).1
  have c46369 := (publicBatchWrapper89_copies1449 a h).2.1
  have c46370 := (publicBatchWrapper89_copies1449 a h).2.2.1
  have c46371 := (publicBatchWrapper89_copies1449 a h).2.2.2.1
  have c46372 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.1
  have c46373 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_12 := arithEq_of_rows h (row := 739) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_12
  simp only [← c46368, ← c46369, ← c46370] at e_739_12
  have e_739_13 := arithEq_of_rows h (row := 739) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_13
  simp only [← c46371, ← c46372, k2, ← c46373] at e_739_13
  have hr := e_739_13
  simp only [e_739_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4922)) := by
  have c46374 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.1
  have c46375 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.1
  have c46376 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.1
  have c46377 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.1
  have c46378 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46379 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_14 := arithEq_of_rows h (row := 739) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_14
  simp only [← c46374, ← c46375, ← c46376] at e_739_14
  have e_739_15 := arithEq_of_rows h (row := 739) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_15
  simp only [← c46377, ← c46378, k2, ← c46379] at e_739_15
  have hr := e_739_15
  simp only [e_739_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4923)) := by
  have c46380 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46381 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46382 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46383 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46384 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46385 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_16 := arithEq_of_rows h (row := 739) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_16
  simp only [← c46380, ← c46381, ← c46382] at e_739_16
  have e_739_17 := arithEq_of_rows h (row := 739) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_17
  simp only [← c46383, ← c46384, k2, ← c46385] at e_739_17
  have hr := e_739_17
  simp only [e_739_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 739 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4924)) := by
  have c46386 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46387 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46388 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46389 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46390 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46391 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_739_18 := arithEq_of_rows h (row := 739) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_18
  simp only [← c46386, ← c46387, ← c46388] at e_739_18
  have e_739_19 := arithEq_of_rows h (row := 739) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_739_19
  simp only [← c46389, ← c46390, k2, ← c46391] at e_739_19
  have hr := e_739_19
  simp only [e_739_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4925)) := by
  have c46392 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46393 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46394 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46395 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46396 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46397 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_0 := arithEq_of_rows h (row := 740) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_0
  simp only [← c46392, ← c46393, ← c46394] at e_740_0
  have e_740_1 := arithEq_of_rows h (row := 740) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_1
  simp only [← c46395, ← c46396, k2, ← c46397] at e_740_1
  have hr := e_740_1
  simp only [e_740_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4926)) := by
  have c46398 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46399 := (publicBatchWrapper89_copies1449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46400 := (publicBatchWrapper89_copies1450 a h).1
  have c46401 := (publicBatchWrapper89_copies1450 a h).2.1
  have c46402 := (publicBatchWrapper89_copies1450 a h).2.2.1
  have c46403 := (publicBatchWrapper89_copies1450 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_2 := arithEq_of_rows h (row := 740) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_2
  simp only [← c46398, ← c46399, ← c46400] at e_740_2
  have e_740_3 := arithEq_of_rows h (row := 740) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_3
  simp only [← c46401, ← c46402, k2, ← c46403] at e_740_3
  have hr := e_740_3
  simp only [e_740_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4927)) := by
  have c46404 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.1
  have c46405 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.1
  have c46406 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.1
  have c46407 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.1
  have c46408 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.1
  have c46409 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_4 := arithEq_of_rows h (row := 740) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_4
  simp only [← c46404, ← c46405, ← c46406] at e_740_4
  have e_740_5 := arithEq_of_rows h (row := 740) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_5
  simp only [← c46407, ← c46408, k2, ← c46409] at e_740_5
  have hr := e_740_5
  simp only [e_740_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4928)) := by
  have c46410 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46411 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46412 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46413 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46414 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46415 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_6 := arithEq_of_rows h (row := 740) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_6
  simp only [← c46410, ← c46411, ← c46412] at e_740_6
  have e_740_7 := arithEq_of_rows h (row := 740) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_7
  simp only [← c46413, ← c46414, k2, ← c46415] at e_740_7
  have hr := e_740_7
  simp only [e_740_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4929)) := by
  have c46416 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46417 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46418 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46419 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46420 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46421 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_8 := arithEq_of_rows h (row := 740) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_8
  simp only [← c46416, ← c46417, ← c46418] at e_740_8
  have e_740_9 := arithEq_of_rows h (row := 740) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_9
  simp only [← c46419, ← c46420, k2, ← c46421] at e_740_9
  have hr := e_740_9
  simp only [e_740_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4930)) := by
  have c46422 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46423 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46424 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46425 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46426 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46427 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_10 := arithEq_of_rows h (row := 740) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_10
  simp only [← c46422, ← c46423, ← c46424] at e_740_10
  have e_740_11 := arithEq_of_rows h (row := 740) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_11
  simp only [← c46425, ← c46426, k2, ← c46427] at e_740_11
  have hr := e_740_11
  simp only [e_740_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4931)) := by
  have c46428 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46429 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46430 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46431 := (publicBatchWrapper89_copies1450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46432 := (publicBatchWrapper89_copies1451 a h).1
  have c46433 := (publicBatchWrapper89_copies1451 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_12 := arithEq_of_rows h (row := 740) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_12
  simp only [← c46428, ← c46429, ← c46430] at e_740_12
  have e_740_13 := arithEq_of_rows h (row := 740) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_13
  simp only [← c46431, ← c46432, k2, ← c46433] at e_740_13
  have hr := e_740_13
  simp only [e_740_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4932)) := by
  have c46434 := (publicBatchWrapper89_copies1451 a h).2.2.1
  have c46435 := (publicBatchWrapper89_copies1451 a h).2.2.2.1
  have c46436 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.1
  have c46437 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.1
  have c46438 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.1
  have c46439 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_14 := arithEq_of_rows h (row := 740) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_14
  simp only [← c46434, ← c46435, ← c46436] at e_740_14
  have e_740_15 := arithEq_of_rows h (row := 740) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_15
  simp only [← c46437, ← c46438, k2, ← c46439] at e_740_15
  have hr := e_740_15
  simp only [e_740_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4933)) := by
  have c46440 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.1
  have c46441 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.1
  have c46442 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46443 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46444 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46445 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_16 := arithEq_of_rows h (row := 740) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_16
  simp only [← c46440, ← c46441, ← c46442] at e_740_16
  have e_740_17 := arithEq_of_rows h (row := 740) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_17
  simp only [← c46443, ← c46444, k2, ← c46445] at e_740_17
  have hr := e_740_17
  simp only [e_740_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 740 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4934)) := by
  have c46446 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46447 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46448 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46449 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46450 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46451 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_740_18 := arithEq_of_rows h (row := 740) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_18
  simp only [← c46446, ← c46447, ← c46448] at e_740_18
  have e_740_19 := arithEq_of_rows h (row := 740) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_740_19
  simp only [← c46449, ← c46450, k2, ← c46451] at e_740_19
  have hr := e_740_19
  simp only [e_740_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4935)) := by
  have c46452 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46453 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46454 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46455 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46456 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46457 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_0 := arithEq_of_rows h (row := 741) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_0
  simp only [← c46452, ← c46453, ← c46454] at e_741_0
  have e_741_1 := arithEq_of_rows h (row := 741) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_1
  simp only [← c46455, ← c46456, k2, ← c46457] at e_741_1
  have hr := e_741_1
  simp only [e_741_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4936)) := by
  have c46458 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46459 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46460 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46461 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46462 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46463 := (publicBatchWrapper89_copies1451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_2 := arithEq_of_rows h (row := 741) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_2
  simp only [← c46458, ← c46459, ← c46460] at e_741_2
  have e_741_3 := arithEq_of_rows h (row := 741) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_3
  simp only [← c46461, ← c46462, k2, ← c46463] at e_741_3
  have hr := e_741_3
  simp only [e_741_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4937)) := by
  have c46464 := (publicBatchWrapper89_copies1452 a h).1
  have c46465 := (publicBatchWrapper89_copies1452 a h).2.1
  have c46466 := (publicBatchWrapper89_copies1452 a h).2.2.1
  have c46467 := (publicBatchWrapper89_copies1452 a h).2.2.2.1
  have c46468 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.1
  have c46469 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_4 := arithEq_of_rows h (row := 741) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_4
  simp only [← c46464, ← c46465, ← c46466] at e_741_4
  have e_741_5 := arithEq_of_rows h (row := 741) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_5
  simp only [← c46467, ← c46468, k2, ← c46469] at e_741_5
  have hr := e_741_5
  simp only [e_741_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4938)) := by
  have c46470 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.1
  have c46471 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.1
  have c46472 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.1
  have c46473 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.1
  have c46474 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46475 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_6 := arithEq_of_rows h (row := 741) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_6
  simp only [← c46470, ← c46471, ← c46472] at e_741_6
  have e_741_7 := arithEq_of_rows h (row := 741) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_7
  simp only [← c46473, ← c46474, k2, ← c46475] at e_741_7
  have hr := e_741_7
  simp only [e_741_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4939)) := by
  have c46476 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46477 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46478 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46479 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46480 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46481 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_8 := arithEq_of_rows h (row := 741) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_8
  simp only [← c46476, ← c46477, ← c46478] at e_741_8
  have e_741_9 := arithEq_of_rows h (row := 741) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_9
  simp only [← c46479, ← c46480, k2, ← c46481] at e_741_9
  have hr := e_741_9
  simp only [e_741_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4940)) := by
  have c46482 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46483 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46484 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46485 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46486 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46487 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_10 := arithEq_of_rows h (row := 741) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_10
  simp only [← c46482, ← c46483, ← c46484] at e_741_10
  have e_741_11 := arithEq_of_rows h (row := 741) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_11
  simp only [← c46485, ← c46486, k2, ← c46487] at e_741_11
  have hr := e_741_11
  simp only [e_741_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4941)) := by
  have c46488 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46489 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46490 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46491 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46492 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46493 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_12 := arithEq_of_rows h (row := 741) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_12
  simp only [← c46488, ← c46489, ← c46490] at e_741_12
  have e_741_13 := arithEq_of_rows h (row := 741) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_13
  simp only [← c46491, ← c46492, k2, ← c46493] at e_741_13
  have hr := e_741_13
  simp only [e_741_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4942)) := by
  have c46494 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46495 := (publicBatchWrapper89_copies1452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46496 := (publicBatchWrapper89_copies1453 a h).1
  have c46497 := (publicBatchWrapper89_copies1453 a h).2.1
  have c46498 := (publicBatchWrapper89_copies1453 a h).2.2.1
  have c46499 := (publicBatchWrapper89_copies1453 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_14 := arithEq_of_rows h (row := 741) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_14
  simp only [← c46494, ← c46495, ← c46496] at e_741_14
  have e_741_15 := arithEq_of_rows h (row := 741) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_15
  simp only [← c46497, ← c46498, k2, ← c46499] at e_741_15
  have hr := e_741_15
  simp only [e_741_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4943)) := by
  have c46500 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.1
  have c46501 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.1
  have c46502 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.1
  have c46503 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.1
  have c46504 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.1
  have c46505 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_16 := arithEq_of_rows h (row := 741) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_16
  simp only [← c46500, ← c46501, ← c46502] at e_741_16
  have e_741_17 := arithEq_of_rows h (row := 741) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_17
  simp only [← c46503, ← c46504, k2, ← c46505] at e_741_17
  have hr := e_741_17
  simp only [e_741_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 741 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4944)) := by
  have c46506 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46507 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46508 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46509 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46510 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46511 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_741_18 := arithEq_of_rows h (row := 741) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_18
  simp only [← c46506, ← c46507, ← c46508] at e_741_18
  have e_741_19 := arithEq_of_rows h (row := 741) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_741_19
  simp only [← c46509, ← c46510, k2, ← c46511] at e_741_19
  have hr := e_741_19
  simp only [e_741_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4945)) := by
  have c46512 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46513 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46514 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46515 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46516 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46517 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_0 := arithEq_of_rows h (row := 742) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_0
  simp only [← c46512, ← c46513, ← c46514] at e_742_0
  have e_742_1 := arithEq_of_rows h (row := 742) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_1
  simp only [← c46515, ← c46516, k2, ← c46517] at e_742_1
  have hr := e_742_1
  simp only [e_742_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4946)) := by
  have c46518 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46519 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46520 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46521 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46522 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46523 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_2 := arithEq_of_rows h (row := 742) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_2
  simp only [← c46518, ← c46519, ← c46520] at e_742_2
  have e_742_3 := arithEq_of_rows h (row := 742) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_3
  simp only [← c46521, ← c46522, k2, ← c46523] at e_742_3
  have hr := e_742_3
  simp only [e_742_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4947)) := by
  have c46524 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46525 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46526 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46527 := (publicBatchWrapper89_copies1453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46528 := (publicBatchWrapper89_copies1454 a h).1
  have c46529 := (publicBatchWrapper89_copies1454 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_4 := arithEq_of_rows h (row := 742) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_4
  simp only [← c46524, ← c46525, ← c46526] at e_742_4
  have e_742_5 := arithEq_of_rows h (row := 742) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_5
  simp only [← c46527, ← c46528, k2, ← c46529] at e_742_5
  have hr := e_742_5
  simp only [e_742_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4948)) := by
  have c46530 := (publicBatchWrapper89_copies1454 a h).2.2.1
  have c46531 := (publicBatchWrapper89_copies1454 a h).2.2.2.1
  have c46532 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.1
  have c46533 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.1
  have c46534 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.1
  have c46535 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_6 := arithEq_of_rows h (row := 742) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_6
  simp only [← c46530, ← c46531, ← c46532] at e_742_6
  have e_742_7 := arithEq_of_rows h (row := 742) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_7
  simp only [← c46533, ← c46534, k2, ← c46535] at e_742_7
  have hr := e_742_7
  simp only [e_742_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4949)) := by
  have c46536 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.1
  have c46537 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.1
  have c46538 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46539 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46540 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46541 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_8 := arithEq_of_rows h (row := 742) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_8
  simp only [← c46536, ← c46537, ← c46538] at e_742_8
  have e_742_9 := arithEq_of_rows h (row := 742) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_9
  simp only [← c46539, ← c46540, k2, ← c46541] at e_742_9
  have hr := e_742_9
  simp only [e_742_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4950)) := by
  have c46542 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46543 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46544 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46545 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46546 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46547 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_10 := arithEq_of_rows h (row := 742) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_10
  simp only [← c46542, ← c46543, ← c46544] at e_742_10
  have e_742_11 := arithEq_of_rows h (row := 742) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_11
  simp only [← c46545, ← c46546, k2, ← c46547] at e_742_11
  have hr := e_742_11
  simp only [e_742_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4951)) := by
  have c46548 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46549 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46550 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46551 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46552 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46553 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_12 := arithEq_of_rows h (row := 742) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_12
  simp only [← c46548, ← c46549, ← c46550] at e_742_12
  have e_742_13 := arithEq_of_rows h (row := 742) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_13
  simp only [← c46551, ← c46552, k2, ← c46553] at e_742_13
  have hr := e_742_13
  simp only [e_742_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4952)) := by
  have c46554 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46555 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46556 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46557 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46558 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46559 := (publicBatchWrapper89_copies1454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_14 := arithEq_of_rows h (row := 742) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_14
  simp only [← c46554, ← c46555, ← c46556] at e_742_14
  have e_742_15 := arithEq_of_rows h (row := 742) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_15
  simp only [← c46557, ← c46558, k2, ← c46559] at e_742_15
  have hr := e_742_15
  simp only [e_742_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4953)) := by
  have c46560 := (publicBatchWrapper89_copies1455 a h).1
  have c46561 := (publicBatchWrapper89_copies1455 a h).2.1
  have c46562 := (publicBatchWrapper89_copies1455 a h).2.2.1
  have c46563 := (publicBatchWrapper89_copies1455 a h).2.2.2.1
  have c46564 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.1
  have c46565 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_16 := arithEq_of_rows h (row := 742) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_16
  simp only [← c46560, ← c46561, ← c46562] at e_742_16
  have e_742_17 := arithEq_of_rows h (row := 742) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_17
  simp only [← c46563, ← c46564, k2, ← c46565] at e_742_17
  have hr := e_742_17
  simp only [e_742_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 742 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4954)) := by
  have c46566 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.1
  have c46567 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.1
  have c46568 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.1
  have c46569 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.1
  have c46570 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46571 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_742_18 := arithEq_of_rows h (row := 742) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_18
  simp only [← c46566, ← c46567, ← c46568] at e_742_18
  have e_742_19 := arithEq_of_rows h (row := 742) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_742_19
  simp only [← c46569, ← c46570, k2, ← c46571] at e_742_19
  have hr := e_742_19
  simp only [e_742_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4955)) := by
  have c46572 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46573 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46574 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46575 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46576 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46577 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_0 := arithEq_of_rows h (row := 743) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_0
  simp only [← c46572, ← c46573, ← c46574] at e_743_0
  have e_743_1 := arithEq_of_rows h (row := 743) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_1
  simp only [← c46575, ← c46576, k2, ← c46577] at e_743_1
  have hr := e_743_1
  simp only [e_743_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4956)) := by
  have c46578 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46579 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46580 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46581 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46582 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46583 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_2 := arithEq_of_rows h (row := 743) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_2
  simp only [← c46578, ← c46579, ← c46580] at e_743_2
  have e_743_3 := arithEq_of_rows h (row := 743) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_3
  simp only [← c46581, ← c46582, k2, ← c46583] at e_743_3
  have hr := e_743_3
  simp only [e_743_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4957)) := by
  have c46584 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46585 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46586 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46587 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46588 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46589 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_4 := arithEq_of_rows h (row := 743) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_4
  simp only [← c46584, ← c46585, ← c46586] at e_743_4
  have e_743_5 := arithEq_of_rows h (row := 743) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_5
  simp only [← c46587, ← c46588, k2, ← c46589] at e_743_5
  have hr := e_743_5
  simp only [e_743_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4958)) := by
  have c46590 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46591 := (publicBatchWrapper89_copies1455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46592 := (publicBatchWrapper89_copies1456 a h).1
  have c46593 := (publicBatchWrapper89_copies1456 a h).2.1
  have c46594 := (publicBatchWrapper89_copies1456 a h).2.2.1
  have c46595 := (publicBatchWrapper89_copies1456 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_6 := arithEq_of_rows h (row := 743) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_6
  simp only [← c46590, ← c46591, ← c46592] at e_743_6
  have e_743_7 := arithEq_of_rows h (row := 743) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_7
  simp only [← c46593, ← c46594, k2, ← c46595] at e_743_7
  have hr := e_743_7
  simp only [e_743_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4959)) := by
  have c46596 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.1
  have c46597 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.1
  have c46598 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.1
  have c46599 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.1
  have c46600 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.1
  have c46601 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_8 := arithEq_of_rows h (row := 743) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_8
  simp only [← c46596, ← c46597, ← c46598] at e_743_8
  have e_743_9 := arithEq_of_rows h (row := 743) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_9
  simp only [← c46599, ← c46600, k2, ← c46601] at e_743_9
  have hr := e_743_9
  simp only [e_743_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4960)) := by
  have c46602 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46603 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46604 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46605 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46606 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46607 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_10 := arithEq_of_rows h (row := 743) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_10
  simp only [← c46602, ← c46603, ← c46604] at e_743_10
  have e_743_11 := arithEq_of_rows h (row := 743) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_11
  simp only [← c46605, ← c46606, k2, ← c46607] at e_743_11
  have hr := e_743_11
  simp only [e_743_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4961)) := by
  have c46608 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46609 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46610 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46611 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46612 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46613 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_12 := arithEq_of_rows h (row := 743) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_12
  simp only [← c46608, ← c46609, ← c46610] at e_743_12
  have e_743_13 := arithEq_of_rows h (row := 743) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_13
  simp only [← c46611, ← c46612, k2, ← c46613] at e_743_13
  have hr := e_743_13
  simp only [e_743_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4962)) := by
  have c46614 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46615 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46616 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46617 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46618 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46619 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_14 := arithEq_of_rows h (row := 743) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_14
  simp only [← c46614, ← c46615, ← c46616] at e_743_14
  have e_743_15 := arithEq_of_rows h (row := 743) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_15
  simp only [← c46617, ← c46618, k2, ← c46619] at e_743_15
  have hr := e_743_15
  simp only [e_743_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4963)) := by
  have c46620 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46621 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46622 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46623 := (publicBatchWrapper89_copies1456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46624 := (publicBatchWrapper89_copies1457 a h).1
  have c46625 := (publicBatchWrapper89_copies1457 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_16 := arithEq_of_rows h (row := 743) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_16
  simp only [← c46620, ← c46621, ← c46622] at e_743_16
  have e_743_17 := arithEq_of_rows h (row := 743) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_17
  simp only [← c46623, ← c46624, k2, ← c46625] at e_743_17
  have hr := e_743_17
  simp only [e_743_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 743 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4964)) := by
  have c46626 := (publicBatchWrapper89_copies1457 a h).2.2.1
  have c46627 := (publicBatchWrapper89_copies1457 a h).2.2.2.1
  have c46628 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.1
  have c46629 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.1
  have c46630 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.1
  have c46631 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_743_18 := arithEq_of_rows h (row := 743) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_18
  simp only [← c46626, ← c46627, ← c46628] at e_743_18
  have e_743_19 := arithEq_of_rows h (row := 743) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_743_19
  simp only [← c46629, ← c46630, k2, ← c46631] at e_743_19
  have hr := e_743_19
  simp only [e_743_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4965)) := by
  have c46632 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.1
  have c46633 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.1
  have c46634 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46635 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46636 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46637 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_0 := arithEq_of_rows h (row := 744) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_0
  simp only [← c46632, ← c46633, ← c46634] at e_744_0
  have e_744_1 := arithEq_of_rows h (row := 744) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_1
  simp only [← c46635, ← c46636, k2, ← c46637] at e_744_1
  have hr := e_744_1
  simp only [e_744_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4966)) := by
  have c46638 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46639 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46640 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46641 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46642 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46643 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_2 := arithEq_of_rows h (row := 744) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_2
  simp only [← c46638, ← c46639, ← c46640] at e_744_2
  have e_744_3 := arithEq_of_rows h (row := 744) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_3
  simp only [← c46641, ← c46642, k2, ← c46643] at e_744_3
  have hr := e_744_3
  simp only [e_744_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4967)) := by
  have c46644 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46645 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46646 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46647 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46648 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46649 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_4 := arithEq_of_rows h (row := 744) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_4
  simp only [← c46644, ← c46645, ← c46646] at e_744_4
  have e_744_5 := arithEq_of_rows h (row := 744) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_5
  simp only [← c46647, ← c46648, k2, ← c46649] at e_744_5
  have hr := e_744_5
  simp only [e_744_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4968)) := by
  have c46650 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46651 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46652 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46653 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46654 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46655 := (publicBatchWrapper89_copies1457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_6 := arithEq_of_rows h (row := 744) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_6
  simp only [← c46650, ← c46651, ← c46652] at e_744_6
  have e_744_7 := arithEq_of_rows h (row := 744) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_7
  simp only [← c46653, ← c46654, k2, ← c46655] at e_744_7
  have hr := e_744_7
  simp only [e_744_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4969)) := by
  have c46656 := (publicBatchWrapper89_copies1458 a h).1
  have c46657 := (publicBatchWrapper89_copies1458 a h).2.1
  have c46658 := (publicBatchWrapper89_copies1458 a h).2.2.1
  have c46659 := (publicBatchWrapper89_copies1458 a h).2.2.2.1
  have c46660 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.1
  have c46661 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_8 := arithEq_of_rows h (row := 744) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_8
  simp only [← c46656, ← c46657, ← c46658] at e_744_8
  have e_744_9 := arithEq_of_rows h (row := 744) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_9
  simp only [← c46659, ← c46660, k2, ← c46661] at e_744_9
  have hr := e_744_9
  simp only [e_744_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5010)) := by
  have c46662 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.1
  have c46663 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.1
  have c46664 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.1
  have c46665 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.1
  have c46666 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46667 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_10 := arithEq_of_rows h (row := 744) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_10
  simp only [← c46662, ← c46663, ← c46664] at e_744_10
  have e_744_11 := arithEq_of_rows h (row := 744) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_11
  simp only [← c46665, ← c46666, k2, ← c46667] at e_744_11
  have hr := e_744_11
  simp only [e_744_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5011)) := by
  have c46668 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46669 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46670 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46671 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46672 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46673 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_12 := arithEq_of_rows h (row := 744) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_12
  simp only [← c46668, ← c46669, ← c46670] at e_744_12
  have e_744_13 := arithEq_of_rows h (row := 744) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_13
  simp only [← c46671, ← c46672, k2, ← c46673] at e_744_13
  have hr := e_744_13
  simp only [e_744_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5012)) := by
  have c46674 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46675 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46676 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46677 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46678 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46679 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_14 := arithEq_of_rows h (row := 744) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_14
  simp only [← c46674, ← c46675, ← c46676] at e_744_14
  have e_744_15 := arithEq_of_rows h (row := 744) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_15
  simp only [← c46677, ← c46678, k2, ← c46679] at e_744_15
  have hr := e_744_15
  simp only [e_744_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5013)) := by
  have c46680 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46681 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46682 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46683 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46684 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46685 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_16 := arithEq_of_rows h (row := 744) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_16
  simp only [← c46680, ← c46681, ← c46682] at e_744_16
  have e_744_17 := arithEq_of_rows h (row := 744) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_17
  simp only [← c46683, ← c46684, k2, ← c46685] at e_744_17
  have hr := e_744_17
  simp only [e_744_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 744 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5014)) := by
  have c46686 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46687 := (publicBatchWrapper89_copies1458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46688 := (publicBatchWrapper89_copies1459 a h).1
  have c46689 := (publicBatchWrapper89_copies1459 a h).2.1
  have c46690 := (publicBatchWrapper89_copies1459 a h).2.2.1
  have c46691 := (publicBatchWrapper89_copies1459 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_744_18 := arithEq_of_rows h (row := 744) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_18
  simp only [← c46686, ← c46687, ← c46688] at e_744_18
  have e_744_19 := arithEq_of_rows h (row := 744) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_744_19
  simp only [← c46689, ← c46690, k2, ← c46691] at e_744_19
  have hr := e_744_19
  simp only [e_744_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5015)) := by
  have c46692 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.1
  have c46693 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.1
  have c46694 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.1
  have c46695 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.1
  have c46696 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.1
  have c46697 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_0 := arithEq_of_rows h (row := 745) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_0
  simp only [← c46692, ← c46693, ← c46694] at e_745_0
  have e_745_1 := arithEq_of_rows h (row := 745) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_1
  simp only [← c46695, ← c46696, k2, ← c46697] at e_745_1
  have hr := e_745_1
  simp only [e_745_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5016)) := by
  have c46698 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46699 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46700 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46701 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46702 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46703 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_2 := arithEq_of_rows h (row := 745) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_2
  simp only [← c46698, ← c46699, ← c46700] at e_745_2
  have e_745_3 := arithEq_of_rows h (row := 745) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_3
  simp only [← c46701, ← c46702, k2, ← c46703] at e_745_3
  have hr := e_745_3
  simp only [e_745_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5017)) := by
  have c46704 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46705 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46706 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46707 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46708 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46709 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_4 := arithEq_of_rows h (row := 745) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_4
  simp only [← c46704, ← c46705, ← c46706] at e_745_4
  have e_745_5 := arithEq_of_rows h (row := 745) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_5
  simp only [← c46707, ← c46708, k2, ← c46709] at e_745_5
  have hr := e_745_5
  simp only [e_745_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5018)) := by
  have c46710 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46711 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46712 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46713 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46714 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46715 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_6 := arithEq_of_rows h (row := 745) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_6
  simp only [← c46710, ← c46711, ← c46712] at e_745_6
  have e_745_7 := arithEq_of_rows h (row := 745) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_7
  simp only [← c46713, ← c46714, k2, ← c46715] at e_745_7
  have hr := e_745_7
  simp only [e_745_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5019)) := by
  have c46716 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46717 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46718 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46719 := (publicBatchWrapper89_copies1459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46720 := (publicBatchWrapper89_copies1460 a h).1
  have c46721 := (publicBatchWrapper89_copies1460 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_8 := arithEq_of_rows h (row := 745) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_8
  simp only [← c46716, ← c46717, ← c46718] at e_745_8
  have e_745_9 := arithEq_of_rows h (row := 745) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_9
  simp only [← c46719, ← c46720, k2, ← c46721] at e_745_9
  have hr := e_745_9
  simp only [e_745_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5020)) := by
  have c46722 := (publicBatchWrapper89_copies1460 a h).2.2.1
  have c46723 := (publicBatchWrapper89_copies1460 a h).2.2.2.1
  have c46724 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.1
  have c46725 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.1
  have c46726 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.1
  have c46727 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_10 := arithEq_of_rows h (row := 745) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_10
  simp only [← c46722, ← c46723, ← c46724] at e_745_10
  have e_745_11 := arithEq_of_rows h (row := 745) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_11
  simp only [← c46725, ← c46726, k2, ← c46727] at e_745_11
  have hr := e_745_11
  simp only [e_745_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5021)) := by
  have c46728 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.1
  have c46729 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.1
  have c46730 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46731 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46732 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46733 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_12 := arithEq_of_rows h (row := 745) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_12
  simp only [← c46728, ← c46729, ← c46730] at e_745_12
  have e_745_13 := arithEq_of_rows h (row := 745) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_13
  simp only [← c46731, ← c46732, k2, ← c46733] at e_745_13
  have hr := e_745_13
  simp only [e_745_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5022)) := by
  have c46734 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46735 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46736 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46737 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46738 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46739 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_14 := arithEq_of_rows h (row := 745) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_14
  simp only [← c46734, ← c46735, ← c46736] at e_745_14
  have e_745_15 := arithEq_of_rows h (row := 745) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_15
  simp only [← c46737, ← c46738, k2, ← c46739] at e_745_15
  have hr := e_745_15
  simp only [e_745_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5023)) := by
  have c46740 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46741 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46742 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46743 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46744 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46745 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_16 := arithEq_of_rows h (row := 745) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_16
  simp only [← c46740, ← c46741, ← c46742] at e_745_16
  have e_745_17 := arithEq_of_rows h (row := 745) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_17
  simp only [← c46743, ← c46744, k2, ← c46745] at e_745_17
  have hr := e_745_17
  simp only [e_745_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 745 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5024)) := by
  have c46746 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46747 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46748 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46749 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46750 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46751 := (publicBatchWrapper89_copies1460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_745_18 := arithEq_of_rows h (row := 745) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_18
  simp only [← c46746, ← c46747, ← c46748] at e_745_18
  have e_745_19 := arithEq_of_rows h (row := 745) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_745_19
  simp only [← c46749, ← c46750, k2, ← c46751] at e_745_19
  have hr := e_745_19
  simp only [e_745_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5025)) := by
  have c46752 := (publicBatchWrapper89_copies1461 a h).1
  have c46753 := (publicBatchWrapper89_copies1461 a h).2.1
  have c46754 := (publicBatchWrapper89_copies1461 a h).2.2.1
  have c46755 := (publicBatchWrapper89_copies1461 a h).2.2.2.1
  have c46756 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.1
  have c46757 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_0 := arithEq_of_rows h (row := 746) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_0
  simp only [← c46752, ← c46753, ← c46754] at e_746_0
  have e_746_1 := arithEq_of_rows h (row := 746) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_1
  simp only [← c46755, ← c46756, k2, ← c46757] at e_746_1
  have hr := e_746_1
  simp only [e_746_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5026)) := by
  have c46758 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.1
  have c46759 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.1
  have c46760 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.1
  have c46761 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.1
  have c46762 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46763 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_2 := arithEq_of_rows h (row := 746) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_2
  simp only [← c46758, ← c46759, ← c46760] at e_746_2
  have e_746_3 := arithEq_of_rows h (row := 746) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_3
  simp only [← c46761, ← c46762, k2, ← c46763] at e_746_3
  have hr := e_746_3
  simp only [e_746_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5027)) := by
  have c46764 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46765 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46766 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46767 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46768 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46769 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_4 := arithEq_of_rows h (row := 746) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_4
  simp only [← c46764, ← c46765, ← c46766] at e_746_4
  have e_746_5 := arithEq_of_rows h (row := 746) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_5
  simp only [← c46767, ← c46768, k2, ← c46769] at e_746_5
  have hr := e_746_5
  simp only [e_746_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5028)) := by
  have c46770 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46771 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46772 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46773 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46774 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46775 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_6 := arithEq_of_rows h (row := 746) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_6
  simp only [← c46770, ← c46771, ← c46772] at e_746_6
  have e_746_7 := arithEq_of_rows h (row := 746) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_7
  simp only [← c46773, ← c46774, k2, ← c46775] at e_746_7
  have hr := e_746_7
  simp only [e_746_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5029)) := by
  have c46776 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46777 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46778 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46779 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46780 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46781 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_8 := arithEq_of_rows h (row := 746) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_8
  simp only [← c46776, ← c46777, ← c46778] at e_746_8
  have e_746_9 := arithEq_of_rows h (row := 746) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_9
  simp only [← c46779, ← c46780, k2, ← c46781] at e_746_9
  have hr := e_746_9
  simp only [e_746_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5030)) := by
  have c46782 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46783 := (publicBatchWrapper89_copies1461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46784 := (publicBatchWrapper89_copies1462 a h).1
  have c46785 := (publicBatchWrapper89_copies1462 a h).2.1
  have c46786 := (publicBatchWrapper89_copies1462 a h).2.2.1
  have c46787 := (publicBatchWrapper89_copies1462 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_10 := arithEq_of_rows h (row := 746) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_10
  simp only [← c46782, ← c46783, ← c46784] at e_746_10
  have e_746_11 := arithEq_of_rows h (row := 746) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_11
  simp only [← c46785, ← c46786, k2, ← c46787] at e_746_11
  have hr := e_746_11
  simp only [e_746_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5031)) := by
  have c46788 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.1
  have c46789 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.1
  have c46790 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.1
  have c46791 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.1
  have c46792 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.1
  have c46793 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_12 := arithEq_of_rows h (row := 746) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_12
  simp only [← c46788, ← c46789, ← c46790] at e_746_12
  have e_746_13 := arithEq_of_rows h (row := 746) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_13
  simp only [← c46791, ← c46792, k2, ← c46793] at e_746_13
  have hr := e_746_13
  simp only [e_746_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5032)) := by
  have c46794 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46795 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46796 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46797 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46798 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46799 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_14 := arithEq_of_rows h (row := 746) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_14
  simp only [← c46794, ← c46795, ← c46796] at e_746_14
  have e_746_15 := arithEq_of_rows h (row := 746) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_15
  simp only [← c46797, ← c46798, k2, ← c46799] at e_746_15
  have hr := e_746_15
  simp only [e_746_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5033)) := by
  have c46800 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46801 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46802 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46803 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46804 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46805 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_16 := arithEq_of_rows h (row := 746) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_16
  simp only [← c46800, ← c46801, ← c46802] at e_746_16
  have e_746_17 := arithEq_of_rows h (row := 746) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_17
  simp only [← c46803, ← c46804, k2, ← c46805] at e_746_17
  have hr := e_746_17
  simp only [e_746_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 746 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5034)) := by
  have c46806 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46807 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46808 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46809 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46810 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46811 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_746_18 := arithEq_of_rows h (row := 746) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_18
  simp only [← c46806, ← c46807, ← c46808] at e_746_18
  have e_746_19 := arithEq_of_rows h (row := 746) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_746_19
  simp only [← c46809, ← c46810, k2, ← c46811] at e_746_19
  have hr := e_746_19
  simp only [e_746_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5035)) := by
  have c46812 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46813 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46814 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46815 := (publicBatchWrapper89_copies1462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46816 := (publicBatchWrapper89_copies1463 a h).1
  have c46817 := (publicBatchWrapper89_copies1463 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_0 := arithEq_of_rows h (row := 747) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_0
  simp only [← c46812, ← c46813, ← c46814] at e_747_0
  have e_747_1 := arithEq_of_rows h (row := 747) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_1
  simp only [← c46815, ← c46816, k2, ← c46817] at e_747_1
  have hr := e_747_1
  simp only [e_747_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5036)) := by
  have c46818 := (publicBatchWrapper89_copies1463 a h).2.2.1
  have c46819 := (publicBatchWrapper89_copies1463 a h).2.2.2.1
  have c46820 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.1
  have c46821 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.1
  have c46822 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.1
  have c46823 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_2 := arithEq_of_rows h (row := 747) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_2
  simp only [← c46818, ← c46819, ← c46820] at e_747_2
  have e_747_3 := arithEq_of_rows h (row := 747) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_3
  simp only [← c46821, ← c46822, k2, ← c46823] at e_747_3
  have hr := e_747_3
  simp only [e_747_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5037)) := by
  have c46824 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.1
  have c46825 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.1
  have c46826 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46827 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46828 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46829 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_4 := arithEq_of_rows h (row := 747) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_4
  simp only [← c46824, ← c46825, ← c46826] at e_747_4
  have e_747_5 := arithEq_of_rows h (row := 747) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_5
  simp only [← c46827, ← c46828, k2, ← c46829] at e_747_5
  have hr := e_747_5
  simp only [e_747_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5038)) := by
  have c46830 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46831 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46832 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46833 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46834 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46835 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_6 := arithEq_of_rows h (row := 747) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_6
  simp only [← c46830, ← c46831, ← c46832] at e_747_6
  have e_747_7 := arithEq_of_rows h (row := 747) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_7
  simp only [← c46833, ← c46834, k2, ← c46835] at e_747_7
  have hr := e_747_7
  simp only [e_747_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5039)) := by
  have c46836 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46837 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46838 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46839 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46840 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46841 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_8 := arithEq_of_rows h (row := 747) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_8
  simp only [← c46836, ← c46837, ← c46838] at e_747_8
  have e_747_9 := arithEq_of_rows h (row := 747) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_9
  simp only [← c46839, ← c46840, k2, ← c46841] at e_747_9
  have hr := e_747_9
  simp only [e_747_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5040)) := by
  have c46842 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46843 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46844 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46845 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46846 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46847 := (publicBatchWrapper89_copies1463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_10 := arithEq_of_rows h (row := 747) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_10
  simp only [← c46842, ← c46843, ← c46844] at e_747_10
  have e_747_11 := arithEq_of_rows h (row := 747) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_11
  simp only [← c46845, ← c46846, k2, ← c46847] at e_747_11
  have hr := e_747_11
  simp only [e_747_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5041)) := by
  have c46848 := (publicBatchWrapper89_copies1464 a h).1
  have c46849 := (publicBatchWrapper89_copies1464 a h).2.1
  have c46850 := (publicBatchWrapper89_copies1464 a h).2.2.1
  have c46851 := (publicBatchWrapper89_copies1464 a h).2.2.2.1
  have c46852 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.1
  have c46853 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_12 := arithEq_of_rows h (row := 747) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_12
  simp only [← c46848, ← c46849, ← c46850] at e_747_12
  have e_747_13 := arithEq_of_rows h (row := 747) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_13
  simp only [← c46851, ← c46852, k2, ← c46853] at e_747_13
  have hr := e_747_13
  simp only [e_747_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5042)) := by
  have c46854 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.1
  have c46855 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.1
  have c46856 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.1
  have c46857 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.1
  have c46858 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46859 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_14 := arithEq_of_rows h (row := 747) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_14
  simp only [← c46854, ← c46855, ← c46856] at e_747_14
  have e_747_15 := arithEq_of_rows h (row := 747) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_15
  simp only [← c46857, ← c46858, k2, ← c46859] at e_747_15
  have hr := e_747_15
  simp only [e_747_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5043)) := by
  have c46860 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46861 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46862 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46863 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46864 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46865 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_16 := arithEq_of_rows h (row := 747) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_16
  simp only [← c46860, ← c46861, ← c46862] at e_747_16
  have e_747_17 := arithEq_of_rows h (row := 747) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_17
  simp only [← c46863, ← c46864, k2, ← c46865] at e_747_17
  have hr := e_747_17
  simp only [e_747_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 747 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5044)) := by
  have c46866 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46867 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46868 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46869 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46870 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46871 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_747_18 := arithEq_of_rows h (row := 747) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_18
  simp only [← c46866, ← c46867, ← c46868] at e_747_18
  have e_747_19 := arithEq_of_rows h (row := 747) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_747_19
  simp only [← c46869, ← c46870, k2, ← c46871] at e_747_19
  have hr := e_747_19
  simp only [e_747_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5045)) := by
  have c46872 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46873 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46874 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46875 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46876 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46877 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_0 := arithEq_of_rows h (row := 748) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_0
  simp only [← c46872, ← c46873, ← c46874] at e_748_0
  have e_748_1 := arithEq_of_rows h (row := 748) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_1
  simp only [← c46875, ← c46876, k2, ← c46877] at e_748_1
  have hr := e_748_1
  simp only [e_748_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5046)) := by
  have c46878 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46879 := (publicBatchWrapper89_copies1464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46880 := (publicBatchWrapper89_copies1465 a h).1
  have c46881 := (publicBatchWrapper89_copies1465 a h).2.1
  have c46882 := (publicBatchWrapper89_copies1465 a h).2.2.1
  have c46883 := (publicBatchWrapper89_copies1465 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_2 := arithEq_of_rows h (row := 748) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_2
  simp only [← c46878, ← c46879, ← c46880] at e_748_2
  have e_748_3 := arithEq_of_rows h (row := 748) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_3
  simp only [← c46881, ← c46882, k2, ← c46883] at e_748_3
  have hr := e_748_3
  simp only [e_748_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5047)) := by
  have c46884 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.1
  have c46885 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.1
  have c46886 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.1
  have c46887 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.1
  have c46888 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.1
  have c46889 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_4 := arithEq_of_rows h (row := 748) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_4
  simp only [← c46884, ← c46885, ← c46886] at e_748_4
  have e_748_5 := arithEq_of_rows h (row := 748) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_5
  simp only [← c46887, ← c46888, k2, ← c46889] at e_748_5
  have hr := e_748_5
  simp only [e_748_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5048)) := by
  have c46890 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46891 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46892 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46893 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46894 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46895 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_6 := arithEq_of_rows h (row := 748) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_6
  simp only [← c46890, ← c46891, ← c46892] at e_748_6
  have e_748_7 := arithEq_of_rows h (row := 748) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_7
  simp only [← c46893, ← c46894, k2, ← c46895] at e_748_7
  have hr := e_748_7
  simp only [e_748_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5049)) := by
  have c46896 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46897 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46898 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46899 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46900 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46901 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_8 := arithEq_of_rows h (row := 748) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_8
  simp only [← c46896, ← c46897, ← c46898] at e_748_8
  have e_748_9 := arithEq_of_rows h (row := 748) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_9
  simp only [← c46899, ← c46900, k2, ← c46901] at e_748_9
  have hr := e_748_9
  simp only [e_748_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5050)) := by
  have c46902 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46903 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46904 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46905 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46906 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46907 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_10 := arithEq_of_rows h (row := 748) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_10
  simp only [← c46902, ← c46903, ← c46904] at e_748_10
  have e_748_11 := arithEq_of_rows h (row := 748) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_11
  simp only [← c46905, ← c46906, k2, ← c46907] at e_748_11
  have hr := e_748_11
  simp only [e_748_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5051)) := by
  have c46908 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46909 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46910 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46911 := (publicBatchWrapper89_copies1465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46912 := (publicBatchWrapper89_copies1466 a h).1
  have c46913 := (publicBatchWrapper89_copies1466 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_12 := arithEq_of_rows h (row := 748) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_12
  simp only [← c46908, ← c46909, ← c46910] at e_748_12
  have e_748_13 := arithEq_of_rows h (row := 748) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_13
  simp only [← c46911, ← c46912, k2, ← c46913] at e_748_13
  have hr := e_748_13
  simp only [e_748_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5052)) := by
  have c46914 := (publicBatchWrapper89_copies1466 a h).2.2.1
  have c46915 := (publicBatchWrapper89_copies1466 a h).2.2.2.1
  have c46916 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.1
  have c46917 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.1
  have c46918 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.1
  have c46919 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_14 := arithEq_of_rows h (row := 748) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_14
  simp only [← c46914, ← c46915, ← c46916] at e_748_14
  have e_748_15 := arithEq_of_rows h (row := 748) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_15
  simp only [← c46917, ← c46918, k2, ← c46919] at e_748_15
  have hr := e_748_15
  simp only [e_748_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5053)) := by
  have c46920 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.1
  have c46921 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.1
  have c46922 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46923 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46924 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46925 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_16 := arithEq_of_rows h (row := 748) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_16
  simp only [← c46920, ← c46921, ← c46922] at e_748_16
  have e_748_17 := arithEq_of_rows h (row := 748) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_17
  simp only [← c46923, ← c46924, k2, ← c46925] at e_748_17
  have hr := e_748_17
  simp only [e_748_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 748 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5054)) := by
  have c46926 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46927 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46928 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46929 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46930 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46931 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_748_18 := arithEq_of_rows h (row := 748) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_18
  simp only [← c46926, ← c46927, ← c46928] at e_748_18
  have e_748_19 := arithEq_of_rows h (row := 748) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_748_19
  simp only [← c46929, ← c46930, k2, ← c46931] at e_748_19
  have hr := e_748_19
  simp only [e_748_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5055)) := by
  have c46932 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46933 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46934 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46935 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46936 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46937 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_0 := arithEq_of_rows h (row := 749) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_0
  simp only [← c46932, ← c46933, ← c46934] at e_749_0
  have e_749_1 := arithEq_of_rows h (row := 749) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_1
  simp only [← c46935, ← c46936, k2, ← c46937] at e_749_1
  have hr := e_749_1
  simp only [e_749_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5056)) := by
  have c46938 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46939 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46940 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46941 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46942 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46943 := (publicBatchWrapper89_copies1466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_2 := arithEq_of_rows h (row := 749) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_2
  simp only [← c46938, ← c46939, ← c46940] at e_749_2
  have e_749_3 := arithEq_of_rows h (row := 749) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_3
  simp only [← c46941, ← c46942, k2, ← c46943] at e_749_3
  have hr := e_749_3
  simp only [e_749_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5057)) := by
  have c46944 := (publicBatchWrapper89_copies1467 a h).1
  have c46945 := (publicBatchWrapper89_copies1467 a h).2.1
  have c46946 := (publicBatchWrapper89_copies1467 a h).2.2.1
  have c46947 := (publicBatchWrapper89_copies1467 a h).2.2.2.1
  have c46948 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.1
  have c46949 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_4 := arithEq_of_rows h (row := 749) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_4
  simp only [← c46944, ← c46945, ← c46946] at e_749_4
  have e_749_5 := arithEq_of_rows h (row := 749) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_5
  simp only [← c46947, ← c46948, k2, ← c46949] at e_749_5
  have hr := e_749_5
  simp only [e_749_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5058)) := by
  have c46950 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.1
  have c46951 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.1
  have c46952 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.1
  have c46953 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.1
  have c46954 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46955 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_6 := arithEq_of_rows h (row := 749) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_6
  simp only [← c46950, ← c46951, ← c46952] at e_749_6
  have e_749_7 := arithEq_of_rows h (row := 749) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_7
  simp only [← c46953, ← c46954, k2, ← c46955] at e_749_7
  have hr := e_749_7
  simp only [e_749_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5059)) := by
  have c46956 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46957 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46958 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46959 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46960 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46961 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_8 := arithEq_of_rows h (row := 749) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_8
  simp only [← c46956, ← c46957, ← c46958] at e_749_8
  have e_749_9 := arithEq_of_rows h (row := 749) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_9
  simp only [← c46959, ← c46960, k2, ← c46961] at e_749_9
  have hr := e_749_9
  simp only [e_749_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5060)) := by
  have c46962 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46963 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46964 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46965 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46966 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46967 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_10 := arithEq_of_rows h (row := 749) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_10
  simp only [← c46962, ← c46963, ← c46964] at e_749_10
  have e_749_11 := arithEq_of_rows h (row := 749) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_11
  simp only [← c46965, ← c46966, k2, ← c46967] at e_749_11
  have hr := e_749_11
  simp only [e_749_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5061)) := by
  have c46968 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46969 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46970 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46971 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46972 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46973 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_12 := arithEq_of_rows h (row := 749) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_12
  simp only [← c46968, ← c46969, ← c46970] at e_749_12
  have e_749_13 := arithEq_of_rows h (row := 749) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_13
  simp only [← c46971, ← c46972, k2, ← c46973] at e_749_13
  have hr := e_749_13
  simp only [e_749_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5062)) := by
  have c46974 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46975 := (publicBatchWrapper89_copies1467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c46976 := (publicBatchWrapper89_copies1468 a h).1
  have c46977 := (publicBatchWrapper89_copies1468 a h).2.1
  have c46978 := (publicBatchWrapper89_copies1468 a h).2.2.1
  have c46979 := (publicBatchWrapper89_copies1468 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_14 := arithEq_of_rows h (row := 749) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_14
  simp only [← c46974, ← c46975, ← c46976] at e_749_14
  have e_749_15 := arithEq_of_rows h (row := 749) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_15
  simp only [← c46977, ← c46978, k2, ← c46979] at e_749_15
  have hr := e_749_15
  simp only [e_749_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5063)) := by
  have c46980 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.1
  have c46981 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.1
  have c46982 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.1
  have c46983 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.1
  have c46984 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.1
  have c46985 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_16 := arithEq_of_rows h (row := 749) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_16
  simp only [← c46980, ← c46981, ← c46982] at e_749_16
  have e_749_17 := arithEq_of_rows h (row := 749) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_17
  simp only [← c46983, ← c46984, k2, ← c46985] at e_749_17
  have hr := e_749_17
  simp only [e_749_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 749 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5064)) := by
  have c46986 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c46987 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c46988 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46989 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46990 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46991 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_749_18 := arithEq_of_rows h (row := 749) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_18
  simp only [← c46986, ← c46987, ← c46988] at e_749_18
  have e_749_19 := arithEq_of_rows h (row := 749) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_749_19
  simp only [← c46989, ← c46990, k2, ← c46991] at e_749_19
  have hr := e_749_19
  simp only [e_749_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5065)) := by
  have c46992 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46993 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46994 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46995 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46996 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46997 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_0 := arithEq_of_rows h (row := 750) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_0
  simp only [← c46992, ← c46993, ← c46994] at e_750_0
  have e_750_1 := arithEq_of_rows h (row := 750) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_1
  simp only [← c46995, ← c46996, k2, ← c46997] at e_750_1
  have hr := e_750_1
  simp only [e_750_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5066)) := by
  have c46998 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46999 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47000 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47001 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47002 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47003 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_2 := arithEq_of_rows h (row := 750) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_2
  simp only [← c46998, ← c46999, ← c47000] at e_750_2
  have e_750_3 := arithEq_of_rows h (row := 750) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_3
  simp only [← c47001, ← c47002, k2, ← c47003] at e_750_3
  have hr := e_750_3
  simp only [e_750_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5067)) := by
  have c47004 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47005 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47006 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47007 := (publicBatchWrapper89_copies1468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47008 := (publicBatchWrapper89_copies1469 a h).1
  have c47009 := (publicBatchWrapper89_copies1469 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_4 := arithEq_of_rows h (row := 750) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_4
  simp only [← c47004, ← c47005, ← c47006] at e_750_4
  have e_750_5 := arithEq_of_rows h (row := 750) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_5
  simp only [← c47007, ← c47008, k2, ← c47009] at e_750_5
  have hr := e_750_5
  simp only [e_750_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5068)) := by
  have c47010 := (publicBatchWrapper89_copies1469 a h).2.2.1
  have c47011 := (publicBatchWrapper89_copies1469 a h).2.2.2.1
  have c47012 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.1
  have c47013 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.1
  have c47014 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.1
  have c47015 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_6 := arithEq_of_rows h (row := 750) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_6
  simp only [← c47010, ← c47011, ← c47012] at e_750_6
  have e_750_7 := arithEq_of_rows h (row := 750) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_7
  simp only [← c47013, ← c47014, k2, ← c47015] at e_750_7
  have hr := e_750_7
  simp only [e_750_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5069)) := by
  have c47016 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.1
  have c47017 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.1
  have c47018 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47019 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47020 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47021 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_8 := arithEq_of_rows h (row := 750) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_8
  simp only [← c47016, ← c47017, ← c47018] at e_750_8
  have e_750_9 := arithEq_of_rows h (row := 750) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_9
  simp only [← c47019, ← c47020, k2, ← c47021] at e_750_9
  have hr := e_750_9
  simp only [e_750_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5070)) := by
  have c47022 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47023 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47024 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47025 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47026 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47027 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_10 := arithEq_of_rows h (row := 750) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_10
  simp only [← c47022, ← c47023, ← c47024] at e_750_10
  have e_750_11 := arithEq_of_rows h (row := 750) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_11
  simp only [← c47025, ← c47026, k2, ← c47027] at e_750_11
  have hr := e_750_11
  simp only [e_750_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5071)) := by
  have c47028 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47029 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47030 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47031 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47032 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47033 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_12 := arithEq_of_rows h (row := 750) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_12
  simp only [← c47028, ← c47029, ← c47030] at e_750_12
  have e_750_13 := arithEq_of_rows h (row := 750) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_13
  simp only [← c47031, ← c47032, k2, ← c47033] at e_750_13
  have hr := e_750_13
  simp only [e_750_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5072)) := by
  have c47034 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47035 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47036 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47037 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47038 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47039 := (publicBatchWrapper89_copies1469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_14 := arithEq_of_rows h (row := 750) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_14
  simp only [← c47034, ← c47035, ← c47036] at e_750_14
  have e_750_15 := arithEq_of_rows h (row := 750) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_15
  simp only [← c47037, ← c47038, k2, ← c47039] at e_750_15
  have hr := e_750_15
  simp only [e_750_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5073)) := by
  have c47040 := (publicBatchWrapper89_copies1470 a h).1
  have c47041 := (publicBatchWrapper89_copies1470 a h).2.1
  have c47042 := (publicBatchWrapper89_copies1470 a h).2.2.1
  have c47043 := (publicBatchWrapper89_copies1470 a h).2.2.2.1
  have c47044 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.1
  have c47045 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_16 := arithEq_of_rows h (row := 750) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_16
  simp only [← c47040, ← c47041, ← c47042] at e_750_16
  have e_750_17 := arithEq_of_rows h (row := 750) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_17
  simp only [← c47043, ← c47044, k2, ← c47045] at e_750_17
  have hr := e_750_17
  simp only [e_750_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 750 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5074)) := by
  have c47046 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.1
  have c47047 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.1
  have c47048 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.1
  have c47049 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.1
  have c47050 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47051 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_750_18 := arithEq_of_rows h (row := 750) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_18
  simp only [← c47046, ← c47047, ← c47048] at e_750_18
  have e_750_19 := arithEq_of_rows h (row := 750) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_750_19
  simp only [← c47049, ← c47050, k2, ← c47051] at e_750_19
  have hr := e_750_19
  simp only [e_750_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5075)) := by
  have c47052 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47053 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47054 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47055 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47056 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47057 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_0 := arithEq_of_rows h (row := 751) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_0
  simp only [← c47052, ← c47053, ← c47054] at e_751_0
  have e_751_1 := arithEq_of_rows h (row := 751) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_1
  simp only [← c47055, ← c47056, k2, ← c47057] at e_751_1
  have hr := e_751_1
  simp only [e_751_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5076)) := by
  have c47058 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47059 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47060 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47061 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47062 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47063 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_2 := arithEq_of_rows h (row := 751) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_2
  simp only [← c47058, ← c47059, ← c47060] at e_751_2
  have e_751_3 := arithEq_of_rows h (row := 751) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_3
  simp only [← c47061, ← c47062, k2, ← c47063] at e_751_3
  have hr := e_751_3
  simp only [e_751_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5077)) := by
  have c47064 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47065 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47066 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47067 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47068 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47069 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_4 := arithEq_of_rows h (row := 751) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_4
  simp only [← c47064, ← c47065, ← c47066] at e_751_4
  have e_751_5 := arithEq_of_rows h (row := 751) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_5
  simp only [← c47067, ← c47068, k2, ← c47069] at e_751_5
  have hr := e_751_5
  simp only [e_751_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5078)) := by
  have c47070 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47071 := (publicBatchWrapper89_copies1470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47072 := (publicBatchWrapper89_copies1471 a h).1
  have c47073 := (publicBatchWrapper89_copies1471 a h).2.1
  have c47074 := (publicBatchWrapper89_copies1471 a h).2.2.1
  have c47075 := (publicBatchWrapper89_copies1471 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_6 := arithEq_of_rows h (row := 751) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_6
  simp only [← c47070, ← c47071, ← c47072] at e_751_6
  have e_751_7 := arithEq_of_rows h (row := 751) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_7
  simp only [← c47073, ← c47074, k2, ← c47075] at e_751_7
  have hr := e_751_7
  simp only [e_751_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5079)) := by
  have c47076 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.1
  have c47077 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.1
  have c47078 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.1
  have c47079 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.1
  have c47080 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.1
  have c47081 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_8 := arithEq_of_rows h (row := 751) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_8
  simp only [← c47076, ← c47077, ← c47078] at e_751_8
  have e_751_9 := arithEq_of_rows h (row := 751) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_9
  simp only [← c47079, ← c47080, k2, ← c47081] at e_751_9
  have hr := e_751_9
  simp only [e_751_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5080)) := by
  have c47082 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47083 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47084 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47085 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47086 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47087 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_10 := arithEq_of_rows h (row := 751) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_10
  simp only [← c47082, ← c47083, ← c47084] at e_751_10
  have e_751_11 := arithEq_of_rows h (row := 751) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_11
  simp only [← c47085, ← c47086, k2, ← c47087] at e_751_11
  have hr := e_751_11
  simp only [e_751_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5081)) := by
  have c47088 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47089 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47090 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47091 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47092 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47093 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_12 := arithEq_of_rows h (row := 751) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_12
  simp only [← c47088, ← c47089, ← c47090] at e_751_12
  have e_751_13 := arithEq_of_rows h (row := 751) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_13
  simp only [← c47091, ← c47092, k2, ← c47093] at e_751_13
  have hr := e_751_13
  simp only [e_751_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5082)) := by
  have c47094 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47095 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47096 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47097 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47098 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47099 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_14 := arithEq_of_rows h (row := 751) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_14
  simp only [← c47094, ← c47095, ← c47096] at e_751_14
  have e_751_15 := arithEq_of_rows h (row := 751) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_15
  simp only [← c47097, ← c47098, k2, ← c47099] at e_751_15
  have hr := e_751_15
  simp only [e_751_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5083)) := by
  have c47100 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47101 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47102 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47103 := (publicBatchWrapper89_copies1471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47104 := (publicBatchWrapper89_copies1472 a h).1
  have c47105 := (publicBatchWrapper89_copies1472 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_16 := arithEq_of_rows h (row := 751) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_16
  simp only [← c47100, ← c47101, ← c47102] at e_751_16
  have e_751_17 := arithEq_of_rows h (row := 751) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_17
  simp only [← c47103, ← c47104, k2, ← c47105] at e_751_17
  have hr := e_751_17
  simp only [e_751_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 751 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5084)) := by
  have c47106 := (publicBatchWrapper89_copies1472 a h).2.2.1
  have c47107 := (publicBatchWrapper89_copies1472 a h).2.2.2.1
  have c47108 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.1
  have c47109 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.1
  have c47110 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.1
  have c47111 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_751_18 := arithEq_of_rows h (row := 751) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_18
  simp only [← c47106, ← c47107, ← c47108] at e_751_18
  have e_751_19 := arithEq_of_rows h (row := 751) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_751_19
  simp only [← c47109, ← c47110, k2, ← c47111] at e_751_19
  have hr := e_751_19
  simp only [e_751_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5085)) := by
  have c47112 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.1
  have c47113 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.1
  have c47114 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47115 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47116 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47117 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_0 := arithEq_of_rows h (row := 752) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_0
  simp only [← c47112, ← c47113, ← c47114] at e_752_0
  have e_752_1 := arithEq_of_rows h (row := 752) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_1
  simp only [← c47115, ← c47116, k2, ← c47117] at e_752_1
  have hr := e_752_1
  simp only [e_752_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5086)) := by
  have c47118 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47119 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47120 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47121 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47122 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47123 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_2 := arithEq_of_rows h (row := 752) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_2
  simp only [← c47118, ← c47119, ← c47120] at e_752_2
  have e_752_3 := arithEq_of_rows h (row := 752) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_3
  simp only [← c47121, ← c47122, k2, ← c47123] at e_752_3
  have hr := e_752_3
  simp only [e_752_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5087)) := by
  have c47124 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47125 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47126 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47127 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47128 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47129 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_4 := arithEq_of_rows h (row := 752) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_4
  simp only [← c47124, ← c47125, ← c47126] at e_752_4
  have e_752_5 := arithEq_of_rows h (row := 752) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_5
  simp only [← c47127, ← c47128, k2, ← c47129] at e_752_5
  have hr := e_752_5
  simp only [e_752_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5088)) := by
  have c47130 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47131 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47132 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47133 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47134 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47135 := (publicBatchWrapper89_copies1472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_6 := arithEq_of_rows h (row := 752) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_6
  simp only [← c47130, ← c47131, ← c47132] at e_752_6
  have e_752_7 := arithEq_of_rows h (row := 752) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_7
  simp only [← c47133, ← c47134, k2, ← c47135] at e_752_7
  have hr := e_752_7
  simp only [e_752_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5089)) := by
  have c47136 := (publicBatchWrapper89_copies1473 a h).1
  have c47137 := (publicBatchWrapper89_copies1473 a h).2.1
  have c47138 := (publicBatchWrapper89_copies1473 a h).2.2.1
  have c47139 := (publicBatchWrapper89_copies1473 a h).2.2.2.1
  have c47140 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.1
  have c47141 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_8 := arithEq_of_rows h (row := 752) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_8
  simp only [← c47136, ← c47137, ← c47138] at e_752_8
  have e_752_9 := arithEq_of_rows h (row := 752) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_9
  simp only [← c47139, ← c47140, k2, ← c47141] at e_752_9
  have hr := e_752_9
  simp only [e_752_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5090)) := by
  have c47142 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.1
  have c47143 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.1
  have c47144 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.1
  have c47145 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.1
  have c47146 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47147 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_10 := arithEq_of_rows h (row := 752) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_10
  simp only [← c47142, ← c47143, ← c47144] at e_752_10
  have e_752_11 := arithEq_of_rows h (row := 752) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_11
  simp only [← c47145, ← c47146, k2, ← c47147] at e_752_11
  have hr := e_752_11
  simp only [e_752_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5091)) := by
  have c47148 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47149 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47150 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47151 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47152 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47153 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_12 := arithEq_of_rows h (row := 752) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_12
  simp only [← c47148, ← c47149, ← c47150] at e_752_12
  have e_752_13 := arithEq_of_rows h (row := 752) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_13
  simp only [← c47151, ← c47152, k2, ← c47153] at e_752_13
  have hr := e_752_13
  simp only [e_752_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5092)) := by
  have c47154 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47155 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47156 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47157 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47158 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47159 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_14 := arithEq_of_rows h (row := 752) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_14
  simp only [← c47154, ← c47155, ← c47156] at e_752_14
  have e_752_15 := arithEq_of_rows h (row := 752) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_15
  simp only [← c47157, ← c47158, k2, ← c47159] at e_752_15
  have hr := e_752_15
  simp only [e_752_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5093)) := by
  have c47160 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47161 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47162 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47163 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47164 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47165 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_16 := arithEq_of_rows h (row := 752) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_16
  simp only [← c47160, ← c47161, ← c47162] at e_752_16
  have e_752_17 := arithEq_of_rows h (row := 752) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_17
  simp only [← c47163, ← c47164, k2, ← c47165] at e_752_17
  have hr := e_752_17
  simp only [e_752_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 752 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5094)) := by
  have c47166 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47167 := (publicBatchWrapper89_copies1473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47168 := (publicBatchWrapper89_copies1474 a h).1
  have c47169 := (publicBatchWrapper89_copies1474 a h).2.1
  have c47170 := (publicBatchWrapper89_copies1474 a h).2.2.1
  have c47171 := (publicBatchWrapper89_copies1474 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_752_18 := arithEq_of_rows h (row := 752) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_18
  simp only [← c47166, ← c47167, ← c47168] at e_752_18
  have e_752_19 := arithEq_of_rows h (row := 752) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_752_19
  simp only [← c47169, ← c47170, k2, ← c47171] at e_752_19
  have hr := e_752_19
  simp only [e_752_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5095)) := by
  have c47172 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.1
  have c47173 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.1
  have c47174 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.1
  have c47175 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.1
  have c47176 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.1
  have c47177 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_0 := arithEq_of_rows h (row := 753) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_0
  simp only [← c47172, ← c47173, ← c47174] at e_753_0
  have e_753_1 := arithEq_of_rows h (row := 753) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_1
  simp only [← c47175, ← c47176, k2, ← c47177] at e_753_1
  have hr := e_753_1
  simp only [e_753_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5096)) := by
  have c47178 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47179 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47180 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47181 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47182 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47183 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_2 := arithEq_of_rows h (row := 753) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_2
  simp only [← c47178, ← c47179, ← c47180] at e_753_2
  have e_753_3 := arithEq_of_rows h (row := 753) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_3
  simp only [← c47181, ← c47182, k2, ← c47183] at e_753_3
  have hr := e_753_3
  simp only [e_753_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5097)) := by
  have c47184 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47185 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47186 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47187 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47188 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47189 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_4 := arithEq_of_rows h (row := 753) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_4
  simp only [← c47184, ← c47185, ← c47186] at e_753_4
  have e_753_5 := arithEq_of_rows h (row := 753) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_5
  simp only [← c47187, ← c47188, k2, ← c47189] at e_753_5
  have hr := e_753_5
  simp only [e_753_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5098)) := by
  have c47190 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47191 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47192 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47193 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47194 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47195 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_6 := arithEq_of_rows h (row := 753) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_6
  simp only [← c47190, ← c47191, ← c47192] at e_753_6
  have e_753_7 := arithEq_of_rows h (row := 753) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_7
  simp only [← c47193, ← c47194, k2, ← c47195] at e_753_7
  have hr := e_753_7
  simp only [e_753_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5099)) := by
  have c47196 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47197 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47198 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47199 := (publicBatchWrapper89_copies1474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47200 := (publicBatchWrapper89_copies1475 a h).1
  have c47201 := (publicBatchWrapper89_copies1475 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_8 := arithEq_of_rows h (row := 753) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_8
  simp only [← c47196, ← c47197, ← c47198] at e_753_8
  have e_753_9 := arithEq_of_rows h (row := 753) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_9
  simp only [← c47199, ← c47200, k2, ← c47201] at e_753_9
  have hr := e_753_9
  simp only [e_753_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5100)) := by
  have c47202 := (publicBatchWrapper89_copies1475 a h).2.2.1
  have c47203 := (publicBatchWrapper89_copies1475 a h).2.2.2.1
  have c47204 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.1
  have c47205 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.1
  have c47206 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.1
  have c47207 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_10 := arithEq_of_rows h (row := 753) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_10
  simp only [← c47202, ← c47203, ← c47204] at e_753_10
  have e_753_11 := arithEq_of_rows h (row := 753) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_11
  simp only [← c47205, ← c47206, k2, ← c47207] at e_753_11
  have hr := e_753_11
  simp only [e_753_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5101)) := by
  have c47208 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.1
  have c47209 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.1
  have c47210 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47211 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47212 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47213 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_12 := arithEq_of_rows h (row := 753) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_12
  simp only [← c47208, ← c47209, ← c47210] at e_753_12
  have e_753_13 := arithEq_of_rows h (row := 753) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_13
  simp only [← c47211, ← c47212, k2, ← c47213] at e_753_13
  have hr := e_753_13
  simp only [e_753_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5102)) := by
  have c47214 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47215 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47216 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47217 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47218 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47219 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_14 := arithEq_of_rows h (row := 753) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_14
  simp only [← c47214, ← c47215, ← c47216] at e_753_14
  have e_753_15 := arithEq_of_rows h (row := 753) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_15
  simp only [← c47217, ← c47218, k2, ← c47219] at e_753_15
  have hr := e_753_15
  simp only [e_753_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5103)) := by
  have c47220 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47221 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47222 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47223 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47224 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47225 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_16 := arithEq_of_rows h (row := 753) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_16
  simp only [← c47220, ← c47221, ← c47222] at e_753_16
  have e_753_17 := arithEq_of_rows h (row := 753) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_17
  simp only [← c47223, ← c47224, k2, ← c47225] at e_753_17
  have hr := e_753_17
  simp only [e_753_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 753 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5104)) := by
  have c47226 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47227 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47228 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47229 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47230 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47231 := (publicBatchWrapper89_copies1475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_753_18 := arithEq_of_rows h (row := 753) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_18
  simp only [← c47226, ← c47227, ← c47228] at e_753_18
  have e_753_19 := arithEq_of_rows h (row := 753) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_753_19
  simp only [← c47229, ← c47230, k2, ← c47231] at e_753_19
  have hr := e_753_19
  simp only [e_753_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5105)) := by
  have c47232 := (publicBatchWrapper89_copies1476 a h).1
  have c47233 := (publicBatchWrapper89_copies1476 a h).2.1
  have c47234 := (publicBatchWrapper89_copies1476 a h).2.2.1
  have c47235 := (publicBatchWrapper89_copies1476 a h).2.2.2.1
  have c47236 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.1
  have c47237 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_0 := arithEq_of_rows h (row := 754) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_0
  simp only [← c47232, ← c47233, ← c47234] at e_754_0
  have e_754_1 := arithEq_of_rows h (row := 754) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_1
  simp only [← c47235, ← c47236, k2, ← c47237] at e_754_1
  have hr := e_754_1
  simp only [e_754_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5106)) := by
  have c47238 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.1
  have c47239 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.1
  have c47240 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.1
  have c47241 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.1
  have c47242 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47243 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_2 := arithEq_of_rows h (row := 754) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_2
  simp only [← c47238, ← c47239, ← c47240] at e_754_2
  have e_754_3 := arithEq_of_rows h (row := 754) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_3
  simp only [← c47241, ← c47242, k2, ← c47243] at e_754_3
  have hr := e_754_3
  simp only [e_754_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5107)) := by
  have c47244 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47245 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47246 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47247 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47248 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47249 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_4 := arithEq_of_rows h (row := 754) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_4
  simp only [← c47244, ← c47245, ← c47246] at e_754_4
  have e_754_5 := arithEq_of_rows h (row := 754) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_5
  simp only [← c47247, ← c47248, k2, ← c47249] at e_754_5
  have hr := e_754_5
  simp only [e_754_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5108)) := by
  have c47250 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47251 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47252 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47253 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47254 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47255 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_6 := arithEq_of_rows h (row := 754) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_6
  simp only [← c47250, ← c47251, ← c47252] at e_754_6
  have e_754_7 := arithEq_of_rows h (row := 754) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_7
  simp only [← c47253, ← c47254, k2, ← c47255] at e_754_7
  have hr := e_754_7
  simp only [e_754_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5109)) := by
  have c47256 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47257 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47258 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47259 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47260 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47261 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_8 := arithEq_of_rows h (row := 754) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_8
  simp only [← c47256, ← c47257, ← c47258] at e_754_8
  have e_754_9 := arithEq_of_rows h (row := 754) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_9
  simp only [← c47259, ← c47260, k2, ← c47261] at e_754_9
  have hr := e_754_9
  simp only [e_754_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5110)) := by
  have c47262 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47263 := (publicBatchWrapper89_copies1476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47264 := (publicBatchWrapper89_copies1477 a h).1
  have c47265 := (publicBatchWrapper89_copies1477 a h).2.1
  have c47266 := (publicBatchWrapper89_copies1477 a h).2.2.1
  have c47267 := (publicBatchWrapper89_copies1477 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_10 := arithEq_of_rows h (row := 754) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_10
  simp only [← c47262, ← c47263, ← c47264] at e_754_10
  have e_754_11 := arithEq_of_rows h (row := 754) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_11
  simp only [← c47265, ← c47266, k2, ← c47267] at e_754_11
  have hr := e_754_11
  simp only [e_754_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5111)) := by
  have c47268 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.1
  have c47269 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.1
  have c47270 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.1
  have c47271 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.1
  have c47272 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.1
  have c47273 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_12 := arithEq_of_rows h (row := 754) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_12
  simp only [← c47268, ← c47269, ← c47270] at e_754_12
  have e_754_13 := arithEq_of_rows h (row := 754) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_13
  simp only [← c47271, ← c47272, k2, ← c47273] at e_754_13
  have hr := e_754_13
  simp only [e_754_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5112)) := by
  have c47274 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47275 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47276 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47277 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47278 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47279 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_14 := arithEq_of_rows h (row := 754) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_14
  simp only [← c47274, ← c47275, ← c47276] at e_754_14
  have e_754_15 := arithEq_of_rows h (row := 754) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_15
  simp only [← c47277, ← c47278, k2, ← c47279] at e_754_15
  have hr := e_754_15
  simp only [e_754_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5113)) := by
  have c47280 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47281 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47282 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47283 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47284 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47285 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_16 := arithEq_of_rows h (row := 754) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_16
  simp only [← c47280, ← c47281, ← c47282] at e_754_16
  have e_754_17 := arithEq_of_rows h (row := 754) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_17
  simp only [← c47283, ← c47284, k2, ← c47285] at e_754_17
  have hr := e_754_17
  simp only [e_754_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 754 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5114)) := by
  have c47286 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47287 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47288 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47289 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47290 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47291 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_754_18 := arithEq_of_rows h (row := 754) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_18
  simp only [← c47286, ← c47287, ← c47288] at e_754_18
  have e_754_19 := arithEq_of_rows h (row := 754) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_754_19
  simp only [← c47289, ← c47290, k2, ← c47291] at e_754_19
  have hr := e_754_19
  simp only [e_754_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5115)) := by
  have c47292 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47293 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47294 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47295 := (publicBatchWrapper89_copies1477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47296 := (publicBatchWrapper89_copies1478 a h).1
  have c47297 := (publicBatchWrapper89_copies1478 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_0 := arithEq_of_rows h (row := 755) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_0
  simp only [← c47292, ← c47293, ← c47294] at e_755_0
  have e_755_1 := arithEq_of_rows h (row := 755) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_1
  simp only [← c47295, ← c47296, k2, ← c47297] at e_755_1
  have hr := e_755_1
  simp only [e_755_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5116)) := by
  have c47298 := (publicBatchWrapper89_copies1478 a h).2.2.1
  have c47299 := (publicBatchWrapper89_copies1478 a h).2.2.2.1
  have c47300 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.1
  have c47301 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.1
  have c47302 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.1
  have c47303 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_2 := arithEq_of_rows h (row := 755) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_2
  simp only [← c47298, ← c47299, ← c47300] at e_755_2
  have e_755_3 := arithEq_of_rows h (row := 755) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_3
  simp only [← c47301, ← c47302, k2, ← c47303] at e_755_3
  have hr := e_755_3
  simp only [e_755_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5117)) := by
  have c47304 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.1
  have c47305 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.1
  have c47306 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47307 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47308 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47309 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_4 := arithEq_of_rows h (row := 755) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_4
  simp only [← c47304, ← c47305, ← c47306] at e_755_4
  have e_755_5 := arithEq_of_rows h (row := 755) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_5
  simp only [← c47307, ← c47308, k2, ← c47309] at e_755_5
  have hr := e_755_5
  simp only [e_755_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5118)) := by
  have c47310 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47311 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47312 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47313 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47314 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47315 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_6 := arithEq_of_rows h (row := 755) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_6
  simp only [← c47310, ← c47311, ← c47312] at e_755_6
  have e_755_7 := arithEq_of_rows h (row := 755) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_7
  simp only [← c47313, ← c47314, k2, ← c47315] at e_755_7
  have hr := e_755_7
  simp only [e_755_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5119)) := by
  have c47316 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47317 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47318 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47319 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47320 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47321 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_8 := arithEq_of_rows h (row := 755) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_8
  simp only [← c47316, ← c47317, ← c47318] at e_755_8
  have e_755_9 := arithEq_of_rows h (row := 755) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_9
  simp only [← c47319, ← c47320, k2, ← c47321] at e_755_9
  have hr := e_755_9
  simp only [e_755_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5120)) := by
  have c47322 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47323 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47324 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47325 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47326 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47327 := (publicBatchWrapper89_copies1478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_10 := arithEq_of_rows h (row := 755) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_10
  simp only [← c47322, ← c47323, ← c47324] at e_755_10
  have e_755_11 := arithEq_of_rows h (row := 755) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_11
  simp only [← c47325, ← c47326, k2, ← c47327] at e_755_11
  have hr := e_755_11
  simp only [e_755_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5121)) := by
  have c47328 := (publicBatchWrapper89_copies1479 a h).1
  have c47329 := (publicBatchWrapper89_copies1479 a h).2.1
  have c47330 := (publicBatchWrapper89_copies1479 a h).2.2.1
  have c47331 := (publicBatchWrapper89_copies1479 a h).2.2.2.1
  have c47332 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.1
  have c47333 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_12 := arithEq_of_rows h (row := 755) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_12
  simp only [← c47328, ← c47329, ← c47330] at e_755_12
  have e_755_13 := arithEq_of_rows h (row := 755) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_13
  simp only [← c47331, ← c47332, k2, ← c47333] at e_755_13
  have hr := e_755_13
  simp only [e_755_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5122)) := by
  have c47334 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.1
  have c47335 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.1
  have c47336 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.1
  have c47337 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.1
  have c47338 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47339 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_14 := arithEq_of_rows h (row := 755) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_14
  simp only [← c47334, ← c47335, ← c47336] at e_755_14
  have e_755_15 := arithEq_of_rows h (row := 755) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_15
  simp only [← c47337, ← c47338, k2, ← c47339] at e_755_15
  have hr := e_755_15
  simp only [e_755_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5123)) := by
  have c47340 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47341 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47342 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47343 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47344 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47345 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_16 := arithEq_of_rows h (row := 755) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_16
  simp only [← c47340, ← c47341, ← c47342] at e_755_16
  have e_755_17 := arithEq_of_rows h (row := 755) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_17
  simp only [← c47343, ← c47344, k2, ← c47345] at e_755_17
  have hr := e_755_17
  simp only [e_755_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 755 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5124)) := by
  have c47346 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47347 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47348 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47349 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47350 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47351 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_755_18 := arithEq_of_rows h (row := 755) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_18
  simp only [← c47346, ← c47347, ← c47348] at e_755_18
  have e_755_19 := arithEq_of_rows h (row := 755) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_755_19
  simp only [← c47349, ← c47350, k2, ← c47351] at e_755_19
  have hr := e_755_19
  simp only [e_755_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5125)) := by
  have c47352 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47353 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47354 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47355 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47356 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47357 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_0 := arithEq_of_rows h (row := 756) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_0
  simp only [← c47352, ← c47353, ← c47354] at e_756_0
  have e_756_1 := arithEq_of_rows h (row := 756) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_1
  simp only [← c47355, ← c47356, k2, ← c47357] at e_756_1
  have hr := e_756_1
  simp only [e_756_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5126)) := by
  have c47358 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47359 := (publicBatchWrapper89_copies1479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47360 := (publicBatchWrapper89_copies1480 a h).1
  have c47361 := (publicBatchWrapper89_copies1480 a h).2.1
  have c47362 := (publicBatchWrapper89_copies1480 a h).2.2.1
  have c47363 := (publicBatchWrapper89_copies1480 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_2 := arithEq_of_rows h (row := 756) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_2
  simp only [← c47358, ← c47359, ← c47360] at e_756_2
  have e_756_3 := arithEq_of_rows h (row := 756) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_3
  simp only [← c47361, ← c47362, k2, ← c47363] at e_756_3
  have hr := e_756_3
  simp only [e_756_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5127)) := by
  have c47364 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.1
  have c47365 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.1
  have c47366 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.1
  have c47367 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.1
  have c47368 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.1
  have c47369 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_4 := arithEq_of_rows h (row := 756) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_4
  simp only [← c47364, ← c47365, ← c47366] at e_756_4
  have e_756_5 := arithEq_of_rows h (row := 756) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_5
  simp only [← c47367, ← c47368, k2, ← c47369] at e_756_5
  have hr := e_756_5
  simp only [e_756_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5128)) := by
  have c47370 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47371 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47372 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47373 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47374 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47375 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_6 := arithEq_of_rows h (row := 756) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_6
  simp only [← c47370, ← c47371, ← c47372] at e_756_6
  have e_756_7 := arithEq_of_rows h (row := 756) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_7
  simp only [← c47373, ← c47374, k2, ← c47375] at e_756_7
  have hr := e_756_7
  simp only [e_756_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5129)) := by
  have c47376 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47377 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47378 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47379 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47380 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47381 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_8 := arithEq_of_rows h (row := 756) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_8
  simp only [← c47376, ← c47377, ← c47378] at e_756_8
  have e_756_9 := arithEq_of_rows h (row := 756) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_9
  simp only [← c47379, ← c47380, k2, ← c47381] at e_756_9
  have hr := e_756_9
  simp only [e_756_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5130)) := by
  have c47382 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47383 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47384 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47385 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47386 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47387 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_10 := arithEq_of_rows h (row := 756) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_10
  simp only [← c47382, ← c47383, ← c47384] at e_756_10
  have e_756_11 := arithEq_of_rows h (row := 756) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_11
  simp only [← c47385, ← c47386, k2, ← c47387] at e_756_11
  have hr := e_756_11
  simp only [e_756_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5131)) := by
  have c47388 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47389 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47390 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47391 := (publicBatchWrapper89_copies1480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47392 := (publicBatchWrapper89_copies1481 a h).1
  have c47393 := (publicBatchWrapper89_copies1481 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_12 := arithEq_of_rows h (row := 756) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_12
  simp only [← c47388, ← c47389, ← c47390] at e_756_12
  have e_756_13 := arithEq_of_rows h (row := 756) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_13
  simp only [← c47391, ← c47392, k2, ← c47393] at e_756_13
  have hr := e_756_13
  simp only [e_756_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5132)) := by
  have c47394 := (publicBatchWrapper89_copies1481 a h).2.2.1
  have c47395 := (publicBatchWrapper89_copies1481 a h).2.2.2.1
  have c47396 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.1
  have c47397 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.1
  have c47398 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.1
  have c47399 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_14 := arithEq_of_rows h (row := 756) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_14
  simp only [← c47394, ← c47395, ← c47396] at e_756_14
  have e_756_15 := arithEq_of_rows h (row := 756) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_15
  simp only [← c47397, ← c47398, k2, ← c47399] at e_756_15
  have hr := e_756_15
  simp only [e_756_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5133)) := by
  have c47400 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.1
  have c47401 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.1
  have c47402 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47403 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47404 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47405 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_16 := arithEq_of_rows h (row := 756) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_16
  simp only [← c47400, ← c47401, ← c47402] at e_756_16
  have e_756_17 := arithEq_of_rows h (row := 756) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_17
  simp only [← c47403, ← c47404, k2, ← c47405] at e_756_17
  have hr := e_756_17
  simp only [e_756_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 756 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5134)) := by
  have c47406 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47407 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47408 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47409 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47410 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47411 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_756_18 := arithEq_of_rows h (row := 756) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_18
  simp only [← c47406, ← c47407, ← c47408] at e_756_18
  have e_756_19 := arithEq_of_rows h (row := 756) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_756_19
  simp only [← c47409, ← c47410, k2, ← c47411] at e_756_19
  have hr := e_756_19
  simp only [e_756_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5135)) := by
  have c47412 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47413 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47414 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47415 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47416 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47417 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_0 := arithEq_of_rows h (row := 757) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_0
  simp only [← c47412, ← c47413, ← c47414] at e_757_0
  have e_757_1 := arithEq_of_rows h (row := 757) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_1
  simp only [← c47415, ← c47416, k2, ← c47417] at e_757_1
  have hr := e_757_1
  simp only [e_757_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5136)) := by
  have c47418 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47419 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47420 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47421 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47422 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47423 := (publicBatchWrapper89_copies1481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_2 := arithEq_of_rows h (row := 757) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_2
  simp only [← c47418, ← c47419, ← c47420] at e_757_2
  have e_757_3 := arithEq_of_rows h (row := 757) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_3
  simp only [← c47421, ← c47422, k2, ← c47423] at e_757_3
  have hr := e_757_3
  simp only [e_757_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5137)) := by
  have c47424 := (publicBatchWrapper89_copies1482 a h).1
  have c47425 := (publicBatchWrapper89_copies1482 a h).2.1
  have c47426 := (publicBatchWrapper89_copies1482 a h).2.2.1
  have c47427 := (publicBatchWrapper89_copies1482 a h).2.2.2.1
  have c47428 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.1
  have c47429 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_4 := arithEq_of_rows h (row := 757) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_4
  simp only [← c47424, ← c47425, ← c47426] at e_757_4
  have e_757_5 := arithEq_of_rows h (row := 757) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_5
  simp only [← c47427, ← c47428, k2, ← c47429] at e_757_5
  have hr := e_757_5
  simp only [e_757_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5138)) := by
  have c47430 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.1
  have c47431 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.1
  have c47432 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.1
  have c47433 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.1
  have c47434 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47435 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_6 := arithEq_of_rows h (row := 757) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_6
  simp only [← c47430, ← c47431, ← c47432] at e_757_6
  have e_757_7 := arithEq_of_rows h (row := 757) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_7
  simp only [← c47433, ← c47434, k2, ← c47435] at e_757_7
  have hr := e_757_7
  simp only [e_757_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5139)) := by
  have c47436 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47437 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47438 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47439 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47440 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47441 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_8 := arithEq_of_rows h (row := 757) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_8
  simp only [← c47436, ← c47437, ← c47438] at e_757_8
  have e_757_9 := arithEq_of_rows h (row := 757) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_9
  simp only [← c47439, ← c47440, k2, ← c47441] at e_757_9
  have hr := e_757_9
  simp only [e_757_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5140)) := by
  have c47442 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47443 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47444 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47445 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47446 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47447 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_10 := arithEq_of_rows h (row := 757) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_10
  simp only [← c47442, ← c47443, ← c47444] at e_757_10
  have e_757_11 := arithEq_of_rows h (row := 757) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_11
  simp only [← c47445, ← c47446, k2, ← c47447] at e_757_11
  have hr := e_757_11
  simp only [e_757_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5141)) := by
  have c47448 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47449 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47450 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47451 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47452 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47453 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_12 := arithEq_of_rows h (row := 757) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_12
  simp only [← c47448, ← c47449, ← c47450] at e_757_12
  have e_757_13 := arithEq_of_rows h (row := 757) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_13
  simp only [← c47451, ← c47452, k2, ← c47453] at e_757_13
  have hr := e_757_13
  simp only [e_757_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5142)) := by
  have c47454 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47455 := (publicBatchWrapper89_copies1482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47456 := (publicBatchWrapper89_copies1483 a h).1
  have c47457 := (publicBatchWrapper89_copies1483 a h).2.1
  have c47458 := (publicBatchWrapper89_copies1483 a h).2.2.1
  have c47459 := (publicBatchWrapper89_copies1483 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_14 := arithEq_of_rows h (row := 757) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_14
  simp only [← c47454, ← c47455, ← c47456] at e_757_14
  have e_757_15 := arithEq_of_rows h (row := 757) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_15
  simp only [← c47457, ← c47458, k2, ← c47459] at e_757_15
  have hr := e_757_15
  simp only [e_757_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5143)) := by
  have c47460 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.1
  have c47461 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.1
  have c47462 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.1
  have c47463 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.1
  have c47464 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.1
  have c47465 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_16 := arithEq_of_rows h (row := 757) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_16
  simp only [← c47460, ← c47461, ← c47462] at e_757_16
  have e_757_17 := arithEq_of_rows h (row := 757) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_17
  simp only [← c47463, ← c47464, k2, ← c47465] at e_757_17
  have hr := e_757_17
  simp only [e_757_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 757 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5144)) := by
  have c47466 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47467 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47468 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47469 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47470 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47471 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_757_18 := arithEq_of_rows h (row := 757) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_18
  simp only [← c47466, ← c47467, ← c47468] at e_757_18
  have e_757_19 := arithEq_of_rows h (row := 757) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_757_19
  simp only [← c47469, ← c47470, k2, ← c47471] at e_757_19
  have hr := e_757_19
  simp only [e_757_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5145)) := by
  have c47472 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47473 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47474 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47475 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47476 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47477 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_0 := arithEq_of_rows h (row := 758) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_0
  simp only [← c47472, ← c47473, ← c47474] at e_758_0
  have e_758_1 := arithEq_of_rows h (row := 758) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_1
  simp only [← c47475, ← c47476, k2, ← c47477] at e_758_1
  have hr := e_758_1
  simp only [e_758_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5146)) := by
  have c47478 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47479 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47480 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47481 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47482 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47483 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_2 := arithEq_of_rows h (row := 758) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_2
  simp only [← c47478, ← c47479, ← c47480] at e_758_2
  have e_758_3 := arithEq_of_rows h (row := 758) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_3
  simp only [← c47481, ← c47482, k2, ← c47483] at e_758_3
  have hr := e_758_3
  simp only [e_758_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5147)) := by
  have c47484 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47485 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47486 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47487 := (publicBatchWrapper89_copies1483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47488 := (publicBatchWrapper89_copies1484 a h).1
  have c47489 := (publicBatchWrapper89_copies1484 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_4 := arithEq_of_rows h (row := 758) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_4
  simp only [← c47484, ← c47485, ← c47486] at e_758_4
  have e_758_5 := arithEq_of_rows h (row := 758) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_5
  simp only [← c47487, ← c47488, k2, ← c47489] at e_758_5
  have hr := e_758_5
  simp only [e_758_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5148)) := by
  have c47490 := (publicBatchWrapper89_copies1484 a h).2.2.1
  have c47491 := (publicBatchWrapper89_copies1484 a h).2.2.2.1
  have c47492 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.1
  have c47493 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.1
  have c47494 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.1
  have c47495 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_6 := arithEq_of_rows h (row := 758) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_6
  simp only [← c47490, ← c47491, ← c47492] at e_758_6
  have e_758_7 := arithEq_of_rows h (row := 758) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_7
  simp only [← c47493, ← c47494, k2, ← c47495] at e_758_7
  have hr := e_758_7
  simp only [e_758_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5149)) := by
  have c47496 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.1
  have c47497 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.1
  have c47498 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47499 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47500 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47501 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_8 := arithEq_of_rows h (row := 758) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_8
  simp only [← c47496, ← c47497, ← c47498] at e_758_8
  have e_758_9 := arithEq_of_rows h (row := 758) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_9
  simp only [← c47499, ← c47500, k2, ← c47501] at e_758_9
  have hr := e_758_9
  simp only [e_758_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5150)) := by
  have c47502 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47503 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47504 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47505 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47506 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47507 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_10 := arithEq_of_rows h (row := 758) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_10
  simp only [← c47502, ← c47503, ← c47504] at e_758_10
  have e_758_11 := arithEq_of_rows h (row := 758) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_11
  simp only [← c47505, ← c47506, k2, ← c47507] at e_758_11
  have hr := e_758_11
  simp only [e_758_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5151)) := by
  have c47508 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47509 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47510 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47511 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47512 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47513 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_12 := arithEq_of_rows h (row := 758) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_12
  simp only [← c47508, ← c47509, ← c47510] at e_758_12
  have e_758_13 := arithEq_of_rows h (row := 758) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_13
  simp only [← c47511, ← c47512, k2, ← c47513] at e_758_13
  have hr := e_758_13
  simp only [e_758_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5152)) := by
  have c47514 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47515 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47516 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47517 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47518 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47519 := (publicBatchWrapper89_copies1484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_14 := arithEq_of_rows h (row := 758) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_14
  simp only [← c47514, ← c47515, ← c47516] at e_758_14
  have e_758_15 := arithEq_of_rows h (row := 758) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_15
  simp only [← c47517, ← c47518, k2, ← c47519] at e_758_15
  have hr := e_758_15
  simp only [e_758_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5153)) := by
  have c47520 := (publicBatchWrapper89_copies1485 a h).1
  have c47521 := (publicBatchWrapper89_copies1485 a h).2.1
  have c47522 := (publicBatchWrapper89_copies1485 a h).2.2.1
  have c47523 := (publicBatchWrapper89_copies1485 a h).2.2.2.1
  have c47524 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.1
  have c47525 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_16 := arithEq_of_rows h (row := 758) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_16
  simp only [← c47520, ← c47521, ← c47522] at e_758_16
  have e_758_17 := arithEq_of_rows h (row := 758) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_17
  simp only [← c47523, ← c47524, k2, ← c47525] at e_758_17
  have hr := e_758_17
  simp only [e_758_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 758 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5154)) := by
  have c47526 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.1
  have c47527 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.1
  have c47528 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.1
  have c47529 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.1
  have c47530 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47531 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_758_18 := arithEq_of_rows h (row := 758) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_18
  simp only [← c47526, ← c47527, ← c47528] at e_758_18
  have e_758_19 := arithEq_of_rows h (row := 758) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_758_19
  simp only [← c47529, ← c47530, k2, ← c47531] at e_758_19
  have hr := e_758_19
  simp only [e_758_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5155)) := by
  have c47532 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47533 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47534 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47535 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47536 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47537 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_0 := arithEq_of_rows h (row := 759) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_0
  simp only [← c47532, ← c47533, ← c47534] at e_759_0
  have e_759_1 := arithEq_of_rows h (row := 759) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_1
  simp only [← c47535, ← c47536, k2, ← c47537] at e_759_1
  have hr := e_759_1
  simp only [e_759_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5156)) := by
  have c47538 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47539 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47540 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47541 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47542 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47543 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_2 := arithEq_of_rows h (row := 759) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_2
  simp only [← c47538, ← c47539, ← c47540] at e_759_2
  have e_759_3 := arithEq_of_rows h (row := 759) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_3
  simp only [← c47541, ← c47542, k2, ← c47543] at e_759_3
  have hr := e_759_3
  simp only [e_759_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5157)) := by
  have c47544 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47545 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47546 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47547 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47548 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47549 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_4 := arithEq_of_rows h (row := 759) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_4
  simp only [← c47544, ← c47545, ← c47546] at e_759_4
  have e_759_5 := arithEq_of_rows h (row := 759) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_5
  simp only [← c47547, ← c47548, k2, ← c47549] at e_759_5
  have hr := e_759_5
  simp only [e_759_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5158)) := by
  have c47550 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47551 := (publicBatchWrapper89_copies1485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47552 := (publicBatchWrapper89_copies1486 a h).1
  have c47553 := (publicBatchWrapper89_copies1486 a h).2.1
  have c47554 := (publicBatchWrapper89_copies1486 a h).2.2.1
  have c47555 := (publicBatchWrapper89_copies1486 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_6 := arithEq_of_rows h (row := 759) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_6
  simp only [← c47550, ← c47551, ← c47552] at e_759_6
  have e_759_7 := arithEq_of_rows h (row := 759) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_7
  simp only [← c47553, ← c47554, k2, ← c47555] at e_759_7
  have hr := e_759_7
  simp only [e_759_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5159)) := by
  have c47556 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.1
  have c47557 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.1
  have c47558 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.1
  have c47559 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.1
  have c47560 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.1
  have c47561 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_8 := arithEq_of_rows h (row := 759) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_8
  simp only [← c47556, ← c47557, ← c47558] at e_759_8
  have e_759_9 := arithEq_of_rows h (row := 759) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_9
  simp only [← c47559, ← c47560, k2, ← c47561] at e_759_9
  have hr := e_759_9
  simp only [e_759_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5160)) := by
  have c47562 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47563 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47564 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47565 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47566 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47567 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_10 := arithEq_of_rows h (row := 759) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_10
  simp only [← c47562, ← c47563, ← c47564] at e_759_10
  have e_759_11 := arithEq_of_rows h (row := 759) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_11
  simp only [← c47565, ← c47566, k2, ← c47567] at e_759_11
  have hr := e_759_11
  simp only [e_759_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5161)) := by
  have c47568 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47569 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47570 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47571 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47572 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47573 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_12 := arithEq_of_rows h (row := 759) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_12
  simp only [← c47568, ← c47569, ← c47570] at e_759_12
  have e_759_13 := arithEq_of_rows h (row := 759) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_13
  simp only [← c47571, ← c47572, k2, ← c47573] at e_759_13
  have hr := e_759_13
  simp only [e_759_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5162)) := by
  have c47574 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47575 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47576 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47577 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47578 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47579 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_14 := arithEq_of_rows h (row := 759) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_14
  simp only [← c47574, ← c47575, ← c47576] at e_759_14
  have e_759_15 := arithEq_of_rows h (row := 759) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_15
  simp only [← c47577, ← c47578, k2, ← c47579] at e_759_15
  have hr := e_759_15
  simp only [e_759_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5163)) := by
  have c47580 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47581 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47582 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47583 := (publicBatchWrapper89_copies1486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47584 := (publicBatchWrapper89_copies1487 a h).1
  have c47585 := (publicBatchWrapper89_copies1487 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_16 := arithEq_of_rows h (row := 759) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_16
  simp only [← c47580, ← c47581, ← c47582] at e_759_16
  have e_759_17 := arithEq_of_rows h (row := 759) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_17
  simp only [← c47583, ← c47584, k2, ← c47585] at e_759_17
  have hr := e_759_17
  simp only [e_759_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 759 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5164)) := by
  have c47586 := (publicBatchWrapper89_copies1487 a h).2.2.1
  have c47587 := (publicBatchWrapper89_copies1487 a h).2.2.2.1
  have c47588 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.1
  have c47589 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.1
  have c47590 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.1
  have c47591 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_759_18 := arithEq_of_rows h (row := 759) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_18
  simp only [← c47586, ← c47587, ← c47588] at e_759_18
  have e_759_19 := arithEq_of_rows h (row := 759) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_759_19
  simp only [← c47589, ← c47590, k2, ← c47591] at e_759_19
  have hr := e_759_19
  simp only [e_759_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5165)) := by
  have c47592 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.1
  have c47593 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.1
  have c47594 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47595 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47596 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47597 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_0 := arithEq_of_rows h (row := 760) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_0
  simp only [← c47592, ← c47593, ← c47594] at e_760_0
  have e_760_1 := arithEq_of_rows h (row := 760) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_1
  simp only [← c47595, ← c47596, k2, ← c47597] at e_760_1
  have hr := e_760_1
  simp only [e_760_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5166)) := by
  have c47598 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47599 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47600 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47601 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47602 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47603 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_2 := arithEq_of_rows h (row := 760) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_2
  simp only [← c47598, ← c47599, ← c47600] at e_760_2
  have e_760_3 := arithEq_of_rows h (row := 760) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_3
  simp only [← c47601, ← c47602, k2, ← c47603] at e_760_3
  have hr := e_760_3
  simp only [e_760_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5167)) := by
  have c47604 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47605 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47606 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47607 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47608 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47609 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_4 := arithEq_of_rows h (row := 760) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_4
  simp only [← c47604, ← c47605, ← c47606] at e_760_4
  have e_760_5 := arithEq_of_rows h (row := 760) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_5
  simp only [← c47607, ← c47608, k2, ← c47609] at e_760_5
  have hr := e_760_5
  simp only [e_760_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5168)) := by
  have c47610 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47611 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47612 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47613 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47614 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47615 := (publicBatchWrapper89_copies1487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_6 := arithEq_of_rows h (row := 760) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_6
  simp only [← c47610, ← c47611, ← c47612] at e_760_6
  have e_760_7 := arithEq_of_rows h (row := 760) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_7
  simp only [← c47613, ← c47614, k2, ← c47615] at e_760_7
  have hr := e_760_7
  simp only [e_760_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5169)) := by
  have c47616 := (publicBatchWrapper89_copies1488 a h).1
  have c47617 := (publicBatchWrapper89_copies1488 a h).2.1
  have c47618 := (publicBatchWrapper89_copies1488 a h).2.2.1
  have c47619 := (publicBatchWrapper89_copies1488 a h).2.2.2.1
  have c47620 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.1
  have c47621 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_8 := arithEq_of_rows h (row := 760) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_8
  simp only [← c47616, ← c47617, ← c47618] at e_760_8
  have e_760_9 := arithEq_of_rows h (row := 760) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_9
  simp only [← c47619, ← c47620, k2, ← c47621] at e_760_9
  have hr := e_760_9
  simp only [e_760_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5210)) := by
  have c47622 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.1
  have c47623 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.1
  have c47624 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.1
  have c47625 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.1
  have c47626 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47627 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_10 := arithEq_of_rows h (row := 760) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_10
  simp only [← c47622, ← c47623, ← c47624] at e_760_10
  have e_760_11 := arithEq_of_rows h (row := 760) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_11
  simp only [← c47625, ← c47626, k2, ← c47627] at e_760_11
  have hr := e_760_11
  simp only [e_760_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5211)) := by
  have c47628 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47629 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47630 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47631 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47632 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47633 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_12 := arithEq_of_rows h (row := 760) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_12
  simp only [← c47628, ← c47629, ← c47630] at e_760_12
  have e_760_13 := arithEq_of_rows h (row := 760) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_13
  simp only [← c47631, ← c47632, k2, ← c47633] at e_760_13
  have hr := e_760_13
  simp only [e_760_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5212)) := by
  have c47634 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47635 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47636 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47637 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47638 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47639 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_14 := arithEq_of_rows h (row := 760) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_14
  simp only [← c47634, ← c47635, ← c47636] at e_760_14
  have e_760_15 := arithEq_of_rows h (row := 760) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_15
  simp only [← c47637, ← c47638, k2, ← c47639] at e_760_15
  have hr := e_760_15
  simp only [e_760_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5213)) := by
  have c47640 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47641 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47642 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47643 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47644 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47645 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_16 := arithEq_of_rows h (row := 760) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_16
  simp only [← c47640, ← c47641, ← c47642] at e_760_16
  have e_760_17 := arithEq_of_rows h (row := 760) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_17
  simp only [← c47643, ← c47644, k2, ← c47645] at e_760_17
  have hr := e_760_17
  simp only [e_760_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 760 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5214)) := by
  have c47646 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47647 := (publicBatchWrapper89_copies1488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47648 := (publicBatchWrapper89_copies1489 a h).1
  have c47649 := (publicBatchWrapper89_copies1489 a h).2.1
  have c47650 := (publicBatchWrapper89_copies1489 a h).2.2.1
  have c47651 := (publicBatchWrapper89_copies1489 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_760_18 := arithEq_of_rows h (row := 760) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_18
  simp only [← c47646, ← c47647, ← c47648] at e_760_18
  have e_760_19 := arithEq_of_rows h (row := 760) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_760_19
  simp only [← c47649, ← c47650, k2, ← c47651] at e_760_19
  have hr := e_760_19
  simp only [e_760_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5215)) := by
  have c47652 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.1
  have c47653 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.1
  have c47654 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.1
  have c47655 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.1
  have c47656 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.1
  have c47657 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_0 := arithEq_of_rows h (row := 761) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_0
  simp only [← c47652, ← c47653, ← c47654] at e_761_0
  have e_761_1 := arithEq_of_rows h (row := 761) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_1
  simp only [← c47655, ← c47656, k2, ← c47657] at e_761_1
  have hr := e_761_1
  simp only [e_761_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5216)) := by
  have c47658 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47659 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47660 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47661 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47662 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47663 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_2 := arithEq_of_rows h (row := 761) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_2
  simp only [← c47658, ← c47659, ← c47660] at e_761_2
  have e_761_3 := arithEq_of_rows h (row := 761) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_3
  simp only [← c47661, ← c47662, k2, ← c47663] at e_761_3
  have hr := e_761_3
  simp only [e_761_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5217)) := by
  have c47664 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47665 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47666 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47667 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47668 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47669 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_4 := arithEq_of_rows h (row := 761) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_4
  simp only [← c47664, ← c47665, ← c47666] at e_761_4
  have e_761_5 := arithEq_of_rows h (row := 761) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_5
  simp only [← c47667, ← c47668, k2, ← c47669] at e_761_5
  have hr := e_761_5
  simp only [e_761_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5218)) := by
  have c47670 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47671 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47672 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47673 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47674 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47675 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_6 := arithEq_of_rows h (row := 761) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_6
  simp only [← c47670, ← c47671, ← c47672] at e_761_6
  have e_761_7 := arithEq_of_rows h (row := 761) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_7
  simp only [← c47673, ← c47674, k2, ← c47675] at e_761_7
  have hr := e_761_7
  simp only [e_761_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5219)) := by
  have c47676 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47677 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47678 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47679 := (publicBatchWrapper89_copies1489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47680 := (publicBatchWrapper89_copies1490 a h).1
  have c47681 := (publicBatchWrapper89_copies1490 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_8 := arithEq_of_rows h (row := 761) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_8
  simp only [← c47676, ← c47677, ← c47678] at e_761_8
  have e_761_9 := arithEq_of_rows h (row := 761) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_9
  simp only [← c47679, ← c47680, k2, ← c47681] at e_761_9
  have hr := e_761_9
  simp only [e_761_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5220)) := by
  have c47682 := (publicBatchWrapper89_copies1490 a h).2.2.1
  have c47683 := (publicBatchWrapper89_copies1490 a h).2.2.2.1
  have c47684 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.1
  have c47685 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.1
  have c47686 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.1
  have c47687 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_10 := arithEq_of_rows h (row := 761) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_10
  simp only [← c47682, ← c47683, ← c47684] at e_761_10
  have e_761_11 := arithEq_of_rows h (row := 761) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_11
  simp only [← c47685, ← c47686, k2, ← c47687] at e_761_11
  have hr := e_761_11
  simp only [e_761_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5221)) := by
  have c47688 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.1
  have c47689 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.1
  have c47690 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47691 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47692 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47693 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_12 := arithEq_of_rows h (row := 761) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_12
  simp only [← c47688, ← c47689, ← c47690] at e_761_12
  have e_761_13 := arithEq_of_rows h (row := 761) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_13
  simp only [← c47691, ← c47692, k2, ← c47693] at e_761_13
  have hr := e_761_13
  simp only [e_761_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5222)) := by
  have c47694 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47695 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47696 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47697 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47698 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47699 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_14 := arithEq_of_rows h (row := 761) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_14
  simp only [← c47694, ← c47695, ← c47696] at e_761_14
  have e_761_15 := arithEq_of_rows h (row := 761) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_15
  simp only [← c47697, ← c47698, k2, ← c47699] at e_761_15
  have hr := e_761_15
  simp only [e_761_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5223)) := by
  have c47700 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47701 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47702 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47703 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47704 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47705 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_16 := arithEq_of_rows h (row := 761) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_16
  simp only [← c47700, ← c47701, ← c47702] at e_761_16
  have e_761_17 := arithEq_of_rows h (row := 761) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_17
  simp only [← c47703, ← c47704, k2, ← c47705] at e_761_17
  have hr := e_761_17
  simp only [e_761_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 761 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5224)) := by
  have c47706 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47707 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47708 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47709 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47710 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47711 := (publicBatchWrapper89_copies1490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_761_18 := arithEq_of_rows h (row := 761) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_18
  simp only [← c47706, ← c47707, ← c47708] at e_761_18
  have e_761_19 := arithEq_of_rows h (row := 761) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_761_19
  simp only [← c47709, ← c47710, k2, ← c47711] at e_761_19
  have hr := e_761_19
  simp only [e_761_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5225)) := by
  have c47712 := (publicBatchWrapper89_copies1491 a h).1
  have c47713 := (publicBatchWrapper89_copies1491 a h).2.1
  have c47714 := (publicBatchWrapper89_copies1491 a h).2.2.1
  have c47715 := (publicBatchWrapper89_copies1491 a h).2.2.2.1
  have c47716 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.1
  have c47717 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_0 := arithEq_of_rows h (row := 762) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_0
  simp only [← c47712, ← c47713, ← c47714] at e_762_0
  have e_762_1 := arithEq_of_rows h (row := 762) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_1
  simp only [← c47715, ← c47716, k2, ← c47717] at e_762_1
  have hr := e_762_1
  simp only [e_762_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5226)) := by
  have c47718 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.1
  have c47719 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.1
  have c47720 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.1
  have c47721 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.1
  have c47722 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47723 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_2 := arithEq_of_rows h (row := 762) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_2
  simp only [← c47718, ← c47719, ← c47720] at e_762_2
  have e_762_3 := arithEq_of_rows h (row := 762) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_3
  simp only [← c47721, ← c47722, k2, ← c47723] at e_762_3
  have hr := e_762_3
  simp only [e_762_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5227)) := by
  have c47724 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47725 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47726 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47727 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47728 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47729 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_4 := arithEq_of_rows h (row := 762) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_4
  simp only [← c47724, ← c47725, ← c47726] at e_762_4
  have e_762_5 := arithEq_of_rows h (row := 762) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_5
  simp only [← c47727, ← c47728, k2, ← c47729] at e_762_5
  have hr := e_762_5
  simp only [e_762_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5228)) := by
  have c47730 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47731 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47732 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47733 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47734 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47735 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_6 := arithEq_of_rows h (row := 762) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_6
  simp only [← c47730, ← c47731, ← c47732] at e_762_6
  have e_762_7 := arithEq_of_rows h (row := 762) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_7
  simp only [← c47733, ← c47734, k2, ← c47735] at e_762_7
  have hr := e_762_7
  simp only [e_762_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5229)) := by
  have c47736 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47737 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47738 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47739 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47740 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47741 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_8 := arithEq_of_rows h (row := 762) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_8
  simp only [← c47736, ← c47737, ← c47738] at e_762_8
  have e_762_9 := arithEq_of_rows h (row := 762) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_9
  simp only [← c47739, ← c47740, k2, ← c47741] at e_762_9
  have hr := e_762_9
  simp only [e_762_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5230)) := by
  have c47742 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47743 := (publicBatchWrapper89_copies1491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47744 := (publicBatchWrapper89_copies1492 a h).1
  have c47745 := (publicBatchWrapper89_copies1492 a h).2.1
  have c47746 := (publicBatchWrapper89_copies1492 a h).2.2.1
  have c47747 := (publicBatchWrapper89_copies1492 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_10 := arithEq_of_rows h (row := 762) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_10
  simp only [← c47742, ← c47743, ← c47744] at e_762_10
  have e_762_11 := arithEq_of_rows h (row := 762) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_11
  simp only [← c47745, ← c47746, k2, ← c47747] at e_762_11
  have hr := e_762_11
  simp only [e_762_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5231)) := by
  have c47748 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.1
  have c47749 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.1
  have c47750 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.1
  have c47751 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.1
  have c47752 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.1
  have c47753 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_12 := arithEq_of_rows h (row := 762) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_12
  simp only [← c47748, ← c47749, ← c47750] at e_762_12
  have e_762_13 := arithEq_of_rows h (row := 762) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_13
  simp only [← c47751, ← c47752, k2, ← c47753] at e_762_13
  have hr := e_762_13
  simp only [e_762_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5232)) := by
  have c47754 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47755 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47756 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47757 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47758 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47759 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_14 := arithEq_of_rows h (row := 762) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_14
  simp only [← c47754, ← c47755, ← c47756] at e_762_14
  have e_762_15 := arithEq_of_rows h (row := 762) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_15
  simp only [← c47757, ← c47758, k2, ← c47759] at e_762_15
  have hr := e_762_15
  simp only [e_762_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5233)) := by
  have c47760 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47761 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47762 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47763 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47764 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47765 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_16 := arithEq_of_rows h (row := 762) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_16
  simp only [← c47760, ← c47761, ← c47762] at e_762_16
  have e_762_17 := arithEq_of_rows h (row := 762) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_17
  simp only [← c47763, ← c47764, k2, ← c47765] at e_762_17
  have hr := e_762_17
  simp only [e_762_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 762 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5234)) := by
  have c47766 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47767 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47768 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47769 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47770 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47771 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_762_18 := arithEq_of_rows h (row := 762) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_18
  simp only [← c47766, ← c47767, ← c47768] at e_762_18
  have e_762_19 := arithEq_of_rows h (row := 762) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_762_19
  simp only [← c47769, ← c47770, k2, ← c47771] at e_762_19
  have hr := e_762_19
  simp only [e_762_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5235)) := by
  have c47772 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47773 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47774 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47775 := (publicBatchWrapper89_copies1492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47776 := (publicBatchWrapper89_copies1493 a h).1
  have c47777 := (publicBatchWrapper89_copies1493 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_0 := arithEq_of_rows h (row := 763) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_0
  simp only [← c47772, ← c47773, ← c47774] at e_763_0
  have e_763_1 := arithEq_of_rows h (row := 763) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_1
  simp only [← c47775, ← c47776, k2, ← c47777] at e_763_1
  have hr := e_763_1
  simp only [e_763_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5236)) := by
  have c47778 := (publicBatchWrapper89_copies1493 a h).2.2.1
  have c47779 := (publicBatchWrapper89_copies1493 a h).2.2.2.1
  have c47780 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.1
  have c47781 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.1
  have c47782 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.1
  have c47783 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_2 := arithEq_of_rows h (row := 763) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_2
  simp only [← c47778, ← c47779, ← c47780] at e_763_2
  have e_763_3 := arithEq_of_rows h (row := 763) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_3
  simp only [← c47781, ← c47782, k2, ← c47783] at e_763_3
  have hr := e_763_3
  simp only [e_763_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5237)) := by
  have c47784 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.1
  have c47785 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.1
  have c47786 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47787 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47788 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47789 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_4 := arithEq_of_rows h (row := 763) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_4
  simp only [← c47784, ← c47785, ← c47786] at e_763_4
  have e_763_5 := arithEq_of_rows h (row := 763) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_5
  simp only [← c47787, ← c47788, k2, ← c47789] at e_763_5
  have hr := e_763_5
  simp only [e_763_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5238)) := by
  have c47790 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47791 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47792 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47793 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47794 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47795 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_6 := arithEq_of_rows h (row := 763) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_6
  simp only [← c47790, ← c47791, ← c47792] at e_763_6
  have e_763_7 := arithEq_of_rows h (row := 763) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_7
  simp only [← c47793, ← c47794, k2, ← c47795] at e_763_7
  have hr := e_763_7
  simp only [e_763_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5239)) := by
  have c47796 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47797 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47798 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47799 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47800 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47801 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_8 := arithEq_of_rows h (row := 763) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_8
  simp only [← c47796, ← c47797, ← c47798] at e_763_8
  have e_763_9 := arithEq_of_rows h (row := 763) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_9
  simp only [← c47799, ← c47800, k2, ← c47801] at e_763_9
  have hr := e_763_9
  simp only [e_763_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5240)) := by
  have c47802 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47803 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47804 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47805 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47806 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47807 := (publicBatchWrapper89_copies1493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_10 := arithEq_of_rows h (row := 763) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_10
  simp only [← c47802, ← c47803, ← c47804] at e_763_10
  have e_763_11 := arithEq_of_rows h (row := 763) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_11
  simp only [← c47805, ← c47806, k2, ← c47807] at e_763_11
  have hr := e_763_11
  simp only [e_763_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5241)) := by
  have c47808 := (publicBatchWrapper89_copies1494 a h).1
  have c47809 := (publicBatchWrapper89_copies1494 a h).2.1
  have c47810 := (publicBatchWrapper89_copies1494 a h).2.2.1
  have c47811 := (publicBatchWrapper89_copies1494 a h).2.2.2.1
  have c47812 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.1
  have c47813 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_12 := arithEq_of_rows h (row := 763) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_12
  simp only [← c47808, ← c47809, ← c47810] at e_763_12
  have e_763_13 := arithEq_of_rows h (row := 763) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_13
  simp only [← c47811, ← c47812, k2, ← c47813] at e_763_13
  have hr := e_763_13
  simp only [e_763_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5242)) := by
  have c47814 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.1
  have c47815 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.1
  have c47816 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.1
  have c47817 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.1
  have c47818 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47819 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_14 := arithEq_of_rows h (row := 763) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_14
  simp only [← c47814, ← c47815, ← c47816] at e_763_14
  have e_763_15 := arithEq_of_rows h (row := 763) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_15
  simp only [← c47817, ← c47818, k2, ← c47819] at e_763_15
  have hr := e_763_15
  simp only [e_763_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5243)) := by
  have c47820 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47821 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47822 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47823 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47824 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47825 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_16 := arithEq_of_rows h (row := 763) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_16
  simp only [← c47820, ← c47821, ← c47822] at e_763_16
  have e_763_17 := arithEq_of_rows h (row := 763) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_17
  simp only [← c47823, ← c47824, k2, ← c47825] at e_763_17
  have hr := e_763_17
  simp only [e_763_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 763 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5244)) := by
  have c47826 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47827 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47828 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47829 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47830 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47831 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_763_18 := arithEq_of_rows h (row := 763) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_18
  simp only [← c47826, ← c47827, ← c47828] at e_763_18
  have e_763_19 := arithEq_of_rows h (row := 763) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_763_19
  simp only [← c47829, ← c47830, k2, ← c47831] at e_763_19
  have hr := e_763_19
  simp only [e_763_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5245)) := by
  have c47832 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47833 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47834 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47835 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47836 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47837 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_0 := arithEq_of_rows h (row := 764) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_0
  simp only [← c47832, ← c47833, ← c47834] at e_764_0
  have e_764_1 := arithEq_of_rows h (row := 764) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_1
  simp only [← c47835, ← c47836, k2, ← c47837] at e_764_1
  have hr := e_764_1
  simp only [e_764_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5246)) := by
  have c47838 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47839 := (publicBatchWrapper89_copies1494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47840 := (publicBatchWrapper89_copies1495 a h).1
  have c47841 := (publicBatchWrapper89_copies1495 a h).2.1
  have c47842 := (publicBatchWrapper89_copies1495 a h).2.2.1
  have c47843 := (publicBatchWrapper89_copies1495 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_2 := arithEq_of_rows h (row := 764) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_2
  simp only [← c47838, ← c47839, ← c47840] at e_764_2
  have e_764_3 := arithEq_of_rows h (row := 764) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_3
  simp only [← c47841, ← c47842, k2, ← c47843] at e_764_3
  have hr := e_764_3
  simp only [e_764_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5247)) := by
  have c47844 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.1
  have c47845 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.1
  have c47846 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.1
  have c47847 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.1
  have c47848 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.1
  have c47849 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_4 := arithEq_of_rows h (row := 764) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_4
  simp only [← c47844, ← c47845, ← c47846] at e_764_4
  have e_764_5 := arithEq_of_rows h (row := 764) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_5
  simp only [← c47847, ← c47848, k2, ← c47849] at e_764_5
  have hr := e_764_5
  simp only [e_764_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5248)) := by
  have c47850 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47851 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47852 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47853 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47854 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47855 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_6 := arithEq_of_rows h (row := 764) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_6
  simp only [← c47850, ← c47851, ← c47852] at e_764_6
  have e_764_7 := arithEq_of_rows h (row := 764) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_7
  simp only [← c47853, ← c47854, k2, ← c47855] at e_764_7
  have hr := e_764_7
  simp only [e_764_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5249)) := by
  have c47856 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47857 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47858 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47859 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47860 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47861 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_8 := arithEq_of_rows h (row := 764) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_8
  simp only [← c47856, ← c47857, ← c47858] at e_764_8
  have e_764_9 := arithEq_of_rows h (row := 764) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_9
  simp only [← c47859, ← c47860, k2, ← c47861] at e_764_9
  have hr := e_764_9
  simp only [e_764_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5250)) := by
  have c47862 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47863 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47864 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47865 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47866 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47867 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_10 := arithEq_of_rows h (row := 764) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_10
  simp only [← c47862, ← c47863, ← c47864] at e_764_10
  have e_764_11 := arithEq_of_rows h (row := 764) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_11
  simp only [← c47865, ← c47866, k2, ← c47867] at e_764_11
  have hr := e_764_11
  simp only [e_764_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5251)) := by
  have c47868 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47869 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47870 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47871 := (publicBatchWrapper89_copies1495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47872 := (publicBatchWrapper89_copies1496 a h).1
  have c47873 := (publicBatchWrapper89_copies1496 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_12 := arithEq_of_rows h (row := 764) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_12
  simp only [← c47868, ← c47869, ← c47870] at e_764_12
  have e_764_13 := arithEq_of_rows h (row := 764) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_13
  simp only [← c47871, ← c47872, k2, ← c47873] at e_764_13
  have hr := e_764_13
  simp only [e_764_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5252)) := by
  have c47874 := (publicBatchWrapper89_copies1496 a h).2.2.1
  have c47875 := (publicBatchWrapper89_copies1496 a h).2.2.2.1
  have c47876 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.1
  have c47877 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.1
  have c47878 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.1
  have c47879 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_14 := arithEq_of_rows h (row := 764) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_14
  simp only [← c47874, ← c47875, ← c47876] at e_764_14
  have e_764_15 := arithEq_of_rows h (row := 764) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_15
  simp only [← c47877, ← c47878, k2, ← c47879] at e_764_15
  have hr := e_764_15
  simp only [e_764_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5253)) := by
  have c47880 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.1
  have c47881 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.1
  have c47882 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47883 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47884 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47885 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_16 := arithEq_of_rows h (row := 764) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_16
  simp only [← c47880, ← c47881, ← c47882] at e_764_16
  have e_764_17 := arithEq_of_rows h (row := 764) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_17
  simp only [← c47883, ← c47884, k2, ← c47885] at e_764_17
  have hr := e_764_17
  simp only [e_764_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 764 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5254)) := by
  have c47886 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47887 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47888 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47889 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47890 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47891 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_764_18 := arithEq_of_rows h (row := 764) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_18
  simp only [← c47886, ← c47887, ← c47888] at e_764_18
  have e_764_19 := arithEq_of_rows h (row := 764) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_764_19
  simp only [← c47889, ← c47890, k2, ← c47891] at e_764_19
  have hr := e_764_19
  simp only [e_764_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5255)) := by
  have c47892 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47893 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47894 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47895 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47896 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47897 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_0 := arithEq_of_rows h (row := 765) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_0
  simp only [← c47892, ← c47893, ← c47894] at e_765_0
  have e_765_1 := arithEq_of_rows h (row := 765) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_1
  simp only [← c47895, ← c47896, k2, ← c47897] at e_765_1
  have hr := e_765_1
  simp only [e_765_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5256)) := by
  have c47898 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47899 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47900 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47901 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47902 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47903 := (publicBatchWrapper89_copies1496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_2 := arithEq_of_rows h (row := 765) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_2
  simp only [← c47898, ← c47899, ← c47900] at e_765_2
  have e_765_3 := arithEq_of_rows h (row := 765) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_3
  simp only [← c47901, ← c47902, k2, ← c47903] at e_765_3
  have hr := e_765_3
  simp only [e_765_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5257)) := by
  have c47904 := (publicBatchWrapper89_copies1497 a h).1
  have c47905 := (publicBatchWrapper89_copies1497 a h).2.1
  have c47906 := (publicBatchWrapper89_copies1497 a h).2.2.1
  have c47907 := (publicBatchWrapper89_copies1497 a h).2.2.2.1
  have c47908 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.1
  have c47909 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_4 := arithEq_of_rows h (row := 765) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_4
  simp only [← c47904, ← c47905, ← c47906] at e_765_4
  have e_765_5 := arithEq_of_rows h (row := 765) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_5
  simp only [← c47907, ← c47908, k2, ← c47909] at e_765_5
  have hr := e_765_5
  simp only [e_765_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5258)) := by
  have c47910 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.1
  have c47911 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.1
  have c47912 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.1
  have c47913 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.1
  have c47914 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47915 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_6 := arithEq_of_rows h (row := 765) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_6
  simp only [← c47910, ← c47911, ← c47912] at e_765_6
  have e_765_7 := arithEq_of_rows h (row := 765) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_7
  simp only [← c47913, ← c47914, k2, ← c47915] at e_765_7
  have hr := e_765_7
  simp only [e_765_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5259)) := by
  have c47916 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47917 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47918 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47919 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47920 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47921 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_8 := arithEq_of_rows h (row := 765) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_8
  simp only [← c47916, ← c47917, ← c47918] at e_765_8
  have e_765_9 := arithEq_of_rows h (row := 765) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_9
  simp only [← c47919, ← c47920, k2, ← c47921] at e_765_9
  have hr := e_765_9
  simp only [e_765_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5260)) := by
  have c47922 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47923 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47924 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47925 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47926 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47927 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_10 := arithEq_of_rows h (row := 765) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_10
  simp only [← c47922, ← c47923, ← c47924] at e_765_10
  have e_765_11 := arithEq_of_rows h (row := 765) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_11
  simp only [← c47925, ← c47926, k2, ← c47927] at e_765_11
  have hr := e_765_11
  simp only [e_765_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5261)) := by
  have c47928 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47929 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47930 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47931 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47932 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47933 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_12 := arithEq_of_rows h (row := 765) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_12
  simp only [← c47928, ← c47929, ← c47930] at e_765_12
  have e_765_13 := arithEq_of_rows h (row := 765) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_13
  simp only [← c47931, ← c47932, k2, ← c47933] at e_765_13
  have hr := e_765_13
  simp only [e_765_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5262)) := by
  have c47934 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47935 := (publicBatchWrapper89_copies1497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47936 := (publicBatchWrapper89_copies1498 a h).1
  have c47937 := (publicBatchWrapper89_copies1498 a h).2.1
  have c47938 := (publicBatchWrapper89_copies1498 a h).2.2.1
  have c47939 := (publicBatchWrapper89_copies1498 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_14 := arithEq_of_rows h (row := 765) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_14
  simp only [← c47934, ← c47935, ← c47936] at e_765_14
  have e_765_15 := arithEq_of_rows h (row := 765) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_15
  simp only [← c47937, ← c47938, k2, ← c47939] at e_765_15
  have hr := e_765_15
  simp only [e_765_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5263)) := by
  have c47940 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.1
  have c47941 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.1
  have c47942 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.1
  have c47943 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.1
  have c47944 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.1
  have c47945 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_16 := arithEq_of_rows h (row := 765) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_16
  simp only [← c47940, ← c47941, ← c47942] at e_765_16
  have e_765_17 := arithEq_of_rows h (row := 765) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_17
  simp only [← c47943, ← c47944, k2, ← c47945] at e_765_17
  have hr := e_765_17
  simp only [e_765_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 765 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5264)) := by
  have c47946 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47947 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47948 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47949 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47950 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47951 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_765_18 := arithEq_of_rows h (row := 765) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_18
  simp only [← c47946, ← c47947, ← c47948] at e_765_18
  have e_765_19 := arithEq_of_rows h (row := 765) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_765_19
  simp only [← c47949, ← c47950, k2, ← c47951] at e_765_19
  have hr := e_765_19
  simp only [e_765_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5265)) := by
  have c47952 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47953 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47954 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47955 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47956 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47957 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_0 := arithEq_of_rows h (row := 766) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_0
  simp only [← c47952, ← c47953, ← c47954] at e_766_0
  have e_766_1 := arithEq_of_rows h (row := 766) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_1
  simp only [← c47955, ← c47956, k2, ← c47957] at e_766_1
  have hr := e_766_1
  simp only [e_766_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5266)) := by
  have c47958 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47959 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47960 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47961 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47962 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47963 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_2 := arithEq_of_rows h (row := 766) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_2
  simp only [← c47958, ← c47959, ← c47960] at e_766_2
  have e_766_3 := arithEq_of_rows h (row := 766) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_3
  simp only [← c47961, ← c47962, k2, ← c47963] at e_766_3
  have hr := e_766_3
  simp only [e_766_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5267)) := by
  have c47964 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47965 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47966 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47967 := (publicBatchWrapper89_copies1498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c47968 := (publicBatchWrapper89_copies1499 a h).1
  have c47969 := (publicBatchWrapper89_copies1499 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_4 := arithEq_of_rows h (row := 766) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_4
  simp only [← c47964, ← c47965, ← c47966] at e_766_4
  have e_766_5 := arithEq_of_rows h (row := 766) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_5
  simp only [← c47967, ← c47968, k2, ← c47969] at e_766_5
  have hr := e_766_5
  simp only [e_766_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5268)) := by
  have c47970 := (publicBatchWrapper89_copies1499 a h).2.2.1
  have c47971 := (publicBatchWrapper89_copies1499 a h).2.2.2.1
  have c47972 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.1
  have c47973 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.1
  have c47974 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.1
  have c47975 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_6 := arithEq_of_rows h (row := 766) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_6
  simp only [← c47970, ← c47971, ← c47972] at e_766_6
  have e_766_7 := arithEq_of_rows h (row := 766) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_7
  simp only [← c47973, ← c47974, k2, ← c47975] at e_766_7
  have hr := e_766_7
  simp only [e_766_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5269)) := by
  have c47976 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.1
  have c47977 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.1
  have c47978 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c47979 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c47980 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47981 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_8 := arithEq_of_rows h (row := 766) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_8
  simp only [← c47976, ← c47977, ← c47978] at e_766_8
  have e_766_9 := arithEq_of_rows h (row := 766) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_9
  simp only [← c47979, ← c47980, k2, ← c47981] at e_766_9
  have hr := e_766_9
  simp only [e_766_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5270)) := by
  have c47982 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47983 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47984 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47985 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47986 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47987 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_10 := arithEq_of_rows h (row := 766) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_10
  simp only [← c47982, ← c47983, ← c47984] at e_766_10
  have e_766_11 := arithEq_of_rows h (row := 766) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_11
  simp only [← c47985, ← c47986, k2, ← c47987] at e_766_11
  have hr := e_766_11
  simp only [e_766_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5271)) := by
  have c47988 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47989 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47990 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47991 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47992 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47993 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_12 := arithEq_of_rows h (row := 766) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_12
  simp only [← c47988, ← c47989, ← c47990] at e_766_12
  have e_766_13 := arithEq_of_rows h (row := 766) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_13
  simp only [← c47991, ← c47992, k2, ← c47993] at e_766_13
  have hr := e_766_13
  simp only [e_766_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5272)) := by
  have c47994 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47995 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47996 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47997 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47998 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47999 := (publicBatchWrapper89_copies1499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_14 := arithEq_of_rows h (row := 766) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_14
  simp only [← c47994, ← c47995, ← c47996] at e_766_14
  have e_766_15 := arithEq_of_rows h (row := 766) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_15
  simp only [← c47997, ← c47998, k2, ← c47999] at e_766_15
  have hr := e_766_15
  simp only [e_766_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5273)) := by
  have c48000 := (publicBatchWrapper89_copies1500 a h).1
  have c48001 := (publicBatchWrapper89_copies1500 a h).2.1
  have c48002 := (publicBatchWrapper89_copies1500 a h).2.2.1
  have c48003 := (publicBatchWrapper89_copies1500 a h).2.2.2.1
  have c48004 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.1
  have c48005 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_16 := arithEq_of_rows h (row := 766) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_16
  simp only [← c48000, ← c48001, ← c48002] at e_766_16
  have e_766_17 := arithEq_of_rows h (row := 766) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_17
  simp only [← c48003, ← c48004, k2, ← c48005] at e_766_17
  have hr := e_766_17
  simp only [e_766_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 766 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5274)) := by
  have c48006 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.1
  have c48007 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.1
  have c48008 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.1
  have c48009 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.1
  have c48010 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48011 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_766_18 := arithEq_of_rows h (row := 766) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_18
  simp only [← c48006, ← c48007, ← c48008] at e_766_18
  have e_766_19 := arithEq_of_rows h (row := 766) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_766_19
  simp only [← c48009, ← c48010, k2, ← c48011] at e_766_19
  have hr := e_766_19
  simp only [e_766_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5275)) := by
  have c48012 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48013 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48014 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48015 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48016 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48017 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_0 := arithEq_of_rows h (row := 767) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_0
  simp only [← c48012, ← c48013, ← c48014] at e_767_0
  have e_767_1 := arithEq_of_rows h (row := 767) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_1
  simp only [← c48015, ← c48016, k2, ← c48017] at e_767_1
  have hr := e_767_1
  simp only [e_767_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5276)) := by
  have c48018 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48019 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48020 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48021 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48022 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48023 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_2 := arithEq_of_rows h (row := 767) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_2
  simp only [← c48018, ← c48019, ← c48020] at e_767_2
  have e_767_3 := arithEq_of_rows h (row := 767) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_3
  simp only [← c48021, ← c48022, k2, ← c48023] at e_767_3
  have hr := e_767_3
  simp only [e_767_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5277)) := by
  have c48024 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48025 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48026 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48027 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48028 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48029 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_4 := arithEq_of_rows h (row := 767) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_4
  simp only [← c48024, ← c48025, ← c48026] at e_767_4
  have e_767_5 := arithEq_of_rows h (row := 767) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_5
  simp only [← c48027, ← c48028, k2, ← c48029] at e_767_5
  have hr := e_767_5
  simp only [e_767_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5278)) := by
  have c48030 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48031 := (publicBatchWrapper89_copies1500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48032 := (publicBatchWrapper89_copies1501 a h).1
  have c48033 := (publicBatchWrapper89_copies1501 a h).2.1
  have c48034 := (publicBatchWrapper89_copies1501 a h).2.2.1
  have c48035 := (publicBatchWrapper89_copies1501 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_6 := arithEq_of_rows h (row := 767) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_6
  simp only [← c48030, ← c48031, ← c48032] at e_767_6
  have e_767_7 := arithEq_of_rows h (row := 767) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_7
  simp only [← c48033, ← c48034, k2, ← c48035] at e_767_7
  have hr := e_767_7
  simp only [e_767_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5279)) := by
  have c48036 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.1
  have c48037 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.1
  have c48038 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.1
  have c48039 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.1
  have c48040 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.1
  have c48041 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_8 := arithEq_of_rows h (row := 767) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_8
  simp only [← c48036, ← c48037, ← c48038] at e_767_8
  have e_767_9 := arithEq_of_rows h (row := 767) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_9
  simp only [← c48039, ← c48040, k2, ← c48041] at e_767_9
  have hr := e_767_9
  simp only [e_767_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5280)) := by
  have c48042 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48043 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48044 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48045 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48046 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48047 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_10 := arithEq_of_rows h (row := 767) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_10
  simp only [← c48042, ← c48043, ← c48044] at e_767_10
  have e_767_11 := arithEq_of_rows h (row := 767) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_11
  simp only [← c48045, ← c48046, k2, ← c48047] at e_767_11
  have hr := e_767_11
  simp only [e_767_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5281)) := by
  have c48048 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48049 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48050 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48051 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48052 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48053 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_12 := arithEq_of_rows h (row := 767) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_12
  simp only [← c48048, ← c48049, ← c48050] at e_767_12
  have e_767_13 := arithEq_of_rows h (row := 767) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_13
  simp only [← c48051, ← c48052, k2, ← c48053] at e_767_13
  have hr := e_767_13
  simp only [e_767_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5282)) := by
  have c48054 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48055 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48056 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48057 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48058 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48059 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_14 := arithEq_of_rows h (row := 767) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_14
  simp only [← c48054, ← c48055, ← c48056] at e_767_14
  have e_767_15 := arithEq_of_rows h (row := 767) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_15
  simp only [← c48057, ← c48058, k2, ← c48059] at e_767_15
  have hr := e_767_15
  simp only [e_767_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5283)) := by
  have c48060 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48061 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48062 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48063 := (publicBatchWrapper89_copies1501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48064 := (publicBatchWrapper89_copies1502 a h).1
  have c48065 := (publicBatchWrapper89_copies1502 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_16 := arithEq_of_rows h (row := 767) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_16
  simp only [← c48060, ← c48061, ← c48062] at e_767_16
  have e_767_17 := arithEq_of_rows h (row := 767) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_17
  simp only [← c48063, ← c48064, k2, ← c48065] at e_767_17
  have hr := e_767_17
  simp only [e_767_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 767 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5284)) := by
  have c48066 := (publicBatchWrapper89_copies1502 a h).2.2.1
  have c48067 := (publicBatchWrapper89_copies1502 a h).2.2.2.1
  have c48068 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.1
  have c48069 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.1
  have c48070 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.1
  have c48071 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_767_18 := arithEq_of_rows h (row := 767) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_18
  simp only [← c48066, ← c48067, ← c48068] at e_767_18
  have e_767_19 := arithEq_of_rows h (row := 767) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_767_19
  simp only [← c48069, ← c48070, k2, ← c48071] at e_767_19
  have hr := e_767_19
  simp only [e_767_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5285)) := by
  have c48072 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.1
  have c48073 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.1
  have c48074 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48075 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48076 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48077 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_0 := arithEq_of_rows h (row := 768) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_0
  simp only [← c48072, ← c48073, ← c48074] at e_768_0
  have e_768_1 := arithEq_of_rows h (row := 768) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_1
  simp only [← c48075, ← c48076, k2, ← c48077] at e_768_1
  have hr := e_768_1
  simp only [e_768_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5286)) := by
  have c48078 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48079 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48080 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48081 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48082 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48083 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_2 := arithEq_of_rows h (row := 768) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_2
  simp only [← c48078, ← c48079, ← c48080] at e_768_2
  have e_768_3 := arithEq_of_rows h (row := 768) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_3
  simp only [← c48081, ← c48082, k2, ← c48083] at e_768_3
  have hr := e_768_3
  simp only [e_768_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5287)) := by
  have c48084 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48085 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48086 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48087 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48088 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48089 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_4 := arithEq_of_rows h (row := 768) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_4
  simp only [← c48084, ← c48085, ← c48086] at e_768_4
  have e_768_5 := arithEq_of_rows h (row := 768) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_5
  simp only [← c48087, ← c48088, k2, ← c48089] at e_768_5
  have hr := e_768_5
  simp only [e_768_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5288)) := by
  have c48090 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48091 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48092 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48093 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48094 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48095 := (publicBatchWrapper89_copies1502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_6 := arithEq_of_rows h (row := 768) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_6
  simp only [← c48090, ← c48091, ← c48092] at e_768_6
  have e_768_7 := arithEq_of_rows h (row := 768) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_7
  simp only [← c48093, ← c48094, k2, ← c48095] at e_768_7
  have hr := e_768_7
  simp only [e_768_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5289)) := by
  have c48096 := (publicBatchWrapper89_copies1503 a h).1
  have c48097 := (publicBatchWrapper89_copies1503 a h).2.1
  have c48098 := (publicBatchWrapper89_copies1503 a h).2.2.1
  have c48099 := (publicBatchWrapper89_copies1503 a h).2.2.2.1
  have c48100 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.1
  have c48101 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_8 := arithEq_of_rows h (row := 768) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_8
  simp only [← c48096, ← c48097, ← c48098] at e_768_8
  have e_768_9 := arithEq_of_rows h (row := 768) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_9
  simp only [← c48099, ← c48100, k2, ← c48101] at e_768_9
  have hr := e_768_9
  simp only [e_768_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5290)) := by
  have c48102 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.1
  have c48103 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.1
  have c48104 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.1
  have c48105 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.1
  have c48106 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48107 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_10 := arithEq_of_rows h (row := 768) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_10
  simp only [← c48102, ← c48103, ← c48104] at e_768_10
  have e_768_11 := arithEq_of_rows h (row := 768) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_11
  simp only [← c48105, ← c48106, k2, ← c48107] at e_768_11
  have hr := e_768_11
  simp only [e_768_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5291)) := by
  have c48108 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48109 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48110 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48111 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48112 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48113 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_12 := arithEq_of_rows h (row := 768) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_12
  simp only [← c48108, ← c48109, ← c48110] at e_768_12
  have e_768_13 := arithEq_of_rows h (row := 768) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_13
  simp only [← c48111, ← c48112, k2, ← c48113] at e_768_13
  have hr := e_768_13
  simp only [e_768_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5292)) := by
  have c48114 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48115 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48116 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48117 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48118 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48119 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_14 := arithEq_of_rows h (row := 768) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_14
  simp only [← c48114, ← c48115, ← c48116] at e_768_14
  have e_768_15 := arithEq_of_rows h (row := 768) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_15
  simp only [← c48117, ← c48118, k2, ← c48119] at e_768_15
  have hr := e_768_15
  simp only [e_768_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5293)) := by
  have c48120 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48121 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48122 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48123 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48124 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48125 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_16 := arithEq_of_rows h (row := 768) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_16
  simp only [← c48120, ← c48121, ← c48122] at e_768_16
  have e_768_17 := arithEq_of_rows h (row := 768) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_17
  simp only [← c48123, ← c48124, k2, ← c48125] at e_768_17
  have hr := e_768_17
  simp only [e_768_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 768 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5294)) := by
  have c48126 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48127 := (publicBatchWrapper89_copies1503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48128 := (publicBatchWrapper89_copies1504 a h).1
  have c48129 := (publicBatchWrapper89_copies1504 a h).2.1
  have c48130 := (publicBatchWrapper89_copies1504 a h).2.2.1
  have c48131 := (publicBatchWrapper89_copies1504 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_768_18 := arithEq_of_rows h (row := 768) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_18
  simp only [← c48126, ← c48127, ← c48128] at e_768_18
  have e_768_19 := arithEq_of_rows h (row := 768) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_768_19
  simp only [← c48129, ← c48130, k2, ← c48131] at e_768_19
  have hr := e_768_19
  simp only [e_768_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5295)) := by
  have c48132 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.1
  have c48133 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.1
  have c48134 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.1
  have c48135 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.1
  have c48136 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.1
  have c48137 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_0 := arithEq_of_rows h (row := 769) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_0
  simp only [← c48132, ← c48133, ← c48134] at e_769_0
  have e_769_1 := arithEq_of_rows h (row := 769) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_1
  simp only [← c48135, ← c48136, k2, ← c48137] at e_769_1
  have hr := e_769_1
  simp only [e_769_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5296)) := by
  have c48138 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48139 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48140 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48141 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48142 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48143 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_2 := arithEq_of_rows h (row := 769) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_2
  simp only [← c48138, ← c48139, ← c48140] at e_769_2
  have e_769_3 := arithEq_of_rows h (row := 769) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_3
  simp only [← c48141, ← c48142, k2, ← c48143] at e_769_3
  have hr := e_769_3
  simp only [e_769_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5297)) := by
  have c48144 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48145 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48146 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48147 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48148 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48149 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_4 := arithEq_of_rows h (row := 769) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_4
  simp only [← c48144, ← c48145, ← c48146] at e_769_4
  have e_769_5 := arithEq_of_rows h (row := 769) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_5
  simp only [← c48147, ← c48148, k2, ← c48149] at e_769_5
  have hr := e_769_5
  simp only [e_769_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5298)) := by
  have c48150 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48151 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48152 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48153 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48154 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48155 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_6 := arithEq_of_rows h (row := 769) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_6
  simp only [← c48150, ← c48151, ← c48152] at e_769_6
  have e_769_7 := arithEq_of_rows h (row := 769) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_7
  simp only [← c48153, ← c48154, k2, ← c48155] at e_769_7
  have hr := e_769_7
  simp only [e_769_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5299)) := by
  have c48156 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48157 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48158 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48159 := (publicBatchWrapper89_copies1504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48160 := (publicBatchWrapper89_copies1505 a h).1
  have c48161 := (publicBatchWrapper89_copies1505 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_8 := arithEq_of_rows h (row := 769) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_8
  simp only [← c48156, ← c48157, ← c48158] at e_769_8
  have e_769_9 := arithEq_of_rows h (row := 769) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_9
  simp only [← c48159, ← c48160, k2, ← c48161] at e_769_9
  have hr := e_769_9
  simp only [e_769_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5300)) := by
  have c48162 := (publicBatchWrapper89_copies1505 a h).2.2.1
  have c48163 := (publicBatchWrapper89_copies1505 a h).2.2.2.1
  have c48164 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.1
  have c48165 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.1
  have c48166 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.1
  have c48167 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_10 := arithEq_of_rows h (row := 769) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_10
  simp only [← c48162, ← c48163, ← c48164] at e_769_10
  have e_769_11 := arithEq_of_rows h (row := 769) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_11
  simp only [← c48165, ← c48166, k2, ← c48167] at e_769_11
  have hr := e_769_11
  simp only [e_769_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5301)) := by
  have c48168 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.1
  have c48169 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.1
  have c48170 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48171 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48172 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48173 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_12 := arithEq_of_rows h (row := 769) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_12
  simp only [← c48168, ← c48169, ← c48170] at e_769_12
  have e_769_13 := arithEq_of_rows h (row := 769) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_13
  simp only [← c48171, ← c48172, k2, ← c48173] at e_769_13
  have hr := e_769_13
  simp only [e_769_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5302)) := by
  have c48174 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48175 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48176 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48177 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48178 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48179 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_14 := arithEq_of_rows h (row := 769) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_14
  simp only [← c48174, ← c48175, ← c48176] at e_769_14
  have e_769_15 := arithEq_of_rows h (row := 769) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_15
  simp only [← c48177, ← c48178, k2, ← c48179] at e_769_15
  have hr := e_769_15
  simp only [e_769_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5303)) := by
  have c48180 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48181 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48182 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48183 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48184 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48185 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_16 := arithEq_of_rows h (row := 769) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_16
  simp only [← c48180, ← c48181, ← c48182] at e_769_16
  have e_769_17 := arithEq_of_rows h (row := 769) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_17
  simp only [← c48183, ← c48184, k2, ← c48185] at e_769_17
  have hr := e_769_17
  simp only [e_769_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 769 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5304)) := by
  have c48186 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48187 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48188 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48189 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48190 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48191 := (publicBatchWrapper89_copies1505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_769_18 := arithEq_of_rows h (row := 769) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_18
  simp only [← c48186, ← c48187, ← c48188] at e_769_18
  have e_769_19 := arithEq_of_rows h (row := 769) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_769_19
  simp only [← c48189, ← c48190, k2, ← c48191] at e_769_19
  have hr := e_769_19
  simp only [e_769_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5305)) := by
  have c48192 := (publicBatchWrapper89_copies1506 a h).1
  have c48193 := (publicBatchWrapper89_copies1506 a h).2.1
  have c48194 := (publicBatchWrapper89_copies1506 a h).2.2.1
  have c48195 := (publicBatchWrapper89_copies1506 a h).2.2.2.1
  have c48196 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.1
  have c48197 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_0 := arithEq_of_rows h (row := 770) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_0
  simp only [← c48192, ← c48193, ← c48194] at e_770_0
  have e_770_1 := arithEq_of_rows h (row := 770) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_1
  simp only [← c48195, ← c48196, k2, ← c48197] at e_770_1
  have hr := e_770_1
  simp only [e_770_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5306)) := by
  have c48198 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.1
  have c48199 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.1
  have c48200 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.1
  have c48201 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.1
  have c48202 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48203 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_2 := arithEq_of_rows h (row := 770) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_2
  simp only [← c48198, ← c48199, ← c48200] at e_770_2
  have e_770_3 := arithEq_of_rows h (row := 770) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_3
  simp only [← c48201, ← c48202, k2, ← c48203] at e_770_3
  have hr := e_770_3
  simp only [e_770_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5307)) := by
  have c48204 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48205 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48206 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48207 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48208 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48209 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_4 := arithEq_of_rows h (row := 770) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_4
  simp only [← c48204, ← c48205, ← c48206] at e_770_4
  have e_770_5 := arithEq_of_rows h (row := 770) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_5
  simp only [← c48207, ← c48208, k2, ← c48209] at e_770_5
  have hr := e_770_5
  simp only [e_770_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5308)) := by
  have c48210 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48211 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48212 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48213 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48214 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48215 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_6 := arithEq_of_rows h (row := 770) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_6
  simp only [← c48210, ← c48211, ← c48212] at e_770_6
  have e_770_7 := arithEq_of_rows h (row := 770) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_7
  simp only [← c48213, ← c48214, k2, ← c48215] at e_770_7
  have hr := e_770_7
  simp only [e_770_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5309)) := by
  have c48216 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48217 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48218 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48219 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48220 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48221 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_8 := arithEq_of_rows h (row := 770) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_8
  simp only [← c48216, ← c48217, ← c48218] at e_770_8
  have e_770_9 := arithEq_of_rows h (row := 770) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_9
  simp only [← c48219, ← c48220, k2, ← c48221] at e_770_9
  have hr := e_770_9
  simp only [e_770_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5310)) := by
  have c48222 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48223 := (publicBatchWrapper89_copies1506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48224 := (publicBatchWrapper89_copies1507 a h).1
  have c48225 := (publicBatchWrapper89_copies1507 a h).2.1
  have c48226 := (publicBatchWrapper89_copies1507 a h).2.2.1
  have c48227 := (publicBatchWrapper89_copies1507 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_10 := arithEq_of_rows h (row := 770) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_10
  simp only [← c48222, ← c48223, ← c48224] at e_770_10
  have e_770_11 := arithEq_of_rows h (row := 770) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_11
  simp only [← c48225, ← c48226, k2, ← c48227] at e_770_11
  have hr := e_770_11
  simp only [e_770_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5311)) := by
  have c48228 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.1
  have c48229 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.1
  have c48230 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.1
  have c48231 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.1
  have c48232 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.1
  have c48233 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_12 := arithEq_of_rows h (row := 770) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_12
  simp only [← c48228, ← c48229, ← c48230] at e_770_12
  have e_770_13 := arithEq_of_rows h (row := 770) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_13
  simp only [← c48231, ← c48232, k2, ← c48233] at e_770_13
  have hr := e_770_13
  simp only [e_770_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5312)) := by
  have c48234 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48235 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48236 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48237 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48238 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48239 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_14 := arithEq_of_rows h (row := 770) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_14
  simp only [← c48234, ← c48235, ← c48236] at e_770_14
  have e_770_15 := arithEq_of_rows h (row := 770) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_15
  simp only [← c48237, ← c48238, k2, ← c48239] at e_770_15
  have hr := e_770_15
  simp only [e_770_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5313)) := by
  have c48240 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48241 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48242 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48243 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48244 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48245 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_16 := arithEq_of_rows h (row := 770) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_16
  simp only [← c48240, ← c48241, ← c48242] at e_770_16
  have e_770_17 := arithEq_of_rows h (row := 770) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_17
  simp only [← c48243, ← c48244, k2, ← c48245] at e_770_17
  have hr := e_770_17
  simp only [e_770_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 770 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5314)) := by
  have c48246 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48247 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48248 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48249 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48250 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48251 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_770_18 := arithEq_of_rows h (row := 770) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_18
  simp only [← c48246, ← c48247, ← c48248] at e_770_18
  have e_770_19 := arithEq_of_rows h (row := 770) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_770_19
  simp only [← c48249, ← c48250, k2, ← c48251] at e_770_19
  have hr := e_770_19
  simp only [e_770_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5315)) := by
  have c48252 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48253 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48254 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48255 := (publicBatchWrapper89_copies1507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48256 := (publicBatchWrapper89_copies1508 a h).1
  have c48257 := (publicBatchWrapper89_copies1508 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_0 := arithEq_of_rows h (row := 771) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_0
  simp only [← c48252, ← c48253, ← c48254] at e_771_0
  have e_771_1 := arithEq_of_rows h (row := 771) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_1
  simp only [← c48255, ← c48256, k2, ← c48257] at e_771_1
  have hr := e_771_1
  simp only [e_771_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5316)) := by
  have c48258 := (publicBatchWrapper89_copies1508 a h).2.2.1
  have c48259 := (publicBatchWrapper89_copies1508 a h).2.2.2.1
  have c48260 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.1
  have c48261 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.1
  have c48262 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.1
  have c48263 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_2 := arithEq_of_rows h (row := 771) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_2
  simp only [← c48258, ← c48259, ← c48260] at e_771_2
  have e_771_3 := arithEq_of_rows h (row := 771) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_3
  simp only [← c48261, ← c48262, k2, ← c48263] at e_771_3
  have hr := e_771_3
  simp only [e_771_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5317)) := by
  have c48264 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.1
  have c48265 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.1
  have c48266 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48267 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48268 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48269 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_4 := arithEq_of_rows h (row := 771) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_4
  simp only [← c48264, ← c48265, ← c48266] at e_771_4
  have e_771_5 := arithEq_of_rows h (row := 771) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_5
  simp only [← c48267, ← c48268, k2, ← c48269] at e_771_5
  have hr := e_771_5
  simp only [e_771_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5318)) := by
  have c48270 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48271 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48272 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48273 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48274 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48275 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_6 := arithEq_of_rows h (row := 771) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_6
  simp only [← c48270, ← c48271, ← c48272] at e_771_6
  have e_771_7 := arithEq_of_rows h (row := 771) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_7
  simp only [← c48273, ← c48274, k2, ← c48275] at e_771_7
  have hr := e_771_7
  simp only [e_771_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5319)) := by
  have c48276 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48277 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48278 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48279 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48280 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48281 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_8 := arithEq_of_rows h (row := 771) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_8
  simp only [← c48276, ← c48277, ← c48278] at e_771_8
  have e_771_9 := arithEq_of_rows h (row := 771) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_9
  simp only [← c48279, ← c48280, k2, ← c48281] at e_771_9
  have hr := e_771_9
  simp only [e_771_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5320)) := by
  have c48282 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48283 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48284 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48285 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48286 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48287 := (publicBatchWrapper89_copies1508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_10 := arithEq_of_rows h (row := 771) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_10
  simp only [← c48282, ← c48283, ← c48284] at e_771_10
  have e_771_11 := arithEq_of_rows h (row := 771) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_11
  simp only [← c48285, ← c48286, k2, ← c48287] at e_771_11
  have hr := e_771_11
  simp only [e_771_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5321)) := by
  have c48288 := (publicBatchWrapper89_copies1509 a h).1
  have c48289 := (publicBatchWrapper89_copies1509 a h).2.1
  have c48290 := (publicBatchWrapper89_copies1509 a h).2.2.1
  have c48291 := (publicBatchWrapper89_copies1509 a h).2.2.2.1
  have c48292 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.1
  have c48293 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_12 := arithEq_of_rows h (row := 771) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_12
  simp only [← c48288, ← c48289, ← c48290] at e_771_12
  have e_771_13 := arithEq_of_rows h (row := 771) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_13
  simp only [← c48291, ← c48292, k2, ← c48293] at e_771_13
  have hr := e_771_13
  simp only [e_771_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5322)) := by
  have c48294 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.1
  have c48295 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.1
  have c48296 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.1
  have c48297 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.1
  have c48298 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48299 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_14 := arithEq_of_rows h (row := 771) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_14
  simp only [← c48294, ← c48295, ← c48296] at e_771_14
  have e_771_15 := arithEq_of_rows h (row := 771) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_15
  simp only [← c48297, ← c48298, k2, ← c48299] at e_771_15
  have hr := e_771_15
  simp only [e_771_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5323)) := by
  have c48300 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48301 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48302 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48303 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48304 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48305 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_16 := arithEq_of_rows h (row := 771) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_16
  simp only [← c48300, ← c48301, ← c48302] at e_771_16
  have e_771_17 := arithEq_of_rows h (row := 771) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_17
  simp only [← c48303, ← c48304, k2, ← c48305] at e_771_17
  have hr := e_771_17
  simp only [e_771_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 771 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5324)) := by
  have c48306 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48307 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48308 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48309 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48310 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48311 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_771_18 := arithEq_of_rows h (row := 771) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_18
  simp only [← c48306, ← c48307, ← c48308] at e_771_18
  have e_771_19 := arithEq_of_rows h (row := 771) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_771_19
  simp only [← c48309, ← c48310, k2, ← c48311] at e_771_19
  have hr := e_771_19
  simp only [e_771_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5325)) := by
  have c48312 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48313 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48314 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48315 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48316 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48317 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_0 := arithEq_of_rows h (row := 772) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_0
  simp only [← c48312, ← c48313, ← c48314] at e_772_0
  have e_772_1 := arithEq_of_rows h (row := 772) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_1
  simp only [← c48315, ← c48316, k2, ← c48317] at e_772_1
  have hr := e_772_1
  simp only [e_772_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5326)) := by
  have c48318 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48319 := (publicBatchWrapper89_copies1509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48320 := (publicBatchWrapper89_copies1510 a h).1
  have c48321 := (publicBatchWrapper89_copies1510 a h).2.1
  have c48322 := (publicBatchWrapper89_copies1510 a h).2.2.1
  have c48323 := (publicBatchWrapper89_copies1510 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_2 := arithEq_of_rows h (row := 772) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_2
  simp only [← c48318, ← c48319, ← c48320] at e_772_2
  have e_772_3 := arithEq_of_rows h (row := 772) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_3
  simp only [← c48321, ← c48322, k2, ← c48323] at e_772_3
  have hr := e_772_3
  simp only [e_772_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5327)) := by
  have c48324 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.1
  have c48325 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.1
  have c48326 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.1
  have c48327 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.1
  have c48328 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.1
  have c48329 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_4 := arithEq_of_rows h (row := 772) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_4
  simp only [← c48324, ← c48325, ← c48326] at e_772_4
  have e_772_5 := arithEq_of_rows h (row := 772) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_5
  simp only [← c48327, ← c48328, k2, ← c48329] at e_772_5
  have hr := e_772_5
  simp only [e_772_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5328)) := by
  have c48330 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48331 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48332 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48333 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48334 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48335 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_6 := arithEq_of_rows h (row := 772) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_6
  simp only [← c48330, ← c48331, ← c48332] at e_772_6
  have e_772_7 := arithEq_of_rows h (row := 772) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_7
  simp only [← c48333, ← c48334, k2, ← c48335] at e_772_7
  have hr := e_772_7
  simp only [e_772_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5329)) := by
  have c48336 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48337 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48338 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48339 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48340 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48341 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_8 := arithEq_of_rows h (row := 772) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_8
  simp only [← c48336, ← c48337, ← c48338] at e_772_8
  have e_772_9 := arithEq_of_rows h (row := 772) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_9
  simp only [← c48339, ← c48340, k2, ← c48341] at e_772_9
  have hr := e_772_9
  simp only [e_772_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5330)) := by
  have c48342 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48343 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48344 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48345 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48346 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48347 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_10 := arithEq_of_rows h (row := 772) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_10
  simp only [← c48342, ← c48343, ← c48344] at e_772_10
  have e_772_11 := arithEq_of_rows h (row := 772) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_11
  simp only [← c48345, ← c48346, k2, ← c48347] at e_772_11
  have hr := e_772_11
  simp only [e_772_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5331)) := by
  have c48348 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48349 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48350 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48351 := (publicBatchWrapper89_copies1510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48352 := (publicBatchWrapper89_copies1511 a h).1
  have c48353 := (publicBatchWrapper89_copies1511 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_12 := arithEq_of_rows h (row := 772) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_12
  simp only [← c48348, ← c48349, ← c48350] at e_772_12
  have e_772_13 := arithEq_of_rows h (row := 772) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_13
  simp only [← c48351, ← c48352, k2, ← c48353] at e_772_13
  have hr := e_772_13
  simp only [e_772_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5332)) := by
  have c48354 := (publicBatchWrapper89_copies1511 a h).2.2.1
  have c48355 := (publicBatchWrapper89_copies1511 a h).2.2.2.1
  have c48356 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.1
  have c48357 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.1
  have c48358 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.1
  have c48359 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_14 := arithEq_of_rows h (row := 772) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_14
  simp only [← c48354, ← c48355, ← c48356] at e_772_14
  have e_772_15 := arithEq_of_rows h (row := 772) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_15
  simp only [← c48357, ← c48358, k2, ← c48359] at e_772_15
  have hr := e_772_15
  simp only [e_772_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5333)) := by
  have c48360 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.1
  have c48361 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.1
  have c48362 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48363 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48364 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48365 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_16 := arithEq_of_rows h (row := 772) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_16
  simp only [← c48360, ← c48361, ← c48362] at e_772_16
  have e_772_17 := arithEq_of_rows h (row := 772) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_17
  simp only [← c48363, ← c48364, k2, ← c48365] at e_772_17
  have hr := e_772_17
  simp only [e_772_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 772 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5334)) := by
  have c48366 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48367 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48368 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48369 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48370 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48371 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_772_18 := arithEq_of_rows h (row := 772) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_18
  simp only [← c48366, ← c48367, ← c48368] at e_772_18
  have e_772_19 := arithEq_of_rows h (row := 772) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_772_19
  simp only [← c48369, ← c48370, k2, ← c48371] at e_772_19
  have hr := e_772_19
  simp only [e_772_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5335)) := by
  have c48372 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48373 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48374 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48375 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48376 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48377 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_0 := arithEq_of_rows h (row := 773) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_0
  simp only [← c48372, ← c48373, ← c48374] at e_773_0
  have e_773_1 := arithEq_of_rows h (row := 773) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_1
  simp only [← c48375, ← c48376, k2, ← c48377] at e_773_1
  have hr := e_773_1
  simp only [e_773_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5336)) := by
  have c48378 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48379 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48380 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48381 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48382 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48383 := (publicBatchWrapper89_copies1511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_2 := arithEq_of_rows h (row := 773) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_2
  simp only [← c48378, ← c48379, ← c48380] at e_773_2
  have e_773_3 := arithEq_of_rows h (row := 773) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_3
  simp only [← c48381, ← c48382, k2, ← c48383] at e_773_3
  have hr := e_773_3
  simp only [e_773_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5337)) := by
  have c48384 := (publicBatchWrapper89_copies1512 a h).1
  have c48385 := (publicBatchWrapper89_copies1512 a h).2.1
  have c48386 := (publicBatchWrapper89_copies1512 a h).2.2.1
  have c48387 := (publicBatchWrapper89_copies1512 a h).2.2.2.1
  have c48388 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.1
  have c48389 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_4 := arithEq_of_rows h (row := 773) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_4
  simp only [← c48384, ← c48385, ← c48386] at e_773_4
  have e_773_5 := arithEq_of_rows h (row := 773) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_5
  simp only [← c48387, ← c48388, k2, ← c48389] at e_773_5
  have hr := e_773_5
  simp only [e_773_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5338)) := by
  have c48390 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.1
  have c48391 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.1
  have c48392 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.1
  have c48393 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.1
  have c48394 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48395 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_6 := arithEq_of_rows h (row := 773) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_6
  simp only [← c48390, ← c48391, ← c48392] at e_773_6
  have e_773_7 := arithEq_of_rows h (row := 773) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_7
  simp only [← c48393, ← c48394, k2, ← c48395] at e_773_7
  have hr := e_773_7
  simp only [e_773_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5339)) := by
  have c48396 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48397 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48398 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48399 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48400 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48401 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_8 := arithEq_of_rows h (row := 773) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_8
  simp only [← c48396, ← c48397, ← c48398] at e_773_8
  have e_773_9 := arithEq_of_rows h (row := 773) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_9
  simp only [← c48399, ← c48400, k2, ← c48401] at e_773_9
  have hr := e_773_9
  simp only [e_773_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5340)) := by
  have c48402 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48403 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48404 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48405 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48406 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48407 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_10 := arithEq_of_rows h (row := 773) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_10
  simp only [← c48402, ← c48403, ← c48404] at e_773_10
  have e_773_11 := arithEq_of_rows h (row := 773) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_11
  simp only [← c48405, ← c48406, k2, ← c48407] at e_773_11
  have hr := e_773_11
  simp only [e_773_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5341)) := by
  have c48408 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48409 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48410 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48411 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48412 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48413 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_12 := arithEq_of_rows h (row := 773) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_12
  simp only [← c48408, ← c48409, ← c48410] at e_773_12
  have e_773_13 := arithEq_of_rows h (row := 773) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_13
  simp only [← c48411, ← c48412, k2, ← c48413] at e_773_13
  have hr := e_773_13
  simp only [e_773_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5342)) := by
  have c48414 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48415 := (publicBatchWrapper89_copies1512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48416 := (publicBatchWrapper89_copies1513 a h).1
  have c48417 := (publicBatchWrapper89_copies1513 a h).2.1
  have c48418 := (publicBatchWrapper89_copies1513 a h).2.2.1
  have c48419 := (publicBatchWrapper89_copies1513 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_14 := arithEq_of_rows h (row := 773) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_14
  simp only [← c48414, ← c48415, ← c48416] at e_773_14
  have e_773_15 := arithEq_of_rows h (row := 773) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_15
  simp only [← c48417, ← c48418, k2, ← c48419] at e_773_15
  have hr := e_773_15
  simp only [e_773_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5343)) := by
  have c48420 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.1
  have c48421 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.1
  have c48422 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.1
  have c48423 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.1
  have c48424 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.1
  have c48425 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_16 := arithEq_of_rows h (row := 773) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_16
  simp only [← c48420, ← c48421, ← c48422] at e_773_16
  have e_773_17 := arithEq_of_rows h (row := 773) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_17
  simp only [← c48423, ← c48424, k2, ← c48425] at e_773_17
  have hr := e_773_17
  simp only [e_773_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 773 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5344)) := by
  have c48426 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48427 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48428 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48429 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48430 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48431 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_773_18 := arithEq_of_rows h (row := 773) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_18
  simp only [← c48426, ← c48427, ← c48428] at e_773_18
  have e_773_19 := arithEq_of_rows h (row := 773) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_773_19
  simp only [← c48429, ← c48430, k2, ← c48431] at e_773_19
  have hr := e_773_19
  simp only [e_773_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5345)) := by
  have c48432 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48433 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48434 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48435 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48436 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48437 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_0 := arithEq_of_rows h (row := 774) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_0
  simp only [← c48432, ← c48433, ← c48434] at e_774_0
  have e_774_1 := arithEq_of_rows h (row := 774) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_1
  simp only [← c48435, ← c48436, k2, ← c48437] at e_774_1
  have hr := e_774_1
  simp only [e_774_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5346)) := by
  have c48438 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48439 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48440 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48441 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48442 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48443 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_2 := arithEq_of_rows h (row := 774) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_2
  simp only [← c48438, ← c48439, ← c48440] at e_774_2
  have e_774_3 := arithEq_of_rows h (row := 774) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_3
  simp only [← c48441, ← c48442, k2, ← c48443] at e_774_3
  have hr := e_774_3
  simp only [e_774_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5347)) := by
  have c48444 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48445 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48446 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48447 := (publicBatchWrapper89_copies1513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48448 := (publicBatchWrapper89_copies1514 a h).1
  have c48449 := (publicBatchWrapper89_copies1514 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_4 := arithEq_of_rows h (row := 774) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_4
  simp only [← c48444, ← c48445, ← c48446] at e_774_4
  have e_774_5 := arithEq_of_rows h (row := 774) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_5
  simp only [← c48447, ← c48448, k2, ← c48449] at e_774_5
  have hr := e_774_5
  simp only [e_774_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5348)) := by
  have c48450 := (publicBatchWrapper89_copies1514 a h).2.2.1
  have c48451 := (publicBatchWrapper89_copies1514 a h).2.2.2.1
  have c48452 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.1
  have c48453 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.1
  have c48454 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.1
  have c48455 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_6 := arithEq_of_rows h (row := 774) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_6
  simp only [← c48450, ← c48451, ← c48452] at e_774_6
  have e_774_7 := arithEq_of_rows h (row := 774) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_7
  simp only [← c48453, ← c48454, k2, ← c48455] at e_774_7
  have hr := e_774_7
  simp only [e_774_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5349)) := by
  have c48456 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.1
  have c48457 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.1
  have c48458 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48459 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48460 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48461 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_8 := arithEq_of_rows h (row := 774) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_8
  simp only [← c48456, ← c48457, ← c48458] at e_774_8
  have e_774_9 := arithEq_of_rows h (row := 774) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_9
  simp only [← c48459, ← c48460, k2, ← c48461] at e_774_9
  have hr := e_774_9
  simp only [e_774_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5350)) := by
  have c48462 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48463 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48464 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48465 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48466 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48467 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_10 := arithEq_of_rows h (row := 774) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_10
  simp only [← c48462, ← c48463, ← c48464] at e_774_10
  have e_774_11 := arithEq_of_rows h (row := 774) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_11
  simp only [← c48465, ← c48466, k2, ← c48467] at e_774_11
  have hr := e_774_11
  simp only [e_774_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5351)) := by
  have c48468 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48469 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48470 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48471 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48472 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48473 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_12 := arithEq_of_rows h (row := 774) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_12
  simp only [← c48468, ← c48469, ← c48470] at e_774_12
  have e_774_13 := arithEq_of_rows h (row := 774) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_13
  simp only [← c48471, ← c48472, k2, ← c48473] at e_774_13
  have hr := e_774_13
  simp only [e_774_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5352)) := by
  have c48474 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48475 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48476 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48477 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48478 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48479 := (publicBatchWrapper89_copies1514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_14 := arithEq_of_rows h (row := 774) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_14
  simp only [← c48474, ← c48475, ← c48476] at e_774_14
  have e_774_15 := arithEq_of_rows h (row := 774) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_15
  simp only [← c48477, ← c48478, k2, ← c48479] at e_774_15
  have hr := e_774_15
  simp only [e_774_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5353)) := by
  have c48480 := (publicBatchWrapper89_copies1515 a h).1
  have c48481 := (publicBatchWrapper89_copies1515 a h).2.1
  have c48482 := (publicBatchWrapper89_copies1515 a h).2.2.1
  have c48483 := (publicBatchWrapper89_copies1515 a h).2.2.2.1
  have c48484 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.1
  have c48485 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_16 := arithEq_of_rows h (row := 774) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_16
  simp only [← c48480, ← c48481, ← c48482] at e_774_16
  have e_774_17 := arithEq_of_rows h (row := 774) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_17
  simp only [← c48483, ← c48484, k2, ← c48485] at e_774_17
  have hr := e_774_17
  simp only [e_774_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 774 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5354)) := by
  have c48486 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.1
  have c48487 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.1
  have c48488 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.1
  have c48489 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.1
  have c48490 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48491 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_774_18 := arithEq_of_rows h (row := 774) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_18
  simp only [← c48486, ← c48487, ← c48488] at e_774_18
  have e_774_19 := arithEq_of_rows h (row := 774) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_774_19
  simp only [← c48489, ← c48490, k2, ← c48491] at e_774_19
  have hr := e_774_19
  simp only [e_774_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5355)) := by
  have c48492 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48493 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48494 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48495 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48496 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48497 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_0 := arithEq_of_rows h (row := 775) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_0
  simp only [← c48492, ← c48493, ← c48494] at e_775_0
  have e_775_1 := arithEq_of_rows h (row := 775) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_1
  simp only [← c48495, ← c48496, k2, ← c48497] at e_775_1
  have hr := e_775_1
  simp only [e_775_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5356)) := by
  have c48498 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48499 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48500 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48501 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48502 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48503 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_2 := arithEq_of_rows h (row := 775) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_2
  simp only [← c48498, ← c48499, ← c48500] at e_775_2
  have e_775_3 := arithEq_of_rows h (row := 775) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_3
  simp only [← c48501, ← c48502, k2, ← c48503] at e_775_3
  have hr := e_775_3
  simp only [e_775_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5357)) := by
  have c48504 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48505 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48506 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48507 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48508 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48509 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_4 := arithEq_of_rows h (row := 775) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_4
  simp only [← c48504, ← c48505, ← c48506] at e_775_4
  have e_775_5 := arithEq_of_rows h (row := 775) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_5
  simp only [← c48507, ← c48508, k2, ← c48509] at e_775_5
  have hr := e_775_5
  simp only [e_775_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5358)) := by
  have c48510 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48511 := (publicBatchWrapper89_copies1515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48512 := (publicBatchWrapper89_copies1516 a h).1
  have c48513 := (publicBatchWrapper89_copies1516 a h).2.1
  have c48514 := (publicBatchWrapper89_copies1516 a h).2.2.1
  have c48515 := (publicBatchWrapper89_copies1516 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_6 := arithEq_of_rows h (row := 775) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_6
  simp only [← c48510, ← c48511, ← c48512] at e_775_6
  have e_775_7 := arithEq_of_rows h (row := 775) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_7
  simp only [← c48513, ← c48514, k2, ← c48515] at e_775_7
  have hr := e_775_7
  simp only [e_775_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5359)) := by
  have c48516 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.1
  have c48517 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.1
  have c48518 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.1
  have c48519 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.1
  have c48520 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.1
  have c48521 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_8 := arithEq_of_rows h (row := 775) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_8
  simp only [← c48516, ← c48517, ← c48518] at e_775_8
  have e_775_9 := arithEq_of_rows h (row := 775) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_9
  simp only [← c48519, ← c48520, k2, ← c48521] at e_775_9
  have hr := e_775_9
  simp only [e_775_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5360)) := by
  have c48522 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48523 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48524 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48525 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48526 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48527 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_10 := arithEq_of_rows h (row := 775) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_10
  simp only [← c48522, ← c48523, ← c48524] at e_775_10
  have e_775_11 := arithEq_of_rows h (row := 775) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_11
  simp only [← c48525, ← c48526, k2, ← c48527] at e_775_11
  have hr := e_775_11
  simp only [e_775_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5361)) := by
  have c48528 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48529 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48530 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48531 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48532 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48533 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_12 := arithEq_of_rows h (row := 775) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_12
  simp only [← c48528, ← c48529, ← c48530] at e_775_12
  have e_775_13 := arithEq_of_rows h (row := 775) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_13
  simp only [← c48531, ← c48532, k2, ← c48533] at e_775_13
  have hr := e_775_13
  simp only [e_775_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5362)) := by
  have c48534 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48535 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48536 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48537 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48538 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48539 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_14 := arithEq_of_rows h (row := 775) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_14
  simp only [← c48534, ← c48535, ← c48536] at e_775_14
  have e_775_15 := arithEq_of_rows h (row := 775) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_15
  simp only [← c48537, ← c48538, k2, ← c48539] at e_775_15
  have hr := e_775_15
  simp only [e_775_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5363)) := by
  have c48540 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48541 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48542 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48543 := (publicBatchWrapper89_copies1516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48544 := (publicBatchWrapper89_copies1517 a h).1
  have c48545 := (publicBatchWrapper89_copies1517 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_16 := arithEq_of_rows h (row := 775) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_16
  simp only [← c48540, ← c48541, ← c48542] at e_775_16
  have e_775_17 := arithEq_of_rows h (row := 775) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_17
  simp only [← c48543, ← c48544, k2, ← c48545] at e_775_17
  have hr := e_775_17
  simp only [e_775_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 775 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5364)) := by
  have c48546 := (publicBatchWrapper89_copies1517 a h).2.2.1
  have c48547 := (publicBatchWrapper89_copies1517 a h).2.2.2.1
  have c48548 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.1
  have c48549 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.1
  have c48550 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.1
  have c48551 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_775_18 := arithEq_of_rows h (row := 775) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_18
  simp only [← c48546, ← c48547, ← c48548] at e_775_18
  have e_775_19 := arithEq_of_rows h (row := 775) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_775_19
  simp only [← c48549, ← c48550, k2, ← c48551] at e_775_19
  have hr := e_775_19
  simp only [e_775_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5365)) := by
  have c48552 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.1
  have c48553 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.1
  have c48554 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48555 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48556 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48557 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_0 := arithEq_of_rows h (row := 776) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_0
  simp only [← c48552, ← c48553, ← c48554] at e_776_0
  have e_776_1 := arithEq_of_rows h (row := 776) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_1
  simp only [← c48555, ← c48556, k2, ← c48557] at e_776_1
  have hr := e_776_1
  simp only [e_776_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5366)) := by
  have c48558 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48559 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48560 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48561 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48562 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48563 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_2 := arithEq_of_rows h (row := 776) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_2
  simp only [← c48558, ← c48559, ← c48560] at e_776_2
  have e_776_3 := arithEq_of_rows h (row := 776) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_3
  simp only [← c48561, ← c48562, k2, ← c48563] at e_776_3
  have hr := e_776_3
  simp only [e_776_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5367)) := by
  have c48564 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48565 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48566 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48567 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48568 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48569 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_4 := arithEq_of_rows h (row := 776) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_4
  simp only [← c48564, ← c48565, ← c48566] at e_776_4
  have e_776_5 := arithEq_of_rows h (row := 776) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_5
  simp only [← c48567, ← c48568, k2, ← c48569] at e_776_5
  have hr := e_776_5
  simp only [e_776_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5368)) := by
  have c48570 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48571 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48572 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48573 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48574 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48575 := (publicBatchWrapper89_copies1517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_6 := arithEq_of_rows h (row := 776) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_6
  simp only [← c48570, ← c48571, ← c48572] at e_776_6
  have e_776_7 := arithEq_of_rows h (row := 776) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_7
  simp only [← c48573, ← c48574, k2, ← c48575] at e_776_7
  have hr := e_776_7
  simp only [e_776_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5369)) := by
  have c48576 := (publicBatchWrapper89_copies1518 a h).1
  have c48577 := (publicBatchWrapper89_copies1518 a h).2.1
  have c48578 := (publicBatchWrapper89_copies1518 a h).2.2.1
  have c48579 := (publicBatchWrapper89_copies1518 a h).2.2.2.1
  have c48580 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.1
  have c48581 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_8 := arithEq_of_rows h (row := 776) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_8
  simp only [← c48576, ← c48577, ← c48578] at e_776_8
  have e_776_9 := arithEq_of_rows h (row := 776) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_9
  simp only [← c48579, ← c48580, k2, ← c48581] at e_776_9
  have hr := e_776_9
  simp only [e_776_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5410)) := by
  have c48582 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.1
  have c48583 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.1
  have c48584 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.1
  have c48585 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.1
  have c48586 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48587 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_10 := arithEq_of_rows h (row := 776) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_10
  simp only [← c48582, ← c48583, ← c48584] at e_776_10
  have e_776_11 := arithEq_of_rows h (row := 776) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_11
  simp only [← c48585, ← c48586, k2, ← c48587] at e_776_11
  have hr := e_776_11
  simp only [e_776_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5411)) := by
  have c48588 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48589 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48590 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48591 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48592 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48593 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_12 := arithEq_of_rows h (row := 776) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_12
  simp only [← c48588, ← c48589, ← c48590] at e_776_12
  have e_776_13 := arithEq_of_rows h (row := 776) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_13
  simp only [← c48591, ← c48592, k2, ← c48593] at e_776_13
  have hr := e_776_13
  simp only [e_776_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5412)) := by
  have c48594 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48595 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48596 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48597 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48598 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48599 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_14 := arithEq_of_rows h (row := 776) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_14
  simp only [← c48594, ← c48595, ← c48596] at e_776_14
  have e_776_15 := arithEq_of_rows h (row := 776) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_15
  simp only [← c48597, ← c48598, k2, ← c48599] at e_776_15
  have hr := e_776_15
  simp only [e_776_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5413)) := by
  have c48600 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48601 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48602 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48603 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48604 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48605 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_16 := arithEq_of_rows h (row := 776) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_16
  simp only [← c48600, ← c48601, ← c48602] at e_776_16
  have e_776_17 := arithEq_of_rows h (row := 776) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_17
  simp only [← c48603, ← c48604, k2, ← c48605] at e_776_17
  have hr := e_776_17
  simp only [e_776_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 776 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5414)) := by
  have c48606 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48607 := (publicBatchWrapper89_copies1518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48608 := (publicBatchWrapper89_copies1519 a h).1
  have c48609 := (publicBatchWrapper89_copies1519 a h).2.1
  have c48610 := (publicBatchWrapper89_copies1519 a h).2.2.1
  have c48611 := (publicBatchWrapper89_copies1519 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_776_18 := arithEq_of_rows h (row := 776) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_18
  simp only [← c48606, ← c48607, ← c48608] at e_776_18
  have e_776_19 := arithEq_of_rows h (row := 776) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_776_19
  simp only [← c48609, ← c48610, k2, ← c48611] at e_776_19
  have hr := e_776_19
  simp only [e_776_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5415)) := by
  have c48612 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.1
  have c48613 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.1
  have c48614 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.1
  have c48615 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.1
  have c48616 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.1
  have c48617 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_0 := arithEq_of_rows h (row := 777) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_0
  simp only [← c48612, ← c48613, ← c48614] at e_777_0
  have e_777_1 := arithEq_of_rows h (row := 777) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_1
  simp only [← c48615, ← c48616, k2, ← c48617] at e_777_1
  have hr := e_777_1
  simp only [e_777_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5416)) := by
  have c48618 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48619 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48620 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48621 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48622 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48623 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_2 := arithEq_of_rows h (row := 777) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_2
  simp only [← c48618, ← c48619, ← c48620] at e_777_2
  have e_777_3 := arithEq_of_rows h (row := 777) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_3
  simp only [← c48621, ← c48622, k2, ← c48623] at e_777_3
  have hr := e_777_3
  simp only [e_777_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5417)) := by
  have c48624 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48625 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48626 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48627 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48628 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48629 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_4 := arithEq_of_rows h (row := 777) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_4
  simp only [← c48624, ← c48625, ← c48626] at e_777_4
  have e_777_5 := arithEq_of_rows h (row := 777) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_5
  simp only [← c48627, ← c48628, k2, ← c48629] at e_777_5
  have hr := e_777_5
  simp only [e_777_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5418)) := by
  have c48630 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48631 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48632 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48633 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48634 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48635 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_6 := arithEq_of_rows h (row := 777) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_6
  simp only [← c48630, ← c48631, ← c48632] at e_777_6
  have e_777_7 := arithEq_of_rows h (row := 777) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_7
  simp only [← c48633, ← c48634, k2, ← c48635] at e_777_7
  have hr := e_777_7
  simp only [e_777_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5419)) := by
  have c48636 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48637 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48638 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48639 := (publicBatchWrapper89_copies1519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48640 := (publicBatchWrapper89_copies1520 a h).1
  have c48641 := (publicBatchWrapper89_copies1520 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_8 := arithEq_of_rows h (row := 777) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_8
  simp only [← c48636, ← c48637, ← c48638] at e_777_8
  have e_777_9 := arithEq_of_rows h (row := 777) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_9
  simp only [← c48639, ← c48640, k2, ← c48641] at e_777_9
  have hr := e_777_9
  simp only [e_777_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5420)) := by
  have c48642 := (publicBatchWrapper89_copies1520 a h).2.2.1
  have c48643 := (publicBatchWrapper89_copies1520 a h).2.2.2.1
  have c48644 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.1
  have c48645 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.1
  have c48646 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.1
  have c48647 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_10 := arithEq_of_rows h (row := 777) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_10
  simp only [← c48642, ← c48643, ← c48644] at e_777_10
  have e_777_11 := arithEq_of_rows h (row := 777) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_11
  simp only [← c48645, ← c48646, k2, ← c48647] at e_777_11
  have hr := e_777_11
  simp only [e_777_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5421)) := by
  have c48648 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.1
  have c48649 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.1
  have c48650 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48651 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48652 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48653 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_12 := arithEq_of_rows h (row := 777) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_12
  simp only [← c48648, ← c48649, ← c48650] at e_777_12
  have e_777_13 := arithEq_of_rows h (row := 777) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_13
  simp only [← c48651, ← c48652, k2, ← c48653] at e_777_13
  have hr := e_777_13
  simp only [e_777_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5422)) := by
  have c48654 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48655 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48656 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48657 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48658 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48659 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_14 := arithEq_of_rows h (row := 777) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_14
  simp only [← c48654, ← c48655, ← c48656] at e_777_14
  have e_777_15 := arithEq_of_rows h (row := 777) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_15
  simp only [← c48657, ← c48658, k2, ← c48659] at e_777_15
  have hr := e_777_15
  simp only [e_777_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5423)) := by
  have c48660 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48661 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48662 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48663 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48664 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48665 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_16 := arithEq_of_rows h (row := 777) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_16
  simp only [← c48660, ← c48661, ← c48662] at e_777_16
  have e_777_17 := arithEq_of_rows h (row := 777) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_17
  simp only [← c48663, ← c48664, k2, ← c48665] at e_777_17
  have hr := e_777_17
  simp only [e_777_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 777 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5424)) := by
  have c48666 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48667 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48668 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48669 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48670 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48671 := (publicBatchWrapper89_copies1520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_777_18 := arithEq_of_rows h (row := 777) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_18
  simp only [← c48666, ← c48667, ← c48668] at e_777_18
  have e_777_19 := arithEq_of_rows h (row := 777) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_777_19
  simp only [← c48669, ← c48670, k2, ← c48671] at e_777_19
  have hr := e_777_19
  simp only [e_777_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5425)) := by
  have c48672 := (publicBatchWrapper89_copies1521 a h).1
  have c48673 := (publicBatchWrapper89_copies1521 a h).2.1
  have c48674 := (publicBatchWrapper89_copies1521 a h).2.2.1
  have c48675 := (publicBatchWrapper89_copies1521 a h).2.2.2.1
  have c48676 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.1
  have c48677 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_0 := arithEq_of_rows h (row := 778) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_0
  simp only [← c48672, ← c48673, ← c48674] at e_778_0
  have e_778_1 := arithEq_of_rows h (row := 778) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_1
  simp only [← c48675, ← c48676, k2, ← c48677] at e_778_1
  have hr := e_778_1
  simp only [e_778_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5426)) := by
  have c48678 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.1
  have c48679 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.1
  have c48680 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.1
  have c48681 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.1
  have c48682 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48683 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_2 := arithEq_of_rows h (row := 778) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_2
  simp only [← c48678, ← c48679, ← c48680] at e_778_2
  have e_778_3 := arithEq_of_rows h (row := 778) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_3
  simp only [← c48681, ← c48682, k2, ← c48683] at e_778_3
  have hr := e_778_3
  simp only [e_778_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5427)) := by
  have c48684 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48685 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48686 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48687 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48688 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48689 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_4 := arithEq_of_rows h (row := 778) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_4
  simp only [← c48684, ← c48685, ← c48686] at e_778_4
  have e_778_5 := arithEq_of_rows h (row := 778) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_5
  simp only [← c48687, ← c48688, k2, ← c48689] at e_778_5
  have hr := e_778_5
  simp only [e_778_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5428)) := by
  have c48690 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48691 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48692 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48693 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48694 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48695 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_6 := arithEq_of_rows h (row := 778) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_6
  simp only [← c48690, ← c48691, ← c48692] at e_778_6
  have e_778_7 := arithEq_of_rows h (row := 778) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_7
  simp only [← c48693, ← c48694, k2, ← c48695] at e_778_7
  have hr := e_778_7
  simp only [e_778_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5429)) := by
  have c48696 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48697 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48698 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48699 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48700 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48701 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_8 := arithEq_of_rows h (row := 778) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_8
  simp only [← c48696, ← c48697, ← c48698] at e_778_8
  have e_778_9 := arithEq_of_rows h (row := 778) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_9
  simp only [← c48699, ← c48700, k2, ← c48701] at e_778_9
  have hr := e_778_9
  simp only [e_778_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5430)) := by
  have c48702 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48703 := (publicBatchWrapper89_copies1521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48704 := (publicBatchWrapper89_copies1522 a h).1
  have c48705 := (publicBatchWrapper89_copies1522 a h).2.1
  have c48706 := (publicBatchWrapper89_copies1522 a h).2.2.1
  have c48707 := (publicBatchWrapper89_copies1522 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_10 := arithEq_of_rows h (row := 778) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_10
  simp only [← c48702, ← c48703, ← c48704] at e_778_10
  have e_778_11 := arithEq_of_rows h (row := 778) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_11
  simp only [← c48705, ← c48706, k2, ← c48707] at e_778_11
  have hr := e_778_11
  simp only [e_778_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5431)) := by
  have c48708 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.1
  have c48709 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.1
  have c48710 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.1
  have c48711 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.1
  have c48712 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.1
  have c48713 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_12 := arithEq_of_rows h (row := 778) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_12
  simp only [← c48708, ← c48709, ← c48710] at e_778_12
  have e_778_13 := arithEq_of_rows h (row := 778) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_13
  simp only [← c48711, ← c48712, k2, ← c48713] at e_778_13
  have hr := e_778_13
  simp only [e_778_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5432)) := by
  have c48714 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48715 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48716 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48717 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48718 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48719 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_14 := arithEq_of_rows h (row := 778) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_14
  simp only [← c48714, ← c48715, ← c48716] at e_778_14
  have e_778_15 := arithEq_of_rows h (row := 778) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_15
  simp only [← c48717, ← c48718, k2, ← c48719] at e_778_15
  have hr := e_778_15
  simp only [e_778_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5433)) := by
  have c48720 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48721 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48722 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48723 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48724 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48725 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_16 := arithEq_of_rows h (row := 778) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_16
  simp only [← c48720, ← c48721, ← c48722] at e_778_16
  have e_778_17 := arithEq_of_rows h (row := 778) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_17
  simp only [← c48723, ← c48724, k2, ← c48725] at e_778_17
  have hr := e_778_17
  simp only [e_778_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 778 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5434)) := by
  have c48726 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48727 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48728 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48729 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48730 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48731 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_778_18 := arithEq_of_rows h (row := 778) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_18
  simp only [← c48726, ← c48727, ← c48728] at e_778_18
  have e_778_19 := arithEq_of_rows h (row := 778) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_778_19
  simp only [← c48729, ← c48730, k2, ← c48731] at e_778_19
  have hr := e_778_19
  simp only [e_778_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5435)) := by
  have c48732 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48733 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48734 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48735 := (publicBatchWrapper89_copies1522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48736 := (publicBatchWrapper89_copies1523 a h).1
  have c48737 := (publicBatchWrapper89_copies1523 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_0 := arithEq_of_rows h (row := 779) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_0
  simp only [← c48732, ← c48733, ← c48734] at e_779_0
  have e_779_1 := arithEq_of_rows h (row := 779) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_1
  simp only [← c48735, ← c48736, k2, ← c48737] at e_779_1
  have hr := e_779_1
  simp only [e_779_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5436)) := by
  have c48738 := (publicBatchWrapper89_copies1523 a h).2.2.1
  have c48739 := (publicBatchWrapper89_copies1523 a h).2.2.2.1
  have c48740 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.1
  have c48741 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.1
  have c48742 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.1
  have c48743 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_2 := arithEq_of_rows h (row := 779) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_2
  simp only [← c48738, ← c48739, ← c48740] at e_779_2
  have e_779_3 := arithEq_of_rows h (row := 779) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_3
  simp only [← c48741, ← c48742, k2, ← c48743] at e_779_3
  have hr := e_779_3
  simp only [e_779_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5437)) := by
  have c48744 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.1
  have c48745 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.1
  have c48746 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48747 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48748 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48749 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_4 := arithEq_of_rows h (row := 779) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_4
  simp only [← c48744, ← c48745, ← c48746] at e_779_4
  have e_779_5 := arithEq_of_rows h (row := 779) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_5
  simp only [← c48747, ← c48748, k2, ← c48749] at e_779_5
  have hr := e_779_5
  simp only [e_779_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5438)) := by
  have c48750 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48751 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48752 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48753 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48754 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48755 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_6 := arithEq_of_rows h (row := 779) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_6
  simp only [← c48750, ← c48751, ← c48752] at e_779_6
  have e_779_7 := arithEq_of_rows h (row := 779) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_7
  simp only [← c48753, ← c48754, k2, ← c48755] at e_779_7
  have hr := e_779_7
  simp only [e_779_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5439)) := by
  have c48756 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48757 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48758 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48759 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48760 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48761 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_8 := arithEq_of_rows h (row := 779) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_8
  simp only [← c48756, ← c48757, ← c48758] at e_779_8
  have e_779_9 := arithEq_of_rows h (row := 779) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_9
  simp only [← c48759, ← c48760, k2, ← c48761] at e_779_9
  have hr := e_779_9
  simp only [e_779_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5440)) := by
  have c48762 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48763 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48764 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48765 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48766 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48767 := (publicBatchWrapper89_copies1523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_10 := arithEq_of_rows h (row := 779) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_10
  simp only [← c48762, ← c48763, ← c48764] at e_779_10
  have e_779_11 := arithEq_of_rows h (row := 779) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_11
  simp only [← c48765, ← c48766, k2, ← c48767] at e_779_11
  have hr := e_779_11
  simp only [e_779_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5441)) := by
  have c48768 := (publicBatchWrapper89_copies1524 a h).1
  have c48769 := (publicBatchWrapper89_copies1524 a h).2.1
  have c48770 := (publicBatchWrapper89_copies1524 a h).2.2.1
  have c48771 := (publicBatchWrapper89_copies1524 a h).2.2.2.1
  have c48772 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.1
  have c48773 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_12 := arithEq_of_rows h (row := 779) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_12
  simp only [← c48768, ← c48769, ← c48770] at e_779_12
  have e_779_13 := arithEq_of_rows h (row := 779) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_13
  simp only [← c48771, ← c48772, k2, ← c48773] at e_779_13
  have hr := e_779_13
  simp only [e_779_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5442)) := by
  have c48774 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.1
  have c48775 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.1
  have c48776 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.1
  have c48777 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.1
  have c48778 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48779 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_14 := arithEq_of_rows h (row := 779) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_14
  simp only [← c48774, ← c48775, ← c48776] at e_779_14
  have e_779_15 := arithEq_of_rows h (row := 779) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_15
  simp only [← c48777, ← c48778, k2, ← c48779] at e_779_15
  have hr := e_779_15
  simp only [e_779_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5443)) := by
  have c48780 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48781 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48782 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48783 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48784 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48785 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_16 := arithEq_of_rows h (row := 779) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_16
  simp only [← c48780, ← c48781, ← c48782] at e_779_16
  have e_779_17 := arithEq_of_rows h (row := 779) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_17
  simp only [← c48783, ← c48784, k2, ← c48785] at e_779_17
  have hr := e_779_17
  simp only [e_779_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 779 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5444)) := by
  have c48786 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48787 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48788 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48789 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48790 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48791 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_779_18 := arithEq_of_rows h (row := 779) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_18
  simp only [← c48786, ← c48787, ← c48788] at e_779_18
  have e_779_19 := arithEq_of_rows h (row := 779) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_779_19
  simp only [← c48789, ← c48790, k2, ← c48791] at e_779_19
  have hr := e_779_19
  simp only [e_779_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5445)) := by
  have c48792 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48793 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48794 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48795 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48796 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48797 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_0 := arithEq_of_rows h (row := 780) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_0
  simp only [← c48792, ← c48793, ← c48794] at e_780_0
  have e_780_1 := arithEq_of_rows h (row := 780) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_1
  simp only [← c48795, ← c48796, k2, ← c48797] at e_780_1
  have hr := e_780_1
  simp only [e_780_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5446)) := by
  have c48798 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48799 := (publicBatchWrapper89_copies1524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48800 := (publicBatchWrapper89_copies1525 a h).1
  have c48801 := (publicBatchWrapper89_copies1525 a h).2.1
  have c48802 := (publicBatchWrapper89_copies1525 a h).2.2.1
  have c48803 := (publicBatchWrapper89_copies1525 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_2 := arithEq_of_rows h (row := 780) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_2
  simp only [← c48798, ← c48799, ← c48800] at e_780_2
  have e_780_3 := arithEq_of_rows h (row := 780) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_3
  simp only [← c48801, ← c48802, k2, ← c48803] at e_780_3
  have hr := e_780_3
  simp only [e_780_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5447)) := by
  have c48804 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.1
  have c48805 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.1
  have c48806 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.1
  have c48807 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.1
  have c48808 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.1
  have c48809 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_4 := arithEq_of_rows h (row := 780) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_4
  simp only [← c48804, ← c48805, ← c48806] at e_780_4
  have e_780_5 := arithEq_of_rows h (row := 780) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_5
  simp only [← c48807, ← c48808, k2, ← c48809] at e_780_5
  have hr := e_780_5
  simp only [e_780_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5448)) := by
  have c48810 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48811 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48812 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48813 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48814 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48815 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_6 := arithEq_of_rows h (row := 780) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_6
  simp only [← c48810, ← c48811, ← c48812] at e_780_6
  have e_780_7 := arithEq_of_rows h (row := 780) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_7
  simp only [← c48813, ← c48814, k2, ← c48815] at e_780_7
  have hr := e_780_7
  simp only [e_780_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5449)) := by
  have c48816 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48817 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48818 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48819 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48820 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48821 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_8 := arithEq_of_rows h (row := 780) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_8
  simp only [← c48816, ← c48817, ← c48818] at e_780_8
  have e_780_9 := arithEq_of_rows h (row := 780) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_9
  simp only [← c48819, ← c48820, k2, ← c48821] at e_780_9
  have hr := e_780_9
  simp only [e_780_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5450)) := by
  have c48822 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48823 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48824 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48825 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48826 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48827 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_10 := arithEq_of_rows h (row := 780) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_10
  simp only [← c48822, ← c48823, ← c48824] at e_780_10
  have e_780_11 := arithEq_of_rows h (row := 780) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_11
  simp only [← c48825, ← c48826, k2, ← c48827] at e_780_11
  have hr := e_780_11
  simp only [e_780_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5451)) := by
  have c48828 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48829 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48830 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48831 := (publicBatchWrapper89_copies1525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48832 := (publicBatchWrapper89_copies1526 a h).1
  have c48833 := (publicBatchWrapper89_copies1526 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_12 := arithEq_of_rows h (row := 780) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_12
  simp only [← c48828, ← c48829, ← c48830] at e_780_12
  have e_780_13 := arithEq_of_rows h (row := 780) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_13
  simp only [← c48831, ← c48832, k2, ← c48833] at e_780_13
  have hr := e_780_13
  simp only [e_780_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5452)) := by
  have c48834 := (publicBatchWrapper89_copies1526 a h).2.2.1
  have c48835 := (publicBatchWrapper89_copies1526 a h).2.2.2.1
  have c48836 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.1
  have c48837 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.1
  have c48838 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.1
  have c48839 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_14 := arithEq_of_rows h (row := 780) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_14
  simp only [← c48834, ← c48835, ← c48836] at e_780_14
  have e_780_15 := arithEq_of_rows h (row := 780) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_15
  simp only [← c48837, ← c48838, k2, ← c48839] at e_780_15
  have hr := e_780_15
  simp only [e_780_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5453)) := by
  have c48840 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.1
  have c48841 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.1
  have c48842 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48843 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48844 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48845 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_16 := arithEq_of_rows h (row := 780) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_16
  simp only [← c48840, ← c48841, ← c48842] at e_780_16
  have e_780_17 := arithEq_of_rows h (row := 780) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_17
  simp only [← c48843, ← c48844, k2, ← c48845] at e_780_17
  have hr := e_780_17
  simp only [e_780_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 780 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5454)) := by
  have c48846 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48847 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48848 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48849 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48850 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48851 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_780_18 := arithEq_of_rows h (row := 780) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_18
  simp only [← c48846, ← c48847, ← c48848] at e_780_18
  have e_780_19 := arithEq_of_rows h (row := 780) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_780_19
  simp only [← c48849, ← c48850, k2, ← c48851] at e_780_19
  have hr := e_780_19
  simp only [e_780_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5455)) := by
  have c48852 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48853 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48854 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48855 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48856 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48857 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_0 := arithEq_of_rows h (row := 781) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_0
  simp only [← c48852, ← c48853, ← c48854] at e_781_0
  have e_781_1 := arithEq_of_rows h (row := 781) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_1
  simp only [← c48855, ← c48856, k2, ← c48857] at e_781_1
  have hr := e_781_1
  simp only [e_781_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5456)) := by
  have c48858 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48859 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48860 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48861 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48862 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48863 := (publicBatchWrapper89_copies1526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_2 := arithEq_of_rows h (row := 781) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_2
  simp only [← c48858, ← c48859, ← c48860] at e_781_2
  have e_781_3 := arithEq_of_rows h (row := 781) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_3
  simp only [← c48861, ← c48862, k2, ← c48863] at e_781_3
  have hr := e_781_3
  simp only [e_781_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5457)) := by
  have c48864 := (publicBatchWrapper89_copies1527 a h).1
  have c48865 := (publicBatchWrapper89_copies1527 a h).2.1
  have c48866 := (publicBatchWrapper89_copies1527 a h).2.2.1
  have c48867 := (publicBatchWrapper89_copies1527 a h).2.2.2.1
  have c48868 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.1
  have c48869 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_4 := arithEq_of_rows h (row := 781) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_4
  simp only [← c48864, ← c48865, ← c48866] at e_781_4
  have e_781_5 := arithEq_of_rows h (row := 781) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_5
  simp only [← c48867, ← c48868, k2, ← c48869] at e_781_5
  have hr := e_781_5
  simp only [e_781_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5458)) := by
  have c48870 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.1
  have c48871 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.1
  have c48872 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.1
  have c48873 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.1
  have c48874 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48875 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_6 := arithEq_of_rows h (row := 781) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_6
  simp only [← c48870, ← c48871, ← c48872] at e_781_6
  have e_781_7 := arithEq_of_rows h (row := 781) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_7
  simp only [← c48873, ← c48874, k2, ← c48875] at e_781_7
  have hr := e_781_7
  simp only [e_781_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5459)) := by
  have c48876 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48877 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48878 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48879 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48880 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48881 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_8 := arithEq_of_rows h (row := 781) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_8
  simp only [← c48876, ← c48877, ← c48878] at e_781_8
  have e_781_9 := arithEq_of_rows h (row := 781) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_9
  simp only [← c48879, ← c48880, k2, ← c48881] at e_781_9
  have hr := e_781_9
  simp only [e_781_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5460)) := by
  have c48882 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48883 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48884 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48885 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48886 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48887 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_10 := arithEq_of_rows h (row := 781) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_10
  simp only [← c48882, ← c48883, ← c48884] at e_781_10
  have e_781_11 := arithEq_of_rows h (row := 781) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_11
  simp only [← c48885, ← c48886, k2, ← c48887] at e_781_11
  have hr := e_781_11
  simp only [e_781_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5461)) := by
  have c48888 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48889 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48890 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48891 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48892 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48893 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_12 := arithEq_of_rows h (row := 781) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_12
  simp only [← c48888, ← c48889, ← c48890] at e_781_12
  have e_781_13 := arithEq_of_rows h (row := 781) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_13
  simp only [← c48891, ← c48892, k2, ← c48893] at e_781_13
  have hr := e_781_13
  simp only [e_781_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5462)) := by
  have c48894 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48895 := (publicBatchWrapper89_copies1527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48896 := (publicBatchWrapper89_copies1528 a h).1
  have c48897 := (publicBatchWrapper89_copies1528 a h).2.1
  have c48898 := (publicBatchWrapper89_copies1528 a h).2.2.1
  have c48899 := (publicBatchWrapper89_copies1528 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_14 := arithEq_of_rows h (row := 781) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_14
  simp only [← c48894, ← c48895, ← c48896] at e_781_14
  have e_781_15 := arithEq_of_rows h (row := 781) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_15
  simp only [← c48897, ← c48898, k2, ← c48899] at e_781_15
  have hr := e_781_15
  simp only [e_781_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5463)) := by
  have c48900 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.1
  have c48901 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.1
  have c48902 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.1
  have c48903 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.1
  have c48904 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.1
  have c48905 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_16 := arithEq_of_rows h (row := 781) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_16
  simp only [← c48900, ← c48901, ← c48902] at e_781_16
  have e_781_17 := arithEq_of_rows h (row := 781) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_17
  simp only [← c48903, ← c48904, k2, ← c48905] at e_781_17
  have hr := e_781_17
  simp only [e_781_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 781 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5464)) := by
  have c48906 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48907 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48908 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48909 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48910 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48911 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_781_18 := arithEq_of_rows h (row := 781) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_18
  simp only [← c48906, ← c48907, ← c48908] at e_781_18
  have e_781_19 := arithEq_of_rows h (row := 781) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_781_19
  simp only [← c48909, ← c48910, k2, ← c48911] at e_781_19
  have hr := e_781_19
  simp only [e_781_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5465)) := by
  have c48912 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48913 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48914 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48915 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48916 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48917 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_0 := arithEq_of_rows h (row := 782) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_0
  simp only [← c48912, ← c48913, ← c48914] at e_782_0
  have e_782_1 := arithEq_of_rows h (row := 782) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_1
  simp only [← c48915, ← c48916, k2, ← c48917] at e_782_1
  have hr := e_782_1
  simp only [e_782_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5466)) := by
  have c48918 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48919 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48920 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48921 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48922 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48923 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_2 := arithEq_of_rows h (row := 782) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_2
  simp only [← c48918, ← c48919, ← c48920] at e_782_2
  have e_782_3 := arithEq_of_rows h (row := 782) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_3
  simp only [← c48921, ← c48922, k2, ← c48923] at e_782_3
  have hr := e_782_3
  simp only [e_782_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5467)) := by
  have c48924 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48925 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48926 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48927 := (publicBatchWrapper89_copies1528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48928 := (publicBatchWrapper89_copies1529 a h).1
  have c48929 := (publicBatchWrapper89_copies1529 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_4 := arithEq_of_rows h (row := 782) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_4
  simp only [← c48924, ← c48925, ← c48926] at e_782_4
  have e_782_5 := arithEq_of_rows h (row := 782) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_5
  simp only [← c48927, ← c48928, k2, ← c48929] at e_782_5
  have hr := e_782_5
  simp only [e_782_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5468)) := by
  have c48930 := (publicBatchWrapper89_copies1529 a h).2.2.1
  have c48931 := (publicBatchWrapper89_copies1529 a h).2.2.2.1
  have c48932 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.1
  have c48933 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.1
  have c48934 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.1
  have c48935 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_6 := arithEq_of_rows h (row := 782) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_6
  simp only [← c48930, ← c48931, ← c48932] at e_782_6
  have e_782_7 := arithEq_of_rows h (row := 782) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_7
  simp only [← c48933, ← c48934, k2, ← c48935] at e_782_7
  have hr := e_782_7
  simp only [e_782_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5469)) := by
  have c48936 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.1
  have c48937 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.1
  have c48938 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48939 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c48940 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48941 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_8 := arithEq_of_rows h (row := 782) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_8
  simp only [← c48936, ← c48937, ← c48938] at e_782_8
  have e_782_9 := arithEq_of_rows h (row := 782) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_9
  simp only [← c48939, ← c48940, k2, ← c48941] at e_782_9
  have hr := e_782_9
  simp only [e_782_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5470)) := by
  have c48942 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48943 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48944 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48945 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48946 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48947 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_10 := arithEq_of_rows h (row := 782) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_10
  simp only [← c48942, ← c48943, ← c48944] at e_782_10
  have e_782_11 := arithEq_of_rows h (row := 782) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_11
  simp only [← c48945, ← c48946, k2, ← c48947] at e_782_11
  have hr := e_782_11
  simp only [e_782_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5471)) := by
  have c48948 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48949 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48950 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48951 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48952 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48953 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_12 := arithEq_of_rows h (row := 782) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_12
  simp only [← c48948, ← c48949, ← c48950] at e_782_12
  have e_782_13 := arithEq_of_rows h (row := 782) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_13
  simp only [← c48951, ← c48952, k2, ← c48953] at e_782_13
  have hr := e_782_13
  simp only [e_782_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5472)) := by
  have c48954 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48955 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48956 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48957 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48958 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48959 := (publicBatchWrapper89_copies1529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_14 := arithEq_of_rows h (row := 782) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_14
  simp only [← c48954, ← c48955, ← c48956] at e_782_14
  have e_782_15 := arithEq_of_rows h (row := 782) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_15
  simp only [← c48957, ← c48958, k2, ← c48959] at e_782_15
  have hr := e_782_15
  simp only [e_782_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5473)) := by
  have c48960 := (publicBatchWrapper89_copies1530 a h).1
  have c48961 := (publicBatchWrapper89_copies1530 a h).2.1
  have c48962 := (publicBatchWrapper89_copies1530 a h).2.2.1
  have c48963 := (publicBatchWrapper89_copies1530 a h).2.2.2.1
  have c48964 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.1
  have c48965 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_16 := arithEq_of_rows h (row := 782) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_16
  simp only [← c48960, ← c48961, ← c48962] at e_782_16
  have e_782_17 := arithEq_of_rows h (row := 782) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_17
  simp only [← c48963, ← c48964, k2, ← c48965] at e_782_17
  have hr := e_782_17
  simp only [e_782_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 782 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5474)) := by
  have c48966 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.1
  have c48967 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.1
  have c48968 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.1
  have c48969 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.1
  have c48970 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c48971 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_782_18 := arithEq_of_rows h (row := 782) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_18
  simp only [← c48966, ← c48967, ← c48968] at e_782_18
  have e_782_19 := arithEq_of_rows h (row := 782) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_782_19
  simp only [← c48969, ← c48970, k2, ← c48971] at e_782_19
  have hr := e_782_19
  simp only [e_782_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5475)) := by
  have c48972 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48973 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48974 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48975 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48976 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48977 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_0 := arithEq_of_rows h (row := 783) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_0
  simp only [← c48972, ← c48973, ← c48974] at e_783_0
  have e_783_1 := arithEq_of_rows h (row := 783) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_1
  simp only [← c48975, ← c48976, k2, ← c48977] at e_783_1
  have hr := e_783_1
  simp only [e_783_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5476)) := by
  have c48978 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48979 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48980 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48981 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48982 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48983 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_2 := arithEq_of_rows h (row := 783) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_2
  simp only [← c48978, ← c48979, ← c48980] at e_783_2
  have e_783_3 := arithEq_of_rows h (row := 783) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_3
  simp only [← c48981, ← c48982, k2, ← c48983] at e_783_3
  have hr := e_783_3
  simp only [e_783_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5477)) := by
  have c48984 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48985 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48986 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48987 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48988 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48989 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_4 := arithEq_of_rows h (row := 783) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_4
  simp only [← c48984, ← c48985, ← c48986] at e_783_4
  have e_783_5 := arithEq_of_rows h (row := 783) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_5
  simp only [← c48987, ← c48988, k2, ← c48989] at e_783_5
  have hr := e_783_5
  simp only [e_783_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5478)) := by
  have c48990 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48991 := (publicBatchWrapper89_copies1530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c48992 := (publicBatchWrapper89_copies1531 a h).1
  have c48993 := (publicBatchWrapper89_copies1531 a h).2.1
  have c48994 := (publicBatchWrapper89_copies1531 a h).2.2.1
  have c48995 := (publicBatchWrapper89_copies1531 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_6 := arithEq_of_rows h (row := 783) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_6
  simp only [← c48990, ← c48991, ← c48992] at e_783_6
  have e_783_7 := arithEq_of_rows h (row := 783) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_7
  simp only [← c48993, ← c48994, k2, ← c48995] at e_783_7
  have hr := e_783_7
  simp only [e_783_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5479)) := by
  have c48996 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.1
  have c48997 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.1
  have c48998 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.1
  have c48999 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.1
  have c49000 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.1
  have c49001 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_8 := arithEq_of_rows h (row := 783) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_8
  simp only [← c48996, ← c48997, ← c48998] at e_783_8
  have e_783_9 := arithEq_of_rows h (row := 783) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_9
  simp only [← c48999, ← c49000, k2, ← c49001] at e_783_9
  have hr := e_783_9
  simp only [e_783_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5480)) := by
  have c49002 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49003 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49004 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49005 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49006 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49007 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_10 := arithEq_of_rows h (row := 783) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_10
  simp only [← c49002, ← c49003, ← c49004] at e_783_10
  have e_783_11 := arithEq_of_rows h (row := 783) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_11
  simp only [← c49005, ← c49006, k2, ← c49007] at e_783_11
  have hr := e_783_11
  simp only [e_783_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5481)) := by
  have c49008 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49009 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49010 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49011 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49012 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49013 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_12 := arithEq_of_rows h (row := 783) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_12
  simp only [← c49008, ← c49009, ← c49010] at e_783_12
  have e_783_13 := arithEq_of_rows h (row := 783) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_13
  simp only [← c49011, ← c49012, k2, ← c49013] at e_783_13
  have hr := e_783_13
  simp only [e_783_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5482)) := by
  have c49014 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49015 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49016 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49017 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49018 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49019 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_14 := arithEq_of_rows h (row := 783) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_14
  simp only [← c49014, ← c49015, ← c49016] at e_783_14
  have e_783_15 := arithEq_of_rows h (row := 783) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_15
  simp only [← c49017, ← c49018, k2, ← c49019] at e_783_15
  have hr := e_783_15
  simp only [e_783_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5483)) := by
  have c49020 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49021 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49022 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49023 := (publicBatchWrapper89_copies1531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49024 := (publicBatchWrapper89_copies1532 a h).1
  have c49025 := (publicBatchWrapper89_copies1532 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_16 := arithEq_of_rows h (row := 783) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_16
  simp only [← c49020, ← c49021, ← c49022] at e_783_16
  have e_783_17 := arithEq_of_rows h (row := 783) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_17
  simp only [← c49023, ← c49024, k2, ← c49025] at e_783_17
  have hr := e_783_17
  simp only [e_783_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 783 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5484)) := by
  have c49026 := (publicBatchWrapper89_copies1532 a h).2.2.1
  have c49027 := (publicBatchWrapper89_copies1532 a h).2.2.2.1
  have c49028 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.1
  have c49029 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.1
  have c49030 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.1
  have c49031 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_783_18 := arithEq_of_rows h (row := 783) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_18
  simp only [← c49026, ← c49027, ← c49028] at e_783_18
  have e_783_19 := arithEq_of_rows h (row := 783) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_783_19
  simp only [← c49029, ← c49030, k2, ← c49031] at e_783_19
  have hr := e_783_19
  simp only [e_783_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5485)) := by
  have c49032 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.1
  have c49033 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.1
  have c49034 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49035 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49036 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49037 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_0 := arithEq_of_rows h (row := 784) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_0
  simp only [← c49032, ← c49033, ← c49034] at e_784_0
  have e_784_1 := arithEq_of_rows h (row := 784) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_1
  simp only [← c49035, ← c49036, k2, ← c49037] at e_784_1
  have hr := e_784_1
  simp only [e_784_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5486)) := by
  have c49038 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49039 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49040 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49041 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49042 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49043 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_2 := arithEq_of_rows h (row := 784) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_2
  simp only [← c49038, ← c49039, ← c49040] at e_784_2
  have e_784_3 := arithEq_of_rows h (row := 784) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_3
  simp only [← c49041, ← c49042, k2, ← c49043] at e_784_3
  have hr := e_784_3
  simp only [e_784_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5487)) := by
  have c49044 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49045 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49046 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49047 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49048 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49049 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_4 := arithEq_of_rows h (row := 784) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_4
  simp only [← c49044, ← c49045, ← c49046] at e_784_4
  have e_784_5 := arithEq_of_rows h (row := 784) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_5
  simp only [← c49047, ← c49048, k2, ← c49049] at e_784_5
  have hr := e_784_5
  simp only [e_784_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5488)) := by
  have c49050 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49051 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49052 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49053 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49054 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49055 := (publicBatchWrapper89_copies1532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_6 := arithEq_of_rows h (row := 784) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_6
  simp only [← c49050, ← c49051, ← c49052] at e_784_6
  have e_784_7 := arithEq_of_rows h (row := 784) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_7
  simp only [← c49053, ← c49054, k2, ← c49055] at e_784_7
  have hr := e_784_7
  simp only [e_784_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5489)) := by
  have c49056 := (publicBatchWrapper89_copies1533 a h).1
  have c49057 := (publicBatchWrapper89_copies1533 a h).2.1
  have c49058 := (publicBatchWrapper89_copies1533 a h).2.2.1
  have c49059 := (publicBatchWrapper89_copies1533 a h).2.2.2.1
  have c49060 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.1
  have c49061 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_8 := arithEq_of_rows h (row := 784) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_8
  simp only [← c49056, ← c49057, ← c49058] at e_784_8
  have e_784_9 := arithEq_of_rows h (row := 784) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_9
  simp only [← c49059, ← c49060, k2, ← c49061] at e_784_9
  have hr := e_784_9
  simp only [e_784_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5490)) := by
  have c49062 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.1
  have c49063 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.1
  have c49064 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.1
  have c49065 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.1
  have c49066 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49067 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_10 := arithEq_of_rows h (row := 784) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_10
  simp only [← c49062, ← c49063, ← c49064] at e_784_10
  have e_784_11 := arithEq_of_rows h (row := 784) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_11
  simp only [← c49065, ← c49066, k2, ← c49067] at e_784_11
  have hr := e_784_11
  simp only [e_784_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5491)) := by
  have c49068 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49069 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49070 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49071 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49072 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49073 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_12 := arithEq_of_rows h (row := 784) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_12
  simp only [← c49068, ← c49069, ← c49070] at e_784_12
  have e_784_13 := arithEq_of_rows h (row := 784) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_13
  simp only [← c49071, ← c49072, k2, ← c49073] at e_784_13
  have hr := e_784_13
  simp only [e_784_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5492)) := by
  have c49074 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49075 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49076 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49077 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49078 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49079 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_14 := arithEq_of_rows h (row := 784) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_14
  simp only [← c49074, ← c49075, ← c49076] at e_784_14
  have e_784_15 := arithEq_of_rows h (row := 784) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_15
  simp only [← c49077, ← c49078, k2, ← c49079] at e_784_15
  have hr := e_784_15
  simp only [e_784_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5493)) := by
  have c49080 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49081 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49082 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49083 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49084 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49085 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_16 := arithEq_of_rows h (row := 784) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_16
  simp only [← c49080, ← c49081, ← c49082] at e_784_16
  have e_784_17 := arithEq_of_rows h (row := 784) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_17
  simp only [← c49083, ← c49084, k2, ← c49085] at e_784_17
  have hr := e_784_17
  simp only [e_784_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 784 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5494)) := by
  have c49086 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49087 := (publicBatchWrapper89_copies1533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49088 := (publicBatchWrapper89_copies1534 a h).1
  have c49089 := (publicBatchWrapper89_copies1534 a h).2.1
  have c49090 := (publicBatchWrapper89_copies1534 a h).2.2.1
  have c49091 := (publicBatchWrapper89_copies1534 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_784_18 := arithEq_of_rows h (row := 784) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_18
  simp only [← c49086, ← c49087, ← c49088] at e_784_18
  have e_784_19 := arithEq_of_rows h (row := 784) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_784_19
  simp only [← c49089, ← c49090, k2, ← c49091] at e_784_19
  have hr := e_784_19
  simp only [e_784_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5495)) := by
  have c49092 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.1
  have c49093 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.1
  have c49094 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.1
  have c49095 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.1
  have c49096 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.1
  have c49097 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_0 := arithEq_of_rows h (row := 785) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_0
  simp only [← c49092, ← c49093, ← c49094] at e_785_0
  have e_785_1 := arithEq_of_rows h (row := 785) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_1
  simp only [← c49095, ← c49096, k2, ← c49097] at e_785_1
  have hr := e_785_1
  simp only [e_785_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5496)) := by
  have c49098 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49099 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49100 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49101 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49102 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49103 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_2 := arithEq_of_rows h (row := 785) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_2
  simp only [← c49098, ← c49099, ← c49100] at e_785_2
  have e_785_3 := arithEq_of_rows h (row := 785) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_3
  simp only [← c49101, ← c49102, k2, ← c49103] at e_785_3
  have hr := e_785_3
  simp only [e_785_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5497)) := by
  have c49104 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49105 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49106 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49107 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49108 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49109 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_4 := arithEq_of_rows h (row := 785) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_4
  simp only [← c49104, ← c49105, ← c49106] at e_785_4
  have e_785_5 := arithEq_of_rows h (row := 785) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_5
  simp only [← c49107, ← c49108, k2, ← c49109] at e_785_5
  have hr := e_785_5
  simp only [e_785_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5498)) := by
  have c49110 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49111 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49112 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49113 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49114 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49115 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_6 := arithEq_of_rows h (row := 785) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_6
  simp only [← c49110, ← c49111, ← c49112] at e_785_6
  have e_785_7 := arithEq_of_rows h (row := 785) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_7
  simp only [← c49113, ← c49114, k2, ← c49115] at e_785_7
  have hr := e_785_7
  simp only [e_785_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5499)) := by
  have c49116 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49117 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49118 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49119 := (publicBatchWrapper89_copies1534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49120 := (publicBatchWrapper89_copies1535 a h).1
  have c49121 := (publicBatchWrapper89_copies1535 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_8 := arithEq_of_rows h (row := 785) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_8
  simp only [← c49116, ← c49117, ← c49118] at e_785_8
  have e_785_9 := arithEq_of_rows h (row := 785) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_9
  simp only [← c49119, ← c49120, k2, ← c49121] at e_785_9
  have hr := e_785_9
  simp only [e_785_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5500)) := by
  have c49122 := (publicBatchWrapper89_copies1535 a h).2.2.1
  have c49123 := (publicBatchWrapper89_copies1535 a h).2.2.2.1
  have c49124 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.1
  have c49125 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.1
  have c49126 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.1
  have c49127 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_10 := arithEq_of_rows h (row := 785) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_10
  simp only [← c49122, ← c49123, ← c49124] at e_785_10
  have e_785_11 := arithEq_of_rows h (row := 785) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_11
  simp only [← c49125, ← c49126, k2, ← c49127] at e_785_11
  have hr := e_785_11
  simp only [e_785_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5501)) := by
  have c49128 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.1
  have c49129 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.1
  have c49130 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49131 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49132 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49133 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_12 := arithEq_of_rows h (row := 785) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_12
  simp only [← c49128, ← c49129, ← c49130] at e_785_12
  have e_785_13 := arithEq_of_rows h (row := 785) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_13
  simp only [← c49131, ← c49132, k2, ← c49133] at e_785_13
  have hr := e_785_13
  simp only [e_785_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5502)) := by
  have c49134 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49135 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49136 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49137 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49138 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49139 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_14 := arithEq_of_rows h (row := 785) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_14
  simp only [← c49134, ← c49135, ← c49136] at e_785_14
  have e_785_15 := arithEq_of_rows h (row := 785) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_15
  simp only [← c49137, ← c49138, k2, ← c49139] at e_785_15
  have hr := e_785_15
  simp only [e_785_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5503)) := by
  have c49140 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49141 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49142 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49143 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49144 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49145 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_16 := arithEq_of_rows h (row := 785) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_16
  simp only [← c49140, ← c49141, ← c49142] at e_785_16
  have e_785_17 := arithEq_of_rows h (row := 785) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_17
  simp only [← c49143, ← c49144, k2, ← c49145] at e_785_17
  have hr := e_785_17
  simp only [e_785_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 785 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5504)) := by
  have c49146 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49147 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49148 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49149 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49150 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49151 := (publicBatchWrapper89_copies1535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_785_18 := arithEq_of_rows h (row := 785) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_18
  simp only [← c49146, ← c49147, ← c49148] at e_785_18
  have e_785_19 := arithEq_of_rows h (row := 785) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_785_19
  simp only [← c49149, ← c49150, k2, ← c49151] at e_785_19
  have hr := e_785_19
  simp only [e_785_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5505)) := by
  have c49152 := (publicBatchWrapper89_copies1536 a h).1
  have c49153 := (publicBatchWrapper89_copies1536 a h).2.1
  have c49154 := (publicBatchWrapper89_copies1536 a h).2.2.1
  have c49155 := (publicBatchWrapper89_copies1536 a h).2.2.2.1
  have c49156 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.1
  have c49157 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_0 := arithEq_of_rows h (row := 786) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_0
  simp only [← c49152, ← c49153, ← c49154] at e_786_0
  have e_786_1 := arithEq_of_rows h (row := 786) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_1
  simp only [← c49155, ← c49156, k2, ← c49157] at e_786_1
  have hr := e_786_1
  simp only [e_786_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5506)) := by
  have c49158 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.1
  have c49159 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.1
  have c49160 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.1
  have c49161 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.1
  have c49162 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49163 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_2 := arithEq_of_rows h (row := 786) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_2
  simp only [← c49158, ← c49159, ← c49160] at e_786_2
  have e_786_3 := arithEq_of_rows h (row := 786) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_3
  simp only [← c49161, ← c49162, k2, ← c49163] at e_786_3
  have hr := e_786_3
  simp only [e_786_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5507)) := by
  have c49164 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49165 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49166 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49167 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49168 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49169 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_4 := arithEq_of_rows h (row := 786) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_4
  simp only [← c49164, ← c49165, ← c49166] at e_786_4
  have e_786_5 := arithEq_of_rows h (row := 786) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_5
  simp only [← c49167, ← c49168, k2, ← c49169] at e_786_5
  have hr := e_786_5
  simp only [e_786_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5508)) := by
  have c49170 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49171 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49172 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49173 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49174 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49175 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_6 := arithEq_of_rows h (row := 786) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_6
  simp only [← c49170, ← c49171, ← c49172] at e_786_6
  have e_786_7 := arithEq_of_rows h (row := 786) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_7
  simp only [← c49173, ← c49174, k2, ← c49175] at e_786_7
  have hr := e_786_7
  simp only [e_786_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5509)) := by
  have c49176 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49177 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49178 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49179 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49180 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49181 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_8 := arithEq_of_rows h (row := 786) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_8
  simp only [← c49176, ← c49177, ← c49178] at e_786_8
  have e_786_9 := arithEq_of_rows h (row := 786) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_9
  simp only [← c49179, ← c49180, k2, ← c49181] at e_786_9
  have hr := e_786_9
  simp only [e_786_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5510)) := by
  have c49182 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49183 := (publicBatchWrapper89_copies1536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49184 := (publicBatchWrapper89_copies1537 a h).1
  have c49185 := (publicBatchWrapper89_copies1537 a h).2.1
  have c49186 := (publicBatchWrapper89_copies1537 a h).2.2.1
  have c49187 := (publicBatchWrapper89_copies1537 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_10 := arithEq_of_rows h (row := 786) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_10
  simp only [← c49182, ← c49183, ← c49184] at e_786_10
  have e_786_11 := arithEq_of_rows h (row := 786) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_11
  simp only [← c49185, ← c49186, k2, ← c49187] at e_786_11
  have hr := e_786_11
  simp only [e_786_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5511)) := by
  have c49188 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.1
  have c49189 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.1
  have c49190 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.1
  have c49191 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.1
  have c49192 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.1
  have c49193 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_12 := arithEq_of_rows h (row := 786) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_12
  simp only [← c49188, ← c49189, ← c49190] at e_786_12
  have e_786_13 := arithEq_of_rows h (row := 786) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_13
  simp only [← c49191, ← c49192, k2, ← c49193] at e_786_13
  have hr := e_786_13
  simp only [e_786_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5512)) := by
  have c49194 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49195 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49196 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49197 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49198 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49199 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_14 := arithEq_of_rows h (row := 786) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_14
  simp only [← c49194, ← c49195, ← c49196] at e_786_14
  have e_786_15 := arithEq_of_rows h (row := 786) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_15
  simp only [← c49197, ← c49198, k2, ← c49199] at e_786_15
  have hr := e_786_15
  simp only [e_786_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5513)) := by
  have c49200 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49201 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49202 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49203 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49204 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49205 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_16 := arithEq_of_rows h (row := 786) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_16
  simp only [← c49200, ← c49201, ← c49202] at e_786_16
  have e_786_17 := arithEq_of_rows h (row := 786) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_17
  simp only [← c49203, ← c49204, k2, ← c49205] at e_786_17
  have hr := e_786_17
  simp only [e_786_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 786 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5514)) := by
  have c49206 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49207 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49208 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49209 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49210 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49211 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_786_18 := arithEq_of_rows h (row := 786) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_18
  simp only [← c49206, ← c49207, ← c49208] at e_786_18
  have e_786_19 := arithEq_of_rows h (row := 786) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_786_19
  simp only [← c49209, ← c49210, k2, ← c49211] at e_786_19
  have hr := e_786_19
  simp only [e_786_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5515)) := by
  have c49212 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49213 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49214 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49215 := (publicBatchWrapper89_copies1537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49216 := (publicBatchWrapper89_copies1538 a h).1
  have c49217 := (publicBatchWrapper89_copies1538 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_0 := arithEq_of_rows h (row := 787) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_0
  simp only [← c49212, ← c49213, ← c49214] at e_787_0
  have e_787_1 := arithEq_of_rows h (row := 787) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_1
  simp only [← c49215, ← c49216, k2, ← c49217] at e_787_1
  have hr := e_787_1
  simp only [e_787_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5516)) := by
  have c49218 := (publicBatchWrapper89_copies1538 a h).2.2.1
  have c49219 := (publicBatchWrapper89_copies1538 a h).2.2.2.1
  have c49220 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.1
  have c49221 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.1
  have c49222 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.1
  have c49223 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_2 := arithEq_of_rows h (row := 787) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_2
  simp only [← c49218, ← c49219, ← c49220] at e_787_2
  have e_787_3 := arithEq_of_rows h (row := 787) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_3
  simp only [← c49221, ← c49222, k2, ← c49223] at e_787_3
  have hr := e_787_3
  simp only [e_787_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5517)) := by
  have c49224 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.1
  have c49225 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.1
  have c49226 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49227 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49228 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49229 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_4 := arithEq_of_rows h (row := 787) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_4
  simp only [← c49224, ← c49225, ← c49226] at e_787_4
  have e_787_5 := arithEq_of_rows h (row := 787) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_5
  simp only [← c49227, ← c49228, k2, ← c49229] at e_787_5
  have hr := e_787_5
  simp only [e_787_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5518)) := by
  have c49230 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49231 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49232 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49233 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49234 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49235 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_6 := arithEq_of_rows h (row := 787) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_6
  simp only [← c49230, ← c49231, ← c49232] at e_787_6
  have e_787_7 := arithEq_of_rows h (row := 787) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_7
  simp only [← c49233, ← c49234, k2, ← c49235] at e_787_7
  have hr := e_787_7
  simp only [e_787_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5519)) := by
  have c49236 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49237 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49238 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49239 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49240 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49241 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_8 := arithEq_of_rows h (row := 787) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_8
  simp only [← c49236, ← c49237, ← c49238] at e_787_8
  have e_787_9 := arithEq_of_rows h (row := 787) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_9
  simp only [← c49239, ← c49240, k2, ← c49241] at e_787_9
  have hr := e_787_9
  simp only [e_787_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5520)) := by
  have c49242 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49243 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49244 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49245 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49246 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49247 := (publicBatchWrapper89_copies1538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_10 := arithEq_of_rows h (row := 787) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_10
  simp only [← c49242, ← c49243, ← c49244] at e_787_10
  have e_787_11 := arithEq_of_rows h (row := 787) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_11
  simp only [← c49245, ← c49246, k2, ← c49247] at e_787_11
  have hr := e_787_11
  simp only [e_787_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5521)) := by
  have c49248 := (publicBatchWrapper89_copies1539 a h).1
  have c49249 := (publicBatchWrapper89_copies1539 a h).2.1
  have c49250 := (publicBatchWrapper89_copies1539 a h).2.2.1
  have c49251 := (publicBatchWrapper89_copies1539 a h).2.2.2.1
  have c49252 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.1
  have c49253 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_12 := arithEq_of_rows h (row := 787) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_12
  simp only [← c49248, ← c49249, ← c49250] at e_787_12
  have e_787_13 := arithEq_of_rows h (row := 787) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_13
  simp only [← c49251, ← c49252, k2, ← c49253] at e_787_13
  have hr := e_787_13
  simp only [e_787_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5522)) := by
  have c49254 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.1
  have c49255 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.1
  have c49256 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.1
  have c49257 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.1
  have c49258 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49259 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_14 := arithEq_of_rows h (row := 787) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_14
  simp only [← c49254, ← c49255, ← c49256] at e_787_14
  have e_787_15 := arithEq_of_rows h (row := 787) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_15
  simp only [← c49257, ← c49258, k2, ← c49259] at e_787_15
  have hr := e_787_15
  simp only [e_787_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5523)) := by
  have c49260 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49261 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49262 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49263 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49264 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49265 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_16 := arithEq_of_rows h (row := 787) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_16
  simp only [← c49260, ← c49261, ← c49262] at e_787_16
  have e_787_17 := arithEq_of_rows h (row := 787) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_17
  simp only [← c49263, ← c49264, k2, ← c49265] at e_787_17
  have hr := e_787_17
  simp only [e_787_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 787 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5524)) := by
  have c49266 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49267 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49268 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49269 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49270 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49271 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_787_18 := arithEq_of_rows h (row := 787) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_18
  simp only [← c49266, ← c49267, ← c49268] at e_787_18
  have e_787_19 := arithEq_of_rows h (row := 787) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_787_19
  simp only [← c49269, ← c49270, k2, ← c49271] at e_787_19
  have hr := e_787_19
  simp only [e_787_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5525)) := by
  have c49272 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49273 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49274 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49275 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49276 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49277 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_0 := arithEq_of_rows h (row := 788) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_0
  simp only [← c49272, ← c49273, ← c49274] at e_788_0
  have e_788_1 := arithEq_of_rows h (row := 788) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_1
  simp only [← c49275, ← c49276, k2, ← c49277] at e_788_1
  have hr := e_788_1
  simp only [e_788_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5526)) := by
  have c49278 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49279 := (publicBatchWrapper89_copies1539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49280 := (publicBatchWrapper89_copies1540 a h).1
  have c49281 := (publicBatchWrapper89_copies1540 a h).2.1
  have c49282 := (publicBatchWrapper89_copies1540 a h).2.2.1
  have c49283 := (publicBatchWrapper89_copies1540 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_2 := arithEq_of_rows h (row := 788) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_2
  simp only [← c49278, ← c49279, ← c49280] at e_788_2
  have e_788_3 := arithEq_of_rows h (row := 788) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_3
  simp only [← c49281, ← c49282, k2, ← c49283] at e_788_3
  have hr := e_788_3
  simp only [e_788_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5527)) := by
  have c49284 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.1
  have c49285 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.1
  have c49286 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.1
  have c49287 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.1
  have c49288 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.1
  have c49289 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_4 := arithEq_of_rows h (row := 788) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_4
  simp only [← c49284, ← c49285, ← c49286] at e_788_4
  have e_788_5 := arithEq_of_rows h (row := 788) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_5
  simp only [← c49287, ← c49288, k2, ← c49289] at e_788_5
  have hr := e_788_5
  simp only [e_788_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5528)) := by
  have c49290 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49291 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49292 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49293 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49294 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49295 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_6 := arithEq_of_rows h (row := 788) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_6
  simp only [← c49290, ← c49291, ← c49292] at e_788_6
  have e_788_7 := arithEq_of_rows h (row := 788) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_7
  simp only [← c49293, ← c49294, k2, ← c49295] at e_788_7
  have hr := e_788_7
  simp only [e_788_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5529)) := by
  have c49296 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49297 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49298 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49299 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49300 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49301 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_8 := arithEq_of_rows h (row := 788) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_8
  simp only [← c49296, ← c49297, ← c49298] at e_788_8
  have e_788_9 := arithEq_of_rows h (row := 788) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_9
  simp only [← c49299, ← c49300, k2, ← c49301] at e_788_9
  have hr := e_788_9
  simp only [e_788_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5530)) := by
  have c49302 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49303 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49304 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49305 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49306 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49307 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_10 := arithEq_of_rows h (row := 788) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_10
  simp only [← c49302, ← c49303, ← c49304] at e_788_10
  have e_788_11 := arithEq_of_rows h (row := 788) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_11
  simp only [← c49305, ← c49306, k2, ← c49307] at e_788_11
  have hr := e_788_11
  simp only [e_788_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5531)) := by
  have c49308 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49309 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49310 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49311 := (publicBatchWrapper89_copies1540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49312 := (publicBatchWrapper89_copies1541 a h).1
  have c49313 := (publicBatchWrapper89_copies1541 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_12 := arithEq_of_rows h (row := 788) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_12
  simp only [← c49308, ← c49309, ← c49310] at e_788_12
  have e_788_13 := arithEq_of_rows h (row := 788) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_13
  simp only [← c49311, ← c49312, k2, ← c49313] at e_788_13
  have hr := e_788_13
  simp only [e_788_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5532)) := by
  have c49314 := (publicBatchWrapper89_copies1541 a h).2.2.1
  have c49315 := (publicBatchWrapper89_copies1541 a h).2.2.2.1
  have c49316 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.1
  have c49317 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.1
  have c49318 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.1
  have c49319 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_14 := arithEq_of_rows h (row := 788) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_14
  simp only [← c49314, ← c49315, ← c49316] at e_788_14
  have e_788_15 := arithEq_of_rows h (row := 788) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_15
  simp only [← c49317, ← c49318, k2, ← c49319] at e_788_15
  have hr := e_788_15
  simp only [e_788_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5533)) := by
  have c49320 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.1
  have c49321 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.1
  have c49322 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49323 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49324 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49325 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_16 := arithEq_of_rows h (row := 788) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_16
  simp only [← c49320, ← c49321, ← c49322] at e_788_16
  have e_788_17 := arithEq_of_rows h (row := 788) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_17
  simp only [← c49323, ← c49324, k2, ← c49325] at e_788_17
  have hr := e_788_17
  simp only [e_788_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 788 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5534)) := by
  have c49326 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49327 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49328 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49329 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49330 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49331 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_788_18 := arithEq_of_rows h (row := 788) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_18
  simp only [← c49326, ← c49327, ← c49328] at e_788_18
  have e_788_19 := arithEq_of_rows h (row := 788) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_788_19
  simp only [← c49329, ← c49330, k2, ← c49331] at e_788_19
  have hr := e_788_19
  simp only [e_788_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5535)) := by
  have c49332 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49333 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49334 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49335 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49336 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49337 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_0 := arithEq_of_rows h (row := 789) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_0
  simp only [← c49332, ← c49333, ← c49334] at e_789_0
  have e_789_1 := arithEq_of_rows h (row := 789) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_1
  simp only [← c49335, ← c49336, k2, ← c49337] at e_789_1
  have hr := e_789_1
  simp only [e_789_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5536)) := by
  have c49338 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49339 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49340 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49341 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49342 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49343 := (publicBatchWrapper89_copies1541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_2 := arithEq_of_rows h (row := 789) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_2
  simp only [← c49338, ← c49339, ← c49340] at e_789_2
  have e_789_3 := arithEq_of_rows h (row := 789) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_3
  simp only [← c49341, ← c49342, k2, ← c49343] at e_789_3
  have hr := e_789_3
  simp only [e_789_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5537)) := by
  have c49344 := (publicBatchWrapper89_copies1542 a h).1
  have c49345 := (publicBatchWrapper89_copies1542 a h).2.1
  have c49346 := (publicBatchWrapper89_copies1542 a h).2.2.1
  have c49347 := (publicBatchWrapper89_copies1542 a h).2.2.2.1
  have c49348 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.1
  have c49349 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_4 := arithEq_of_rows h (row := 789) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_4
  simp only [← c49344, ← c49345, ← c49346] at e_789_4
  have e_789_5 := arithEq_of_rows h (row := 789) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_5
  simp only [← c49347, ← c49348, k2, ← c49349] at e_789_5
  have hr := e_789_5
  simp only [e_789_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5538)) := by
  have c49350 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.1
  have c49351 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.1
  have c49352 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.1
  have c49353 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.1
  have c49354 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49355 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_6 := arithEq_of_rows h (row := 789) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_6
  simp only [← c49350, ← c49351, ← c49352] at e_789_6
  have e_789_7 := arithEq_of_rows h (row := 789) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_7
  simp only [← c49353, ← c49354, k2, ← c49355] at e_789_7
  have hr := e_789_7
  simp only [e_789_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5539)) := by
  have c49356 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49357 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49358 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49359 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49360 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49361 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_8 := arithEq_of_rows h (row := 789) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_8
  simp only [← c49356, ← c49357, ← c49358] at e_789_8
  have e_789_9 := arithEq_of_rows h (row := 789) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_9
  simp only [← c49359, ← c49360, k2, ← c49361] at e_789_9
  have hr := e_789_9
  simp only [e_789_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5540)) := by
  have c49362 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49363 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49364 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49365 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49366 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49367 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_10 := arithEq_of_rows h (row := 789) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_10
  simp only [← c49362, ← c49363, ← c49364] at e_789_10
  have e_789_11 := arithEq_of_rows h (row := 789) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_11
  simp only [← c49365, ← c49366, k2, ← c49367] at e_789_11
  have hr := e_789_11
  simp only [e_789_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5541)) := by
  have c49368 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49369 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49370 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49371 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49372 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49373 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_12 := arithEq_of_rows h (row := 789) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_12
  simp only [← c49368, ← c49369, ← c49370] at e_789_12
  have e_789_13 := arithEq_of_rows h (row := 789) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_13
  simp only [← c49371, ← c49372, k2, ← c49373] at e_789_13
  have hr := e_789_13
  simp only [e_789_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5542)) := by
  have c49374 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49375 := (publicBatchWrapper89_copies1542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49376 := (publicBatchWrapper89_copies1543 a h).1
  have c49377 := (publicBatchWrapper89_copies1543 a h).2.1
  have c49378 := (publicBatchWrapper89_copies1543 a h).2.2.1
  have c49379 := (publicBatchWrapper89_copies1543 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_14 := arithEq_of_rows h (row := 789) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_14
  simp only [← c49374, ← c49375, ← c49376] at e_789_14
  have e_789_15 := arithEq_of_rows h (row := 789) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_15
  simp only [← c49377, ← c49378, k2, ← c49379] at e_789_15
  have hr := e_789_15
  simp only [e_789_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5543)) := by
  have c49380 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.1
  have c49381 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.1
  have c49382 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.1
  have c49383 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.1
  have c49384 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.1
  have c49385 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_16 := arithEq_of_rows h (row := 789) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_16
  simp only [← c49380, ← c49381, ← c49382] at e_789_16
  have e_789_17 := arithEq_of_rows h (row := 789) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_17
  simp only [← c49383, ← c49384, k2, ← c49385] at e_789_17
  have hr := e_789_17
  simp only [e_789_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 789 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5544)) := by
  have c49386 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49387 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49388 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49389 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49390 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49391 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_789_18 := arithEq_of_rows h (row := 789) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_18
  simp only [← c49386, ← c49387, ← c49388] at e_789_18
  have e_789_19 := arithEq_of_rows h (row := 789) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_789_19
  simp only [← c49389, ← c49390, k2, ← c49391] at e_789_19
  have hr := e_789_19
  simp only [e_789_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5545)) := by
  have c49392 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49393 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49394 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49395 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49396 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49397 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_0 := arithEq_of_rows h (row := 790) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_0
  simp only [← c49392, ← c49393, ← c49394] at e_790_0
  have e_790_1 := arithEq_of_rows h (row := 790) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_1
  simp only [← c49395, ← c49396, k2, ← c49397] at e_790_1
  have hr := e_790_1
  simp only [e_790_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5546)) := by
  have c49398 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49399 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49400 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49401 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49402 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49403 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_2 := arithEq_of_rows h (row := 790) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_2
  simp only [← c49398, ← c49399, ← c49400] at e_790_2
  have e_790_3 := arithEq_of_rows h (row := 790) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_3
  simp only [← c49401, ← c49402, k2, ← c49403] at e_790_3
  have hr := e_790_3
  simp only [e_790_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5547)) := by
  have c49404 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49405 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49406 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49407 := (publicBatchWrapper89_copies1543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49408 := (publicBatchWrapper89_copies1544 a h).1
  have c49409 := (publicBatchWrapper89_copies1544 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_4 := arithEq_of_rows h (row := 790) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_4
  simp only [← c49404, ← c49405, ← c49406] at e_790_4
  have e_790_5 := arithEq_of_rows h (row := 790) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_5
  simp only [← c49407, ← c49408, k2, ← c49409] at e_790_5
  have hr := e_790_5
  simp only [e_790_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5548)) := by
  have c49410 := (publicBatchWrapper89_copies1544 a h).2.2.1
  have c49411 := (publicBatchWrapper89_copies1544 a h).2.2.2.1
  have c49412 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.1
  have c49413 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.1
  have c49414 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.1
  have c49415 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_6 := arithEq_of_rows h (row := 790) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_6
  simp only [← c49410, ← c49411, ← c49412] at e_790_6
  have e_790_7 := arithEq_of_rows h (row := 790) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_7
  simp only [← c49413, ← c49414, k2, ← c49415] at e_790_7
  have hr := e_790_7
  simp only [e_790_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5549)) := by
  have c49416 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.1
  have c49417 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.1
  have c49418 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49419 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49420 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49421 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_8 := arithEq_of_rows h (row := 790) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_8
  simp only [← c49416, ← c49417, ← c49418] at e_790_8
  have e_790_9 := arithEq_of_rows h (row := 790) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_9
  simp only [← c49419, ← c49420, k2, ← c49421] at e_790_9
  have hr := e_790_9
  simp only [e_790_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5550)) := by
  have c49422 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49423 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49424 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49425 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49426 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49427 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_10 := arithEq_of_rows h (row := 790) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_10
  simp only [← c49422, ← c49423, ← c49424] at e_790_10
  have e_790_11 := arithEq_of_rows h (row := 790) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_11
  simp only [← c49425, ← c49426, k2, ← c49427] at e_790_11
  have hr := e_790_11
  simp only [e_790_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5551)) := by
  have c49428 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49429 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49430 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49431 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49432 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49433 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_12 := arithEq_of_rows h (row := 790) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_12
  simp only [← c49428, ← c49429, ← c49430] at e_790_12
  have e_790_13 := arithEq_of_rows h (row := 790) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_13
  simp only [← c49431, ← c49432, k2, ← c49433] at e_790_13
  have hr := e_790_13
  simp only [e_790_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5552)) := by
  have c49434 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49435 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49436 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49437 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49438 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49439 := (publicBatchWrapper89_copies1544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_14 := arithEq_of_rows h (row := 790) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_14
  simp only [← c49434, ← c49435, ← c49436] at e_790_14
  have e_790_15 := arithEq_of_rows h (row := 790) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_15
  simp only [← c49437, ← c49438, k2, ← c49439] at e_790_15
  have hr := e_790_15
  simp only [e_790_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5553)) := by
  have c49440 := (publicBatchWrapper89_copies1545 a h).1
  have c49441 := (publicBatchWrapper89_copies1545 a h).2.1
  have c49442 := (publicBatchWrapper89_copies1545 a h).2.2.1
  have c49443 := (publicBatchWrapper89_copies1545 a h).2.2.2.1
  have c49444 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.1
  have c49445 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_16 := arithEq_of_rows h (row := 790) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_16
  simp only [← c49440, ← c49441, ← c49442] at e_790_16
  have e_790_17 := arithEq_of_rows h (row := 790) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_17
  simp only [← c49443, ← c49444, k2, ← c49445] at e_790_17
  have hr := e_790_17
  simp only [e_790_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 790 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5554)) := by
  have c49446 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.1
  have c49447 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.1
  have c49448 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.1
  have c49449 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.1
  have c49450 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49451 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_790_18 := arithEq_of_rows h (row := 790) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_18
  simp only [← c49446, ← c49447, ← c49448] at e_790_18
  have e_790_19 := arithEq_of_rows h (row := 790) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_790_19
  simp only [← c49449, ← c49450, k2, ← c49451] at e_790_19
  have hr := e_790_19
  simp only [e_790_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5555)) := by
  have c49452 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49453 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49454 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49455 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49456 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49457 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_0 := arithEq_of_rows h (row := 791) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_0
  simp only [← c49452, ← c49453, ← c49454] at e_791_0
  have e_791_1 := arithEq_of_rows h (row := 791) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_1
  simp only [← c49455, ← c49456, k2, ← c49457] at e_791_1
  have hr := e_791_1
  simp only [e_791_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5556)) := by
  have c49458 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49459 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49460 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49461 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49462 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49463 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_2 := arithEq_of_rows h (row := 791) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_2
  simp only [← c49458, ← c49459, ← c49460] at e_791_2
  have e_791_3 := arithEq_of_rows h (row := 791) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_3
  simp only [← c49461, ← c49462, k2, ← c49463] at e_791_3
  have hr := e_791_3
  simp only [e_791_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5557)) := by
  have c49464 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49465 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49466 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49467 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49468 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49469 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_4 := arithEq_of_rows h (row := 791) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_4
  simp only [← c49464, ← c49465, ← c49466] at e_791_4
  have e_791_5 := arithEq_of_rows h (row := 791) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_5
  simp only [← c49467, ← c49468, k2, ← c49469] at e_791_5
  have hr := e_791_5
  simp only [e_791_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5558)) := by
  have c49470 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49471 := (publicBatchWrapper89_copies1545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49472 := (publicBatchWrapper89_copies1546 a h).1
  have c49473 := (publicBatchWrapper89_copies1546 a h).2.1
  have c49474 := (publicBatchWrapper89_copies1546 a h).2.2.1
  have c49475 := (publicBatchWrapper89_copies1546 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_6 := arithEq_of_rows h (row := 791) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_6
  simp only [← c49470, ← c49471, ← c49472] at e_791_6
  have e_791_7 := arithEq_of_rows h (row := 791) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_7
  simp only [← c49473, ← c49474, k2, ← c49475] at e_791_7
  have hr := e_791_7
  simp only [e_791_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5559)) := by
  have c49476 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.1
  have c49477 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.1
  have c49478 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.1
  have c49479 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.1
  have c49480 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.1
  have c49481 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_8 := arithEq_of_rows h (row := 791) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_8
  simp only [← c49476, ← c49477, ← c49478] at e_791_8
  have e_791_9 := arithEq_of_rows h (row := 791) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_9
  simp only [← c49479, ← c49480, k2, ← c49481] at e_791_9
  have hr := e_791_9
  simp only [e_791_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5560)) := by
  have c49482 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49483 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49484 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49485 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49486 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49487 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_10 := arithEq_of_rows h (row := 791) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_10
  simp only [← c49482, ← c49483, ← c49484] at e_791_10
  have e_791_11 := arithEq_of_rows h (row := 791) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_11
  simp only [← c49485, ← c49486, k2, ← c49487] at e_791_11
  have hr := e_791_11
  simp only [e_791_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5561)) := by
  have c49488 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49489 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49490 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49491 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49492 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49493 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_12 := arithEq_of_rows h (row := 791) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_12
  simp only [← c49488, ← c49489, ← c49490] at e_791_12
  have e_791_13 := arithEq_of_rows h (row := 791) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_13
  simp only [← c49491, ← c49492, k2, ← c49493] at e_791_13
  have hr := e_791_13
  simp only [e_791_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5562)) := by
  have c49494 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49495 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49496 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49497 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49498 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49499 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_14 := arithEq_of_rows h (row := 791) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_14
  simp only [← c49494, ← c49495, ← c49496] at e_791_14
  have e_791_15 := arithEq_of_rows h (row := 791) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_15
  simp only [← c49497, ← c49498, k2, ← c49499] at e_791_15
  have hr := e_791_15
  simp only [e_791_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5563)) := by
  have c49500 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49501 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49502 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49503 := (publicBatchWrapper89_copies1546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49504 := (publicBatchWrapper89_copies1547 a h).1
  have c49505 := (publicBatchWrapper89_copies1547 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_16 := arithEq_of_rows h (row := 791) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_16
  simp only [← c49500, ← c49501, ← c49502] at e_791_16
  have e_791_17 := arithEq_of_rows h (row := 791) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_17
  simp only [← c49503, ← c49504, k2, ← c49505] at e_791_17
  have hr := e_791_17
  simp only [e_791_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 791 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5564)) := by
  have c49506 := (publicBatchWrapper89_copies1547 a h).2.2.1
  have c49507 := (publicBatchWrapper89_copies1547 a h).2.2.2.1
  have c49508 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.1
  have c49509 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.1
  have c49510 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.1
  have c49511 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_791_18 := arithEq_of_rows h (row := 791) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_18
  simp only [← c49506, ← c49507, ← c49508] at e_791_18
  have e_791_19 := arithEq_of_rows h (row := 791) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_791_19
  simp only [← c49509, ← c49510, k2, ← c49511] at e_791_19
  have hr := e_791_19
  simp only [e_791_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5565)) := by
  have c49512 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.1
  have c49513 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.1
  have c49514 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49515 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49516 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49517 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_0 := arithEq_of_rows h (row := 792) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_0
  simp only [← c49512, ← c49513, ← c49514] at e_792_0
  have e_792_1 := arithEq_of_rows h (row := 792) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_1
  simp only [← c49515, ← c49516, k2, ← c49517] at e_792_1
  have hr := e_792_1
  simp only [e_792_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5566)) := by
  have c49518 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49519 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49520 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49521 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49522 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49523 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_2 := arithEq_of_rows h (row := 792) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_2
  simp only [← c49518, ← c49519, ← c49520] at e_792_2
  have e_792_3 := arithEq_of_rows h (row := 792) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_3
  simp only [← c49521, ← c49522, k2, ← c49523] at e_792_3
  have hr := e_792_3
  simp only [e_792_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5567)) := by
  have c49524 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49525 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49526 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49527 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49528 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49529 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_4 := arithEq_of_rows h (row := 792) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_4
  simp only [← c49524, ← c49525, ← c49526] at e_792_4
  have e_792_5 := arithEq_of_rows h (row := 792) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_5
  simp only [← c49527, ← c49528, k2, ← c49529] at e_792_5
  have hr := e_792_5
  simp only [e_792_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5568)) := by
  have c49530 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49531 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49532 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49533 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49534 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49535 := (publicBatchWrapper89_copies1547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_6 := arithEq_of_rows h (row := 792) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_6
  simp only [← c49530, ← c49531, ← c49532] at e_792_6
  have e_792_7 := arithEq_of_rows h (row := 792) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_7
  simp only [← c49533, ← c49534, k2, ← c49535] at e_792_7
  have hr := e_792_7
  simp only [e_792_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5569)) := by
  have c49536 := (publicBatchWrapper89_copies1548 a h).1
  have c49537 := (publicBatchWrapper89_copies1548 a h).2.1
  have c49538 := (publicBatchWrapper89_copies1548 a h).2.2.1
  have c49539 := (publicBatchWrapper89_copies1548 a h).2.2.2.1
  have c49540 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.1
  have c49541 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_8 := arithEq_of_rows h (row := 792) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_8
  simp only [← c49536, ← c49537, ← c49538] at e_792_8
  have e_792_9 := arithEq_of_rows h (row := 792) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_9
  simp only [← c49539, ← c49540, k2, ← c49541] at e_792_9
  have hr := e_792_9
  simp only [e_792_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5610)) := by
  have c49542 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.1
  have c49543 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.1
  have c49544 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.1
  have c49545 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.1
  have c49546 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49547 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_10 := arithEq_of_rows h (row := 792) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_10
  simp only [← c49542, ← c49543, ← c49544] at e_792_10
  have e_792_11 := arithEq_of_rows h (row := 792) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_11
  simp only [← c49545, ← c49546, k2, ← c49547] at e_792_11
  have hr := e_792_11
  simp only [e_792_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5611)) := by
  have c49548 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49549 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49550 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49551 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49552 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49553 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_12 := arithEq_of_rows h (row := 792) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_12
  simp only [← c49548, ← c49549, ← c49550] at e_792_12
  have e_792_13 := arithEq_of_rows h (row := 792) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_13
  simp only [← c49551, ← c49552, k2, ← c49553] at e_792_13
  have hr := e_792_13
  simp only [e_792_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5612)) := by
  have c49554 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49555 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49556 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49557 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49558 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49559 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_14 := arithEq_of_rows h (row := 792) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_14
  simp only [← c49554, ← c49555, ← c49556] at e_792_14
  have e_792_15 := arithEq_of_rows h (row := 792) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_15
  simp only [← c49557, ← c49558, k2, ← c49559] at e_792_15
  have hr := e_792_15
  simp only [e_792_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5613)) := by
  have c49560 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49561 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49562 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49563 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49564 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49565 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_16 := arithEq_of_rows h (row := 792) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_16
  simp only [← c49560, ← c49561, ← c49562] at e_792_16
  have e_792_17 := arithEq_of_rows h (row := 792) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_17
  simp only [← c49563, ← c49564, k2, ← c49565] at e_792_17
  have hr := e_792_17
  simp only [e_792_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 792 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5614)) := by
  have c49566 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49567 := (publicBatchWrapper89_copies1548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49568 := (publicBatchWrapper89_copies1549 a h).1
  have c49569 := (publicBatchWrapper89_copies1549 a h).2.1
  have c49570 := (publicBatchWrapper89_copies1549 a h).2.2.1
  have c49571 := (publicBatchWrapper89_copies1549 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_792_18 := arithEq_of_rows h (row := 792) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_18
  simp only [← c49566, ← c49567, ← c49568] at e_792_18
  have e_792_19 := arithEq_of_rows h (row := 792) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_792_19
  simp only [← c49569, ← c49570, k2, ← c49571] at e_792_19
  have hr := e_792_19
  simp only [e_792_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5615)) := by
  have c49572 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.1
  have c49573 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.1
  have c49574 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.1
  have c49575 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.1
  have c49576 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.1
  have c49577 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_0 := arithEq_of_rows h (row := 793) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_0
  simp only [← c49572, ← c49573, ← c49574] at e_793_0
  have e_793_1 := arithEq_of_rows h (row := 793) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_1
  simp only [← c49575, ← c49576, k2, ← c49577] at e_793_1
  have hr := e_793_1
  simp only [e_793_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5616)) := by
  have c49578 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49579 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49580 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49581 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49582 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49583 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_2 := arithEq_of_rows h (row := 793) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_2
  simp only [← c49578, ← c49579, ← c49580] at e_793_2
  have e_793_3 := arithEq_of_rows h (row := 793) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_3
  simp only [← c49581, ← c49582, k2, ← c49583] at e_793_3
  have hr := e_793_3
  simp only [e_793_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5617)) := by
  have c49584 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49585 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49586 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49587 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49588 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49589 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_4 := arithEq_of_rows h (row := 793) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_4
  simp only [← c49584, ← c49585, ← c49586] at e_793_4
  have e_793_5 := arithEq_of_rows h (row := 793) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_5
  simp only [← c49587, ← c49588, k2, ← c49589] at e_793_5
  have hr := e_793_5
  simp only [e_793_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5618)) := by
  have c49590 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49591 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49592 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49593 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49594 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49595 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_6 := arithEq_of_rows h (row := 793) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_6
  simp only [← c49590, ← c49591, ← c49592] at e_793_6
  have e_793_7 := arithEq_of_rows h (row := 793) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_7
  simp only [← c49593, ← c49594, k2, ← c49595] at e_793_7
  have hr := e_793_7
  simp only [e_793_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5619)) := by
  have c49596 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49597 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49598 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49599 := (publicBatchWrapper89_copies1549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49600 := (publicBatchWrapper89_copies1550 a h).1
  have c49601 := (publicBatchWrapper89_copies1550 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_8 := arithEq_of_rows h (row := 793) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_8
  simp only [← c49596, ← c49597, ← c49598] at e_793_8
  have e_793_9 := arithEq_of_rows h (row := 793) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_9
  simp only [← c49599, ← c49600, k2, ← c49601] at e_793_9
  have hr := e_793_9
  simp only [e_793_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5620)) := by
  have c49602 := (publicBatchWrapper89_copies1550 a h).2.2.1
  have c49603 := (publicBatchWrapper89_copies1550 a h).2.2.2.1
  have c49604 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.1
  have c49605 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.1
  have c49606 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.1
  have c49607 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_10 := arithEq_of_rows h (row := 793) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_10
  simp only [← c49602, ← c49603, ← c49604] at e_793_10
  have e_793_11 := arithEq_of_rows h (row := 793) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_11
  simp only [← c49605, ← c49606, k2, ← c49607] at e_793_11
  have hr := e_793_11
  simp only [e_793_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5621)) := by
  have c49608 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.1
  have c49609 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.1
  have c49610 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49611 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49612 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49613 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_12 := arithEq_of_rows h (row := 793) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_12
  simp only [← c49608, ← c49609, ← c49610] at e_793_12
  have e_793_13 := arithEq_of_rows h (row := 793) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_13
  simp only [← c49611, ← c49612, k2, ← c49613] at e_793_13
  have hr := e_793_13
  simp only [e_793_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5622)) := by
  have c49614 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49615 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49616 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49617 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49618 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49619 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_14 := arithEq_of_rows h (row := 793) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_14
  simp only [← c49614, ← c49615, ← c49616] at e_793_14
  have e_793_15 := arithEq_of_rows h (row := 793) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_15
  simp only [← c49617, ← c49618, k2, ← c49619] at e_793_15
  have hr := e_793_15
  simp only [e_793_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5623)) := by
  have c49620 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49621 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49622 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49623 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49624 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49625 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_16 := arithEq_of_rows h (row := 793) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_16
  simp only [← c49620, ← c49621, ← c49622] at e_793_16
  have e_793_17 := arithEq_of_rows h (row := 793) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_17
  simp only [← c49623, ← c49624, k2, ← c49625] at e_793_17
  have hr := e_793_17
  simp only [e_793_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 793 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5624)) := by
  have c49626 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49627 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49628 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49629 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49630 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49631 := (publicBatchWrapper89_copies1550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_793_18 := arithEq_of_rows h (row := 793) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_18
  simp only [← c49626, ← c49627, ← c49628] at e_793_18
  have e_793_19 := arithEq_of_rows h (row := 793) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_793_19
  simp only [← c49629, ← c49630, k2, ← c49631] at e_793_19
  have hr := e_793_19
  simp only [e_793_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5625)) := by
  have c49632 := (publicBatchWrapper89_copies1551 a h).1
  have c49633 := (publicBatchWrapper89_copies1551 a h).2.1
  have c49634 := (publicBatchWrapper89_copies1551 a h).2.2.1
  have c49635 := (publicBatchWrapper89_copies1551 a h).2.2.2.1
  have c49636 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.1
  have c49637 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_0 := arithEq_of_rows h (row := 794) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_0
  simp only [← c49632, ← c49633, ← c49634] at e_794_0
  have e_794_1 := arithEq_of_rows h (row := 794) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_1
  simp only [← c49635, ← c49636, k2, ← c49637] at e_794_1
  have hr := e_794_1
  simp only [e_794_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5626)) := by
  have c49638 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.1
  have c49639 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.1
  have c49640 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.1
  have c49641 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.1
  have c49642 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49643 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_2 := arithEq_of_rows h (row := 794) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_2
  simp only [← c49638, ← c49639, ← c49640] at e_794_2
  have e_794_3 := arithEq_of_rows h (row := 794) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_3
  simp only [← c49641, ← c49642, k2, ← c49643] at e_794_3
  have hr := e_794_3
  simp only [e_794_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5627)) := by
  have c49644 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49645 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49646 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49647 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49648 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49649 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_4 := arithEq_of_rows h (row := 794) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_4
  simp only [← c49644, ← c49645, ← c49646] at e_794_4
  have e_794_5 := arithEq_of_rows h (row := 794) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_5
  simp only [← c49647, ← c49648, k2, ← c49649] at e_794_5
  have hr := e_794_5
  simp only [e_794_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5628)) := by
  have c49650 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49651 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49652 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49653 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49654 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49655 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_6 := arithEq_of_rows h (row := 794) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_6
  simp only [← c49650, ← c49651, ← c49652] at e_794_6
  have e_794_7 := arithEq_of_rows h (row := 794) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_7
  simp only [← c49653, ← c49654, k2, ← c49655] at e_794_7
  have hr := e_794_7
  simp only [e_794_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5629)) := by
  have c49656 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49657 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49658 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49659 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49660 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49661 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_8 := arithEq_of_rows h (row := 794) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_8
  simp only [← c49656, ← c49657, ← c49658] at e_794_8
  have e_794_9 := arithEq_of_rows h (row := 794) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_9
  simp only [← c49659, ← c49660, k2, ← c49661] at e_794_9
  have hr := e_794_9
  simp only [e_794_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5630)) := by
  have c49662 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49663 := (publicBatchWrapper89_copies1551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49664 := (publicBatchWrapper89_copies1552 a h).1
  have c49665 := (publicBatchWrapper89_copies1552 a h).2.1
  have c49666 := (publicBatchWrapper89_copies1552 a h).2.2.1
  have c49667 := (publicBatchWrapper89_copies1552 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_10 := arithEq_of_rows h (row := 794) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_10
  simp only [← c49662, ← c49663, ← c49664] at e_794_10
  have e_794_11 := arithEq_of_rows h (row := 794) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_11
  simp only [← c49665, ← c49666, k2, ← c49667] at e_794_11
  have hr := e_794_11
  simp only [e_794_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5631)) := by
  have c49668 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.1
  have c49669 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.1
  have c49670 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.1
  have c49671 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.1
  have c49672 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.1
  have c49673 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_12 := arithEq_of_rows h (row := 794) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_12
  simp only [← c49668, ← c49669, ← c49670] at e_794_12
  have e_794_13 := arithEq_of_rows h (row := 794) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_13
  simp only [← c49671, ← c49672, k2, ← c49673] at e_794_13
  have hr := e_794_13
  simp only [e_794_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5632)) := by
  have c49674 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49675 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49676 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49677 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49678 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49679 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_14 := arithEq_of_rows h (row := 794) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_14
  simp only [← c49674, ← c49675, ← c49676] at e_794_14
  have e_794_15 := arithEq_of_rows h (row := 794) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_15
  simp only [← c49677, ← c49678, k2, ← c49679] at e_794_15
  have hr := e_794_15
  simp only [e_794_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5633)) := by
  have c49680 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49681 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49682 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49683 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49684 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49685 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_16 := arithEq_of_rows h (row := 794) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_16
  simp only [← c49680, ← c49681, ← c49682] at e_794_16
  have e_794_17 := arithEq_of_rows h (row := 794) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_17
  simp only [← c49683, ← c49684, k2, ← c49685] at e_794_17
  have hr := e_794_17
  simp only [e_794_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 794 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5634)) := by
  have c49686 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49687 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49688 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49689 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49690 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49691 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_794_18 := arithEq_of_rows h (row := 794) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_18
  simp only [← c49686, ← c49687, ← c49688] at e_794_18
  have e_794_19 := arithEq_of_rows h (row := 794) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_794_19
  simp only [← c49689, ← c49690, k2, ← c49691] at e_794_19
  have hr := e_794_19
  simp only [e_794_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5635)) := by
  have c49692 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49693 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49694 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49695 := (publicBatchWrapper89_copies1552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49696 := (publicBatchWrapper89_copies1553 a h).1
  have c49697 := (publicBatchWrapper89_copies1553 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_0 := arithEq_of_rows h (row := 795) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_0
  simp only [← c49692, ← c49693, ← c49694] at e_795_0
  have e_795_1 := arithEq_of_rows h (row := 795) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_1
  simp only [← c49695, ← c49696, k2, ← c49697] at e_795_1
  have hr := e_795_1
  simp only [e_795_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5636)) := by
  have c49698 := (publicBatchWrapper89_copies1553 a h).2.2.1
  have c49699 := (publicBatchWrapper89_copies1553 a h).2.2.2.1
  have c49700 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.1
  have c49701 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.1
  have c49702 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.1
  have c49703 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_2 := arithEq_of_rows h (row := 795) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_2
  simp only [← c49698, ← c49699, ← c49700] at e_795_2
  have e_795_3 := arithEq_of_rows h (row := 795) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_3
  simp only [← c49701, ← c49702, k2, ← c49703] at e_795_3
  have hr := e_795_3
  simp only [e_795_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5637)) := by
  have c49704 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.1
  have c49705 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.1
  have c49706 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49707 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49708 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49709 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_4 := arithEq_of_rows h (row := 795) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_4
  simp only [← c49704, ← c49705, ← c49706] at e_795_4
  have e_795_5 := arithEq_of_rows h (row := 795) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_5
  simp only [← c49707, ← c49708, k2, ← c49709] at e_795_5
  have hr := e_795_5
  simp only [e_795_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5638)) := by
  have c49710 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49711 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49712 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49713 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49714 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49715 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_6 := arithEq_of_rows h (row := 795) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_6
  simp only [← c49710, ← c49711, ← c49712] at e_795_6
  have e_795_7 := arithEq_of_rows h (row := 795) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_7
  simp only [← c49713, ← c49714, k2, ← c49715] at e_795_7
  have hr := e_795_7
  simp only [e_795_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5639)) := by
  have c49716 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49717 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49718 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49719 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49720 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49721 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_8 := arithEq_of_rows h (row := 795) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_8
  simp only [← c49716, ← c49717, ← c49718] at e_795_8
  have e_795_9 := arithEq_of_rows h (row := 795) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_9
  simp only [← c49719, ← c49720, k2, ← c49721] at e_795_9
  have hr := e_795_9
  simp only [e_795_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5640)) := by
  have c49722 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49723 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49724 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49725 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49726 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49727 := (publicBatchWrapper89_copies1553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_10 := arithEq_of_rows h (row := 795) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_10
  simp only [← c49722, ← c49723, ← c49724] at e_795_10
  have e_795_11 := arithEq_of_rows h (row := 795) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_11
  simp only [← c49725, ← c49726, k2, ← c49727] at e_795_11
  have hr := e_795_11
  simp only [e_795_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5641)) := by
  have c49728 := (publicBatchWrapper89_copies1554 a h).1
  have c49729 := (publicBatchWrapper89_copies1554 a h).2.1
  have c49730 := (publicBatchWrapper89_copies1554 a h).2.2.1
  have c49731 := (publicBatchWrapper89_copies1554 a h).2.2.2.1
  have c49732 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.1
  have c49733 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_12 := arithEq_of_rows h (row := 795) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_12
  simp only [← c49728, ← c49729, ← c49730] at e_795_12
  have e_795_13 := arithEq_of_rows h (row := 795) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_13
  simp only [← c49731, ← c49732, k2, ← c49733] at e_795_13
  have hr := e_795_13
  simp only [e_795_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5642)) := by
  have c49734 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.1
  have c49735 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.1
  have c49736 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.1
  have c49737 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.1
  have c49738 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49739 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_14 := arithEq_of_rows h (row := 795) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_14
  simp only [← c49734, ← c49735, ← c49736] at e_795_14
  have e_795_15 := arithEq_of_rows h (row := 795) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_15
  simp only [← c49737, ← c49738, k2, ← c49739] at e_795_15
  have hr := e_795_15
  simp only [e_795_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5643)) := by
  have c49740 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49741 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49742 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49743 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49744 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49745 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_16 := arithEq_of_rows h (row := 795) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_16
  simp only [← c49740, ← c49741, ← c49742] at e_795_16
  have e_795_17 := arithEq_of_rows h (row := 795) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_17
  simp only [← c49743, ← c49744, k2, ← c49745] at e_795_17
  have hr := e_795_17
  simp only [e_795_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 795 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5644)) := by
  have c49746 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49747 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49748 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49749 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49750 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49751 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_795_18 := arithEq_of_rows h (row := 795) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_18
  simp only [← c49746, ← c49747, ← c49748] at e_795_18
  have e_795_19 := arithEq_of_rows h (row := 795) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_795_19
  simp only [← c49749, ← c49750, k2, ← c49751] at e_795_19
  have hr := e_795_19
  simp only [e_795_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5645)) := by
  have c49752 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49753 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49754 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49755 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49756 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49757 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_0 := arithEq_of_rows h (row := 796) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_0
  simp only [← c49752, ← c49753, ← c49754] at e_796_0
  have e_796_1 := arithEq_of_rows h (row := 796) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_1
  simp only [← c49755, ← c49756, k2, ← c49757] at e_796_1
  have hr := e_796_1
  simp only [e_796_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5646)) := by
  have c49758 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49759 := (publicBatchWrapper89_copies1554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49760 := (publicBatchWrapper89_copies1555 a h).1
  have c49761 := (publicBatchWrapper89_copies1555 a h).2.1
  have c49762 := (publicBatchWrapper89_copies1555 a h).2.2.1
  have c49763 := (publicBatchWrapper89_copies1555 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_2 := arithEq_of_rows h (row := 796) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_2
  simp only [← c49758, ← c49759, ← c49760] at e_796_2
  have e_796_3 := arithEq_of_rows h (row := 796) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_3
  simp only [← c49761, ← c49762, k2, ← c49763] at e_796_3
  have hr := e_796_3
  simp only [e_796_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5647)) := by
  have c49764 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.1
  have c49765 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.1
  have c49766 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.1
  have c49767 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.1
  have c49768 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.1
  have c49769 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_4 := arithEq_of_rows h (row := 796) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_4
  simp only [← c49764, ← c49765, ← c49766] at e_796_4
  have e_796_5 := arithEq_of_rows h (row := 796) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_5
  simp only [← c49767, ← c49768, k2, ← c49769] at e_796_5
  have hr := e_796_5
  simp only [e_796_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5648)) := by
  have c49770 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49771 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49772 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49773 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49774 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49775 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_6 := arithEq_of_rows h (row := 796) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_6
  simp only [← c49770, ← c49771, ← c49772] at e_796_6
  have e_796_7 := arithEq_of_rows h (row := 796) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_7
  simp only [← c49773, ← c49774, k2, ← c49775] at e_796_7
  have hr := e_796_7
  simp only [e_796_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5649)) := by
  have c49776 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49777 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49778 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49779 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49780 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49781 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_8 := arithEq_of_rows h (row := 796) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_8
  simp only [← c49776, ← c49777, ← c49778] at e_796_8
  have e_796_9 := arithEq_of_rows h (row := 796) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_9
  simp only [← c49779, ← c49780, k2, ← c49781] at e_796_9
  have hr := e_796_9
  simp only [e_796_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5650)) := by
  have c49782 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49783 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49784 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49785 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49786 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49787 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_10 := arithEq_of_rows h (row := 796) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_10
  simp only [← c49782, ← c49783, ← c49784] at e_796_10
  have e_796_11 := arithEq_of_rows h (row := 796) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_11
  simp only [← c49785, ← c49786, k2, ← c49787] at e_796_11
  have hr := e_796_11
  simp only [e_796_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5651)) := by
  have c49788 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49789 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49790 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49791 := (publicBatchWrapper89_copies1555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49792 := (publicBatchWrapper89_copies1556 a h).1
  have c49793 := (publicBatchWrapper89_copies1556 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_12 := arithEq_of_rows h (row := 796) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_12
  simp only [← c49788, ← c49789, ← c49790] at e_796_12
  have e_796_13 := arithEq_of_rows h (row := 796) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_13
  simp only [← c49791, ← c49792, k2, ← c49793] at e_796_13
  have hr := e_796_13
  simp only [e_796_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5652)) := by
  have c49794 := (publicBatchWrapper89_copies1556 a h).2.2.1
  have c49795 := (publicBatchWrapper89_copies1556 a h).2.2.2.1
  have c49796 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.1
  have c49797 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.1
  have c49798 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.1
  have c49799 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_14 := arithEq_of_rows h (row := 796) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_14
  simp only [← c49794, ← c49795, ← c49796] at e_796_14
  have e_796_15 := arithEq_of_rows h (row := 796) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_15
  simp only [← c49797, ← c49798, k2, ← c49799] at e_796_15
  have hr := e_796_15
  simp only [e_796_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5653)) := by
  have c49800 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.1
  have c49801 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.1
  have c49802 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49803 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49804 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49805 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_16 := arithEq_of_rows h (row := 796) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_16
  simp only [← c49800, ← c49801, ← c49802] at e_796_16
  have e_796_17 := arithEq_of_rows h (row := 796) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_17
  simp only [← c49803, ← c49804, k2, ← c49805] at e_796_17
  have hr := e_796_17
  simp only [e_796_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 796 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5654)) := by
  have c49806 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49807 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49808 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49809 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49810 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49811 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_796_18 := arithEq_of_rows h (row := 796) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_18
  simp only [← c49806, ← c49807, ← c49808] at e_796_18
  have e_796_19 := arithEq_of_rows h (row := 796) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_796_19
  simp only [← c49809, ← c49810, k2, ← c49811] at e_796_19
  have hr := e_796_19
  simp only [e_796_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5655)) := by
  have c49812 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49813 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49814 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49815 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49816 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49817 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_0 := arithEq_of_rows h (row := 797) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_0
  simp only [← c49812, ← c49813, ← c49814] at e_797_0
  have e_797_1 := arithEq_of_rows h (row := 797) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_1
  simp only [← c49815, ← c49816, k2, ← c49817] at e_797_1
  have hr := e_797_1
  simp only [e_797_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5656)) := by
  have c49818 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49819 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49820 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49821 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49822 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49823 := (publicBatchWrapper89_copies1556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_2 := arithEq_of_rows h (row := 797) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_2
  simp only [← c49818, ← c49819, ← c49820] at e_797_2
  have e_797_3 := arithEq_of_rows h (row := 797) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_3
  simp only [← c49821, ← c49822, k2, ← c49823] at e_797_3
  have hr := e_797_3
  simp only [e_797_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5657)) := by
  have c49824 := (publicBatchWrapper89_copies1557 a h).1
  have c49825 := (publicBatchWrapper89_copies1557 a h).2.1
  have c49826 := (publicBatchWrapper89_copies1557 a h).2.2.1
  have c49827 := (publicBatchWrapper89_copies1557 a h).2.2.2.1
  have c49828 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.1
  have c49829 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_4 := arithEq_of_rows h (row := 797) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_4
  simp only [← c49824, ← c49825, ← c49826] at e_797_4
  have e_797_5 := arithEq_of_rows h (row := 797) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_5
  simp only [← c49827, ← c49828, k2, ← c49829] at e_797_5
  have hr := e_797_5
  simp only [e_797_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5658)) := by
  have c49830 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.1
  have c49831 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.1
  have c49832 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.1
  have c49833 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.1
  have c49834 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49835 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_6 := arithEq_of_rows h (row := 797) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_6
  simp only [← c49830, ← c49831, ← c49832] at e_797_6
  have e_797_7 := arithEq_of_rows h (row := 797) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_7
  simp only [← c49833, ← c49834, k2, ← c49835] at e_797_7
  have hr := e_797_7
  simp only [e_797_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5659)) := by
  have c49836 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49837 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49838 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49839 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49840 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49841 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_8 := arithEq_of_rows h (row := 797) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_8
  simp only [← c49836, ← c49837, ← c49838] at e_797_8
  have e_797_9 := arithEq_of_rows h (row := 797) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_9
  simp only [← c49839, ← c49840, k2, ← c49841] at e_797_9
  have hr := e_797_9
  simp only [e_797_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5660)) := by
  have c49842 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49843 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49844 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49845 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49846 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49847 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_10 := arithEq_of_rows h (row := 797) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_10
  simp only [← c49842, ← c49843, ← c49844] at e_797_10
  have e_797_11 := arithEq_of_rows h (row := 797) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_11
  simp only [← c49845, ← c49846, k2, ← c49847] at e_797_11
  have hr := e_797_11
  simp only [e_797_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5661)) := by
  have c49848 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49849 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49850 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49851 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49852 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49853 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_12 := arithEq_of_rows h (row := 797) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_12
  simp only [← c49848, ← c49849, ← c49850] at e_797_12
  have e_797_13 := arithEq_of_rows h (row := 797) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_13
  simp only [← c49851, ← c49852, k2, ← c49853] at e_797_13
  have hr := e_797_13
  simp only [e_797_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5662)) := by
  have c49854 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49855 := (publicBatchWrapper89_copies1557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49856 := (publicBatchWrapper89_copies1558 a h).1
  have c49857 := (publicBatchWrapper89_copies1558 a h).2.1
  have c49858 := (publicBatchWrapper89_copies1558 a h).2.2.1
  have c49859 := (publicBatchWrapper89_copies1558 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_14 := arithEq_of_rows h (row := 797) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_14
  simp only [← c49854, ← c49855, ← c49856] at e_797_14
  have e_797_15 := arithEq_of_rows h (row := 797) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_15
  simp only [← c49857, ← c49858, k2, ← c49859] at e_797_15
  have hr := e_797_15
  simp only [e_797_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5663)) := by
  have c49860 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.1
  have c49861 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.1
  have c49862 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.1
  have c49863 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.1
  have c49864 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.1
  have c49865 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_16 := arithEq_of_rows h (row := 797) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_16
  simp only [← c49860, ← c49861, ← c49862] at e_797_16
  have e_797_17 := arithEq_of_rows h (row := 797) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_17
  simp only [← c49863, ← c49864, k2, ← c49865] at e_797_17
  have hr := e_797_17
  simp only [e_797_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 797 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5664)) := by
  have c49866 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49867 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49868 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49869 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49870 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49871 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_797_18 := arithEq_of_rows h (row := 797) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_18
  simp only [← c49866, ← c49867, ← c49868] at e_797_18
  have e_797_19 := arithEq_of_rows h (row := 797) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_797_19
  simp only [← c49869, ← c49870, k2, ← c49871] at e_797_19
  have hr := e_797_19
  simp only [e_797_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5665)) := by
  have c49872 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49873 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49874 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49875 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49876 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49877 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_0 := arithEq_of_rows h (row := 798) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_0
  simp only [← c49872, ← c49873, ← c49874] at e_798_0
  have e_798_1 := arithEq_of_rows h (row := 798) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_1
  simp only [← c49875, ← c49876, k2, ← c49877] at e_798_1
  have hr := e_798_1
  simp only [e_798_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5666)) := by
  have c49878 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49879 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49880 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49881 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49882 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49883 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_2 := arithEq_of_rows h (row := 798) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_2
  simp only [← c49878, ← c49879, ← c49880] at e_798_2
  have e_798_3 := arithEq_of_rows h (row := 798) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_3
  simp only [← c49881, ← c49882, k2, ← c49883] at e_798_3
  have hr := e_798_3
  simp only [e_798_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5667)) := by
  have c49884 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49885 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49886 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49887 := (publicBatchWrapper89_copies1558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49888 := (publicBatchWrapper89_copies1559 a h).1
  have c49889 := (publicBatchWrapper89_copies1559 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_4 := arithEq_of_rows h (row := 798) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_4
  simp only [← c49884, ← c49885, ← c49886] at e_798_4
  have e_798_5 := arithEq_of_rows h (row := 798) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_5
  simp only [← c49887, ← c49888, k2, ← c49889] at e_798_5
  have hr := e_798_5
  simp only [e_798_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5668)) := by
  have c49890 := (publicBatchWrapper89_copies1559 a h).2.2.1
  have c49891 := (publicBatchWrapper89_copies1559 a h).2.2.2.1
  have c49892 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.1
  have c49893 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.1
  have c49894 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.1
  have c49895 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_6 := arithEq_of_rows h (row := 798) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_6
  simp only [← c49890, ← c49891, ← c49892] at e_798_6
  have e_798_7 := arithEq_of_rows h (row := 798) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_7
  simp only [← c49893, ← c49894, k2, ← c49895] at e_798_7
  have hr := e_798_7
  simp only [e_798_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5669)) := by
  have c49896 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.1
  have c49897 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.1
  have c49898 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49899 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49900 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49901 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_8 := arithEq_of_rows h (row := 798) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_8
  simp only [← c49896, ← c49897, ← c49898] at e_798_8
  have e_798_9 := arithEq_of_rows h (row := 798) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_9
  simp only [← c49899, ← c49900, k2, ← c49901] at e_798_9
  have hr := e_798_9
  simp only [e_798_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5670)) := by
  have c49902 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49903 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49904 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49905 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49906 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49907 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_10 := arithEq_of_rows h (row := 798) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_10
  simp only [← c49902, ← c49903, ← c49904] at e_798_10
  have e_798_11 := arithEq_of_rows h (row := 798) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_11
  simp only [← c49905, ← c49906, k2, ← c49907] at e_798_11
  have hr := e_798_11
  simp only [e_798_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5671)) := by
  have c49908 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49909 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49910 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49911 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49912 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49913 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_12 := arithEq_of_rows h (row := 798) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_12
  simp only [← c49908, ← c49909, ← c49910] at e_798_12
  have e_798_13 := arithEq_of_rows h (row := 798) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_13
  simp only [← c49911, ← c49912, k2, ← c49913] at e_798_13
  have hr := e_798_13
  simp only [e_798_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5672)) := by
  have c49914 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49915 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49916 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49917 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49918 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49919 := (publicBatchWrapper89_copies1559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_14 := arithEq_of_rows h (row := 798) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_14
  simp only [← c49914, ← c49915, ← c49916] at e_798_14
  have e_798_15 := arithEq_of_rows h (row := 798) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_15
  simp only [← c49917, ← c49918, k2, ← c49919] at e_798_15
  have hr := e_798_15
  simp only [e_798_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5673)) := by
  have c49920 := (publicBatchWrapper89_copies1560 a h).1
  have c49921 := (publicBatchWrapper89_copies1560 a h).2.1
  have c49922 := (publicBatchWrapper89_copies1560 a h).2.2.1
  have c49923 := (publicBatchWrapper89_copies1560 a h).2.2.2.1
  have c49924 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.1
  have c49925 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_16 := arithEq_of_rows h (row := 798) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_16
  simp only [← c49920, ← c49921, ← c49922] at e_798_16
  have e_798_17 := arithEq_of_rows h (row := 798) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_17
  simp only [← c49923, ← c49924, k2, ← c49925] at e_798_17
  have hr := e_798_17
  simp only [e_798_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 798 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5674)) := by
  have c49926 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.1
  have c49927 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.1
  have c49928 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.1
  have c49929 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.1
  have c49930 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49931 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_798_18 := arithEq_of_rows h (row := 798) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_18
  simp only [← c49926, ← c49927, ← c49928] at e_798_18
  have e_798_19 := arithEq_of_rows h (row := 798) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_798_19
  simp only [← c49929, ← c49930, k2, ← c49931] at e_798_19
  have hr := e_798_19
  simp only [e_798_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5675)) := by
  have c49932 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49933 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49934 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49935 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49936 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49937 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_0 := arithEq_of_rows h (row := 799) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_0
  simp only [← c49932, ← c49933, ← c49934] at e_799_0
  have e_799_1 := arithEq_of_rows h (row := 799) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_1
  simp only [← c49935, ← c49936, k2, ← c49937] at e_799_1
  have hr := e_799_1
  simp only [e_799_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5676)) := by
  have c49938 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49939 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49940 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49941 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49942 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49943 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_2 := arithEq_of_rows h (row := 799) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_2
  simp only [← c49938, ← c49939, ← c49940] at e_799_2
  have e_799_3 := arithEq_of_rows h (row := 799) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_3
  simp only [← c49941, ← c49942, k2, ← c49943] at e_799_3
  have hr := e_799_3
  simp only [e_799_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5677)) := by
  have c49944 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49945 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49946 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49947 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49948 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49949 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_4 := arithEq_of_rows h (row := 799) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_4
  simp only [← c49944, ← c49945, ← c49946] at e_799_4
  have e_799_5 := arithEq_of_rows h (row := 799) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_5
  simp only [← c49947, ← c49948, k2, ← c49949] at e_799_5
  have hr := e_799_5
  simp only [e_799_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5678)) := by
  have c49950 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49951 := (publicBatchWrapper89_copies1560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49952 := (publicBatchWrapper89_copies1561 a h).1
  have c49953 := (publicBatchWrapper89_copies1561 a h).2.1
  have c49954 := (publicBatchWrapper89_copies1561 a h).2.2.1
  have c49955 := (publicBatchWrapper89_copies1561 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_6 := arithEq_of_rows h (row := 799) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_6
  simp only [← c49950, ← c49951, ← c49952] at e_799_6
  have e_799_7 := arithEq_of_rows h (row := 799) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_7
  simp only [← c49953, ← c49954, k2, ← c49955] at e_799_7
  have hr := e_799_7
  simp only [e_799_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5679)) := by
  have c49956 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.1
  have c49957 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.1
  have c49958 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.1
  have c49959 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.1
  have c49960 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.1
  have c49961 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_8 := arithEq_of_rows h (row := 799) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_8
  simp only [← c49956, ← c49957, ← c49958] at e_799_8
  have e_799_9 := arithEq_of_rows h (row := 799) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_9
  simp only [← c49959, ← c49960, k2, ← c49961] at e_799_9
  have hr := e_799_9
  simp only [e_799_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5680)) := by
  have c49962 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49963 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49964 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49965 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49966 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49967 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_10 := arithEq_of_rows h (row := 799) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_10
  simp only [← c49962, ← c49963, ← c49964] at e_799_10
  have e_799_11 := arithEq_of_rows h (row := 799) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_11
  simp only [← c49965, ← c49966, k2, ← c49967] at e_799_11
  have hr := e_799_11
  simp only [e_799_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5681)) := by
  have c49968 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49969 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49970 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49971 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49972 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49973 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_12 := arithEq_of_rows h (row := 799) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_12
  simp only [← c49968, ← c49969, ← c49970] at e_799_12
  have e_799_13 := arithEq_of_rows h (row := 799) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_13
  simp only [← c49971, ← c49972, k2, ← c49973] at e_799_13
  have hr := e_799_13
  simp only [e_799_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5682)) := by
  have c49974 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49975 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49976 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49977 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49978 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49979 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_14 := arithEq_of_rows h (row := 799) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_14
  simp only [← c49974, ← c49975, ← c49976] at e_799_14
  have e_799_15 := arithEq_of_rows h (row := 799) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_15
  simp only [← c49977, ← c49978, k2, ← c49979] at e_799_15
  have hr := e_799_15
  simp only [e_799_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5683)) := by
  have c49980 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49981 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49982 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49983 := (publicBatchWrapper89_copies1561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c49984 := (publicBatchWrapper89_copies1562 a h).1
  have c49985 := (publicBatchWrapper89_copies1562 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_16 := arithEq_of_rows h (row := 799) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_16
  simp only [← c49980, ← c49981, ← c49982] at e_799_16
  have e_799_17 := arithEq_of_rows h (row := 799) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_17
  simp only [← c49983, ← c49984, k2, ← c49985] at e_799_17
  have hr := e_799_17
  simp only [e_799_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 799 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5684)) := by
  have c49986 := (publicBatchWrapper89_copies1562 a h).2.2.1
  have c49987 := (publicBatchWrapper89_copies1562 a h).2.2.2.1
  have c49988 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.1
  have c49989 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.1
  have c49990 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.1
  have c49991 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_799_18 := arithEq_of_rows h (row := 799) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_18
  simp only [← c49986, ← c49987, ← c49988] at e_799_18
  have e_799_19 := arithEq_of_rows h (row := 799) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_799_19
  simp only [← c49989, ← c49990, k2, ← c49991] at e_799_19
  have hr := e_799_19
  simp only [e_799_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5685)) := by
  have c49992 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.1
  have c49993 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.1
  have c49994 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c49995 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c49996 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49997 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_0 := arithEq_of_rows h (row := 800) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_0
  simp only [← c49992, ← c49993, ← c49994] at e_800_0
  have e_800_1 := arithEq_of_rows h (row := 800) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_1
  simp only [← c49995, ← c49996, k2, ← c49997] at e_800_1
  have hr := e_800_1
  simp only [e_800_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5686)) := by
  have c49998 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49999 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50000 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50001 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50002 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50003 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_2 := arithEq_of_rows h (row := 800) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_2
  simp only [← c49998, ← c49999, ← c50000] at e_800_2
  have e_800_3 := arithEq_of_rows h (row := 800) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_3
  simp only [← c50001, ← c50002, k2, ← c50003] at e_800_3
  have hr := e_800_3
  simp only [e_800_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5687)) := by
  have c50004 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50005 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50006 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50007 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50008 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50009 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_4 := arithEq_of_rows h (row := 800) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_4
  simp only [← c50004, ← c50005, ← c50006] at e_800_4
  have e_800_5 := arithEq_of_rows h (row := 800) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_5
  simp only [← c50007, ← c50008, k2, ← c50009] at e_800_5
  have hr := e_800_5
  simp only [e_800_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5688)) := by
  have c50010 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50011 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50012 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50013 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50014 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50015 := (publicBatchWrapper89_copies1562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_6 := arithEq_of_rows h (row := 800) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_6
  simp only [← c50010, ← c50011, ← c50012] at e_800_6
  have e_800_7 := arithEq_of_rows h (row := 800) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_7
  simp only [← c50013, ← c50014, k2, ← c50015] at e_800_7
  have hr := e_800_7
  simp only [e_800_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5689)) := by
  have c50016 := (publicBatchWrapper89_copies1563 a h).1
  have c50017 := (publicBatchWrapper89_copies1563 a h).2.1
  have c50018 := (publicBatchWrapper89_copies1563 a h).2.2.1
  have c50019 := (publicBatchWrapper89_copies1563 a h).2.2.2.1
  have c50020 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.1
  have c50021 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_8 := arithEq_of_rows h (row := 800) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_8
  simp only [← c50016, ← c50017, ← c50018] at e_800_8
  have e_800_9 := arithEq_of_rows h (row := 800) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_9
  simp only [← c50019, ← c50020, k2, ← c50021] at e_800_9
  have hr := e_800_9
  simp only [e_800_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5690)) := by
  have c50022 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.1
  have c50023 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.1
  have c50024 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.1
  have c50025 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.1
  have c50026 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50027 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_10 := arithEq_of_rows h (row := 800) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_10
  simp only [← c50022, ← c50023, ← c50024] at e_800_10
  have e_800_11 := arithEq_of_rows h (row := 800) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_11
  simp only [← c50025, ← c50026, k2, ← c50027] at e_800_11
  have hr := e_800_11
  simp only [e_800_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5691)) := by
  have c50028 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50029 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50030 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50031 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50032 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50033 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_12 := arithEq_of_rows h (row := 800) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_12
  simp only [← c50028, ← c50029, ← c50030] at e_800_12
  have e_800_13 := arithEq_of_rows h (row := 800) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_13
  simp only [← c50031, ← c50032, k2, ← c50033] at e_800_13
  have hr := e_800_13
  simp only [e_800_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5692)) := by
  have c50034 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50035 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50036 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50037 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50038 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50039 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_14 := arithEq_of_rows h (row := 800) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_14
  simp only [← c50034, ← c50035, ← c50036] at e_800_14
  have e_800_15 := arithEq_of_rows h (row := 800) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_15
  simp only [← c50037, ← c50038, k2, ← c50039] at e_800_15
  have hr := e_800_15
  simp only [e_800_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5693)) := by
  have c50040 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50041 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50042 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50043 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50044 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50045 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_16 := arithEq_of_rows h (row := 800) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_16
  simp only [← c50040, ← c50041, ← c50042] at e_800_16
  have e_800_17 := arithEq_of_rows h (row := 800) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_17
  simp only [← c50043, ← c50044, k2, ← c50045] at e_800_17
  have hr := e_800_17
  simp only [e_800_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 800 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5694)) := by
  have c50046 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50047 := (publicBatchWrapper89_copies1563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c50048 := (publicBatchWrapper89_copies1564 a h).1
  have c50049 := (publicBatchWrapper89_copies1564 a h).2.1
  have c50050 := (publicBatchWrapper89_copies1564 a h).2.2.1
  have c50051 := (publicBatchWrapper89_copies1564 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_800_18 := arithEq_of_rows h (row := 800) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_18
  simp only [← c50046, ← c50047, ← c50048] at e_800_18
  have e_800_19 := arithEq_of_rows h (row := 800) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_800_19
  simp only [← c50049, ← c50050, k2, ← c50051] at e_800_19
  have hr := e_800_19
  simp only [e_800_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5695)) := by
  have c50052 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.1
  have c50053 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.1
  have c50054 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.1
  have c50055 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.1
  have c50056 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.1
  have c50057 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_0 := arithEq_of_rows h (row := 801) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_0
  simp only [← c50052, ← c50053, ← c50054] at e_801_0
  have e_801_1 := arithEq_of_rows h (row := 801) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_1
  simp only [← c50055, ← c50056, k2, ← c50057] at e_801_1
  have hr := e_801_1
  simp only [e_801_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f7592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 801 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5696)) := by
  have c50058 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.1
  have c50059 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c50060 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50061 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50062 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50063 := (publicBatchWrapper89_copies1564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_801_2 := arithEq_of_rows h (row := 801) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_2
  simp only [← c50058, ← c50059, ← c50060] at e_801_2
  have e_801_3 := arithEq_of_rows h (row := 801) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_801_3
  simp only [← c50061, ← c50062, k2, ← c50063] at e_801_3
  have hr := e_801_3
  simp only [e_801_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
