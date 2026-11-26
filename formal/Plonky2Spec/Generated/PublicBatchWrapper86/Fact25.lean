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

theorem publicBatchWrapper86_f19330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16592)) := by
  have c120333 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120334 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120335 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120336 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120337 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120338 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_16 := arithEq_of_rows h (row := 1973) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_16
  simp only [← c120333, ← c120334, ← c120335] at e_1973_16
  have e_1973_17 := arithEq_of_rows h (row := 1973) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_17
  simp only [← c120336, ← c120337, k2, ← c120338] at e_1973_17
  have hr := e_1973_17
  simp only [e_1973_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16593)) := by
  have c120339 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120340 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120341 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120342 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120343 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120344 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_18 := arithEq_of_rows h (row := 1973) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_18
  simp only [← c120339, ← c120340, ← c120341] at e_1973_18
  have e_1973_19 := arithEq_of_rows h (row := 1973) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_19
  simp only [← c120342, ← c120343, k2, ← c120344] at e_1973_19
  have hr := e_1973_19
  simp only [e_1973_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16594)) := by
  have c120345 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120346 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120347 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120348 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120349 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120350 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_0 := arithEq_of_rows h (row := 1974) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_0
  simp only [← c120345, ← c120346, ← c120347] at e_1974_0
  have e_1974_1 := arithEq_of_rows h (row := 1974) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_1
  simp only [← c120348, ← c120349, k2, ← c120350] at e_1974_1
  have hr := e_1974_1
  simp only [e_1974_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16595)) := by
  have c120351 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120352 := (publicBatchWrapper86_copies3761 a h).1
  have c120353 := (publicBatchWrapper86_copies3761 a h).2.1
  have c120354 := (publicBatchWrapper86_copies3761 a h).2.2.1
  have c120355 := (publicBatchWrapper86_copies3761 a h).2.2.2.1
  have c120356 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_2 := arithEq_of_rows h (row := 1974) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_2
  simp only [← c120351, ← c120352, ← c120353] at e_1974_2
  have e_1974_3 := arithEq_of_rows h (row := 1974) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_3
  simp only [← c120354, ← c120355, k2, ← c120356] at e_1974_3
  have hr := e_1974_3
  simp only [e_1974_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16596)) := by
  have c120357 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.1
  have c120358 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.1
  have c120359 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.1
  have c120360 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.1
  have c120361 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.1
  have c120362 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_4 := arithEq_of_rows h (row := 1974) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_4
  simp only [← c120357, ← c120358, ← c120359] at e_1974_4
  have e_1974_5 := arithEq_of_rows h (row := 1974) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_5
  simp only [← c120360, ← c120361, k2, ← c120362] at e_1974_5
  have hr := e_1974_5
  simp only [e_1974_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16597)) := by
  have c120363 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120364 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120365 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120366 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120367 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120368 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_6 := arithEq_of_rows h (row := 1974) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_6
  simp only [← c120363, ← c120364, ← c120365] at e_1974_6
  have e_1974_7 := arithEq_of_rows h (row := 1974) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_7
  simp only [← c120366, ← c120367, k2, ← c120368] at e_1974_7
  have hr := e_1974_7
  simp only [e_1974_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16598)) := by
  have c120369 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120370 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120371 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120372 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120373 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120374 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_8 := arithEq_of_rows h (row := 1974) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_8
  simp only [← c120369, ← c120370, ← c120371] at e_1974_8
  have e_1974_9 := arithEq_of_rows h (row := 1974) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_9
  simp only [← c120372, ← c120373, k2, ← c120374] at e_1974_9
  have hr := e_1974_9
  simp only [e_1974_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16599)) := by
  have c120375 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120376 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120377 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120378 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120379 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120380 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_10 := arithEq_of_rows h (row := 1974) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_10
  simp only [← c120375, ← c120376, ← c120377] at e_1974_10
  have e_1974_11 := arithEq_of_rows h (row := 1974) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_11
  simp only [← c120378, ← c120379, k2, ← c120380] at e_1974_11
  have hr := e_1974_11
  simp only [e_1974_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16600)) := by
  have c120381 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120382 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120383 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120384 := (publicBatchWrapper86_copies3762 a h).1
  have c120385 := (publicBatchWrapper86_copies3762 a h).2.1
  have c120386 := (publicBatchWrapper86_copies3762 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_12 := arithEq_of_rows h (row := 1974) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_12
  simp only [← c120381, ← c120382, ← c120383] at e_1974_12
  have e_1974_13 := arithEq_of_rows h (row := 1974) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_13
  simp only [← c120384, ← c120385, k2, ← c120386] at e_1974_13
  have hr := e_1974_13
  simp only [e_1974_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16601)) := by
  have c120387 := (publicBatchWrapper86_copies3762 a h).2.2.2.1
  have c120388 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.1
  have c120389 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.1
  have c120390 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.1
  have c120391 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.1
  have c120392 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_14 := arithEq_of_rows h (row := 1974) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_14
  simp only [← c120387, ← c120388, ← c120389] at e_1974_14
  have e_1974_15 := arithEq_of_rows h (row := 1974) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_15
  simp only [← c120390, ← c120391, k2, ← c120392] at e_1974_15
  have hr := e_1974_15
  simp only [e_1974_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16770)) := by
  have c120393 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.1
  have c120394 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120395 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120396 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120397 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120398 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_16 := arithEq_of_rows h (row := 1974) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_16
  simp only [← c120393, ← c120394, ← c120395] at e_1974_16
  have e_1974_17 := arithEq_of_rows h (row := 1974) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_17
  simp only [← c120396, ← c120397, k2, ← c120398] at e_1974_17
  have hr := e_1974_17
  simp only [e_1974_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16771)) := by
  have c120399 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120400 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120401 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120402 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120403 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120404 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_18 := arithEq_of_rows h (row := 1974) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_18
  simp only [← c120399, ← c120400, ← c120401] at e_1974_18
  have e_1974_19 := arithEq_of_rows h (row := 1974) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_19
  simp only [← c120402, ← c120403, k2, ← c120404] at e_1974_19
  have hr := e_1974_19
  simp only [e_1974_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16772)) := by
  have c120405 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120406 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120407 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120408 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120409 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120410 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_0 := arithEq_of_rows h (row := 1975) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_0
  simp only [← c120405, ← c120406, ← c120407] at e_1975_0
  have e_1975_1 := arithEq_of_rows h (row := 1975) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_1
  simp only [← c120408, ← c120409, k2, ← c120410] at e_1975_1
  have hr := e_1975_1
  simp only [e_1975_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16773)) := by
  have c120411 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120412 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120413 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120414 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120415 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120416 := (publicBatchWrapper86_copies3763 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_2 := arithEq_of_rows h (row := 1975) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_2
  simp only [← c120411, ← c120412, ← c120413] at e_1975_2
  have e_1975_3 := arithEq_of_rows h (row := 1975) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_3
  simp only [← c120414, ← c120415, k2, ← c120416] at e_1975_3
  have hr := e_1975_3
  simp only [e_1975_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16774)) := by
  have c120417 := (publicBatchWrapper86_copies3763 a h).2.1
  have c120418 := (publicBatchWrapper86_copies3763 a h).2.2.1
  have c120419 := (publicBatchWrapper86_copies3763 a h).2.2.2.1
  have c120420 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.1
  have c120421 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.1
  have c120422 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_4 := arithEq_of_rows h (row := 1975) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_4
  simp only [← c120417, ← c120418, ← c120419] at e_1975_4
  have e_1975_5 := arithEq_of_rows h (row := 1975) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_5
  simp only [← c120420, ← c120421, k2, ← c120422] at e_1975_5
  have hr := e_1975_5
  simp only [e_1975_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16775)) := by
  have c120423 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.1
  have c120424 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.1
  have c120425 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.1
  have c120426 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120427 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120428 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_6 := arithEq_of_rows h (row := 1975) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_6
  simp only [← c120423, ← c120424, ← c120425] at e_1975_6
  have e_1975_7 := arithEq_of_rows h (row := 1975) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_7
  simp only [← c120426, ← c120427, k2, ← c120428] at e_1975_7
  have hr := e_1975_7
  simp only [e_1975_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16776)) := by
  have c120429 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120430 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120431 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120432 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120433 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120434 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_8 := arithEq_of_rows h (row := 1975) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_8
  simp only [← c120429, ← c120430, ← c120431] at e_1975_8
  have e_1975_9 := arithEq_of_rows h (row := 1975) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_9
  simp only [← c120432, ← c120433, k2, ← c120434] at e_1975_9
  have hr := e_1975_9
  simp only [e_1975_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16777)) := by
  have c120435 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120436 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120437 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120438 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120439 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120440 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_10 := arithEq_of_rows h (row := 1975) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_10
  simp only [← c120435, ← c120436, ← c120437] at e_1975_10
  have e_1975_11 := arithEq_of_rows h (row := 1975) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_11
  simp only [← c120438, ← c120439, k2, ← c120440] at e_1975_11
  have hr := e_1975_11
  simp only [e_1975_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16778)) := by
  have c120441 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120442 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120443 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120444 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120445 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120446 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_12 := arithEq_of_rows h (row := 1975) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_12
  simp only [← c120441, ← c120442, ← c120443] at e_1975_12
  have e_1975_13 := arithEq_of_rows h (row := 1975) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_13
  simp only [← c120444, ← c120445, k2, ← c120446] at e_1975_13
  have hr := e_1975_13
  simp only [e_1975_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16779)) := by
  have c120447 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120448 := (publicBatchWrapper86_copies3764 a h).1
  have c120449 := (publicBatchWrapper86_copies3764 a h).2.1
  have c120450 := (publicBatchWrapper86_copies3764 a h).2.2.1
  have c120451 := (publicBatchWrapper86_copies3764 a h).2.2.2.1
  have c120452 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_14 := arithEq_of_rows h (row := 1975) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_14
  simp only [← c120447, ← c120448, ← c120449] at e_1975_14
  have e_1975_15 := arithEq_of_rows h (row := 1975) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_15
  simp only [← c120450, ← c120451, k2, ← c120452] at e_1975_15
  have hr := e_1975_15
  simp only [e_1975_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16780)) := by
  have c120453 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.1
  have c120454 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.1
  have c120455 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.1
  have c120456 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.1
  have c120457 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.1
  have c120458 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_16 := arithEq_of_rows h (row := 1975) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_16
  simp only [← c120453, ← c120454, ← c120455] at e_1975_16
  have e_1975_17 := arithEq_of_rows h (row := 1975) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_17
  simp only [← c120456, ← c120457, k2, ← c120458] at e_1975_17
  have hr := e_1975_17
  simp only [e_1975_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16781)) := by
  have c120459 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120460 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120461 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120462 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120463 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120464 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_18 := arithEq_of_rows h (row := 1975) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_18
  simp only [← c120459, ← c120460, ← c120461] at e_1975_18
  have e_1975_19 := arithEq_of_rows h (row := 1975) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_19
  simp only [← c120462, ← c120463, k2, ← c120464] at e_1975_19
  have hr := e_1975_19
  simp only [e_1975_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16782)) := by
  have c120465 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120466 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120467 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120468 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120469 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120470 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_0 := arithEq_of_rows h (row := 1976) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_0
  simp only [← c120465, ← c120466, ← c120467] at e_1976_0
  have e_1976_1 := arithEq_of_rows h (row := 1976) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_1
  simp only [← c120468, ← c120469, k2, ← c120470] at e_1976_1
  have hr := e_1976_1
  simp only [e_1976_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16783)) := by
  have c120471 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120472 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120473 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120474 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120475 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120476 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_2 := arithEq_of_rows h (row := 1976) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_2
  simp only [← c120471, ← c120472, ← c120473] at e_1976_2
  have e_1976_3 := arithEq_of_rows h (row := 1976) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_3
  simp only [← c120474, ← c120475, k2, ← c120476] at e_1976_3
  have hr := e_1976_3
  simp only [e_1976_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16784)) := by
  have c120477 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120478 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120479 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120480 := (publicBatchWrapper86_copies3765 a h).1
  have c120481 := (publicBatchWrapper86_copies3765 a h).2.1
  have c120482 := (publicBatchWrapper86_copies3765 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_4 := arithEq_of_rows h (row := 1976) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_4
  simp only [← c120477, ← c120478, ← c120479] at e_1976_4
  have e_1976_5 := arithEq_of_rows h (row := 1976) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_5
  simp only [← c120480, ← c120481, k2, ← c120482] at e_1976_5
  have hr := e_1976_5
  simp only [e_1976_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16785)) := by
  have c120483 := (publicBatchWrapper86_copies3765 a h).2.2.2.1
  have c120484 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.1
  have c120485 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.1
  have c120486 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.1
  have c120487 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.1
  have c120488 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_6 := arithEq_of_rows h (row := 1976) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_6
  simp only [← c120483, ← c120484, ← c120485] at e_1976_6
  have e_1976_7 := arithEq_of_rows h (row := 1976) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_7
  simp only [← c120486, ← c120487, k2, ← c120488] at e_1976_7
  have hr := e_1976_7
  simp only [e_1976_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16786)) := by
  have c120489 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.1
  have c120490 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120491 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120492 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120493 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120494 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_8 := arithEq_of_rows h (row := 1976) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_8
  simp only [← c120489, ← c120490, ← c120491] at e_1976_8
  have e_1976_9 := arithEq_of_rows h (row := 1976) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_9
  simp only [← c120492, ← c120493, k2, ← c120494] at e_1976_9
  have hr := e_1976_9
  simp only [e_1976_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16787)) := by
  have c120495 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120496 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120497 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120498 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120499 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120500 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_10 := arithEq_of_rows h (row := 1976) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_10
  simp only [← c120495, ← c120496, ← c120497] at e_1976_10
  have e_1976_11 := arithEq_of_rows h (row := 1976) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_11
  simp only [← c120498, ← c120499, k2, ← c120500] at e_1976_11
  have hr := e_1976_11
  simp only [e_1976_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16788)) := by
  have c120501 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120502 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120503 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120504 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120505 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120506 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_12 := arithEq_of_rows h (row := 1976) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_12
  simp only [← c120501, ← c120502, ← c120503] at e_1976_12
  have e_1976_13 := arithEq_of_rows h (row := 1976) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_13
  simp only [← c120504, ← c120505, k2, ← c120506] at e_1976_13
  have hr := e_1976_13
  simp only [e_1976_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16789)) := by
  have c120507 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120508 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120509 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120510 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120511 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120512 := (publicBatchWrapper86_copies3766 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_14 := arithEq_of_rows h (row := 1976) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_14
  simp only [← c120507, ← c120508, ← c120509] at e_1976_14
  have e_1976_15 := arithEq_of_rows h (row := 1976) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_15
  simp only [← c120510, ← c120511, k2, ← c120512] at e_1976_15
  have hr := e_1976_15
  simp only [e_1976_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16790)) := by
  have c120513 := (publicBatchWrapper86_copies3766 a h).2.1
  have c120514 := (publicBatchWrapper86_copies3766 a h).2.2.1
  have c120515 := (publicBatchWrapper86_copies3766 a h).2.2.2.1
  have c120516 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.1
  have c120517 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.1
  have c120518 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_16 := arithEq_of_rows h (row := 1976) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_16
  simp only [← c120513, ← c120514, ← c120515] at e_1976_16
  have e_1976_17 := arithEq_of_rows h (row := 1976) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_17
  simp only [← c120516, ← c120517, k2, ← c120518] at e_1976_17
  have hr := e_1976_17
  simp only [e_1976_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16791)) := by
  have c120519 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.1
  have c120520 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.1
  have c120521 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.1
  have c120522 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120523 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120524 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_18 := arithEq_of_rows h (row := 1976) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_18
  simp only [← c120519, ← c120520, ← c120521] at e_1976_18
  have e_1976_19 := arithEq_of_rows h (row := 1976) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_19
  simp only [← c120522, ← c120523, k2, ← c120524] at e_1976_19
  have hr := e_1976_19
  simp only [e_1976_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16792)) := by
  have c120525 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120526 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120527 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120528 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120529 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120530 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_0 := arithEq_of_rows h (row := 1977) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_0
  simp only [← c120525, ← c120526, ← c120527] at e_1977_0
  have e_1977_1 := arithEq_of_rows h (row := 1977) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_1
  simp only [← c120528, ← c120529, k2, ← c120530] at e_1977_1
  have hr := e_1977_1
  simp only [e_1977_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16793)) := by
  have c120531 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120532 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120533 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120534 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120535 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120536 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_2 := arithEq_of_rows h (row := 1977) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_2
  simp only [← c120531, ← c120532, ← c120533] at e_1977_2
  have e_1977_3 := arithEq_of_rows h (row := 1977) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_3
  simp only [← c120534, ← c120535, k2, ← c120536] at e_1977_3
  have hr := e_1977_3
  simp only [e_1977_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16794)) := by
  have c120537 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120538 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120539 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120540 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120541 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120542 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_4 := arithEq_of_rows h (row := 1977) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_4
  simp only [← c120537, ← c120538, ← c120539] at e_1977_4
  have e_1977_5 := arithEq_of_rows h (row := 1977) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_5
  simp only [← c120540, ← c120541, k2, ← c120542] at e_1977_5
  have hr := e_1977_5
  simp only [e_1977_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16795)) := by
  have c120543 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120544 := (publicBatchWrapper86_copies3767 a h).1
  have c120545 := (publicBatchWrapper86_copies3767 a h).2.1
  have c120546 := (publicBatchWrapper86_copies3767 a h).2.2.1
  have c120547 := (publicBatchWrapper86_copies3767 a h).2.2.2.1
  have c120548 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_6 := arithEq_of_rows h (row := 1977) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_6
  simp only [← c120543, ← c120544, ← c120545] at e_1977_6
  have e_1977_7 := arithEq_of_rows h (row := 1977) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_7
  simp only [← c120546, ← c120547, k2, ← c120548] at e_1977_7
  have hr := e_1977_7
  simp only [e_1977_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16796)) := by
  have c120549 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.1
  have c120550 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.1
  have c120551 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.1
  have c120552 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.1
  have c120553 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.1
  have c120554 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_8 := arithEq_of_rows h (row := 1977) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_8
  simp only [← c120549, ← c120550, ← c120551] at e_1977_8
  have e_1977_9 := arithEq_of_rows h (row := 1977) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_9
  simp only [← c120552, ← c120553, k2, ← c120554] at e_1977_9
  have hr := e_1977_9
  simp only [e_1977_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16797)) := by
  have c120555 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120556 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120557 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120558 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120559 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120560 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_10 := arithEq_of_rows h (row := 1977) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_10
  simp only [← c120555, ← c120556, ← c120557] at e_1977_10
  have e_1977_11 := arithEq_of_rows h (row := 1977) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_11
  simp only [← c120558, ← c120559, k2, ← c120560] at e_1977_11
  have hr := e_1977_11
  simp only [e_1977_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16798)) := by
  have c120561 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120562 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120563 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120564 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120565 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120566 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_12 := arithEq_of_rows h (row := 1977) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_12
  simp only [← c120561, ← c120562, ← c120563] at e_1977_12
  have e_1977_13 := arithEq_of_rows h (row := 1977) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_13
  simp only [← c120564, ← c120565, k2, ← c120566] at e_1977_13
  have hr := e_1977_13
  simp only [e_1977_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16799)) := by
  have c120567 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120568 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120569 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120570 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120571 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120572 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_14 := arithEq_of_rows h (row := 1977) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_14
  simp only [← c120567, ← c120568, ← c120569] at e_1977_14
  have e_1977_15 := arithEq_of_rows h (row := 1977) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_15
  simp only [← c120570, ← c120571, k2, ← c120572] at e_1977_15
  have hr := e_1977_15
  simp only [e_1977_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16800)) := by
  have c120573 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120574 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120575 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120576 := (publicBatchWrapper86_copies3768 a h).1
  have c120577 := (publicBatchWrapper86_copies3768 a h).2.1
  have c120578 := (publicBatchWrapper86_copies3768 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_16 := arithEq_of_rows h (row := 1977) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_16
  simp only [← c120573, ← c120574, ← c120575] at e_1977_16
  have e_1977_17 := arithEq_of_rows h (row := 1977) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_17
  simp only [← c120576, ← c120577, k2, ← c120578] at e_1977_17
  have hr := e_1977_17
  simp only [e_1977_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16801)) := by
  have c120579 := (publicBatchWrapper86_copies3768 a h).2.2.2.1
  have c120580 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.1
  have c120581 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.1
  have c120582 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.1
  have c120583 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.1
  have c120584 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_18 := arithEq_of_rows h (row := 1977) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_18
  simp only [← c120579, ← c120580, ← c120581] at e_1977_18
  have e_1977_19 := arithEq_of_rows h (row := 1977) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_19
  simp only [← c120582, ← c120583, k2, ← c120584] at e_1977_19
  have hr := e_1977_19
  simp only [e_1977_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16970)) := by
  have c120585 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.1
  have c120586 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120587 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120588 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120589 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120590 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_0 := arithEq_of_rows h (row := 1978) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_0
  simp only [← c120585, ← c120586, ← c120587] at e_1978_0
  have e_1978_1 := arithEq_of_rows h (row := 1978) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_1
  simp only [← c120588, ← c120589, k2, ← c120590] at e_1978_1
  have hr := e_1978_1
  simp only [e_1978_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16971)) := by
  have c120591 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120592 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120593 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120594 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120595 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120596 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_2 := arithEq_of_rows h (row := 1978) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_2
  simp only [← c120591, ← c120592, ← c120593] at e_1978_2
  have e_1978_3 := arithEq_of_rows h (row := 1978) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_3
  simp only [← c120594, ← c120595, k2, ← c120596] at e_1978_3
  have hr := e_1978_3
  simp only [e_1978_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16972)) := by
  have c120597 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120598 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120599 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120600 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120601 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120602 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_4 := arithEq_of_rows h (row := 1978) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_4
  simp only [← c120597, ← c120598, ← c120599] at e_1978_4
  have e_1978_5 := arithEq_of_rows h (row := 1978) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_5
  simp only [← c120600, ← c120601, k2, ← c120602] at e_1978_5
  have hr := e_1978_5
  simp only [e_1978_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16973)) := by
  have c120603 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120604 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120605 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120606 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120607 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120608 := (publicBatchWrapper86_copies3769 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_6 := arithEq_of_rows h (row := 1978) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_6
  simp only [← c120603, ← c120604, ← c120605] at e_1978_6
  have e_1978_7 := arithEq_of_rows h (row := 1978) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_7
  simp only [← c120606, ← c120607, k2, ← c120608] at e_1978_7
  have hr := e_1978_7
  simp only [e_1978_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16974)) := by
  have c120609 := (publicBatchWrapper86_copies3769 a h).2.1
  have c120610 := (publicBatchWrapper86_copies3769 a h).2.2.1
  have c120611 := (publicBatchWrapper86_copies3769 a h).2.2.2.1
  have c120612 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.1
  have c120613 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.1
  have c120614 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_8 := arithEq_of_rows h (row := 1978) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_8
  simp only [← c120609, ← c120610, ← c120611] at e_1978_8
  have e_1978_9 := arithEq_of_rows h (row := 1978) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_9
  simp only [← c120612, ← c120613, k2, ← c120614] at e_1978_9
  have hr := e_1978_9
  simp only [e_1978_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16975)) := by
  have c120615 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.1
  have c120616 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.1
  have c120617 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.1
  have c120618 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120619 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120620 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_10 := arithEq_of_rows h (row := 1978) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_10
  simp only [← c120615, ← c120616, ← c120617] at e_1978_10
  have e_1978_11 := arithEq_of_rows h (row := 1978) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_11
  simp only [← c120618, ← c120619, k2, ← c120620] at e_1978_11
  have hr := e_1978_11
  simp only [e_1978_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16976)) := by
  have c120621 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120622 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120623 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120624 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120625 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120626 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_12 := arithEq_of_rows h (row := 1978) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_12
  simp only [← c120621, ← c120622, ← c120623] at e_1978_12
  have e_1978_13 := arithEq_of_rows h (row := 1978) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_13
  simp only [← c120624, ← c120625, k2, ← c120626] at e_1978_13
  have hr := e_1978_13
  simp only [e_1978_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16977)) := by
  have c120627 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120628 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120629 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120630 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120631 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120632 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_14 := arithEq_of_rows h (row := 1978) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_14
  simp only [← c120627, ← c120628, ← c120629] at e_1978_14
  have e_1978_15 := arithEq_of_rows h (row := 1978) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_15
  simp only [← c120630, ← c120631, k2, ← c120632] at e_1978_15
  have hr := e_1978_15
  simp only [e_1978_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16978)) := by
  have c120633 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120634 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120635 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120636 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120637 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120638 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_16 := arithEq_of_rows h (row := 1978) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_16
  simp only [← c120633, ← c120634, ← c120635] at e_1978_16
  have e_1978_17 := arithEq_of_rows h (row := 1978) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_17
  simp only [← c120636, ← c120637, k2, ← c120638] at e_1978_17
  have hr := e_1978_17
  simp only [e_1978_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16979)) := by
  have c120639 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120640 := (publicBatchWrapper86_copies3770 a h).1
  have c120641 := (publicBatchWrapper86_copies3770 a h).2.1
  have c120642 := (publicBatchWrapper86_copies3770 a h).2.2.1
  have c120643 := (publicBatchWrapper86_copies3770 a h).2.2.2.1
  have c120644 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_18 := arithEq_of_rows h (row := 1978) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_18
  simp only [← c120639, ← c120640, ← c120641] at e_1978_18
  have e_1978_19 := arithEq_of_rows h (row := 1978) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_19
  simp only [← c120642, ← c120643, k2, ← c120644] at e_1978_19
  have hr := e_1978_19
  simp only [e_1978_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16980)) := by
  have c120645 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.1
  have c120646 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.1
  have c120647 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.1
  have c120648 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.1
  have c120649 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.1
  have c120650 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_0 := arithEq_of_rows h (row := 1979) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_0
  simp only [← c120645, ← c120646, ← c120647] at e_1979_0
  have e_1979_1 := arithEq_of_rows h (row := 1979) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_1
  simp only [← c120648, ← c120649, k2, ← c120650] at e_1979_1
  have hr := e_1979_1
  simp only [e_1979_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16981)) := by
  have c120651 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120652 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120653 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120654 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120655 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120656 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_2 := arithEq_of_rows h (row := 1979) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_2
  simp only [← c120651, ← c120652, ← c120653] at e_1979_2
  have e_1979_3 := arithEq_of_rows h (row := 1979) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_3
  simp only [← c120654, ← c120655, k2, ← c120656] at e_1979_3
  have hr := e_1979_3
  simp only [e_1979_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16982)) := by
  have c120657 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120658 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120659 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120660 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120661 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120662 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_4 := arithEq_of_rows h (row := 1979) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_4
  simp only [← c120657, ← c120658, ← c120659] at e_1979_4
  have e_1979_5 := arithEq_of_rows h (row := 1979) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_5
  simp only [← c120660, ← c120661, k2, ← c120662] at e_1979_5
  have hr := e_1979_5
  simp only [e_1979_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16983)) := by
  have c120663 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120664 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120665 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120666 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120667 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120668 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_6 := arithEq_of_rows h (row := 1979) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_6
  simp only [← c120663, ← c120664, ← c120665] at e_1979_6
  have e_1979_7 := arithEq_of_rows h (row := 1979) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_7
  simp only [← c120666, ← c120667, k2, ← c120668] at e_1979_7
  have hr := e_1979_7
  simp only [e_1979_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16984)) := by
  have c120669 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120670 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120671 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120672 := (publicBatchWrapper86_copies3771 a h).1
  have c120673 := (publicBatchWrapper86_copies3771 a h).2.1
  have c120674 := (publicBatchWrapper86_copies3771 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_8 := arithEq_of_rows h (row := 1979) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_8
  simp only [← c120669, ← c120670, ← c120671] at e_1979_8
  have e_1979_9 := arithEq_of_rows h (row := 1979) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_9
  simp only [← c120672, ← c120673, k2, ← c120674] at e_1979_9
  have hr := e_1979_9
  simp only [e_1979_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16985)) := by
  have c120675 := (publicBatchWrapper86_copies3771 a h).2.2.2.1
  have c120676 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.1
  have c120677 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.1
  have c120678 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.1
  have c120679 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.1
  have c120680 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_10 := arithEq_of_rows h (row := 1979) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_10
  simp only [← c120675, ← c120676, ← c120677] at e_1979_10
  have e_1979_11 := arithEq_of_rows h (row := 1979) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_11
  simp only [← c120678, ← c120679, k2, ← c120680] at e_1979_11
  have hr := e_1979_11
  simp only [e_1979_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16986)) := by
  have c120681 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.1
  have c120682 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120683 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120684 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120685 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120686 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_12 := arithEq_of_rows h (row := 1979) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_12
  simp only [← c120681, ← c120682, ← c120683] at e_1979_12
  have e_1979_13 := arithEq_of_rows h (row := 1979) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_13
  simp only [← c120684, ← c120685, k2, ← c120686] at e_1979_13
  have hr := e_1979_13
  simp only [e_1979_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16987)) := by
  have c120687 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120688 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120689 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120690 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120691 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120692 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_14 := arithEq_of_rows h (row := 1979) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_14
  simp only [← c120687, ← c120688, ← c120689] at e_1979_14
  have e_1979_15 := arithEq_of_rows h (row := 1979) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_15
  simp only [← c120690, ← c120691, k2, ← c120692] at e_1979_15
  have hr := e_1979_15
  simp only [e_1979_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16988)) := by
  have c120693 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120694 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120695 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120696 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120697 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120698 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_16 := arithEq_of_rows h (row := 1979) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_16
  simp only [← c120693, ← c120694, ← c120695] at e_1979_16
  have e_1979_17 := arithEq_of_rows h (row := 1979) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_17
  simp only [← c120696, ← c120697, k2, ← c120698] at e_1979_17
  have hr := e_1979_17
  simp only [e_1979_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16989)) := by
  have c120699 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120700 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120701 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120702 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120703 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120704 := (publicBatchWrapper86_copies3772 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_18 := arithEq_of_rows h (row := 1979) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_18
  simp only [← c120699, ← c120700, ← c120701] at e_1979_18
  have e_1979_19 := arithEq_of_rows h (row := 1979) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_19
  simp only [← c120702, ← c120703, k2, ← c120704] at e_1979_19
  have hr := e_1979_19
  simp only [e_1979_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16990)) := by
  have c120705 := (publicBatchWrapper86_copies3772 a h).2.1
  have c120706 := (publicBatchWrapper86_copies3772 a h).2.2.1
  have c120707 := (publicBatchWrapper86_copies3772 a h).2.2.2.1
  have c120708 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.1
  have c120709 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.1
  have c120710 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_0 := arithEq_of_rows h (row := 1980) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_0
  simp only [← c120705, ← c120706, ← c120707] at e_1980_0
  have e_1980_1 := arithEq_of_rows h (row := 1980) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_1
  simp only [← c120708, ← c120709, k2, ← c120710] at e_1980_1
  have hr := e_1980_1
  simp only [e_1980_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16991)) := by
  have c120711 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.1
  have c120712 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.1
  have c120713 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.1
  have c120714 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120715 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120716 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_2 := arithEq_of_rows h (row := 1980) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_2
  simp only [← c120711, ← c120712, ← c120713] at e_1980_2
  have e_1980_3 := arithEq_of_rows h (row := 1980) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_3
  simp only [← c120714, ← c120715, k2, ← c120716] at e_1980_3
  have hr := e_1980_3
  simp only [e_1980_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16992)) := by
  have c120717 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120718 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120719 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120720 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120721 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120722 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_4 := arithEq_of_rows h (row := 1980) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_4
  simp only [← c120717, ← c120718, ← c120719] at e_1980_4
  have e_1980_5 := arithEq_of_rows h (row := 1980) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_5
  simp only [← c120720, ← c120721, k2, ← c120722] at e_1980_5
  have hr := e_1980_5
  simp only [e_1980_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16993)) := by
  have c120723 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120724 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120725 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120726 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120727 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120728 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_6 := arithEq_of_rows h (row := 1980) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_6
  simp only [← c120723, ← c120724, ← c120725] at e_1980_6
  have e_1980_7 := arithEq_of_rows h (row := 1980) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_7
  simp only [← c120726, ← c120727, k2, ← c120728] at e_1980_7
  have hr := e_1980_7
  simp only [e_1980_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16994)) := by
  have c120729 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120730 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120731 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120732 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120733 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120734 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_8 := arithEq_of_rows h (row := 1980) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_8
  simp only [← c120729, ← c120730, ← c120731] at e_1980_8
  have e_1980_9 := arithEq_of_rows h (row := 1980) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_9
  simp only [← c120732, ← c120733, k2, ← c120734] at e_1980_9
  have hr := e_1980_9
  simp only [e_1980_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16995)) := by
  have c120735 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120736 := (publicBatchWrapper86_copies3773 a h).1
  have c120737 := (publicBatchWrapper86_copies3773 a h).2.1
  have c120738 := (publicBatchWrapper86_copies3773 a h).2.2.1
  have c120739 := (publicBatchWrapper86_copies3773 a h).2.2.2.1
  have c120740 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_10 := arithEq_of_rows h (row := 1980) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_10
  simp only [← c120735, ← c120736, ← c120737] at e_1980_10
  have e_1980_11 := arithEq_of_rows h (row := 1980) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_11
  simp only [← c120738, ← c120739, k2, ← c120740] at e_1980_11
  have hr := e_1980_11
  simp only [e_1980_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16996)) := by
  have c120741 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.1
  have c120742 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.1
  have c120743 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.1
  have c120744 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.1
  have c120745 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.1
  have c120746 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_12 := arithEq_of_rows h (row := 1980) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_12
  simp only [← c120741, ← c120742, ← c120743] at e_1980_12
  have e_1980_13 := arithEq_of_rows h (row := 1980) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_13
  simp only [← c120744, ← c120745, k2, ← c120746] at e_1980_13
  have hr := e_1980_13
  simp only [e_1980_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16997)) := by
  have c120747 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120748 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120749 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120750 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120751 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120752 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_14 := arithEq_of_rows h (row := 1980) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_14
  simp only [← c120747, ← c120748, ← c120749] at e_1980_14
  have e_1980_15 := arithEq_of_rows h (row := 1980) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_15
  simp only [← c120750, ← c120751, k2, ← c120752] at e_1980_15
  have hr := e_1980_15
  simp only [e_1980_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16998)) := by
  have c120753 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120754 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120755 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120756 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120757 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120758 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_16 := arithEq_of_rows h (row := 1980) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_16
  simp only [← c120753, ← c120754, ← c120755] at e_1980_16
  have e_1980_17 := arithEq_of_rows h (row := 1980) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_17
  simp only [← c120756, ← c120757, k2, ← c120758] at e_1980_17
  have hr := e_1980_17
  simp only [e_1980_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16999)) := by
  have c120759 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120760 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120761 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120762 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120763 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120764 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_18 := arithEq_of_rows h (row := 1980) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_18
  simp only [← c120759, ← c120760, ← c120761] at e_1980_18
  have e_1980_19 := arithEq_of_rows h (row := 1980) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_19
  simp only [← c120762, ← c120763, k2, ← c120764] at e_1980_19
  have hr := e_1980_19
  simp only [e_1980_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 17000)) := by
  have c120765 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120766 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120767 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120768 := (publicBatchWrapper86_copies3774 a h).1
  have c120769 := (publicBatchWrapper86_copies3774 a h).2.1
  have c120770 := (publicBatchWrapper86_copies3774 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_0 := arithEq_of_rows h (row := 1981) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_0
  simp only [← c120765, ← c120766, ← c120767] at e_1981_0
  have e_1981_1 := arithEq_of_rows h (row := 1981) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_1
  simp only [← c120768, ← c120769, k2, ← c120770] at e_1981_1
  have hr := e_1981_1
  simp only [e_1981_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 17001)) := by
  have c120771 := (publicBatchWrapper86_copies3774 a h).2.2.2.1
  have c120772 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.1
  have c120773 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.1
  have c120774 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.1
  have c120775 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.1
  have c120776 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_2 := arithEq_of_rows h (row := 1981) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_2
  simp only [← c120771, ← c120772, ← c120773] at e_1981_2
  have e_1981_3 := arithEq_of_rows h (row := 1981) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_3
  simp only [← c120774, ← c120775, k2, ← c120776] at e_1981_3
  have hr := e_1981_3
  simp only [e_1981_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17170)) := by
  have c120777 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.1
  have c120778 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120779 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120780 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120781 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120782 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_4 := arithEq_of_rows h (row := 1981) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_4
  simp only [← c120777, ← c120778, ← c120779] at e_1981_4
  have e_1981_5 := arithEq_of_rows h (row := 1981) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_5
  simp only [← c120780, ← c120781, k2, ← c120782] at e_1981_5
  have hr := e_1981_5
  simp only [e_1981_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17171)) := by
  have c120783 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120784 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120785 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120786 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120787 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120788 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_6 := arithEq_of_rows h (row := 1981) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_6
  simp only [← c120783, ← c120784, ← c120785] at e_1981_6
  have e_1981_7 := arithEq_of_rows h (row := 1981) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_7
  simp only [← c120786, ← c120787, k2, ← c120788] at e_1981_7
  have hr := e_1981_7
  simp only [e_1981_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17172)) := by
  have c120789 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120790 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120791 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120792 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120793 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120794 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_8 := arithEq_of_rows h (row := 1981) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_8
  simp only [← c120789, ← c120790, ← c120791] at e_1981_8
  have e_1981_9 := arithEq_of_rows h (row := 1981) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_9
  simp only [← c120792, ← c120793, k2, ← c120794] at e_1981_9
  have hr := e_1981_9
  simp only [e_1981_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17173)) := by
  have c120795 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120796 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120797 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120798 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120799 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120800 := (publicBatchWrapper86_copies3775 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_10 := arithEq_of_rows h (row := 1981) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_10
  simp only [← c120795, ← c120796, ← c120797] at e_1981_10
  have e_1981_11 := arithEq_of_rows h (row := 1981) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_11
  simp only [← c120798, ← c120799, k2, ← c120800] at e_1981_11
  have hr := e_1981_11
  simp only [e_1981_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17174)) := by
  have c120801 := (publicBatchWrapper86_copies3775 a h).2.1
  have c120802 := (publicBatchWrapper86_copies3775 a h).2.2.1
  have c120803 := (publicBatchWrapper86_copies3775 a h).2.2.2.1
  have c120804 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.1
  have c120805 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.1
  have c120806 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_12 := arithEq_of_rows h (row := 1981) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_12
  simp only [← c120801, ← c120802, ← c120803] at e_1981_12
  have e_1981_13 := arithEq_of_rows h (row := 1981) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_13
  simp only [← c120804, ← c120805, k2, ← c120806] at e_1981_13
  have hr := e_1981_13
  simp only [e_1981_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17175)) := by
  have c120807 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.1
  have c120808 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.1
  have c120809 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.1
  have c120810 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120811 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120812 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_14 := arithEq_of_rows h (row := 1981) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_14
  simp only [← c120807, ← c120808, ← c120809] at e_1981_14
  have e_1981_15 := arithEq_of_rows h (row := 1981) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_15
  simp only [← c120810, ← c120811, k2, ← c120812] at e_1981_15
  have hr := e_1981_15
  simp only [e_1981_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17176)) := by
  have c120813 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120814 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120815 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120816 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120817 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120818 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_16 := arithEq_of_rows h (row := 1981) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_16
  simp only [← c120813, ← c120814, ← c120815] at e_1981_16
  have e_1981_17 := arithEq_of_rows h (row := 1981) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_17
  simp only [← c120816, ← c120817, k2, ← c120818] at e_1981_17
  have hr := e_1981_17
  simp only [e_1981_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17177)) := by
  have c120819 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120820 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120821 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120822 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120823 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120824 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_18 := arithEq_of_rows h (row := 1981) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_18
  simp only [← c120819, ← c120820, ← c120821] at e_1981_18
  have e_1981_19 := arithEq_of_rows h (row := 1981) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_19
  simp only [← c120822, ← c120823, k2, ← c120824] at e_1981_19
  have hr := e_1981_19
  simp only [e_1981_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17178)) := by
  have c120825 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120826 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120827 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120828 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120829 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120830 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_0 := arithEq_of_rows h (row := 1982) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_0
  simp only [← c120825, ← c120826, ← c120827] at e_1982_0
  have e_1982_1 := arithEq_of_rows h (row := 1982) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_1
  simp only [← c120828, ← c120829, k2, ← c120830] at e_1982_1
  have hr := e_1982_1
  simp only [e_1982_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17179)) := by
  have c120831 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120832 := (publicBatchWrapper86_copies3776 a h).1
  have c120833 := (publicBatchWrapper86_copies3776 a h).2.1
  have c120834 := (publicBatchWrapper86_copies3776 a h).2.2.1
  have c120835 := (publicBatchWrapper86_copies3776 a h).2.2.2.1
  have c120836 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_2 := arithEq_of_rows h (row := 1982) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_2
  simp only [← c120831, ← c120832, ← c120833] at e_1982_2
  have e_1982_3 := arithEq_of_rows h (row := 1982) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_3
  simp only [← c120834, ← c120835, k2, ← c120836] at e_1982_3
  have hr := e_1982_3
  simp only [e_1982_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17180)) := by
  have c120837 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.1
  have c120838 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.1
  have c120839 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.1
  have c120840 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.1
  have c120841 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.1
  have c120842 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_4 := arithEq_of_rows h (row := 1982) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_4
  simp only [← c120837, ← c120838, ← c120839] at e_1982_4
  have e_1982_5 := arithEq_of_rows h (row := 1982) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_5
  simp only [← c120840, ← c120841, k2, ← c120842] at e_1982_5
  have hr := e_1982_5
  simp only [e_1982_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17181)) := by
  have c120843 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120844 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120845 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120846 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120847 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120848 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_6 := arithEq_of_rows h (row := 1982) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_6
  simp only [← c120843, ← c120844, ← c120845] at e_1982_6
  have e_1982_7 := arithEq_of_rows h (row := 1982) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_7
  simp only [← c120846, ← c120847, k2, ← c120848] at e_1982_7
  have hr := e_1982_7
  simp only [e_1982_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17182)) := by
  have c120849 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120850 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120851 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120852 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120853 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120854 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_8 := arithEq_of_rows h (row := 1982) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_8
  simp only [← c120849, ← c120850, ← c120851] at e_1982_8
  have e_1982_9 := arithEq_of_rows h (row := 1982) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_9
  simp only [← c120852, ← c120853, k2, ← c120854] at e_1982_9
  have hr := e_1982_9
  simp only [e_1982_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17183)) := by
  have c120855 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120856 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120857 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120858 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120859 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120860 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_10 := arithEq_of_rows h (row := 1982) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_10
  simp only [← c120855, ← c120856, ← c120857] at e_1982_10
  have e_1982_11 := arithEq_of_rows h (row := 1982) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_11
  simp only [← c120858, ← c120859, k2, ← c120860] at e_1982_11
  have hr := e_1982_11
  simp only [e_1982_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17184)) := by
  have c120861 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120862 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120863 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120864 := (publicBatchWrapper86_copies3777 a h).1
  have c120865 := (publicBatchWrapper86_copies3777 a h).2.1
  have c120866 := (publicBatchWrapper86_copies3777 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_12 := arithEq_of_rows h (row := 1982) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_12
  simp only [← c120861, ← c120862, ← c120863] at e_1982_12
  have e_1982_13 := arithEq_of_rows h (row := 1982) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_13
  simp only [← c120864, ← c120865, k2, ← c120866] at e_1982_13
  have hr := e_1982_13
  simp only [e_1982_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17185)) := by
  have c120867 := (publicBatchWrapper86_copies3777 a h).2.2.2.1
  have c120868 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.1
  have c120869 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.1
  have c120870 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.1
  have c120871 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.1
  have c120872 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_14 := arithEq_of_rows h (row := 1982) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_14
  simp only [← c120867, ← c120868, ← c120869] at e_1982_14
  have e_1982_15 := arithEq_of_rows h (row := 1982) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_15
  simp only [← c120870, ← c120871, k2, ← c120872] at e_1982_15
  have hr := e_1982_15
  simp only [e_1982_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17186)) := by
  have c120873 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.1
  have c120874 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120875 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120876 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120877 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120878 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_16 := arithEq_of_rows h (row := 1982) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_16
  simp only [← c120873, ← c120874, ← c120875] at e_1982_16
  have e_1982_17 := arithEq_of_rows h (row := 1982) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_17
  simp only [← c120876, ← c120877, k2, ← c120878] at e_1982_17
  have hr := e_1982_17
  simp only [e_1982_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17187)) := by
  have c120879 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120880 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120881 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120882 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120883 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120884 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_18 := arithEq_of_rows h (row := 1982) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_18
  simp only [← c120879, ← c120880, ← c120881] at e_1982_18
  have e_1982_19 := arithEq_of_rows h (row := 1982) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_19
  simp only [← c120882, ← c120883, k2, ← c120884] at e_1982_19
  have hr := e_1982_19
  simp only [e_1982_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17188)) := by
  have c120885 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120886 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120887 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120888 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120889 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120890 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_0 := arithEq_of_rows h (row := 1983) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_0
  simp only [← c120885, ← c120886, ← c120887] at e_1983_0
  have e_1983_1 := arithEq_of_rows h (row := 1983) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_1
  simp only [← c120888, ← c120889, k2, ← c120890] at e_1983_1
  have hr := e_1983_1
  simp only [e_1983_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17189)) := by
  have c120891 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120892 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120893 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120894 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120895 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120896 := (publicBatchWrapper86_copies3778 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_2 := arithEq_of_rows h (row := 1983) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_2
  simp only [← c120891, ← c120892, ← c120893] at e_1983_2
  have e_1983_3 := arithEq_of_rows h (row := 1983) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_3
  simp only [← c120894, ← c120895, k2, ← c120896] at e_1983_3
  have hr := e_1983_3
  simp only [e_1983_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17190)) := by
  have c120897 := (publicBatchWrapper86_copies3778 a h).2.1
  have c120898 := (publicBatchWrapper86_copies3778 a h).2.2.1
  have c120899 := (publicBatchWrapper86_copies3778 a h).2.2.2.1
  have c120900 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.1
  have c120901 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.1
  have c120902 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_4 := arithEq_of_rows h (row := 1983) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_4
  simp only [← c120897, ← c120898, ← c120899] at e_1983_4
  have e_1983_5 := arithEq_of_rows h (row := 1983) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_5
  simp only [← c120900, ← c120901, k2, ← c120902] at e_1983_5
  have hr := e_1983_5
  simp only [e_1983_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17191)) := by
  have c120903 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.1
  have c120904 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.1
  have c120905 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.1
  have c120906 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120907 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120908 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_6 := arithEq_of_rows h (row := 1983) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_6
  simp only [← c120903, ← c120904, ← c120905] at e_1983_6
  have e_1983_7 := arithEq_of_rows h (row := 1983) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_7
  simp only [← c120906, ← c120907, k2, ← c120908] at e_1983_7
  have hr := e_1983_7
  simp only [e_1983_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17192)) := by
  have c120909 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120910 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120911 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120912 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120913 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120914 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_8 := arithEq_of_rows h (row := 1983) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_8
  simp only [← c120909, ← c120910, ← c120911] at e_1983_8
  have e_1983_9 := arithEq_of_rows h (row := 1983) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_9
  simp only [← c120912, ← c120913, k2, ← c120914] at e_1983_9
  have hr := e_1983_9
  simp only [e_1983_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17193)) := by
  have c120915 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120916 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120917 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120918 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120919 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120920 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_10 := arithEq_of_rows h (row := 1983) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_10
  simp only [← c120915, ← c120916, ← c120917] at e_1983_10
  have e_1983_11 := arithEq_of_rows h (row := 1983) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_11
  simp only [← c120918, ← c120919, k2, ← c120920] at e_1983_11
  have hr := e_1983_11
  simp only [e_1983_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17194)) := by
  have c120921 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120922 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120923 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120924 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120925 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120926 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_12 := arithEq_of_rows h (row := 1983) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_12
  simp only [← c120921, ← c120922, ← c120923] at e_1983_12
  have e_1983_13 := arithEq_of_rows h (row := 1983) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_13
  simp only [← c120924, ← c120925, k2, ← c120926] at e_1983_13
  have hr := e_1983_13
  simp only [e_1983_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17195)) := by
  have c120927 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120928 := (publicBatchWrapper86_copies3779 a h).1
  have c120929 := (publicBatchWrapper86_copies3779 a h).2.1
  have c120930 := (publicBatchWrapper86_copies3779 a h).2.2.1
  have c120931 := (publicBatchWrapper86_copies3779 a h).2.2.2.1
  have c120932 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_14 := arithEq_of_rows h (row := 1983) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_14
  simp only [← c120927, ← c120928, ← c120929] at e_1983_14
  have e_1983_15 := arithEq_of_rows h (row := 1983) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_15
  simp only [← c120930, ← c120931, k2, ← c120932] at e_1983_15
  have hr := e_1983_15
  simp only [e_1983_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17196)) := by
  have c120933 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.1
  have c120934 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.1
  have c120935 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.1
  have c120936 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.1
  have c120937 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.1
  have c120938 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_16 := arithEq_of_rows h (row := 1983) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_16
  simp only [← c120933, ← c120934, ← c120935] at e_1983_16
  have e_1983_17 := arithEq_of_rows h (row := 1983) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_17
  simp only [← c120936, ← c120937, k2, ← c120938] at e_1983_17
  have hr := e_1983_17
  simp only [e_1983_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17197)) := by
  have c120939 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120940 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120941 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120942 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120943 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120944 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_18 := arithEq_of_rows h (row := 1983) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_18
  simp only [← c120939, ← c120940, ← c120941] at e_1983_18
  have e_1983_19 := arithEq_of_rows h (row := 1983) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_19
  simp only [← c120942, ← c120943, k2, ← c120944] at e_1983_19
  have hr := e_1983_19
  simp only [e_1983_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17198)) := by
  have c120945 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120946 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120947 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120948 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120949 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120950 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_0 := arithEq_of_rows h (row := 1984) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_0
  simp only [← c120945, ← c120946, ← c120947] at e_1984_0
  have e_1984_1 := arithEq_of_rows h (row := 1984) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_1
  simp only [← c120948, ← c120949, k2, ← c120950] at e_1984_1
  have hr := e_1984_1
  simp only [e_1984_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17199)) := by
  have c120951 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120952 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120953 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120954 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120955 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120956 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_2 := arithEq_of_rows h (row := 1984) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_2
  simp only [← c120951, ← c120952, ← c120953] at e_1984_2
  have e_1984_3 := arithEq_of_rows h (row := 1984) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_3
  simp only [← c120954, ← c120955, k2, ← c120956] at e_1984_3
  have hr := e_1984_3
  simp only [e_1984_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17200)) := by
  have c120957 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120958 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120959 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120960 := (publicBatchWrapper86_copies3780 a h).1
  have c120961 := (publicBatchWrapper86_copies3780 a h).2.1
  have c120962 := (publicBatchWrapper86_copies3780 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_4 := arithEq_of_rows h (row := 1984) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_4
  simp only [← c120957, ← c120958, ← c120959] at e_1984_4
  have e_1984_5 := arithEq_of_rows h (row := 1984) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_5
  simp only [← c120960, ← c120961, k2, ← c120962] at e_1984_5
  have hr := e_1984_5
  simp only [e_1984_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17201)) := by
  have c120963 := (publicBatchWrapper86_copies3780 a h).2.2.2.1
  have c120964 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.1
  have c120965 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.1
  have c120966 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.1
  have c120967 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.2.1
  have c120968 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_6 := arithEq_of_rows h (row := 1984) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_6
  simp only [← c120963, ← c120964, ← c120965] at e_1984_6
  have e_1984_7 := arithEq_of_rows h (row := 1984) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_7
  simp only [← c120966, ← c120967, k2, ← c120968] at e_1984_7
  have hr := e_1984_7
  simp only [e_1984_6] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
