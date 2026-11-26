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

theorem publicBatchWrapper86_f12321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1272 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11727)) := by
  have c78279 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.1
  have c78280 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.1
  have c78281 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.1
  have c78282 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78283 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78284 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1272_18 := arithEq_of_rows h (row := 1272) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_18
  simp only [← c78279, ← c78280, ← c78281] at e_1272_18
  have e_1272_19 := arithEq_of_rows h (row := 1272) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_19
  simp only [← c78282, ← c78283, k2, ← c78284] at e_1272_19
  have hr := e_1272_19
  simp only [e_1272_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11728)) := by
  have c78285 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78286 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78287 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78288 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78289 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78290 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_0 := arithEq_of_rows h (row := 1273) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_0
  simp only [← c78285, ← c78286, ← c78287] at e_1273_0
  have e_1273_1 := arithEq_of_rows h (row := 1273) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_1
  simp only [← c78288, ← c78289, k2, ← c78290] at e_1273_1
  have hr := e_1273_1
  simp only [e_1273_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11729)) := by
  have c78291 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78292 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78293 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78294 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78295 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78296 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_2 := arithEq_of_rows h (row := 1273) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_2
  simp only [← c78291, ← c78292, ← c78293] at e_1273_2
  have e_1273_3 := arithEq_of_rows h (row := 1273) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_3
  simp only [← c78294, ← c78295, k2, ← c78296] at e_1273_3
  have hr := e_1273_3
  simp only [e_1273_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11730)) := by
  have c78297 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78298 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78299 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78300 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78301 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78302 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_4 := arithEq_of_rows h (row := 1273) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_4
  simp only [← c78297, ← c78298, ← c78299] at e_1273_4
  have e_1273_5 := arithEq_of_rows h (row := 1273) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_5
  simp only [← c78300, ← c78301, k2, ← c78302] at e_1273_5
  have hr := e_1273_5
  simp only [e_1273_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11731)) := by
  have c78303 := (publicBatchWrapper86_copies2446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78304 := (publicBatchWrapper86_copies2447 a h).1
  have c78305 := (publicBatchWrapper86_copies2447 a h).2.1
  have c78306 := (publicBatchWrapper86_copies2447 a h).2.2.1
  have c78307 := (publicBatchWrapper86_copies2447 a h).2.2.2.1
  have c78308 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_6 := arithEq_of_rows h (row := 1273) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_6
  simp only [← c78303, ← c78304, ← c78305] at e_1273_6
  have e_1273_7 := arithEq_of_rows h (row := 1273) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_7
  simp only [← c78306, ← c78307, k2, ← c78308] at e_1273_7
  have hr := e_1273_7
  simp only [e_1273_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11732)) := by
  have c78309 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.1
  have c78310 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.1
  have c78311 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.1
  have c78312 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.1
  have c78313 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.1
  have c78314 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_8 := arithEq_of_rows h (row := 1273) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_8
  simp only [← c78309, ← c78310, ← c78311] at e_1273_8
  have e_1273_9 := arithEq_of_rows h (row := 1273) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_9
  simp only [← c78312, ← c78313, k2, ← c78314] at e_1273_9
  have hr := e_1273_9
  simp only [e_1273_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11733)) := by
  have c78315 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78316 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78317 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78318 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78319 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78320 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_10 := arithEq_of_rows h (row := 1273) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_10
  simp only [← c78315, ← c78316, ← c78317] at e_1273_10
  have e_1273_11 := arithEq_of_rows h (row := 1273) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_11
  simp only [← c78318, ← c78319, k2, ← c78320] at e_1273_11
  have hr := e_1273_11
  simp only [e_1273_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11734)) := by
  have c78321 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78322 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78323 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78324 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78325 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78326 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_12 := arithEq_of_rows h (row := 1273) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_12
  simp only [← c78321, ← c78322, ← c78323] at e_1273_12
  have e_1273_13 := arithEq_of_rows h (row := 1273) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_13
  simp only [← c78324, ← c78325, k2, ← c78326] at e_1273_13
  have hr := e_1273_13
  simp only [e_1273_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11735)) := by
  have c78327 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78328 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78329 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78330 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78331 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78332 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_14 := arithEq_of_rows h (row := 1273) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_14
  simp only [← c78327, ← c78328, ← c78329] at e_1273_14
  have e_1273_15 := arithEq_of_rows h (row := 1273) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_15
  simp only [← c78330, ← c78331, k2, ← c78332] at e_1273_15
  have hr := e_1273_15
  simp only [e_1273_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11736)) := by
  have c78333 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78334 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78335 := (publicBatchWrapper86_copies2447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78336 := (publicBatchWrapper86_copies2448 a h).1
  have c78337 := (publicBatchWrapper86_copies2448 a h).2.1
  have c78338 := (publicBatchWrapper86_copies2448 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_16 := arithEq_of_rows h (row := 1273) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_16
  simp only [← c78333, ← c78334, ← c78335] at e_1273_16
  have e_1273_17 := arithEq_of_rows h (row := 1273) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_17
  simp only [← c78336, ← c78337, k2, ← c78338] at e_1273_17
  have hr := e_1273_17
  simp only [e_1273_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1273 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11737)) := by
  have c78339 := (publicBatchWrapper86_copies2448 a h).2.2.2.1
  have c78340 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.1
  have c78341 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.1
  have c78342 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.1
  have c78343 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.1
  have c78344 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1273_18 := arithEq_of_rows h (row := 1273) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_18
  simp only [← c78339, ← c78340, ← c78341] at e_1273_18
  have e_1273_19 := arithEq_of_rows h (row := 1273) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_19
  simp only [← c78342, ← c78343, k2, ← c78344] at e_1273_19
  have hr := e_1273_19
  simp only [e_1273_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11738)) := by
  have c78345 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.1
  have c78346 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78347 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78348 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78349 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78350 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_0 := arithEq_of_rows h (row := 1274) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_0
  simp only [← c78345, ← c78346, ← c78347] at e_1274_0
  have e_1274_1 := arithEq_of_rows h (row := 1274) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_1
  simp only [← c78348, ← c78349, k2, ← c78350] at e_1274_1
  have hr := e_1274_1
  simp only [e_1274_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11739)) := by
  have c78351 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78352 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78353 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78354 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78355 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78356 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_2 := arithEq_of_rows h (row := 1274) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_2
  simp only [← c78351, ← c78352, ← c78353] at e_1274_2
  have e_1274_3 := arithEq_of_rows h (row := 1274) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_3
  simp only [← c78354, ← c78355, k2, ← c78356] at e_1274_3
  have hr := e_1274_3
  simp only [e_1274_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11740)) := by
  have c78357 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78358 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78359 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78360 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78361 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78362 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_4 := arithEq_of_rows h (row := 1274) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_4
  simp only [← c78357, ← c78358, ← c78359] at e_1274_4
  have e_1274_5 := arithEq_of_rows h (row := 1274) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_5
  simp only [← c78360, ← c78361, k2, ← c78362] at e_1274_5
  have hr := e_1274_5
  simp only [e_1274_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11741)) := by
  have c78363 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78364 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78365 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78366 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78367 := (publicBatchWrapper86_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78368 := (publicBatchWrapper86_copies2449 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_6 := arithEq_of_rows h (row := 1274) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_6
  simp only [← c78363, ← c78364, ← c78365] at e_1274_6
  have e_1274_7 := arithEq_of_rows h (row := 1274) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_7
  simp only [← c78366, ← c78367, k2, ← c78368] at e_1274_7
  have hr := e_1274_7
  simp only [e_1274_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11742)) := by
  have c78369 := (publicBatchWrapper86_copies2449 a h).2.1
  have c78370 := (publicBatchWrapper86_copies2449 a h).2.2.1
  have c78371 := (publicBatchWrapper86_copies2449 a h).2.2.2.1
  have c78372 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.1
  have c78373 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.1
  have c78374 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_8 := arithEq_of_rows h (row := 1274) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_8
  simp only [← c78369, ← c78370, ← c78371] at e_1274_8
  have e_1274_9 := arithEq_of_rows h (row := 1274) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_9
  simp only [← c78372, ← c78373, k2, ← c78374] at e_1274_9
  have hr := e_1274_9
  simp only [e_1274_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11743)) := by
  have c78375 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.1
  have c78376 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.1
  have c78377 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.1
  have c78378 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78379 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78380 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_10 := arithEq_of_rows h (row := 1274) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_10
  simp only [← c78375, ← c78376, ← c78377] at e_1274_10
  have e_1274_11 := arithEq_of_rows h (row := 1274) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_11
  simp only [← c78378, ← c78379, k2, ← c78380] at e_1274_11
  have hr := e_1274_11
  simp only [e_1274_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11744)) := by
  have c78381 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78382 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78383 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78384 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78385 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78386 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_12 := arithEq_of_rows h (row := 1274) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_12
  simp only [← c78381, ← c78382, ← c78383] at e_1274_12
  have e_1274_13 := arithEq_of_rows h (row := 1274) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_13
  simp only [← c78384, ← c78385, k2, ← c78386] at e_1274_13
  have hr := e_1274_13
  simp only [e_1274_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11745)) := by
  have c78387 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78388 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78389 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78390 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78391 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78392 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_14 := arithEq_of_rows h (row := 1274) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_14
  simp only [← c78387, ← c78388, ← c78389] at e_1274_14
  have e_1274_15 := arithEq_of_rows h (row := 1274) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_15
  simp only [← c78390, ← c78391, k2, ← c78392] at e_1274_15
  have hr := e_1274_15
  simp only [e_1274_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11746)) := by
  have c78393 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78394 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78395 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78396 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78397 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78398 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_16 := arithEq_of_rows h (row := 1274) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_16
  simp only [← c78393, ← c78394, ← c78395] at e_1274_16
  have e_1274_17 := arithEq_of_rows h (row := 1274) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_17
  simp only [← c78396, ← c78397, k2, ← c78398] at e_1274_17
  have hr := e_1274_17
  simp only [e_1274_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1274 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11747)) := by
  have c78399 := (publicBatchWrapper86_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78400 := (publicBatchWrapper86_copies2450 a h).1
  have c78401 := (publicBatchWrapper86_copies2450 a h).2.1
  have c78402 := (publicBatchWrapper86_copies2450 a h).2.2.1
  have c78403 := (publicBatchWrapper86_copies2450 a h).2.2.2.1
  have c78404 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1274_18 := arithEq_of_rows h (row := 1274) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_18
  simp only [← c78399, ← c78400, ← c78401] at e_1274_18
  have e_1274_19 := arithEq_of_rows h (row := 1274) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_19
  simp only [← c78402, ← c78403, k2, ← c78404] at e_1274_19
  have hr := e_1274_19
  simp only [e_1274_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11748)) := by
  have c78405 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.1
  have c78406 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.1
  have c78407 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.1
  have c78408 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.1
  have c78409 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.1
  have c78410 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_0 := arithEq_of_rows h (row := 1275) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_0
  simp only [← c78405, ← c78406, ← c78407] at e_1275_0
  have e_1275_1 := arithEq_of_rows h (row := 1275) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_1
  simp only [← c78408, ← c78409, k2, ← c78410] at e_1275_1
  have hr := e_1275_1
  simp only [e_1275_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11749)) := by
  have c78411 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78412 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78413 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78414 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78415 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78416 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_2 := arithEq_of_rows h (row := 1275) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_2
  simp only [← c78411, ← c78412, ← c78413] at e_1275_2
  have e_1275_3 := arithEq_of_rows h (row := 1275) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_3
  simp only [← c78414, ← c78415, k2, ← c78416] at e_1275_3
  have hr := e_1275_3
  simp only [e_1275_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11750)) := by
  have c78417 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78418 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78419 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78420 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78421 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78422 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_4 := arithEq_of_rows h (row := 1275) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_4
  simp only [← c78417, ← c78418, ← c78419] at e_1275_4
  have e_1275_5 := arithEq_of_rows h (row := 1275) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_5
  simp only [← c78420, ← c78421, k2, ← c78422] at e_1275_5
  have hr := e_1275_5
  simp only [e_1275_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11751)) := by
  have c78423 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78424 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78425 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78426 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78427 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78428 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_6 := arithEq_of_rows h (row := 1275) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_6
  simp only [← c78423, ← c78424, ← c78425] at e_1275_6
  have e_1275_7 := arithEq_of_rows h (row := 1275) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_7
  simp only [← c78426, ← c78427, k2, ← c78428] at e_1275_7
  have hr := e_1275_7
  simp only [e_1275_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11752)) := by
  have c78429 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78430 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78431 := (publicBatchWrapper86_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78432 := (publicBatchWrapper86_copies2451 a h).1
  have c78433 := (publicBatchWrapper86_copies2451 a h).2.1
  have c78434 := (publicBatchWrapper86_copies2451 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_8 := arithEq_of_rows h (row := 1275) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_8
  simp only [← c78429, ← c78430, ← c78431] at e_1275_8
  have e_1275_9 := arithEq_of_rows h (row := 1275) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_9
  simp only [← c78432, ← c78433, k2, ← c78434] at e_1275_9
  have hr := e_1275_9
  simp only [e_1275_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11753)) := by
  have c78435 := (publicBatchWrapper86_copies2451 a h).2.2.2.1
  have c78436 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.1
  have c78437 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.1
  have c78438 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.1
  have c78439 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.1
  have c78440 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_10 := arithEq_of_rows h (row := 1275) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_10
  simp only [← c78435, ← c78436, ← c78437] at e_1275_10
  have e_1275_11 := arithEq_of_rows h (row := 1275) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_11
  simp only [← c78438, ← c78439, k2, ← c78440] at e_1275_11
  have hr := e_1275_11
  simp only [e_1275_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11754)) := by
  have c78441 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.1
  have c78442 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78443 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78444 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78445 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78446 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_12 := arithEq_of_rows h (row := 1275) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_12
  simp only [← c78441, ← c78442, ← c78443] at e_1275_12
  have e_1275_13 := arithEq_of_rows h (row := 1275) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_13
  simp only [← c78444, ← c78445, k2, ← c78446] at e_1275_13
  have hr := e_1275_13
  simp only [e_1275_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11755)) := by
  have c78447 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78448 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78449 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78450 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78451 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78452 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_14 := arithEq_of_rows h (row := 1275) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_14
  simp only [← c78447, ← c78448, ← c78449] at e_1275_14
  have e_1275_15 := arithEq_of_rows h (row := 1275) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_15
  simp only [← c78450, ← c78451, k2, ← c78452] at e_1275_15
  have hr := e_1275_15
  simp only [e_1275_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11756)) := by
  have c78453 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78454 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78455 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78456 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78457 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78458 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_16 := arithEq_of_rows h (row := 1275) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_16
  simp only [← c78453, ← c78454, ← c78455] at e_1275_16
  have e_1275_17 := arithEq_of_rows h (row := 1275) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_17
  simp only [← c78456, ← c78457, k2, ← c78458] at e_1275_17
  have hr := e_1275_17
  simp only [e_1275_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1275 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11757)) := by
  have c78459 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78460 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78461 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78462 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78463 := (publicBatchWrapper86_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78464 := (publicBatchWrapper86_copies2452 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1275_18 := arithEq_of_rows h (row := 1275) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_18
  simp only [← c78459, ← c78460, ← c78461] at e_1275_18
  have e_1275_19 := arithEq_of_rows h (row := 1275) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_19
  simp only [← c78462, ← c78463, k2, ← c78464] at e_1275_19
  have hr := e_1275_19
  simp only [e_1275_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11758)) := by
  have c78465 := (publicBatchWrapper86_copies2452 a h).2.1
  have c78466 := (publicBatchWrapper86_copies2452 a h).2.2.1
  have c78467 := (publicBatchWrapper86_copies2452 a h).2.2.2.1
  have c78468 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.1
  have c78469 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.1
  have c78470 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_0 := arithEq_of_rows h (row := 1276) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_0
  simp only [← c78465, ← c78466, ← c78467] at e_1276_0
  have e_1276_1 := arithEq_of_rows h (row := 1276) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_1
  simp only [← c78468, ← c78469, k2, ← c78470] at e_1276_1
  have hr := e_1276_1
  simp only [e_1276_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11759)) := by
  have c78471 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.1
  have c78472 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.1
  have c78473 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.1
  have c78474 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78475 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78476 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_2 := arithEq_of_rows h (row := 1276) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_2
  simp only [← c78471, ← c78472, ← c78473] at e_1276_2
  have e_1276_3 := arithEq_of_rows h (row := 1276) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_3
  simp only [← c78474, ← c78475, k2, ← c78476] at e_1276_3
  have hr := e_1276_3
  simp only [e_1276_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11760)) := by
  have c78477 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78478 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78479 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78480 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78481 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78482 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_4 := arithEq_of_rows h (row := 1276) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_4
  simp only [← c78477, ← c78478, ← c78479] at e_1276_4
  have e_1276_5 := arithEq_of_rows h (row := 1276) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_5
  simp only [← c78480, ← c78481, k2, ← c78482] at e_1276_5
  have hr := e_1276_5
  simp only [e_1276_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11761)) := by
  have c78483 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78484 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78485 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78486 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78487 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78488 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_6 := arithEq_of_rows h (row := 1276) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_6
  simp only [← c78483, ← c78484, ← c78485] at e_1276_6
  have e_1276_7 := arithEq_of_rows h (row := 1276) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_7
  simp only [← c78486, ← c78487, k2, ← c78488] at e_1276_7
  have hr := e_1276_7
  simp only [e_1276_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11762)) := by
  have c78489 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78490 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78491 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78492 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78493 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78494 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_8 := arithEq_of_rows h (row := 1276) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_8
  simp only [← c78489, ← c78490, ← c78491] at e_1276_8
  have e_1276_9 := arithEq_of_rows h (row := 1276) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_9
  simp only [← c78492, ← c78493, k2, ← c78494] at e_1276_9
  have hr := e_1276_9
  simp only [e_1276_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11763)) := by
  have c78495 := (publicBatchWrapper86_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78496 := (publicBatchWrapper86_copies2453 a h).1
  have c78497 := (publicBatchWrapper86_copies2453 a h).2.1
  have c78498 := (publicBatchWrapper86_copies2453 a h).2.2.1
  have c78499 := (publicBatchWrapper86_copies2453 a h).2.2.2.1
  have c78500 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_10 := arithEq_of_rows h (row := 1276) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_10
  simp only [← c78495, ← c78496, ← c78497] at e_1276_10
  have e_1276_11 := arithEq_of_rows h (row := 1276) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_11
  simp only [← c78498, ← c78499, k2, ← c78500] at e_1276_11
  have hr := e_1276_11
  simp only [e_1276_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11764)) := by
  have c78501 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.1
  have c78502 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.1
  have c78503 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.1
  have c78504 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.1
  have c78505 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.1
  have c78506 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_12 := arithEq_of_rows h (row := 1276) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_12
  simp only [← c78501, ← c78502, ← c78503] at e_1276_12
  have e_1276_13 := arithEq_of_rows h (row := 1276) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_13
  simp only [← c78504, ← c78505, k2, ← c78506] at e_1276_13
  have hr := e_1276_13
  simp only [e_1276_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11765)) := by
  have c78507 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78508 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78509 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78510 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78511 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78512 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_14 := arithEq_of_rows h (row := 1276) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_14
  simp only [← c78507, ← c78508, ← c78509] at e_1276_14
  have e_1276_15 := arithEq_of_rows h (row := 1276) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_15
  simp only [← c78510, ← c78511, k2, ← c78512] at e_1276_15
  have hr := e_1276_15
  simp only [e_1276_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11766)) := by
  have c78513 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78514 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78515 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78516 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78517 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78518 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_16 := arithEq_of_rows h (row := 1276) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_16
  simp only [← c78513, ← c78514, ← c78515] at e_1276_16
  have e_1276_17 := arithEq_of_rows h (row := 1276) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_17
  simp only [← c78516, ← c78517, k2, ← c78518] at e_1276_17
  have hr := e_1276_17
  simp only [e_1276_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1276 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11767)) := by
  have c78519 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78520 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78521 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78522 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78523 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78524 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1276_18 := arithEq_of_rows h (row := 1276) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_18
  simp only [← c78519, ← c78520, ← c78521] at e_1276_18
  have e_1276_19 := arithEq_of_rows h (row := 1276) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_19
  simp only [← c78522, ← c78523, k2, ← c78524] at e_1276_19
  have hr := e_1276_19
  simp only [e_1276_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11768)) := by
  have c78525 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78526 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78527 := (publicBatchWrapper86_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78528 := (publicBatchWrapper86_copies2454 a h).1
  have c78529 := (publicBatchWrapper86_copies2454 a h).2.1
  have c78530 := (publicBatchWrapper86_copies2454 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_0 := arithEq_of_rows h (row := 1277) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_0
  simp only [← c78525, ← c78526, ← c78527] at e_1277_0
  have e_1277_1 := arithEq_of_rows h (row := 1277) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_1
  simp only [← c78528, ← c78529, k2, ← c78530] at e_1277_1
  have hr := e_1277_1
  simp only [e_1277_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11769)) := by
  have c78531 := (publicBatchWrapper86_copies2454 a h).2.2.2.1
  have c78532 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.1
  have c78533 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.1
  have c78534 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.1
  have c78535 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.1
  have c78536 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_2 := arithEq_of_rows h (row := 1277) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_2
  simp only [← c78531, ← c78532, ← c78533] at e_1277_2
  have e_1277_3 := arithEq_of_rows h (row := 1277) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_3
  simp only [← c78534, ← c78535, k2, ← c78536] at e_1277_3
  have hr := e_1277_3
  simp only [e_1277_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11810)) := by
  have c78537 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.1
  have c78538 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78539 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78540 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78541 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78542 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_4 := arithEq_of_rows h (row := 1277) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_4
  simp only [← c78537, ← c78538, ← c78539] at e_1277_4
  have e_1277_5 := arithEq_of_rows h (row := 1277) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_5
  simp only [← c78540, ← c78541, k2, ← c78542] at e_1277_5
  have hr := e_1277_5
  simp only [e_1277_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11811)) := by
  have c78543 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78544 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78545 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78546 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78547 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78548 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_6 := arithEq_of_rows h (row := 1277) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_6
  simp only [← c78543, ← c78544, ← c78545] at e_1277_6
  have e_1277_7 := arithEq_of_rows h (row := 1277) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_7
  simp only [← c78546, ← c78547, k2, ← c78548] at e_1277_7
  have hr := e_1277_7
  simp only [e_1277_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11812)) := by
  have c78549 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78550 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78551 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78552 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78553 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78554 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_8 := arithEq_of_rows h (row := 1277) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_8
  simp only [← c78549, ← c78550, ← c78551] at e_1277_8
  have e_1277_9 := arithEq_of_rows h (row := 1277) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_9
  simp only [← c78552, ← c78553, k2, ← c78554] at e_1277_9
  have hr := e_1277_9
  simp only [e_1277_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11813)) := by
  have c78555 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78556 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78557 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78558 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78559 := (publicBatchWrapper86_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78560 := (publicBatchWrapper86_copies2455 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_10 := arithEq_of_rows h (row := 1277) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_10
  simp only [← c78555, ← c78556, ← c78557] at e_1277_10
  have e_1277_11 := arithEq_of_rows h (row := 1277) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_11
  simp only [← c78558, ← c78559, k2, ← c78560] at e_1277_11
  have hr := e_1277_11
  simp only [e_1277_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11814)) := by
  have c78561 := (publicBatchWrapper86_copies2455 a h).2.1
  have c78562 := (publicBatchWrapper86_copies2455 a h).2.2.1
  have c78563 := (publicBatchWrapper86_copies2455 a h).2.2.2.1
  have c78564 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.1
  have c78565 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.1
  have c78566 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_12 := arithEq_of_rows h (row := 1277) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_12
  simp only [← c78561, ← c78562, ← c78563] at e_1277_12
  have e_1277_13 := arithEq_of_rows h (row := 1277) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_13
  simp only [← c78564, ← c78565, k2, ← c78566] at e_1277_13
  have hr := e_1277_13
  simp only [e_1277_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11815)) := by
  have c78567 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.1
  have c78568 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.1
  have c78569 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.1
  have c78570 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78571 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78572 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_14 := arithEq_of_rows h (row := 1277) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_14
  simp only [← c78567, ← c78568, ← c78569] at e_1277_14
  have e_1277_15 := arithEq_of_rows h (row := 1277) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_15
  simp only [← c78570, ← c78571, k2, ← c78572] at e_1277_15
  have hr := e_1277_15
  simp only [e_1277_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11816)) := by
  have c78573 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78574 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78575 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78576 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78577 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78578 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_16 := arithEq_of_rows h (row := 1277) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_16
  simp only [← c78573, ← c78574, ← c78575] at e_1277_16
  have e_1277_17 := arithEq_of_rows h (row := 1277) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_17
  simp only [← c78576, ← c78577, k2, ← c78578] at e_1277_17
  have hr := e_1277_17
  simp only [e_1277_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1277 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11817)) := by
  have c78579 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78580 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78581 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78582 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78583 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78584 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1277_18 := arithEq_of_rows h (row := 1277) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_18
  simp only [← c78579, ← c78580, ← c78581] at e_1277_18
  have e_1277_19 := arithEq_of_rows h (row := 1277) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_19
  simp only [← c78582, ← c78583, k2, ← c78584] at e_1277_19
  have hr := e_1277_19
  simp only [e_1277_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11818)) := by
  have c78585 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78586 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78587 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78588 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78589 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78590 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_0 := arithEq_of_rows h (row := 1278) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_0
  simp only [← c78585, ← c78586, ← c78587] at e_1278_0
  have e_1278_1 := arithEq_of_rows h (row := 1278) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_1
  simp only [← c78588, ← c78589, k2, ← c78590] at e_1278_1
  have hr := e_1278_1
  simp only [e_1278_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11819)) := by
  have c78591 := (publicBatchWrapper86_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78592 := (publicBatchWrapper86_copies2456 a h).1
  have c78593 := (publicBatchWrapper86_copies2456 a h).2.1
  have c78594 := (publicBatchWrapper86_copies2456 a h).2.2.1
  have c78595 := (publicBatchWrapper86_copies2456 a h).2.2.2.1
  have c78596 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_2 := arithEq_of_rows h (row := 1278) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_2
  simp only [← c78591, ← c78592, ← c78593] at e_1278_2
  have e_1278_3 := arithEq_of_rows h (row := 1278) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_3
  simp only [← c78594, ← c78595, k2, ← c78596] at e_1278_3
  have hr := e_1278_3
  simp only [e_1278_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11820)) := by
  have c78597 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.1
  have c78598 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.1
  have c78599 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.1
  have c78600 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.1
  have c78601 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.1
  have c78602 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_4 := arithEq_of_rows h (row := 1278) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_4
  simp only [← c78597, ← c78598, ← c78599] at e_1278_4
  have e_1278_5 := arithEq_of_rows h (row := 1278) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_5
  simp only [← c78600, ← c78601, k2, ← c78602] at e_1278_5
  have hr := e_1278_5
  simp only [e_1278_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11821)) := by
  have c78603 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78604 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78605 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78606 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78607 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78608 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_6 := arithEq_of_rows h (row := 1278) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_6
  simp only [← c78603, ← c78604, ← c78605] at e_1278_6
  have e_1278_7 := arithEq_of_rows h (row := 1278) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_7
  simp only [← c78606, ← c78607, k2, ← c78608] at e_1278_7
  have hr := e_1278_7
  simp only [e_1278_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11822)) := by
  have c78609 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78610 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78611 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78612 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78613 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78614 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_8 := arithEq_of_rows h (row := 1278) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_8
  simp only [← c78609, ← c78610, ← c78611] at e_1278_8
  have e_1278_9 := arithEq_of_rows h (row := 1278) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_9
  simp only [← c78612, ← c78613, k2, ← c78614] at e_1278_9
  have hr := e_1278_9
  simp only [e_1278_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11823)) := by
  have c78615 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78616 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78617 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78618 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78619 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78620 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_10 := arithEq_of_rows h (row := 1278) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_10
  simp only [← c78615, ← c78616, ← c78617] at e_1278_10
  have e_1278_11 := arithEq_of_rows h (row := 1278) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_11
  simp only [← c78618, ← c78619, k2, ← c78620] at e_1278_11
  have hr := e_1278_11
  simp only [e_1278_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11824)) := by
  have c78621 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78622 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78623 := (publicBatchWrapper86_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78624 := (publicBatchWrapper86_copies2457 a h).1
  have c78625 := (publicBatchWrapper86_copies2457 a h).2.1
  have c78626 := (publicBatchWrapper86_copies2457 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_12 := arithEq_of_rows h (row := 1278) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_12
  simp only [← c78621, ← c78622, ← c78623] at e_1278_12
  have e_1278_13 := arithEq_of_rows h (row := 1278) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_13
  simp only [← c78624, ← c78625, k2, ← c78626] at e_1278_13
  have hr := e_1278_13
  simp only [e_1278_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11825)) := by
  have c78627 := (publicBatchWrapper86_copies2457 a h).2.2.2.1
  have c78628 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.1
  have c78629 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.1
  have c78630 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.1
  have c78631 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.1
  have c78632 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_14 := arithEq_of_rows h (row := 1278) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_14
  simp only [← c78627, ← c78628, ← c78629] at e_1278_14
  have e_1278_15 := arithEq_of_rows h (row := 1278) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_15
  simp only [← c78630, ← c78631, k2, ← c78632] at e_1278_15
  have hr := e_1278_15
  simp only [e_1278_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11826)) := by
  have c78633 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.1
  have c78634 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78635 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78636 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78637 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78638 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_16 := arithEq_of_rows h (row := 1278) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_16
  simp only [← c78633, ← c78634, ← c78635] at e_1278_16
  have e_1278_17 := arithEq_of_rows h (row := 1278) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_17
  simp only [← c78636, ← c78637, k2, ← c78638] at e_1278_17
  have hr := e_1278_17
  simp only [e_1278_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1278 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11827)) := by
  have c78639 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78640 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78641 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78642 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78643 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78644 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1278_18 := arithEq_of_rows h (row := 1278) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_18
  simp only [← c78639, ← c78640, ← c78641] at e_1278_18
  have e_1278_19 := arithEq_of_rows h (row := 1278) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_19
  simp only [← c78642, ← c78643, k2, ← c78644] at e_1278_19
  have hr := e_1278_19
  simp only [e_1278_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11828)) := by
  have c78645 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78646 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78647 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78648 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78649 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78650 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_0 := arithEq_of_rows h (row := 1279) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_0
  simp only [← c78645, ← c78646, ← c78647] at e_1279_0
  have e_1279_1 := arithEq_of_rows h (row := 1279) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_1
  simp only [← c78648, ← c78649, k2, ← c78650] at e_1279_1
  have hr := e_1279_1
  simp only [e_1279_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11829)) := by
  have c78651 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78652 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78653 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78654 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78655 := (publicBatchWrapper86_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78656 := (publicBatchWrapper86_copies2458 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_2 := arithEq_of_rows h (row := 1279) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_2
  simp only [← c78651, ← c78652, ← c78653] at e_1279_2
  have e_1279_3 := arithEq_of_rows h (row := 1279) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_3
  simp only [← c78654, ← c78655, k2, ← c78656] at e_1279_3
  have hr := e_1279_3
  simp only [e_1279_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11830)) := by
  have c78657 := (publicBatchWrapper86_copies2458 a h).2.1
  have c78658 := (publicBatchWrapper86_copies2458 a h).2.2.1
  have c78659 := (publicBatchWrapper86_copies2458 a h).2.2.2.1
  have c78660 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.1
  have c78661 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.1
  have c78662 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_4 := arithEq_of_rows h (row := 1279) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_4
  simp only [← c78657, ← c78658, ← c78659] at e_1279_4
  have e_1279_5 := arithEq_of_rows h (row := 1279) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_5
  simp only [← c78660, ← c78661, k2, ← c78662] at e_1279_5
  have hr := e_1279_5
  simp only [e_1279_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11831)) := by
  have c78663 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.1
  have c78664 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.1
  have c78665 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.1
  have c78666 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78667 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78668 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_6 := arithEq_of_rows h (row := 1279) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_6
  simp only [← c78663, ← c78664, ← c78665] at e_1279_6
  have e_1279_7 := arithEq_of_rows h (row := 1279) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_7
  simp only [← c78666, ← c78667, k2, ← c78668] at e_1279_7
  have hr := e_1279_7
  simp only [e_1279_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11832)) := by
  have c78669 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78670 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78671 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78672 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78673 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78674 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_8 := arithEq_of_rows h (row := 1279) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_8
  simp only [← c78669, ← c78670, ← c78671] at e_1279_8
  have e_1279_9 := arithEq_of_rows h (row := 1279) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_9
  simp only [← c78672, ← c78673, k2, ← c78674] at e_1279_9
  have hr := e_1279_9
  simp only [e_1279_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11833)) := by
  have c78675 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78676 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78677 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78678 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78679 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78680 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_10 := arithEq_of_rows h (row := 1279) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_10
  simp only [← c78675, ← c78676, ← c78677] at e_1279_10
  have e_1279_11 := arithEq_of_rows h (row := 1279) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_11
  simp only [← c78678, ← c78679, k2, ← c78680] at e_1279_11
  have hr := e_1279_11
  simp only [e_1279_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11834)) := by
  have c78681 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78682 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78683 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78684 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78685 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78686 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_12 := arithEq_of_rows h (row := 1279) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_12
  simp only [← c78681, ← c78682, ← c78683] at e_1279_12
  have e_1279_13 := arithEq_of_rows h (row := 1279) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_13
  simp only [← c78684, ← c78685, k2, ← c78686] at e_1279_13
  have hr := e_1279_13
  simp only [e_1279_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11835)) := by
  have c78687 := (publicBatchWrapper86_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78688 := (publicBatchWrapper86_copies2459 a h).1
  have c78689 := (publicBatchWrapper86_copies2459 a h).2.1
  have c78690 := (publicBatchWrapper86_copies2459 a h).2.2.1
  have c78691 := (publicBatchWrapper86_copies2459 a h).2.2.2.1
  have c78692 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_14 := arithEq_of_rows h (row := 1279) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_14
  simp only [← c78687, ← c78688, ← c78689] at e_1279_14
  have e_1279_15 := arithEq_of_rows h (row := 1279) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_15
  simp only [← c78690, ← c78691, k2, ← c78692] at e_1279_15
  have hr := e_1279_15
  simp only [e_1279_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11836)) := by
  have c78693 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.1
  have c78694 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.1
  have c78695 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.1
  have c78696 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.1
  have c78697 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.1
  have c78698 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_16 := arithEq_of_rows h (row := 1279) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_16
  simp only [← c78693, ← c78694, ← c78695] at e_1279_16
  have e_1279_17 := arithEq_of_rows h (row := 1279) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_17
  simp only [← c78696, ← c78697, k2, ← c78698] at e_1279_17
  have hr := e_1279_17
  simp only [e_1279_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1279 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11837)) := by
  have c78699 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78700 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78701 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78702 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78703 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78704 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1279_18 := arithEq_of_rows h (row := 1279) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_18
  simp only [← c78699, ← c78700, ← c78701] at e_1279_18
  have e_1279_19 := arithEq_of_rows h (row := 1279) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_19
  simp only [← c78702, ← c78703, k2, ← c78704] at e_1279_19
  have hr := e_1279_19
  simp only [e_1279_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11838)) := by
  have c78705 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78706 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78707 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78708 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78709 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78710 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_0 := arithEq_of_rows h (row := 1280) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_0
  simp only [← c78705, ← c78706, ← c78707] at e_1280_0
  have e_1280_1 := arithEq_of_rows h (row := 1280) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_1
  simp only [← c78708, ← c78709, k2, ← c78710] at e_1280_1
  have hr := e_1280_1
  simp only [e_1280_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11839)) := by
  have c78711 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78712 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78713 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78714 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78715 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78716 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_2 := arithEq_of_rows h (row := 1280) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_2
  simp only [← c78711, ← c78712, ← c78713] at e_1280_2
  have e_1280_3 := arithEq_of_rows h (row := 1280) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_3
  simp only [← c78714, ← c78715, k2, ← c78716] at e_1280_3
  have hr := e_1280_3
  simp only [e_1280_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11840)) := by
  have c78717 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78718 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78719 := (publicBatchWrapper86_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78720 := (publicBatchWrapper86_copies2460 a h).1
  have c78721 := (publicBatchWrapper86_copies2460 a h).2.1
  have c78722 := (publicBatchWrapper86_copies2460 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_4 := arithEq_of_rows h (row := 1280) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_4
  simp only [← c78717, ← c78718, ← c78719] at e_1280_4
  have e_1280_5 := arithEq_of_rows h (row := 1280) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_5
  simp only [← c78720, ← c78721, k2, ← c78722] at e_1280_5
  have hr := e_1280_5
  simp only [e_1280_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11841)) := by
  have c78723 := (publicBatchWrapper86_copies2460 a h).2.2.2.1
  have c78724 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.1
  have c78725 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.1
  have c78726 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.1
  have c78727 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.1
  have c78728 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_6 := arithEq_of_rows h (row := 1280) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_6
  simp only [← c78723, ← c78724, ← c78725] at e_1280_6
  have e_1280_7 := arithEq_of_rows h (row := 1280) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_7
  simp only [← c78726, ← c78727, k2, ← c78728] at e_1280_7
  have hr := e_1280_7
  simp only [e_1280_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11842)) := by
  have c78729 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.1
  have c78730 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78731 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78732 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78733 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78734 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_8 := arithEq_of_rows h (row := 1280) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_8
  simp only [← c78729, ← c78730, ← c78731] at e_1280_8
  have e_1280_9 := arithEq_of_rows h (row := 1280) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_9
  simp only [← c78732, ← c78733, k2, ← c78734] at e_1280_9
  have hr := e_1280_9
  simp only [e_1280_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11843)) := by
  have c78735 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78736 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78737 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78738 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78739 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78740 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_10 := arithEq_of_rows h (row := 1280) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_10
  simp only [← c78735, ← c78736, ← c78737] at e_1280_10
  have e_1280_11 := arithEq_of_rows h (row := 1280) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_11
  simp only [← c78738, ← c78739, k2, ← c78740] at e_1280_11
  have hr := e_1280_11
  simp only [e_1280_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11844)) := by
  have c78741 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78742 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78743 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78744 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78745 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78746 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_12 := arithEq_of_rows h (row := 1280) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_12
  simp only [← c78741, ← c78742, ← c78743] at e_1280_12
  have e_1280_13 := arithEq_of_rows h (row := 1280) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_13
  simp only [← c78744, ← c78745, k2, ← c78746] at e_1280_13
  have hr := e_1280_13
  simp only [e_1280_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11845)) := by
  have c78747 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78748 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78749 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78750 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78751 := (publicBatchWrapper86_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78752 := (publicBatchWrapper86_copies2461 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_14 := arithEq_of_rows h (row := 1280) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_14
  simp only [← c78747, ← c78748, ← c78749] at e_1280_14
  have e_1280_15 := arithEq_of_rows h (row := 1280) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_15
  simp only [← c78750, ← c78751, k2, ← c78752] at e_1280_15
  have hr := e_1280_15
  simp only [e_1280_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11846)) := by
  have c78753 := (publicBatchWrapper86_copies2461 a h).2.1
  have c78754 := (publicBatchWrapper86_copies2461 a h).2.2.1
  have c78755 := (publicBatchWrapper86_copies2461 a h).2.2.2.1
  have c78756 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.1
  have c78757 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.1
  have c78758 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_16 := arithEq_of_rows h (row := 1280) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_16
  simp only [← c78753, ← c78754, ← c78755] at e_1280_16
  have e_1280_17 := arithEq_of_rows h (row := 1280) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_17
  simp only [← c78756, ← c78757, k2, ← c78758] at e_1280_17
  have hr := e_1280_17
  simp only [e_1280_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1280 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11847)) := by
  have c78759 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.1
  have c78760 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.1
  have c78761 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.1
  have c78762 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78763 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78764 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1280_18 := arithEq_of_rows h (row := 1280) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_18
  simp only [← c78759, ← c78760, ← c78761] at e_1280_18
  have e_1280_19 := arithEq_of_rows h (row := 1280) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_19
  simp only [← c78762, ← c78763, k2, ← c78764] at e_1280_19
  have hr := e_1280_19
  simp only [e_1280_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11848)) := by
  have c78765 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78766 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78767 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78768 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78769 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78770 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_0 := arithEq_of_rows h (row := 1281) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_0
  simp only [← c78765, ← c78766, ← c78767] at e_1281_0
  have e_1281_1 := arithEq_of_rows h (row := 1281) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_1
  simp only [← c78768, ← c78769, k2, ← c78770] at e_1281_1
  have hr := e_1281_1
  simp only [e_1281_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11849)) := by
  have c78771 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78772 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78773 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78774 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78775 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78776 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_2 := arithEq_of_rows h (row := 1281) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_2
  simp only [← c78771, ← c78772, ← c78773] at e_1281_2
  have e_1281_3 := arithEq_of_rows h (row := 1281) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_3
  simp only [← c78774, ← c78775, k2, ← c78776] at e_1281_3
  have hr := e_1281_3
  simp only [e_1281_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11850)) := by
  have c78777 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78778 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78779 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78780 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78781 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78782 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_4 := arithEq_of_rows h (row := 1281) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_4
  simp only [← c78777, ← c78778, ← c78779] at e_1281_4
  have e_1281_5 := arithEq_of_rows h (row := 1281) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_5
  simp only [← c78780, ← c78781, k2, ← c78782] at e_1281_5
  have hr := e_1281_5
  simp only [e_1281_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11851)) := by
  have c78783 := (publicBatchWrapper86_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78784 := (publicBatchWrapper86_copies2462 a h).1
  have c78785 := (publicBatchWrapper86_copies2462 a h).2.1
  have c78786 := (publicBatchWrapper86_copies2462 a h).2.2.1
  have c78787 := (publicBatchWrapper86_copies2462 a h).2.2.2.1
  have c78788 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_6 := arithEq_of_rows h (row := 1281) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_6
  simp only [← c78783, ← c78784, ← c78785] at e_1281_6
  have e_1281_7 := arithEq_of_rows h (row := 1281) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_7
  simp only [← c78786, ← c78787, k2, ← c78788] at e_1281_7
  have hr := e_1281_7
  simp only [e_1281_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11852)) := by
  have c78789 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.1
  have c78790 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.1
  have c78791 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.1
  have c78792 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.1
  have c78793 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.1
  have c78794 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_8 := arithEq_of_rows h (row := 1281) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_8
  simp only [← c78789, ← c78790, ← c78791] at e_1281_8
  have e_1281_9 := arithEq_of_rows h (row := 1281) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_9
  simp only [← c78792, ← c78793, k2, ← c78794] at e_1281_9
  have hr := e_1281_9
  simp only [e_1281_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11853)) := by
  have c78795 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78796 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78797 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78798 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78799 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78800 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_10 := arithEq_of_rows h (row := 1281) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_10
  simp only [← c78795, ← c78796, ← c78797] at e_1281_10
  have e_1281_11 := arithEq_of_rows h (row := 1281) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_11
  simp only [← c78798, ← c78799, k2, ← c78800] at e_1281_11
  have hr := e_1281_11
  simp only [e_1281_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11854)) := by
  have c78801 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78802 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78803 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78804 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78805 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78806 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_12 := arithEq_of_rows h (row := 1281) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_12
  simp only [← c78801, ← c78802, ← c78803] at e_1281_12
  have e_1281_13 := arithEq_of_rows h (row := 1281) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_13
  simp only [← c78804, ← c78805, k2, ← c78806] at e_1281_13
  have hr := e_1281_13
  simp only [e_1281_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11855)) := by
  have c78807 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78808 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78809 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78810 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78811 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78812 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_14 := arithEq_of_rows h (row := 1281) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_14
  simp only [← c78807, ← c78808, ← c78809] at e_1281_14
  have e_1281_15 := arithEq_of_rows h (row := 1281) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_15
  simp only [← c78810, ← c78811, k2, ← c78812] at e_1281_15
  have hr := e_1281_15
  simp only [e_1281_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11856)) := by
  have c78813 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78814 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78815 := (publicBatchWrapper86_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78816 := (publicBatchWrapper86_copies2463 a h).1
  have c78817 := (publicBatchWrapper86_copies2463 a h).2.1
  have c78818 := (publicBatchWrapper86_copies2463 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_16 := arithEq_of_rows h (row := 1281) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_16
  simp only [← c78813, ← c78814, ← c78815] at e_1281_16
  have e_1281_17 := arithEq_of_rows h (row := 1281) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_17
  simp only [← c78816, ← c78817, k2, ← c78818] at e_1281_17
  have hr := e_1281_17
  simp only [e_1281_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1281 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11857)) := by
  have c78819 := (publicBatchWrapper86_copies2463 a h).2.2.2.1
  have c78820 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.1
  have c78821 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.1
  have c78822 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.1
  have c78823 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.1
  have c78824 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1281_18 := arithEq_of_rows h (row := 1281) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_18
  simp only [← c78819, ← c78820, ← c78821] at e_1281_18
  have e_1281_19 := arithEq_of_rows h (row := 1281) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_19
  simp only [← c78822, ← c78823, k2, ← c78824] at e_1281_19
  have hr := e_1281_19
  simp only [e_1281_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11858)) := by
  have c78825 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.1
  have c78826 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78827 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78828 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78829 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78830 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_0 := arithEq_of_rows h (row := 1282) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_0
  simp only [← c78825, ← c78826, ← c78827] at e_1282_0
  have e_1282_1 := arithEq_of_rows h (row := 1282) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_1
  simp only [← c78828, ← c78829, k2, ← c78830] at e_1282_1
  have hr := e_1282_1
  simp only [e_1282_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11859)) := by
  have c78831 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78832 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78833 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78834 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78835 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78836 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_2 := arithEq_of_rows h (row := 1282) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_2
  simp only [← c78831, ← c78832, ← c78833] at e_1282_2
  have e_1282_3 := arithEq_of_rows h (row := 1282) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_3
  simp only [← c78834, ← c78835, k2, ← c78836] at e_1282_3
  have hr := e_1282_3
  simp only [e_1282_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11860)) := by
  have c78837 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78838 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78839 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78840 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78841 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78842 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_4 := arithEq_of_rows h (row := 1282) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_4
  simp only [← c78837, ← c78838, ← c78839] at e_1282_4
  have e_1282_5 := arithEq_of_rows h (row := 1282) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_5
  simp only [← c78840, ← c78841, k2, ← c78842] at e_1282_5
  have hr := e_1282_5
  simp only [e_1282_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11861)) := by
  have c78843 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78844 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78845 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78846 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78847 := (publicBatchWrapper86_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78848 := (publicBatchWrapper86_copies2464 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_6 := arithEq_of_rows h (row := 1282) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_6
  simp only [← c78843, ← c78844, ← c78845] at e_1282_6
  have e_1282_7 := arithEq_of_rows h (row := 1282) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_7
  simp only [← c78846, ← c78847, k2, ← c78848] at e_1282_7
  have hr := e_1282_7
  simp only [e_1282_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11862)) := by
  have c78849 := (publicBatchWrapper86_copies2464 a h).2.1
  have c78850 := (publicBatchWrapper86_copies2464 a h).2.2.1
  have c78851 := (publicBatchWrapper86_copies2464 a h).2.2.2.1
  have c78852 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.1
  have c78853 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.1
  have c78854 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_8 := arithEq_of_rows h (row := 1282) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_8
  simp only [← c78849, ← c78850, ← c78851] at e_1282_8
  have e_1282_9 := arithEq_of_rows h (row := 1282) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_9
  simp only [← c78852, ← c78853, k2, ← c78854] at e_1282_9
  have hr := e_1282_9
  simp only [e_1282_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11863)) := by
  have c78855 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.1
  have c78856 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.1
  have c78857 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.1
  have c78858 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78859 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78860 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_10 := arithEq_of_rows h (row := 1282) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_10
  simp only [← c78855, ← c78856, ← c78857] at e_1282_10
  have e_1282_11 := arithEq_of_rows h (row := 1282) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_11
  simp only [← c78858, ← c78859, k2, ← c78860] at e_1282_11
  have hr := e_1282_11
  simp only [e_1282_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11864)) := by
  have c78861 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78862 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78863 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78864 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78865 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78866 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_12 := arithEq_of_rows h (row := 1282) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_12
  simp only [← c78861, ← c78862, ← c78863] at e_1282_12
  have e_1282_13 := arithEq_of_rows h (row := 1282) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_13
  simp only [← c78864, ← c78865, k2, ← c78866] at e_1282_13
  have hr := e_1282_13
  simp only [e_1282_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11865)) := by
  have c78867 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78868 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78869 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78870 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78871 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78872 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_14 := arithEq_of_rows h (row := 1282) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_14
  simp only [← c78867, ← c78868, ← c78869] at e_1282_14
  have e_1282_15 := arithEq_of_rows h (row := 1282) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_15
  simp only [← c78870, ← c78871, k2, ← c78872] at e_1282_15
  have hr := e_1282_15
  simp only [e_1282_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11866)) := by
  have c78873 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78874 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78875 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78876 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78877 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78878 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_16 := arithEq_of_rows h (row := 1282) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_16
  simp only [← c78873, ← c78874, ← c78875] at e_1282_16
  have e_1282_17 := arithEq_of_rows h (row := 1282) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_17
  simp only [← c78876, ← c78877, k2, ← c78878] at e_1282_17
  have hr := e_1282_17
  simp only [e_1282_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1282 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11867)) := by
  have c78879 := (publicBatchWrapper86_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78880 := (publicBatchWrapper86_copies2465 a h).1
  have c78881 := (publicBatchWrapper86_copies2465 a h).2.1
  have c78882 := (publicBatchWrapper86_copies2465 a h).2.2.1
  have c78883 := (publicBatchWrapper86_copies2465 a h).2.2.2.1
  have c78884 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1282_18 := arithEq_of_rows h (row := 1282) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_18
  simp only [← c78879, ← c78880, ← c78881] at e_1282_18
  have e_1282_19 := arithEq_of_rows h (row := 1282) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_19
  simp only [← c78882, ← c78883, k2, ← c78884] at e_1282_19
  have hr := e_1282_19
  simp only [e_1282_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11868)) := by
  have c78885 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.1
  have c78886 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.1
  have c78887 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.1
  have c78888 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.1
  have c78889 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.1
  have c78890 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_0 := arithEq_of_rows h (row := 1283) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_0
  simp only [← c78885, ← c78886, ← c78887] at e_1283_0
  have e_1283_1 := arithEq_of_rows h (row := 1283) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_1
  simp only [← c78888, ← c78889, k2, ← c78890] at e_1283_1
  have hr := e_1283_1
  simp only [e_1283_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11869)) := by
  have c78891 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78892 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78893 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78894 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78895 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78896 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_2 := arithEq_of_rows h (row := 1283) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_2
  simp only [← c78891, ← c78892, ← c78893] at e_1283_2
  have e_1283_3 := arithEq_of_rows h (row := 1283) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_3
  simp only [← c78894, ← c78895, k2, ← c78896] at e_1283_3
  have hr := e_1283_3
  simp only [e_1283_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11870)) := by
  have c78897 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78898 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78899 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78900 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78901 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78902 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_4 := arithEq_of_rows h (row := 1283) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_4
  simp only [← c78897, ← c78898, ← c78899] at e_1283_4
  have e_1283_5 := arithEq_of_rows h (row := 1283) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_5
  simp only [← c78900, ← c78901, k2, ← c78902] at e_1283_5
  have hr := e_1283_5
  simp only [e_1283_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11871)) := by
  have c78903 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78904 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78905 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78906 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78907 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78908 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_6 := arithEq_of_rows h (row := 1283) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_6
  simp only [← c78903, ← c78904, ← c78905] at e_1283_6
  have e_1283_7 := arithEq_of_rows h (row := 1283) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_7
  simp only [← c78906, ← c78907, k2, ← c78908] at e_1283_7
  have hr := e_1283_7
  simp only [e_1283_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11872)) := by
  have c78909 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78910 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78911 := (publicBatchWrapper86_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78912 := (publicBatchWrapper86_copies2466 a h).1
  have c78913 := (publicBatchWrapper86_copies2466 a h).2.1
  have c78914 := (publicBatchWrapper86_copies2466 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_8 := arithEq_of_rows h (row := 1283) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_8
  simp only [← c78909, ← c78910, ← c78911] at e_1283_8
  have e_1283_9 := arithEq_of_rows h (row := 1283) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_9
  simp only [← c78912, ← c78913, k2, ← c78914] at e_1283_9
  have hr := e_1283_9
  simp only [e_1283_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11873)) := by
  have c78915 := (publicBatchWrapper86_copies2466 a h).2.2.2.1
  have c78916 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.1
  have c78917 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.1
  have c78918 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.1
  have c78919 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.1
  have c78920 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_10 := arithEq_of_rows h (row := 1283) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_10
  simp only [← c78915, ← c78916, ← c78917] at e_1283_10
  have e_1283_11 := arithEq_of_rows h (row := 1283) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_11
  simp only [← c78918, ← c78919, k2, ← c78920] at e_1283_11
  have hr := e_1283_11
  simp only [e_1283_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11874)) := by
  have c78921 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.1
  have c78922 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78923 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78924 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78925 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78926 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_12 := arithEq_of_rows h (row := 1283) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_12
  simp only [← c78921, ← c78922, ← c78923] at e_1283_12
  have e_1283_13 := arithEq_of_rows h (row := 1283) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_13
  simp only [← c78924, ← c78925, k2, ← c78926] at e_1283_13
  have hr := e_1283_13
  simp only [e_1283_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11875)) := by
  have c78927 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78928 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78929 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78930 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78931 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78932 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_14 := arithEq_of_rows h (row := 1283) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_14
  simp only [← c78927, ← c78928, ← c78929] at e_1283_14
  have e_1283_15 := arithEq_of_rows h (row := 1283) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_15
  simp only [← c78930, ← c78931, k2, ← c78932] at e_1283_15
  have hr := e_1283_15
  simp only [e_1283_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11876)) := by
  have c78933 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78934 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78935 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78936 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78937 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78938 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_16 := arithEq_of_rows h (row := 1283) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_16
  simp only [← c78933, ← c78934, ← c78935] at e_1283_16
  have e_1283_17 := arithEq_of_rows h (row := 1283) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_17
  simp only [← c78936, ← c78937, k2, ← c78938] at e_1283_17
  have hr := e_1283_17
  simp only [e_1283_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1283 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11877)) := by
  have c78939 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78940 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78941 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78942 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78943 := (publicBatchWrapper86_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78944 := (publicBatchWrapper86_copies2467 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1283_18 := arithEq_of_rows h (row := 1283) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_18
  simp only [← c78939, ← c78940, ← c78941] at e_1283_18
  have e_1283_19 := arithEq_of_rows h (row := 1283) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_19
  simp only [← c78942, ← c78943, k2, ← c78944] at e_1283_19
  have hr := e_1283_19
  simp only [e_1283_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11878)) := by
  have c78945 := (publicBatchWrapper86_copies2467 a h).2.1
  have c78946 := (publicBatchWrapper86_copies2467 a h).2.2.1
  have c78947 := (publicBatchWrapper86_copies2467 a h).2.2.2.1
  have c78948 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.1
  have c78949 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.1
  have c78950 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_0 := arithEq_of_rows h (row := 1284) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_0
  simp only [← c78945, ← c78946, ← c78947] at e_1284_0
  have e_1284_1 := arithEq_of_rows h (row := 1284) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_1
  simp only [← c78948, ← c78949, k2, ← c78950] at e_1284_1
  have hr := e_1284_1
  simp only [e_1284_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11879)) := by
  have c78951 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.1
  have c78952 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.1
  have c78953 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.1
  have c78954 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78955 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78956 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_2 := arithEq_of_rows h (row := 1284) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_2
  simp only [← c78951, ← c78952, ← c78953] at e_1284_2
  have e_1284_3 := arithEq_of_rows h (row := 1284) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_3
  simp only [← c78954, ← c78955, k2, ← c78956] at e_1284_3
  have hr := e_1284_3
  simp only [e_1284_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11880)) := by
  have c78957 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78958 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78959 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78960 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78961 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78962 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_4 := arithEq_of_rows h (row := 1284) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_4
  simp only [← c78957, ← c78958, ← c78959] at e_1284_4
  have e_1284_5 := arithEq_of_rows h (row := 1284) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_5
  simp only [← c78960, ← c78961, k2, ← c78962] at e_1284_5
  have hr := e_1284_5
  simp only [e_1284_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11881)) := by
  have c78963 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78964 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78965 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78966 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78967 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78968 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_6 := arithEq_of_rows h (row := 1284) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_6
  simp only [← c78963, ← c78964, ← c78965] at e_1284_6
  have e_1284_7 := arithEq_of_rows h (row := 1284) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_7
  simp only [← c78966, ← c78967, k2, ← c78968] at e_1284_7
  have hr := e_1284_7
  simp only [e_1284_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11882)) := by
  have c78969 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78970 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78971 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78972 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78973 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78974 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_8 := arithEq_of_rows h (row := 1284) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_8
  simp only [← c78969, ← c78970, ← c78971] at e_1284_8
  have e_1284_9 := arithEq_of_rows h (row := 1284) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_9
  simp only [← c78972, ← c78973, k2, ← c78974] at e_1284_9
  have hr := e_1284_9
  simp only [e_1284_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11883)) := by
  have c78975 := (publicBatchWrapper86_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78976 := (publicBatchWrapper86_copies2468 a h).1
  have c78977 := (publicBatchWrapper86_copies2468 a h).2.1
  have c78978 := (publicBatchWrapper86_copies2468 a h).2.2.1
  have c78979 := (publicBatchWrapper86_copies2468 a h).2.2.2.1
  have c78980 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_10 := arithEq_of_rows h (row := 1284) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_10
  simp only [← c78975, ← c78976, ← c78977] at e_1284_10
  have e_1284_11 := arithEq_of_rows h (row := 1284) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_11
  simp only [← c78978, ← c78979, k2, ← c78980] at e_1284_11
  have hr := e_1284_11
  simp only [e_1284_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11884)) := by
  have c78981 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.1
  have c78982 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.1
  have c78983 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.1
  have c78984 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.1
  have c78985 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.1
  have c78986 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_12 := arithEq_of_rows h (row := 1284) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_12
  simp only [← c78981, ← c78982, ← c78983] at e_1284_12
  have e_1284_13 := arithEq_of_rows h (row := 1284) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_13
  simp only [← c78984, ← c78985, k2, ← c78986] at e_1284_13
  have hr := e_1284_13
  simp only [e_1284_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11885)) := by
  have c78987 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78988 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78989 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78990 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78991 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78992 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_14 := arithEq_of_rows h (row := 1284) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_14
  simp only [← c78987, ← c78988, ← c78989] at e_1284_14
  have e_1284_15 := arithEq_of_rows h (row := 1284) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_15
  simp only [← c78990, ← c78991, k2, ← c78992] at e_1284_15
  have hr := e_1284_15
  simp only [e_1284_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11886)) := by
  have c78993 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78994 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78995 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78996 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78997 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78998 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_16 := arithEq_of_rows h (row := 1284) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_16
  simp only [← c78993, ← c78994, ← c78995] at e_1284_16
  have e_1284_17 := arithEq_of_rows h (row := 1284) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_17
  simp only [← c78996, ← c78997, k2, ← c78998] at e_1284_17
  have hr := e_1284_17
  simp only [e_1284_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1284 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11887)) := by
  have c78999 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79000 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79001 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79002 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79003 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79004 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1284_18 := arithEq_of_rows h (row := 1284) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_18
  simp only [← c78999, ← c79000, ← c79001] at e_1284_18
  have e_1284_19 := arithEq_of_rows h (row := 1284) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_19
  simp only [← c79002, ← c79003, k2, ← c79004] at e_1284_19
  have hr := e_1284_19
  simp only [e_1284_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11888)) := by
  have c79005 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79006 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79007 := (publicBatchWrapper86_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79008 := (publicBatchWrapper86_copies2469 a h).1
  have c79009 := (publicBatchWrapper86_copies2469 a h).2.1
  have c79010 := (publicBatchWrapper86_copies2469 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_0 := arithEq_of_rows h (row := 1285) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_0
  simp only [← c79005, ← c79006, ← c79007] at e_1285_0
  have e_1285_1 := arithEq_of_rows h (row := 1285) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_1
  simp only [← c79008, ← c79009, k2, ← c79010] at e_1285_1
  have hr := e_1285_1
  simp only [e_1285_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11889)) := by
  have c79011 := (publicBatchWrapper86_copies2469 a h).2.2.2.1
  have c79012 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.1
  have c79013 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.1
  have c79014 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.1
  have c79015 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.1
  have c79016 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_2 := arithEq_of_rows h (row := 1285) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_2
  simp only [← c79011, ← c79012, ← c79013] at e_1285_2
  have e_1285_3 := arithEq_of_rows h (row := 1285) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_3
  simp only [← c79014, ← c79015, k2, ← c79016] at e_1285_3
  have hr := e_1285_3
  simp only [e_1285_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11890)) := by
  have c79017 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.1
  have c79018 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79019 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79020 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79021 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79022 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_4 := arithEq_of_rows h (row := 1285) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_4
  simp only [← c79017, ← c79018, ← c79019] at e_1285_4
  have e_1285_5 := arithEq_of_rows h (row := 1285) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_5
  simp only [← c79020, ← c79021, k2, ← c79022] at e_1285_5
  have hr := e_1285_5
  simp only [e_1285_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11891)) := by
  have c79023 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79024 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79025 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79026 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79027 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79028 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_6 := arithEq_of_rows h (row := 1285) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_6
  simp only [← c79023, ← c79024, ← c79025] at e_1285_6
  have e_1285_7 := arithEq_of_rows h (row := 1285) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_7
  simp only [← c79026, ← c79027, k2, ← c79028] at e_1285_7
  have hr := e_1285_7
  simp only [e_1285_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11892)) := by
  have c79029 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79030 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79031 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79032 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79033 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79034 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_8 := arithEq_of_rows h (row := 1285) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_8
  simp only [← c79029, ← c79030, ← c79031] at e_1285_8
  have e_1285_9 := arithEq_of_rows h (row := 1285) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_9
  simp only [← c79032, ← c79033, k2, ← c79034] at e_1285_9
  have hr := e_1285_9
  simp only [e_1285_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11893)) := by
  have c79035 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79036 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79037 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79038 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79039 := (publicBatchWrapper86_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79040 := (publicBatchWrapper86_copies2470 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_10 := arithEq_of_rows h (row := 1285) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_10
  simp only [← c79035, ← c79036, ← c79037] at e_1285_10
  have e_1285_11 := arithEq_of_rows h (row := 1285) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_11
  simp only [← c79038, ← c79039, k2, ← c79040] at e_1285_11
  have hr := e_1285_11
  simp only [e_1285_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11894)) := by
  have c79041 := (publicBatchWrapper86_copies2470 a h).2.1
  have c79042 := (publicBatchWrapper86_copies2470 a h).2.2.1
  have c79043 := (publicBatchWrapper86_copies2470 a h).2.2.2.1
  have c79044 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.1
  have c79045 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.1
  have c79046 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_12 := arithEq_of_rows h (row := 1285) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_12
  simp only [← c79041, ← c79042, ← c79043] at e_1285_12
  have e_1285_13 := arithEq_of_rows h (row := 1285) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_13
  simp only [← c79044, ← c79045, k2, ← c79046] at e_1285_13
  have hr := e_1285_13
  simp only [e_1285_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11895)) := by
  have c79047 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.1
  have c79048 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.1
  have c79049 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.1
  have c79050 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79051 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79052 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_14 := arithEq_of_rows h (row := 1285) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_14
  simp only [← c79047, ← c79048, ← c79049] at e_1285_14
  have e_1285_15 := arithEq_of_rows h (row := 1285) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_15
  simp only [← c79050, ← c79051, k2, ← c79052] at e_1285_15
  have hr := e_1285_15
  simp only [e_1285_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11896)) := by
  have c79053 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79054 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79055 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79056 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79057 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79058 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_16 := arithEq_of_rows h (row := 1285) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_16
  simp only [← c79053, ← c79054, ← c79055] at e_1285_16
  have e_1285_17 := arithEq_of_rows h (row := 1285) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_17
  simp only [← c79056, ← c79057, k2, ← c79058] at e_1285_17
  have hr := e_1285_17
  simp only [e_1285_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1285 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11897)) := by
  have c79059 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79060 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79061 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79062 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79063 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79064 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1285_18 := arithEq_of_rows h (row := 1285) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_18
  simp only [← c79059, ← c79060, ← c79061] at e_1285_18
  have e_1285_19 := arithEq_of_rows h (row := 1285) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_19
  simp only [← c79062, ← c79063, k2, ← c79064] at e_1285_19
  have hr := e_1285_19
  simp only [e_1285_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11898)) := by
  have c79065 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79066 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79067 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79068 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79069 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79070 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_0 := arithEq_of_rows h (row := 1286) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_0
  simp only [← c79065, ← c79066, ← c79067] at e_1286_0
  have e_1286_1 := arithEq_of_rows h (row := 1286) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_1
  simp only [← c79068, ← c79069, k2, ← c79070] at e_1286_1
  have hr := e_1286_1
  simp only [e_1286_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11899)) := by
  have c79071 := (publicBatchWrapper86_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79072 := (publicBatchWrapper86_copies2471 a h).1
  have c79073 := (publicBatchWrapper86_copies2471 a h).2.1
  have c79074 := (publicBatchWrapper86_copies2471 a h).2.2.1
  have c79075 := (publicBatchWrapper86_copies2471 a h).2.2.2.1
  have c79076 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_2 := arithEq_of_rows h (row := 1286) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_2
  simp only [← c79071, ← c79072, ← c79073] at e_1286_2
  have e_1286_3 := arithEq_of_rows h (row := 1286) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_3
  simp only [← c79074, ← c79075, k2, ← c79076] at e_1286_3
  have hr := e_1286_3
  simp only [e_1286_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11900)) := by
  have c79077 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.1
  have c79078 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.1
  have c79079 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.1
  have c79080 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.1
  have c79081 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.1
  have c79082 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_4 := arithEq_of_rows h (row := 1286) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_4
  simp only [← c79077, ← c79078, ← c79079] at e_1286_4
  have e_1286_5 := arithEq_of_rows h (row := 1286) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_5
  simp only [← c79080, ← c79081, k2, ← c79082] at e_1286_5
  have hr := e_1286_5
  simp only [e_1286_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11901)) := by
  have c79083 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79084 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79085 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79086 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79087 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79088 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_6 := arithEq_of_rows h (row := 1286) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_6
  simp only [← c79083, ← c79084, ← c79085] at e_1286_6
  have e_1286_7 := arithEq_of_rows h (row := 1286) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_7
  simp only [← c79086, ← c79087, k2, ← c79088] at e_1286_7
  have hr := e_1286_7
  simp only [e_1286_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11902)) := by
  have c79089 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79090 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79091 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79092 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79093 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79094 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_8 := arithEq_of_rows h (row := 1286) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_8
  simp only [← c79089, ← c79090, ← c79091] at e_1286_8
  have e_1286_9 := arithEq_of_rows h (row := 1286) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_9
  simp only [← c79092, ← c79093, k2, ← c79094] at e_1286_9
  have hr := e_1286_9
  simp only [e_1286_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11903)) := by
  have c79095 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79096 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79097 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79098 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79099 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79100 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_10 := arithEq_of_rows h (row := 1286) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_10
  simp only [← c79095, ← c79096, ← c79097] at e_1286_10
  have e_1286_11 := arithEq_of_rows h (row := 1286) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_11
  simp only [← c79098, ← c79099, k2, ← c79100] at e_1286_11
  have hr := e_1286_11
  simp only [e_1286_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11904)) := by
  have c79101 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79102 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79103 := (publicBatchWrapper86_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79104 := (publicBatchWrapper86_copies2472 a h).1
  have c79105 := (publicBatchWrapper86_copies2472 a h).2.1
  have c79106 := (publicBatchWrapper86_copies2472 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_12 := arithEq_of_rows h (row := 1286) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_12
  simp only [← c79101, ← c79102, ← c79103] at e_1286_12
  have e_1286_13 := arithEq_of_rows h (row := 1286) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_13
  simp only [← c79104, ← c79105, k2, ← c79106] at e_1286_13
  have hr := e_1286_13
  simp only [e_1286_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11905)) := by
  have c79107 := (publicBatchWrapper86_copies2472 a h).2.2.2.1
  have c79108 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.1
  have c79109 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.1
  have c79110 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.1
  have c79111 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.1
  have c79112 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_14 := arithEq_of_rows h (row := 1286) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_14
  simp only [← c79107, ← c79108, ← c79109] at e_1286_14
  have e_1286_15 := arithEq_of_rows h (row := 1286) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_15
  simp only [← c79110, ← c79111, k2, ← c79112] at e_1286_15
  have hr := e_1286_15
  simp only [e_1286_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11906)) := by
  have c79113 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.1
  have c79114 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79115 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79116 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79117 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79118 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_16 := arithEq_of_rows h (row := 1286) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_16
  simp only [← c79113, ← c79114, ← c79115] at e_1286_16
  have e_1286_17 := arithEq_of_rows h (row := 1286) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_17
  simp only [← c79116, ← c79117, k2, ← c79118] at e_1286_17
  have hr := e_1286_17
  simp only [e_1286_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1286 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11907)) := by
  have c79119 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79120 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79121 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79122 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79123 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79124 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1286_18 := arithEq_of_rows h (row := 1286) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_18
  simp only [← c79119, ← c79120, ← c79121] at e_1286_18
  have e_1286_19 := arithEq_of_rows h (row := 1286) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_19
  simp only [← c79122, ← c79123, k2, ← c79124] at e_1286_19
  have hr := e_1286_19
  simp only [e_1286_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11908)) := by
  have c79125 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79126 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79127 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79128 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79129 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79130 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_0 := arithEq_of_rows h (row := 1287) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_0
  simp only [← c79125, ← c79126, ← c79127] at e_1287_0
  have e_1287_1 := arithEq_of_rows h (row := 1287) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_1
  simp only [← c79128, ← c79129, k2, ← c79130] at e_1287_1
  have hr := e_1287_1
  simp only [e_1287_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11909)) := by
  have c79131 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79132 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79133 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79134 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79135 := (publicBatchWrapper86_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79136 := (publicBatchWrapper86_copies2473 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_2 := arithEq_of_rows h (row := 1287) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_2
  simp only [← c79131, ← c79132, ← c79133] at e_1287_2
  have e_1287_3 := arithEq_of_rows h (row := 1287) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_3
  simp only [← c79134, ← c79135, k2, ← c79136] at e_1287_3
  have hr := e_1287_3
  simp only [e_1287_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11910)) := by
  have c79137 := (publicBatchWrapper86_copies2473 a h).2.1
  have c79138 := (publicBatchWrapper86_copies2473 a h).2.2.1
  have c79139 := (publicBatchWrapper86_copies2473 a h).2.2.2.1
  have c79140 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.1
  have c79141 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.1
  have c79142 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_4 := arithEq_of_rows h (row := 1287) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_4
  simp only [← c79137, ← c79138, ← c79139] at e_1287_4
  have e_1287_5 := arithEq_of_rows h (row := 1287) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_5
  simp only [← c79140, ← c79141, k2, ← c79142] at e_1287_5
  have hr := e_1287_5
  simp only [e_1287_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11911)) := by
  have c79143 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.1
  have c79144 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.1
  have c79145 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.1
  have c79146 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79147 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79148 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_6 := arithEq_of_rows h (row := 1287) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_6
  simp only [← c79143, ← c79144, ← c79145] at e_1287_6
  have e_1287_7 := arithEq_of_rows h (row := 1287) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_7
  simp only [← c79146, ← c79147, k2, ← c79148] at e_1287_7
  have hr := e_1287_7
  simp only [e_1287_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11912)) := by
  have c79149 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79150 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79151 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79152 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79153 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79154 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_8 := arithEq_of_rows h (row := 1287) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_8
  simp only [← c79149, ← c79150, ← c79151] at e_1287_8
  have e_1287_9 := arithEq_of_rows h (row := 1287) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_9
  simp only [← c79152, ← c79153, k2, ← c79154] at e_1287_9
  have hr := e_1287_9
  simp only [e_1287_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11913)) := by
  have c79155 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79156 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79157 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79158 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79159 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79160 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_10 := arithEq_of_rows h (row := 1287) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_10
  simp only [← c79155, ← c79156, ← c79157] at e_1287_10
  have e_1287_11 := arithEq_of_rows h (row := 1287) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_11
  simp only [← c79158, ← c79159, k2, ← c79160] at e_1287_11
  have hr := e_1287_11
  simp only [e_1287_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11914)) := by
  have c79161 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79162 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79163 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79164 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79165 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79166 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_12 := arithEq_of_rows h (row := 1287) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_12
  simp only [← c79161, ← c79162, ← c79163] at e_1287_12
  have e_1287_13 := arithEq_of_rows h (row := 1287) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_13
  simp only [← c79164, ← c79165, k2, ← c79166] at e_1287_13
  have hr := e_1287_13
  simp only [e_1287_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11915)) := by
  have c79167 := (publicBatchWrapper86_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79168 := (publicBatchWrapper86_copies2474 a h).1
  have c79169 := (publicBatchWrapper86_copies2474 a h).2.1
  have c79170 := (publicBatchWrapper86_copies2474 a h).2.2.1
  have c79171 := (publicBatchWrapper86_copies2474 a h).2.2.2.1
  have c79172 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_14 := arithEq_of_rows h (row := 1287) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_14
  simp only [← c79167, ← c79168, ← c79169] at e_1287_14
  have e_1287_15 := arithEq_of_rows h (row := 1287) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_15
  simp only [← c79170, ← c79171, k2, ← c79172] at e_1287_15
  have hr := e_1287_15
  simp only [e_1287_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11916)) := by
  have c79173 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.1
  have c79174 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.1
  have c79175 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.1
  have c79176 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.1
  have c79177 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.1
  have c79178 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_16 := arithEq_of_rows h (row := 1287) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_16
  simp only [← c79173, ← c79174, ← c79175] at e_1287_16
  have e_1287_17 := arithEq_of_rows h (row := 1287) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_17
  simp only [← c79176, ← c79177, k2, ← c79178] at e_1287_17
  have hr := e_1287_17
  simp only [e_1287_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1287 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11917)) := by
  have c79179 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79180 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79181 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79182 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79183 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79184 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1287_18 := arithEq_of_rows h (row := 1287) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_18
  simp only [← c79179, ← c79180, ← c79181] at e_1287_18
  have e_1287_19 := arithEq_of_rows h (row := 1287) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_19
  simp only [← c79182, ← c79183, k2, ← c79184] at e_1287_19
  have hr := e_1287_19
  simp only [e_1287_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11918)) := by
  have c79185 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79186 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79187 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79188 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79189 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79190 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_0 := arithEq_of_rows h (row := 1288) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_0
  simp only [← c79185, ← c79186, ← c79187] at e_1288_0
  have e_1288_1 := arithEq_of_rows h (row := 1288) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_1
  simp only [← c79188, ← c79189, k2, ← c79190] at e_1288_1
  have hr := e_1288_1
  simp only [e_1288_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11919)) := by
  have c79191 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79192 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79193 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79194 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79195 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79196 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_2 := arithEq_of_rows h (row := 1288) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_2
  simp only [← c79191, ← c79192, ← c79193] at e_1288_2
  have e_1288_3 := arithEq_of_rows h (row := 1288) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_3
  simp only [← c79194, ← c79195, k2, ← c79196] at e_1288_3
  have hr := e_1288_3
  simp only [e_1288_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11920)) := by
  have c79197 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79198 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79199 := (publicBatchWrapper86_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79200 := (publicBatchWrapper86_copies2475 a h).1
  have c79201 := (publicBatchWrapper86_copies2475 a h).2.1
  have c79202 := (publicBatchWrapper86_copies2475 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_4 := arithEq_of_rows h (row := 1288) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_4
  simp only [← c79197, ← c79198, ← c79199] at e_1288_4
  have e_1288_5 := arithEq_of_rows h (row := 1288) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_5
  simp only [← c79200, ← c79201, k2, ← c79202] at e_1288_5
  have hr := e_1288_5
  simp only [e_1288_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11921)) := by
  have c79203 := (publicBatchWrapper86_copies2475 a h).2.2.2.1
  have c79204 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.1
  have c79205 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.1
  have c79206 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.1
  have c79207 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.1
  have c79208 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_6 := arithEq_of_rows h (row := 1288) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_6
  simp only [← c79203, ← c79204, ← c79205] at e_1288_6
  have e_1288_7 := arithEq_of_rows h (row := 1288) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_7
  simp only [← c79206, ← c79207, k2, ← c79208] at e_1288_7
  have hr := e_1288_7
  simp only [e_1288_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11922)) := by
  have c79209 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.1
  have c79210 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79211 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79212 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79213 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79214 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_8 := arithEq_of_rows h (row := 1288) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_8
  simp only [← c79209, ← c79210, ← c79211] at e_1288_8
  have e_1288_9 := arithEq_of_rows h (row := 1288) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_9
  simp only [← c79212, ← c79213, k2, ← c79214] at e_1288_9
  have hr := e_1288_9
  simp only [e_1288_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11923)) := by
  have c79215 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79216 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79217 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79218 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79219 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79220 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_10 := arithEq_of_rows h (row := 1288) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_10
  simp only [← c79215, ← c79216, ← c79217] at e_1288_10
  have e_1288_11 := arithEq_of_rows h (row := 1288) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_11
  simp only [← c79218, ← c79219, k2, ← c79220] at e_1288_11
  have hr := e_1288_11
  simp only [e_1288_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11924)) := by
  have c79221 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79222 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79223 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79224 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79225 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79226 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_12 := arithEq_of_rows h (row := 1288) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_12
  simp only [← c79221, ← c79222, ← c79223] at e_1288_12
  have e_1288_13 := arithEq_of_rows h (row := 1288) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_13
  simp only [← c79224, ← c79225, k2, ← c79226] at e_1288_13
  have hr := e_1288_13
  simp only [e_1288_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11925)) := by
  have c79227 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79228 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79229 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79230 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79231 := (publicBatchWrapper86_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79232 := (publicBatchWrapper86_copies2476 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_14 := arithEq_of_rows h (row := 1288) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_14
  simp only [← c79227, ← c79228, ← c79229] at e_1288_14
  have e_1288_15 := arithEq_of_rows h (row := 1288) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_15
  simp only [← c79230, ← c79231, k2, ← c79232] at e_1288_15
  have hr := e_1288_15
  simp only [e_1288_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11926)) := by
  have c79233 := (publicBatchWrapper86_copies2476 a h).2.1
  have c79234 := (publicBatchWrapper86_copies2476 a h).2.2.1
  have c79235 := (publicBatchWrapper86_copies2476 a h).2.2.2.1
  have c79236 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.1
  have c79237 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.1
  have c79238 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_16 := arithEq_of_rows h (row := 1288) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_16
  simp only [← c79233, ← c79234, ← c79235] at e_1288_16
  have e_1288_17 := arithEq_of_rows h (row := 1288) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_17
  simp only [← c79236, ← c79237, k2, ← c79238] at e_1288_17
  have hr := e_1288_17
  simp only [e_1288_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1288 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11927)) := by
  have c79239 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.1
  have c79240 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.1
  have c79241 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.1
  have c79242 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79243 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79244 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1288_18 := arithEq_of_rows h (row := 1288) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_18
  simp only [← c79239, ← c79240, ← c79241] at e_1288_18
  have e_1288_19 := arithEq_of_rows h (row := 1288) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_19
  simp only [← c79242, ← c79243, k2, ← c79244] at e_1288_19
  have hr := e_1288_19
  simp only [e_1288_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11928)) := by
  have c79245 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79246 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79247 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79248 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79249 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79250 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_0 := arithEq_of_rows h (row := 1289) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_0
  simp only [← c79245, ← c79246, ← c79247] at e_1289_0
  have e_1289_1 := arithEq_of_rows h (row := 1289) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_1
  simp only [← c79248, ← c79249, k2, ← c79250] at e_1289_1
  have hr := e_1289_1
  simp only [e_1289_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11929)) := by
  have c79251 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79252 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79253 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79254 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79255 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79256 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_2 := arithEq_of_rows h (row := 1289) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_2
  simp only [← c79251, ← c79252, ← c79253] at e_1289_2
  have e_1289_3 := arithEq_of_rows h (row := 1289) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_3
  simp only [← c79254, ← c79255, k2, ← c79256] at e_1289_3
  have hr := e_1289_3
  simp only [e_1289_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11930)) := by
  have c79257 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79258 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79259 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79260 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79261 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79262 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_4 := arithEq_of_rows h (row := 1289) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_4
  simp only [← c79257, ← c79258, ← c79259] at e_1289_4
  have e_1289_5 := arithEq_of_rows h (row := 1289) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_5
  simp only [← c79260, ← c79261, k2, ← c79262] at e_1289_5
  have hr := e_1289_5
  simp only [e_1289_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11931)) := by
  have c79263 := (publicBatchWrapper86_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79264 := (publicBatchWrapper86_copies2477 a h).1
  have c79265 := (publicBatchWrapper86_copies2477 a h).2.1
  have c79266 := (publicBatchWrapper86_copies2477 a h).2.2.1
  have c79267 := (publicBatchWrapper86_copies2477 a h).2.2.2.1
  have c79268 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_6 := arithEq_of_rows h (row := 1289) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_6
  simp only [← c79263, ← c79264, ← c79265] at e_1289_6
  have e_1289_7 := arithEq_of_rows h (row := 1289) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_7
  simp only [← c79266, ← c79267, k2, ← c79268] at e_1289_7
  have hr := e_1289_7
  simp only [e_1289_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11932)) := by
  have c79269 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.1
  have c79270 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.1
  have c79271 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.1
  have c79272 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.1
  have c79273 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.1
  have c79274 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_8 := arithEq_of_rows h (row := 1289) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_8
  simp only [← c79269, ← c79270, ← c79271] at e_1289_8
  have e_1289_9 := arithEq_of_rows h (row := 1289) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_9
  simp only [← c79272, ← c79273, k2, ← c79274] at e_1289_9
  have hr := e_1289_9
  simp only [e_1289_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11933)) := by
  have c79275 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79276 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79277 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79278 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79279 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79280 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_10 := arithEq_of_rows h (row := 1289) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_10
  simp only [← c79275, ← c79276, ← c79277] at e_1289_10
  have e_1289_11 := arithEq_of_rows h (row := 1289) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_11
  simp only [← c79278, ← c79279, k2, ← c79280] at e_1289_11
  have hr := e_1289_11
  simp only [e_1289_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11934)) := by
  have c79281 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79282 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79283 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79284 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79285 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79286 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_12 := arithEq_of_rows h (row := 1289) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_12
  simp only [← c79281, ← c79282, ← c79283] at e_1289_12
  have e_1289_13 := arithEq_of_rows h (row := 1289) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_13
  simp only [← c79284, ← c79285, k2, ← c79286] at e_1289_13
  have hr := e_1289_13
  simp only [e_1289_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11935)) := by
  have c79287 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79288 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79289 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79290 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79291 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79292 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_14 := arithEq_of_rows h (row := 1289) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_14
  simp only [← c79287, ← c79288, ← c79289] at e_1289_14
  have e_1289_15 := arithEq_of_rows h (row := 1289) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_15
  simp only [← c79290, ← c79291, k2, ← c79292] at e_1289_15
  have hr := e_1289_15
  simp only [e_1289_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11936)) := by
  have c79293 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79294 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79295 := (publicBatchWrapper86_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79296 := (publicBatchWrapper86_copies2478 a h).1
  have c79297 := (publicBatchWrapper86_copies2478 a h).2.1
  have c79298 := (publicBatchWrapper86_copies2478 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_16 := arithEq_of_rows h (row := 1289) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_16
  simp only [← c79293, ← c79294, ← c79295] at e_1289_16
  have e_1289_17 := arithEq_of_rows h (row := 1289) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_17
  simp only [← c79296, ← c79297, k2, ← c79298] at e_1289_17
  have hr := e_1289_17
  simp only [e_1289_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1289 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11937)) := by
  have c79299 := (publicBatchWrapper86_copies2478 a h).2.2.2.1
  have c79300 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.1
  have c79301 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.1
  have c79302 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.1
  have c79303 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.1
  have c79304 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1289_18 := arithEq_of_rows h (row := 1289) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_18
  simp only [← c79299, ← c79300, ← c79301] at e_1289_18
  have e_1289_19 := arithEq_of_rows h (row := 1289) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_19
  simp only [← c79302, ← c79303, k2, ← c79304] at e_1289_19
  have hr := e_1289_19
  simp only [e_1289_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11938)) := by
  have c79305 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.1
  have c79306 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79307 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79308 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79309 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79310 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_0 := arithEq_of_rows h (row := 1290) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_0
  simp only [← c79305, ← c79306, ← c79307] at e_1290_0
  have e_1290_1 := arithEq_of_rows h (row := 1290) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_1
  simp only [← c79308, ← c79309, k2, ← c79310] at e_1290_1
  have hr := e_1290_1
  simp only [e_1290_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11939)) := by
  have c79311 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79312 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79313 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79314 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79315 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79316 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_2 := arithEq_of_rows h (row := 1290) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_2
  simp only [← c79311, ← c79312, ← c79313] at e_1290_2
  have e_1290_3 := arithEq_of_rows h (row := 1290) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_3
  simp only [← c79314, ← c79315, k2, ← c79316] at e_1290_3
  have hr := e_1290_3
  simp only [e_1290_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11940)) := by
  have c79317 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79318 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79319 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79320 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79321 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79322 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_4 := arithEq_of_rows h (row := 1290) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_4
  simp only [← c79317, ← c79318, ← c79319] at e_1290_4
  have e_1290_5 := arithEq_of_rows h (row := 1290) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_5
  simp only [← c79320, ← c79321, k2, ← c79322] at e_1290_5
  have hr := e_1290_5
  simp only [e_1290_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11941)) := by
  have c79323 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79324 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79325 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79326 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79327 := (publicBatchWrapper86_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79328 := (publicBatchWrapper86_copies2479 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_6 := arithEq_of_rows h (row := 1290) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_6
  simp only [← c79323, ← c79324, ← c79325] at e_1290_6
  have e_1290_7 := arithEq_of_rows h (row := 1290) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_7
  simp only [← c79326, ← c79327, k2, ← c79328] at e_1290_7
  have hr := e_1290_7
  simp only [e_1290_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11942)) := by
  have c79329 := (publicBatchWrapper86_copies2479 a h).2.1
  have c79330 := (publicBatchWrapper86_copies2479 a h).2.2.1
  have c79331 := (publicBatchWrapper86_copies2479 a h).2.2.2.1
  have c79332 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.1
  have c79333 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.1
  have c79334 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_8 := arithEq_of_rows h (row := 1290) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_8
  simp only [← c79329, ← c79330, ← c79331] at e_1290_8
  have e_1290_9 := arithEq_of_rows h (row := 1290) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_9
  simp only [← c79332, ← c79333, k2, ← c79334] at e_1290_9
  have hr := e_1290_9
  simp only [e_1290_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11943)) := by
  have c79335 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.1
  have c79336 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.1
  have c79337 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.1
  have c79338 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79339 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79340 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_10 := arithEq_of_rows h (row := 1290) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_10
  simp only [← c79335, ← c79336, ← c79337] at e_1290_10
  have e_1290_11 := arithEq_of_rows h (row := 1290) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_11
  simp only [← c79338, ← c79339, k2, ← c79340] at e_1290_11
  have hr := e_1290_11
  simp only [e_1290_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11944)) := by
  have c79341 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79342 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79343 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79344 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79345 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79346 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_12 := arithEq_of_rows h (row := 1290) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_12
  simp only [← c79341, ← c79342, ← c79343] at e_1290_12
  have e_1290_13 := arithEq_of_rows h (row := 1290) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_13
  simp only [← c79344, ← c79345, k2, ← c79346] at e_1290_13
  have hr := e_1290_13
  simp only [e_1290_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11945)) := by
  have c79347 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79348 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79349 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79350 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79351 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79352 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_14 := arithEq_of_rows h (row := 1290) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_14
  simp only [← c79347, ← c79348, ← c79349] at e_1290_14
  have e_1290_15 := arithEq_of_rows h (row := 1290) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_15
  simp only [← c79350, ← c79351, k2, ← c79352] at e_1290_15
  have hr := e_1290_15
  simp only [e_1290_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11946)) := by
  have c79353 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79354 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79355 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79356 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79357 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79358 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_16 := arithEq_of_rows h (row := 1290) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_16
  simp only [← c79353, ← c79354, ← c79355] at e_1290_16
  have e_1290_17 := arithEq_of_rows h (row := 1290) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_17
  simp only [← c79356, ← c79357, k2, ← c79358] at e_1290_17
  have hr := e_1290_17
  simp only [e_1290_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1290 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11947)) := by
  have c79359 := (publicBatchWrapper86_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79360 := (publicBatchWrapper86_copies2480 a h).1
  have c79361 := (publicBatchWrapper86_copies2480 a h).2.1
  have c79362 := (publicBatchWrapper86_copies2480 a h).2.2.1
  have c79363 := (publicBatchWrapper86_copies2480 a h).2.2.2.1
  have c79364 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1290_18 := arithEq_of_rows h (row := 1290) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_18
  simp only [← c79359, ← c79360, ← c79361] at e_1290_18
  have e_1290_19 := arithEq_of_rows h (row := 1290) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_19
  simp only [← c79362, ← c79363, k2, ← c79364] at e_1290_19
  have hr := e_1290_19
  simp only [e_1290_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11948)) := by
  have c79365 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.1
  have c79366 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.1
  have c79367 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.1
  have c79368 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.1
  have c79369 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.1
  have c79370 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_0 := arithEq_of_rows h (row := 1291) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_0
  simp only [← c79365, ← c79366, ← c79367] at e_1291_0
  have e_1291_1 := arithEq_of_rows h (row := 1291) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_1
  simp only [← c79368, ← c79369, k2, ← c79370] at e_1291_1
  have hr := e_1291_1
  simp only [e_1291_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11949)) := by
  have c79371 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79372 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79373 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79374 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79375 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79376 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_2 := arithEq_of_rows h (row := 1291) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_2
  simp only [← c79371, ← c79372, ← c79373] at e_1291_2
  have e_1291_3 := arithEq_of_rows h (row := 1291) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_3
  simp only [← c79374, ← c79375, k2, ← c79376] at e_1291_3
  have hr := e_1291_3
  simp only [e_1291_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11950)) := by
  have c79377 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79378 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79379 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79380 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79381 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79382 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_4 := arithEq_of_rows h (row := 1291) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_4
  simp only [← c79377, ← c79378, ← c79379] at e_1291_4
  have e_1291_5 := arithEq_of_rows h (row := 1291) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_5
  simp only [← c79380, ← c79381, k2, ← c79382] at e_1291_5
  have hr := e_1291_5
  simp only [e_1291_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11951)) := by
  have c79383 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79384 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79385 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79386 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79387 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79388 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_6 := arithEq_of_rows h (row := 1291) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_6
  simp only [← c79383, ← c79384, ← c79385] at e_1291_6
  have e_1291_7 := arithEq_of_rows h (row := 1291) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_7
  simp only [← c79386, ← c79387, k2, ← c79388] at e_1291_7
  have hr := e_1291_7
  simp only [e_1291_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11952)) := by
  have c79389 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79390 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79391 := (publicBatchWrapper86_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79392 := (publicBatchWrapper86_copies2481 a h).1
  have c79393 := (publicBatchWrapper86_copies2481 a h).2.1
  have c79394 := (publicBatchWrapper86_copies2481 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_8 := arithEq_of_rows h (row := 1291) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_8
  simp only [← c79389, ← c79390, ← c79391] at e_1291_8
  have e_1291_9 := arithEq_of_rows h (row := 1291) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_9
  simp only [← c79392, ← c79393, k2, ← c79394] at e_1291_9
  have hr := e_1291_9
  simp only [e_1291_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11953)) := by
  have c79395 := (publicBatchWrapper86_copies2481 a h).2.2.2.1
  have c79396 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.1
  have c79397 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.1
  have c79398 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.1
  have c79399 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.1
  have c79400 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_10 := arithEq_of_rows h (row := 1291) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_10
  simp only [← c79395, ← c79396, ← c79397] at e_1291_10
  have e_1291_11 := arithEq_of_rows h (row := 1291) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_11
  simp only [← c79398, ← c79399, k2, ← c79400] at e_1291_11
  have hr := e_1291_11
  simp only [e_1291_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11954)) := by
  have c79401 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.1
  have c79402 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79403 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79404 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79405 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79406 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_12 := arithEq_of_rows h (row := 1291) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_12
  simp only [← c79401, ← c79402, ← c79403] at e_1291_12
  have e_1291_13 := arithEq_of_rows h (row := 1291) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_13
  simp only [← c79404, ← c79405, k2, ← c79406] at e_1291_13
  have hr := e_1291_13
  simp only [e_1291_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11955)) := by
  have c79407 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79408 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79409 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79410 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79411 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79412 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_14 := arithEq_of_rows h (row := 1291) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_14
  simp only [← c79407, ← c79408, ← c79409] at e_1291_14
  have e_1291_15 := arithEq_of_rows h (row := 1291) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_15
  simp only [← c79410, ← c79411, k2, ← c79412] at e_1291_15
  have hr := e_1291_15
  simp only [e_1291_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11956)) := by
  have c79413 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79414 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79415 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79416 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79417 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79418 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_16 := arithEq_of_rows h (row := 1291) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_16
  simp only [← c79413, ← c79414, ← c79415] at e_1291_16
  have e_1291_17 := arithEq_of_rows h (row := 1291) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_17
  simp only [← c79416, ← c79417, k2, ← c79418] at e_1291_17
  have hr := e_1291_17
  simp only [e_1291_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1291 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11957)) := by
  have c79419 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79420 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79421 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79422 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79423 := (publicBatchWrapper86_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79424 := (publicBatchWrapper86_copies2482 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1291_18 := arithEq_of_rows h (row := 1291) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_18
  simp only [← c79419, ← c79420, ← c79421] at e_1291_18
  have e_1291_19 := arithEq_of_rows h (row := 1291) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_19
  simp only [← c79422, ← c79423, k2, ← c79424] at e_1291_19
  have hr := e_1291_19
  simp only [e_1291_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11958)) := by
  have c79425 := (publicBatchWrapper86_copies2482 a h).2.1
  have c79426 := (publicBatchWrapper86_copies2482 a h).2.2.1
  have c79427 := (publicBatchWrapper86_copies2482 a h).2.2.2.1
  have c79428 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.1
  have c79429 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.1
  have c79430 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_0 := arithEq_of_rows h (row := 1292) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_0
  simp only [← c79425, ← c79426, ← c79427] at e_1292_0
  have e_1292_1 := arithEq_of_rows h (row := 1292) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_1
  simp only [← c79428, ← c79429, k2, ← c79430] at e_1292_1
  have hr := e_1292_1
  simp only [e_1292_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11959)) := by
  have c79431 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.1
  have c79432 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.1
  have c79433 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.1
  have c79434 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79435 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79436 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_2 := arithEq_of_rows h (row := 1292) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_2
  simp only [← c79431, ← c79432, ← c79433] at e_1292_2
  have e_1292_3 := arithEq_of_rows h (row := 1292) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_3
  simp only [← c79434, ← c79435, k2, ← c79436] at e_1292_3
  have hr := e_1292_3
  simp only [e_1292_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11960)) := by
  have c79437 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79438 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79439 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79440 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79441 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79442 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_4 := arithEq_of_rows h (row := 1292) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_4
  simp only [← c79437, ← c79438, ← c79439] at e_1292_4
  have e_1292_5 := arithEq_of_rows h (row := 1292) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_5
  simp only [← c79440, ← c79441, k2, ← c79442] at e_1292_5
  have hr := e_1292_5
  simp only [e_1292_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11961)) := by
  have c79443 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79444 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79445 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79446 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79447 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79448 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_6 := arithEq_of_rows h (row := 1292) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_6
  simp only [← c79443, ← c79444, ← c79445] at e_1292_6
  have e_1292_7 := arithEq_of_rows h (row := 1292) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_7
  simp only [← c79446, ← c79447, k2, ← c79448] at e_1292_7
  have hr := e_1292_7
  simp only [e_1292_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11962)) := by
  have c79449 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79450 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79451 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79452 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79453 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79454 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_8 := arithEq_of_rows h (row := 1292) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_8
  simp only [← c79449, ← c79450, ← c79451] at e_1292_8
  have e_1292_9 := arithEq_of_rows h (row := 1292) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_9
  simp only [← c79452, ← c79453, k2, ← c79454] at e_1292_9
  have hr := e_1292_9
  simp only [e_1292_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11963)) := by
  have c79455 := (publicBatchWrapper86_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79456 := (publicBatchWrapper86_copies2483 a h).1
  have c79457 := (publicBatchWrapper86_copies2483 a h).2.1
  have c79458 := (publicBatchWrapper86_copies2483 a h).2.2.1
  have c79459 := (publicBatchWrapper86_copies2483 a h).2.2.2.1
  have c79460 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_10 := arithEq_of_rows h (row := 1292) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_10
  simp only [← c79455, ← c79456, ← c79457] at e_1292_10
  have e_1292_11 := arithEq_of_rows h (row := 1292) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_11
  simp only [← c79458, ← c79459, k2, ← c79460] at e_1292_11
  have hr := e_1292_11
  simp only [e_1292_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11964)) := by
  have c79461 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.1
  have c79462 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.1
  have c79463 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.1
  have c79464 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.1
  have c79465 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.1
  have c79466 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_12 := arithEq_of_rows h (row := 1292) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_12
  simp only [← c79461, ← c79462, ← c79463] at e_1292_12
  have e_1292_13 := arithEq_of_rows h (row := 1292) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_13
  simp only [← c79464, ← c79465, k2, ← c79466] at e_1292_13
  have hr := e_1292_13
  simp only [e_1292_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11965)) := by
  have c79467 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79468 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79469 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79470 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79471 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79472 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_14 := arithEq_of_rows h (row := 1292) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_14
  simp only [← c79467, ← c79468, ← c79469] at e_1292_14
  have e_1292_15 := arithEq_of_rows h (row := 1292) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_15
  simp only [← c79470, ← c79471, k2, ← c79472] at e_1292_15
  have hr := e_1292_15
  simp only [e_1292_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11966)) := by
  have c79473 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79474 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79475 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79476 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79477 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79478 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_16 := arithEq_of_rows h (row := 1292) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_16
  simp only [← c79473, ← c79474, ← c79475] at e_1292_16
  have e_1292_17 := arithEq_of_rows h (row := 1292) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_17
  simp only [← c79476, ← c79477, k2, ← c79478] at e_1292_17
  have hr := e_1292_17
  simp only [e_1292_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1292 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11967)) := by
  have c79479 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79480 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79481 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79482 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79483 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79484 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1292_18 := arithEq_of_rows h (row := 1292) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_18
  simp only [← c79479, ← c79480, ← c79481] at e_1292_18
  have e_1292_19 := arithEq_of_rows h (row := 1292) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_19
  simp only [← c79482, ← c79483, k2, ← c79484] at e_1292_19
  have hr := e_1292_19
  simp only [e_1292_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11968)) := by
  have c79485 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79486 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79487 := (publicBatchWrapper86_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79488 := (publicBatchWrapper86_copies2484 a h).1
  have c79489 := (publicBatchWrapper86_copies2484 a h).2.1
  have c79490 := (publicBatchWrapper86_copies2484 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_0 := arithEq_of_rows h (row := 1293) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_0
  simp only [← c79485, ← c79486, ← c79487] at e_1293_0
  have e_1293_1 := arithEq_of_rows h (row := 1293) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_1
  simp only [← c79488, ← c79489, k2, ← c79490] at e_1293_1
  have hr := e_1293_1
  simp only [e_1293_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11969)) := by
  have c79491 := (publicBatchWrapper86_copies2484 a h).2.2.2.1
  have c79492 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.1
  have c79493 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.1
  have c79494 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.1
  have c79495 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.1
  have c79496 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_2 := arithEq_of_rows h (row := 1293) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_2
  simp only [← c79491, ← c79492, ← c79493] at e_1293_2
  have e_1293_3 := arithEq_of_rows h (row := 1293) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_3
  simp only [← c79494, ← c79495, k2, ← c79496] at e_1293_3
  have hr := e_1293_3
  simp only [e_1293_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12010)) := by
  have c79497 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.1
  have c79498 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79499 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79500 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79501 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79502 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_4 := arithEq_of_rows h (row := 1293) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_4
  simp only [← c79497, ← c79498, ← c79499] at e_1293_4
  have e_1293_5 := arithEq_of_rows h (row := 1293) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_5
  simp only [← c79500, ← c79501, k2, ← c79502] at e_1293_5
  have hr := e_1293_5
  simp only [e_1293_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12011)) := by
  have c79503 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79504 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79505 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79506 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79507 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79508 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_6 := arithEq_of_rows h (row := 1293) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_6
  simp only [← c79503, ← c79504, ← c79505] at e_1293_6
  have e_1293_7 := arithEq_of_rows h (row := 1293) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_7
  simp only [← c79506, ← c79507, k2, ← c79508] at e_1293_7
  have hr := e_1293_7
  simp only [e_1293_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12012)) := by
  have c79509 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79510 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79511 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79512 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79513 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79514 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_8 := arithEq_of_rows h (row := 1293) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_8
  simp only [← c79509, ← c79510, ← c79511] at e_1293_8
  have e_1293_9 := arithEq_of_rows h (row := 1293) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_9
  simp only [← c79512, ← c79513, k2, ← c79514] at e_1293_9
  have hr := e_1293_9
  simp only [e_1293_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12013)) := by
  have c79515 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79516 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79517 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79518 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79519 := (publicBatchWrapper86_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79520 := (publicBatchWrapper86_copies2485 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_10 := arithEq_of_rows h (row := 1293) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_10
  simp only [← c79515, ← c79516, ← c79517] at e_1293_10
  have e_1293_11 := arithEq_of_rows h (row := 1293) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_11
  simp only [← c79518, ← c79519, k2, ← c79520] at e_1293_11
  have hr := e_1293_11
  simp only [e_1293_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12014)) := by
  have c79521 := (publicBatchWrapper86_copies2485 a h).2.1
  have c79522 := (publicBatchWrapper86_copies2485 a h).2.2.1
  have c79523 := (publicBatchWrapper86_copies2485 a h).2.2.2.1
  have c79524 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.1
  have c79525 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.1
  have c79526 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_12 := arithEq_of_rows h (row := 1293) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_12
  simp only [← c79521, ← c79522, ← c79523] at e_1293_12
  have e_1293_13 := arithEq_of_rows h (row := 1293) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_13
  simp only [← c79524, ← c79525, k2, ← c79526] at e_1293_13
  have hr := e_1293_13
  simp only [e_1293_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12015)) := by
  have c79527 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.1
  have c79528 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.1
  have c79529 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.1
  have c79530 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79531 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79532 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_14 := arithEq_of_rows h (row := 1293) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_14
  simp only [← c79527, ← c79528, ← c79529] at e_1293_14
  have e_1293_15 := arithEq_of_rows h (row := 1293) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_15
  simp only [← c79530, ← c79531, k2, ← c79532] at e_1293_15
  have hr := e_1293_15
  simp only [e_1293_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12016)) := by
  have c79533 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79534 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79535 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79536 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79537 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79538 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_16 := arithEq_of_rows h (row := 1293) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_16
  simp only [← c79533, ← c79534, ← c79535] at e_1293_16
  have e_1293_17 := arithEq_of_rows h (row := 1293) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_17
  simp only [← c79536, ← c79537, k2, ← c79538] at e_1293_17
  have hr := e_1293_17
  simp only [e_1293_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1293 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12017)) := by
  have c79539 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79540 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79541 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79542 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79543 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79544 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1293_18 := arithEq_of_rows h (row := 1293) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_18
  simp only [← c79539, ← c79540, ← c79541] at e_1293_18
  have e_1293_19 := arithEq_of_rows h (row := 1293) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_19
  simp only [← c79542, ← c79543, k2, ← c79544] at e_1293_19
  have hr := e_1293_19
  simp only [e_1293_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12018)) := by
  have c79545 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79546 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79547 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79548 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79549 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79550 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_0 := arithEq_of_rows h (row := 1294) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_0
  simp only [← c79545, ← c79546, ← c79547] at e_1294_0
  have e_1294_1 := arithEq_of_rows h (row := 1294) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_1
  simp only [← c79548, ← c79549, k2, ← c79550] at e_1294_1
  have hr := e_1294_1
  simp only [e_1294_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12019)) := by
  have c79551 := (publicBatchWrapper86_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79552 := (publicBatchWrapper86_copies2486 a h).1
  have c79553 := (publicBatchWrapper86_copies2486 a h).2.1
  have c79554 := (publicBatchWrapper86_copies2486 a h).2.2.1
  have c79555 := (publicBatchWrapper86_copies2486 a h).2.2.2.1
  have c79556 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_2 := arithEq_of_rows h (row := 1294) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_2
  simp only [← c79551, ← c79552, ← c79553] at e_1294_2
  have e_1294_3 := arithEq_of_rows h (row := 1294) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_3
  simp only [← c79554, ← c79555, k2, ← c79556] at e_1294_3
  have hr := e_1294_3
  simp only [e_1294_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12020)) := by
  have c79557 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.1
  have c79558 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.1
  have c79559 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.1
  have c79560 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.1
  have c79561 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.1
  have c79562 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_4 := arithEq_of_rows h (row := 1294) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_4
  simp only [← c79557, ← c79558, ← c79559] at e_1294_4
  have e_1294_5 := arithEq_of_rows h (row := 1294) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_5
  simp only [← c79560, ← c79561, k2, ← c79562] at e_1294_5
  have hr := e_1294_5
  simp only [e_1294_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12021)) := by
  have c79563 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79564 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79565 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79566 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79567 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79568 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_6 := arithEq_of_rows h (row := 1294) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_6
  simp only [← c79563, ← c79564, ← c79565] at e_1294_6
  have e_1294_7 := arithEq_of_rows h (row := 1294) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_7
  simp only [← c79566, ← c79567, k2, ← c79568] at e_1294_7
  have hr := e_1294_7
  simp only [e_1294_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12022)) := by
  have c79569 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79570 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79571 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79572 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79573 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79574 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_8 := arithEq_of_rows h (row := 1294) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_8
  simp only [← c79569, ← c79570, ← c79571] at e_1294_8
  have e_1294_9 := arithEq_of_rows h (row := 1294) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_9
  simp only [← c79572, ← c79573, k2, ← c79574] at e_1294_9
  have hr := e_1294_9
  simp only [e_1294_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12023)) := by
  have c79575 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79576 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79577 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79578 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79579 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79580 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_10 := arithEq_of_rows h (row := 1294) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_10
  simp only [← c79575, ← c79576, ← c79577] at e_1294_10
  have e_1294_11 := arithEq_of_rows h (row := 1294) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_11
  simp only [← c79578, ← c79579, k2, ← c79580] at e_1294_11
  have hr := e_1294_11
  simp only [e_1294_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12024)) := by
  have c79581 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79582 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79583 := (publicBatchWrapper86_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79584 := (publicBatchWrapper86_copies2487 a h).1
  have c79585 := (publicBatchWrapper86_copies2487 a h).2.1
  have c79586 := (publicBatchWrapper86_copies2487 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_12 := arithEq_of_rows h (row := 1294) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_12
  simp only [← c79581, ← c79582, ← c79583] at e_1294_12
  have e_1294_13 := arithEq_of_rows h (row := 1294) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_13
  simp only [← c79584, ← c79585, k2, ← c79586] at e_1294_13
  have hr := e_1294_13
  simp only [e_1294_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12025)) := by
  have c79587 := (publicBatchWrapper86_copies2487 a h).2.2.2.1
  have c79588 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.1
  have c79589 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.1
  have c79590 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.1
  have c79591 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.1
  have c79592 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_14 := arithEq_of_rows h (row := 1294) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_14
  simp only [← c79587, ← c79588, ← c79589] at e_1294_14
  have e_1294_15 := arithEq_of_rows h (row := 1294) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_15
  simp only [← c79590, ← c79591, k2, ← c79592] at e_1294_15
  have hr := e_1294_15
  simp only [e_1294_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12026)) := by
  have c79593 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.1
  have c79594 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79595 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79596 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79597 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79598 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_16 := arithEq_of_rows h (row := 1294) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_16
  simp only [← c79593, ← c79594, ← c79595] at e_1294_16
  have e_1294_17 := arithEq_of_rows h (row := 1294) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_17
  simp only [← c79596, ← c79597, k2, ← c79598] at e_1294_17
  have hr := e_1294_17
  simp only [e_1294_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1294 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12027)) := by
  have c79599 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79600 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79601 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79602 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79603 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79604 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1294_18 := arithEq_of_rows h (row := 1294) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_18
  simp only [← c79599, ← c79600, ← c79601] at e_1294_18
  have e_1294_19 := arithEq_of_rows h (row := 1294) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_19
  simp only [← c79602, ← c79603, k2, ← c79604] at e_1294_19
  have hr := e_1294_19
  simp only [e_1294_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12028)) := by
  have c79605 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79606 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79607 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79608 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79609 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79610 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_0 := arithEq_of_rows h (row := 1295) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_0
  simp only [← c79605, ← c79606, ← c79607] at e_1295_0
  have e_1295_1 := arithEq_of_rows h (row := 1295) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_1
  simp only [← c79608, ← c79609, k2, ← c79610] at e_1295_1
  have hr := e_1295_1
  simp only [e_1295_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12029)) := by
  have c79611 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79612 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79613 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79614 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79615 := (publicBatchWrapper86_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79616 := (publicBatchWrapper86_copies2488 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_2 := arithEq_of_rows h (row := 1295) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_2
  simp only [← c79611, ← c79612, ← c79613] at e_1295_2
  have e_1295_3 := arithEq_of_rows h (row := 1295) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_3
  simp only [← c79614, ← c79615, k2, ← c79616] at e_1295_3
  have hr := e_1295_3
  simp only [e_1295_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12030)) := by
  have c79617 := (publicBatchWrapper86_copies2488 a h).2.1
  have c79618 := (publicBatchWrapper86_copies2488 a h).2.2.1
  have c79619 := (publicBatchWrapper86_copies2488 a h).2.2.2.1
  have c79620 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.1
  have c79621 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.1
  have c79622 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_4 := arithEq_of_rows h (row := 1295) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_4
  simp only [← c79617, ← c79618, ← c79619] at e_1295_4
  have e_1295_5 := arithEq_of_rows h (row := 1295) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_5
  simp only [← c79620, ← c79621, k2, ← c79622] at e_1295_5
  have hr := e_1295_5
  simp only [e_1295_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12031)) := by
  have c79623 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.1
  have c79624 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.1
  have c79625 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.1
  have c79626 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79627 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79628 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_6 := arithEq_of_rows h (row := 1295) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_6
  simp only [← c79623, ← c79624, ← c79625] at e_1295_6
  have e_1295_7 := arithEq_of_rows h (row := 1295) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_7
  simp only [← c79626, ← c79627, k2, ← c79628] at e_1295_7
  have hr := e_1295_7
  simp only [e_1295_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12032)) := by
  have c79629 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79630 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79631 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79632 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79633 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79634 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_8 := arithEq_of_rows h (row := 1295) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_8
  simp only [← c79629, ← c79630, ← c79631] at e_1295_8
  have e_1295_9 := arithEq_of_rows h (row := 1295) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_9
  simp only [← c79632, ← c79633, k2, ← c79634] at e_1295_9
  have hr := e_1295_9
  simp only [e_1295_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12033)) := by
  have c79635 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79636 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79637 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79638 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79639 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79640 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_10 := arithEq_of_rows h (row := 1295) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_10
  simp only [← c79635, ← c79636, ← c79637] at e_1295_10
  have e_1295_11 := arithEq_of_rows h (row := 1295) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_11
  simp only [← c79638, ← c79639, k2, ← c79640] at e_1295_11
  have hr := e_1295_11
  simp only [e_1295_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12034)) := by
  have c79641 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79642 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79643 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79644 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79645 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79646 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_12 := arithEq_of_rows h (row := 1295) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_12
  simp only [← c79641, ← c79642, ← c79643] at e_1295_12
  have e_1295_13 := arithEq_of_rows h (row := 1295) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_13
  simp only [← c79644, ← c79645, k2, ← c79646] at e_1295_13
  have hr := e_1295_13
  simp only [e_1295_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12035)) := by
  have c79647 := (publicBatchWrapper86_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79648 := (publicBatchWrapper86_copies2489 a h).1
  have c79649 := (publicBatchWrapper86_copies2489 a h).2.1
  have c79650 := (publicBatchWrapper86_copies2489 a h).2.2.1
  have c79651 := (publicBatchWrapper86_copies2489 a h).2.2.2.1
  have c79652 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_14 := arithEq_of_rows h (row := 1295) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_14
  simp only [← c79647, ← c79648, ← c79649] at e_1295_14
  have e_1295_15 := arithEq_of_rows h (row := 1295) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_15
  simp only [← c79650, ← c79651, k2, ← c79652] at e_1295_15
  have hr := e_1295_15
  simp only [e_1295_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12036)) := by
  have c79653 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.1
  have c79654 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.1
  have c79655 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.1
  have c79656 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.1
  have c79657 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.1
  have c79658 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_16 := arithEq_of_rows h (row := 1295) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_16
  simp only [← c79653, ← c79654, ← c79655] at e_1295_16
  have e_1295_17 := arithEq_of_rows h (row := 1295) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_17
  simp only [← c79656, ← c79657, k2, ← c79658] at e_1295_17
  have hr := e_1295_17
  simp only [e_1295_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1295 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12037)) := by
  have c79659 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79660 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79661 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79662 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79663 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79664 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1295_18 := arithEq_of_rows h (row := 1295) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_18
  simp only [← c79659, ← c79660, ← c79661] at e_1295_18
  have e_1295_19 := arithEq_of_rows h (row := 1295) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_19
  simp only [← c79662, ← c79663, k2, ← c79664] at e_1295_19
  have hr := e_1295_19
  simp only [e_1295_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12038)) := by
  have c79665 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79666 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79667 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79668 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79669 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79670 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_0 := arithEq_of_rows h (row := 1296) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_0
  simp only [← c79665, ← c79666, ← c79667] at e_1296_0
  have e_1296_1 := arithEq_of_rows h (row := 1296) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_1
  simp only [← c79668, ← c79669, k2, ← c79670] at e_1296_1
  have hr := e_1296_1
  simp only [e_1296_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12039)) := by
  have c79671 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79672 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79673 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79674 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79675 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79676 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_2 := arithEq_of_rows h (row := 1296) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_2
  simp only [← c79671, ← c79672, ← c79673] at e_1296_2
  have e_1296_3 := arithEq_of_rows h (row := 1296) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_3
  simp only [← c79674, ← c79675, k2, ← c79676] at e_1296_3
  have hr := e_1296_3
  simp only [e_1296_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12040)) := by
  have c79677 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79678 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79679 := (publicBatchWrapper86_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79680 := (publicBatchWrapper86_copies2490 a h).1
  have c79681 := (publicBatchWrapper86_copies2490 a h).2.1
  have c79682 := (publicBatchWrapper86_copies2490 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_4 := arithEq_of_rows h (row := 1296) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_4
  simp only [← c79677, ← c79678, ← c79679] at e_1296_4
  have e_1296_5 := arithEq_of_rows h (row := 1296) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_5
  simp only [← c79680, ← c79681, k2, ← c79682] at e_1296_5
  have hr := e_1296_5
  simp only [e_1296_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12041)) := by
  have c79683 := (publicBatchWrapper86_copies2490 a h).2.2.2.1
  have c79684 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.1
  have c79685 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.1
  have c79686 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.1
  have c79687 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.1
  have c79688 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_6 := arithEq_of_rows h (row := 1296) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_6
  simp only [← c79683, ← c79684, ← c79685] at e_1296_6
  have e_1296_7 := arithEq_of_rows h (row := 1296) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_7
  simp only [← c79686, ← c79687, k2, ← c79688] at e_1296_7
  have hr := e_1296_7
  simp only [e_1296_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12042)) := by
  have c79689 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.1
  have c79690 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79691 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79692 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79693 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79694 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_8 := arithEq_of_rows h (row := 1296) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_8
  simp only [← c79689, ← c79690, ← c79691] at e_1296_8
  have e_1296_9 := arithEq_of_rows h (row := 1296) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_9
  simp only [← c79692, ← c79693, k2, ← c79694] at e_1296_9
  have hr := e_1296_9
  simp only [e_1296_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12043)) := by
  have c79695 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79696 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79697 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79698 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79699 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79700 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_10 := arithEq_of_rows h (row := 1296) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_10
  simp only [← c79695, ← c79696, ← c79697] at e_1296_10
  have e_1296_11 := arithEq_of_rows h (row := 1296) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_11
  simp only [← c79698, ← c79699, k2, ← c79700] at e_1296_11
  have hr := e_1296_11
  simp only [e_1296_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12044)) := by
  have c79701 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79702 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79703 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79704 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79705 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79706 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_12 := arithEq_of_rows h (row := 1296) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_12
  simp only [← c79701, ← c79702, ← c79703] at e_1296_12
  have e_1296_13 := arithEq_of_rows h (row := 1296) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_13
  simp only [← c79704, ← c79705, k2, ← c79706] at e_1296_13
  have hr := e_1296_13
  simp only [e_1296_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12045)) := by
  have c79707 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79708 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79709 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79710 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79711 := (publicBatchWrapper86_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79712 := (publicBatchWrapper86_copies2491 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_14 := arithEq_of_rows h (row := 1296) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_14
  simp only [← c79707, ← c79708, ← c79709] at e_1296_14
  have e_1296_15 := arithEq_of_rows h (row := 1296) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_15
  simp only [← c79710, ← c79711, k2, ← c79712] at e_1296_15
  have hr := e_1296_15
  simp only [e_1296_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12046)) := by
  have c79713 := (publicBatchWrapper86_copies2491 a h).2.1
  have c79714 := (publicBatchWrapper86_copies2491 a h).2.2.1
  have c79715 := (publicBatchWrapper86_copies2491 a h).2.2.2.1
  have c79716 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.1
  have c79717 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.1
  have c79718 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_16 := arithEq_of_rows h (row := 1296) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_16
  simp only [← c79713, ← c79714, ← c79715] at e_1296_16
  have e_1296_17 := arithEq_of_rows h (row := 1296) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_17
  simp only [← c79716, ← c79717, k2, ← c79718] at e_1296_17
  have hr := e_1296_17
  simp only [e_1296_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1296 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12047)) := by
  have c79719 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.1
  have c79720 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.1
  have c79721 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.1
  have c79722 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79723 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79724 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1296_18 := arithEq_of_rows h (row := 1296) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_18
  simp only [← c79719, ← c79720, ← c79721] at e_1296_18
  have e_1296_19 := arithEq_of_rows h (row := 1296) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_19
  simp only [← c79722, ← c79723, k2, ← c79724] at e_1296_19
  have hr := e_1296_19
  simp only [e_1296_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12048)) := by
  have c79725 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79726 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79727 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79728 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79729 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79730 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_0 := arithEq_of_rows h (row := 1297) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_0
  simp only [← c79725, ← c79726, ← c79727] at e_1297_0
  have e_1297_1 := arithEq_of_rows h (row := 1297) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_1
  simp only [← c79728, ← c79729, k2, ← c79730] at e_1297_1
  have hr := e_1297_1
  simp only [e_1297_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12049)) := by
  have c79731 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79732 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79733 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79734 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79735 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79736 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_2 := arithEq_of_rows h (row := 1297) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_2
  simp only [← c79731, ← c79732, ← c79733] at e_1297_2
  have e_1297_3 := arithEq_of_rows h (row := 1297) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_3
  simp only [← c79734, ← c79735, k2, ← c79736] at e_1297_3
  have hr := e_1297_3
  simp only [e_1297_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12050)) := by
  have c79737 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79738 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79739 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79740 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79741 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79742 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_4 := arithEq_of_rows h (row := 1297) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_4
  simp only [← c79737, ← c79738, ← c79739] at e_1297_4
  have e_1297_5 := arithEq_of_rows h (row := 1297) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_5
  simp only [← c79740, ← c79741, k2, ← c79742] at e_1297_5
  have hr := e_1297_5
  simp only [e_1297_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12051)) := by
  have c79743 := (publicBatchWrapper86_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79744 := (publicBatchWrapper86_copies2492 a h).1
  have c79745 := (publicBatchWrapper86_copies2492 a h).2.1
  have c79746 := (publicBatchWrapper86_copies2492 a h).2.2.1
  have c79747 := (publicBatchWrapper86_copies2492 a h).2.2.2.1
  have c79748 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_6 := arithEq_of_rows h (row := 1297) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_6
  simp only [← c79743, ← c79744, ← c79745] at e_1297_6
  have e_1297_7 := arithEq_of_rows h (row := 1297) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_7
  simp only [← c79746, ← c79747, k2, ← c79748] at e_1297_7
  have hr := e_1297_7
  simp only [e_1297_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12052)) := by
  have c79749 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.1
  have c79750 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.1
  have c79751 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.1
  have c79752 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.1
  have c79753 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.1
  have c79754 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_8 := arithEq_of_rows h (row := 1297) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_8
  simp only [← c79749, ← c79750, ← c79751] at e_1297_8
  have e_1297_9 := arithEq_of_rows h (row := 1297) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_9
  simp only [← c79752, ← c79753, k2, ← c79754] at e_1297_9
  have hr := e_1297_9
  simp only [e_1297_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12053)) := by
  have c79755 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79756 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79757 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79758 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79759 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79760 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_10 := arithEq_of_rows h (row := 1297) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_10
  simp only [← c79755, ← c79756, ← c79757] at e_1297_10
  have e_1297_11 := arithEq_of_rows h (row := 1297) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_11
  simp only [← c79758, ← c79759, k2, ← c79760] at e_1297_11
  have hr := e_1297_11
  simp only [e_1297_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12054)) := by
  have c79761 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79762 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79763 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79764 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79765 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79766 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_12 := arithEq_of_rows h (row := 1297) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_12
  simp only [← c79761, ← c79762, ← c79763] at e_1297_12
  have e_1297_13 := arithEq_of_rows h (row := 1297) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_13
  simp only [← c79764, ← c79765, k2, ← c79766] at e_1297_13
  have hr := e_1297_13
  simp only [e_1297_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12055)) := by
  have c79767 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79768 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79769 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79770 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79771 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79772 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_14 := arithEq_of_rows h (row := 1297) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_14
  simp only [← c79767, ← c79768, ← c79769] at e_1297_14
  have e_1297_15 := arithEq_of_rows h (row := 1297) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_15
  simp only [← c79770, ← c79771, k2, ← c79772] at e_1297_15
  have hr := e_1297_15
  simp only [e_1297_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12056)) := by
  have c79773 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79774 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79775 := (publicBatchWrapper86_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79776 := (publicBatchWrapper86_copies2493 a h).1
  have c79777 := (publicBatchWrapper86_copies2493 a h).2.1
  have c79778 := (publicBatchWrapper86_copies2493 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_16 := arithEq_of_rows h (row := 1297) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_16
  simp only [← c79773, ← c79774, ← c79775] at e_1297_16
  have e_1297_17 := arithEq_of_rows h (row := 1297) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_17
  simp only [← c79776, ← c79777, k2, ← c79778] at e_1297_17
  have hr := e_1297_17
  simp only [e_1297_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1297 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12057)) := by
  have c79779 := (publicBatchWrapper86_copies2493 a h).2.2.2.1
  have c79780 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.1
  have c79781 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.1
  have c79782 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.1
  have c79783 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.1
  have c79784 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1297_18 := arithEq_of_rows h (row := 1297) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_18
  simp only [← c79779, ← c79780, ← c79781] at e_1297_18
  have e_1297_19 := arithEq_of_rows h (row := 1297) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_19
  simp only [← c79782, ← c79783, k2, ← c79784] at e_1297_19
  have hr := e_1297_19
  simp only [e_1297_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12058)) := by
  have c79785 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.1
  have c79786 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79787 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79788 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79789 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79790 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_0 := arithEq_of_rows h (row := 1298) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_0
  simp only [← c79785, ← c79786, ← c79787] at e_1298_0
  have e_1298_1 := arithEq_of_rows h (row := 1298) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_1
  simp only [← c79788, ← c79789, k2, ← c79790] at e_1298_1
  have hr := e_1298_1
  simp only [e_1298_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12059)) := by
  have c79791 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79792 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79793 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79794 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79795 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79796 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_2 := arithEq_of_rows h (row := 1298) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_2
  simp only [← c79791, ← c79792, ← c79793] at e_1298_2
  have e_1298_3 := arithEq_of_rows h (row := 1298) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_3
  simp only [← c79794, ← c79795, k2, ← c79796] at e_1298_3
  have hr := e_1298_3
  simp only [e_1298_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12060)) := by
  have c79797 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79798 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79799 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79800 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79801 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79802 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_4 := arithEq_of_rows h (row := 1298) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_4
  simp only [← c79797, ← c79798, ← c79799] at e_1298_4
  have e_1298_5 := arithEq_of_rows h (row := 1298) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_5
  simp only [← c79800, ← c79801, k2, ← c79802] at e_1298_5
  have hr := e_1298_5
  simp only [e_1298_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12061)) := by
  have c79803 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79804 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79805 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79806 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79807 := (publicBatchWrapper86_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79808 := (publicBatchWrapper86_copies2494 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_6 := arithEq_of_rows h (row := 1298) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_6
  simp only [← c79803, ← c79804, ← c79805] at e_1298_6
  have e_1298_7 := arithEq_of_rows h (row := 1298) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_7
  simp only [← c79806, ← c79807, k2, ← c79808] at e_1298_7
  have hr := e_1298_7
  simp only [e_1298_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12062)) := by
  have c79809 := (publicBatchWrapper86_copies2494 a h).2.1
  have c79810 := (publicBatchWrapper86_copies2494 a h).2.2.1
  have c79811 := (publicBatchWrapper86_copies2494 a h).2.2.2.1
  have c79812 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.1
  have c79813 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.1
  have c79814 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_8 := arithEq_of_rows h (row := 1298) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_8
  simp only [← c79809, ← c79810, ← c79811] at e_1298_8
  have e_1298_9 := arithEq_of_rows h (row := 1298) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_9
  simp only [← c79812, ← c79813, k2, ← c79814] at e_1298_9
  have hr := e_1298_9
  simp only [e_1298_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12063)) := by
  have c79815 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.1
  have c79816 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.1
  have c79817 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.1
  have c79818 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79819 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79820 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_10 := arithEq_of_rows h (row := 1298) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_10
  simp only [← c79815, ← c79816, ← c79817] at e_1298_10
  have e_1298_11 := arithEq_of_rows h (row := 1298) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_11
  simp only [← c79818, ← c79819, k2, ← c79820] at e_1298_11
  have hr := e_1298_11
  simp only [e_1298_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12064)) := by
  have c79821 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79822 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79823 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79824 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79825 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79826 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_12 := arithEq_of_rows h (row := 1298) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_12
  simp only [← c79821, ← c79822, ← c79823] at e_1298_12
  have e_1298_13 := arithEq_of_rows h (row := 1298) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_13
  simp only [← c79824, ← c79825, k2, ← c79826] at e_1298_13
  have hr := e_1298_13
  simp only [e_1298_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12065)) := by
  have c79827 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79828 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79829 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79830 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79831 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79832 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_14 := arithEq_of_rows h (row := 1298) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_14
  simp only [← c79827, ← c79828, ← c79829] at e_1298_14
  have e_1298_15 := arithEq_of_rows h (row := 1298) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_15
  simp only [← c79830, ← c79831, k2, ← c79832] at e_1298_15
  have hr := e_1298_15
  simp only [e_1298_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12066)) := by
  have c79833 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79834 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79835 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79836 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79837 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79838 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_16 := arithEq_of_rows h (row := 1298) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_16
  simp only [← c79833, ← c79834, ← c79835] at e_1298_16
  have e_1298_17 := arithEq_of_rows h (row := 1298) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_17
  simp only [← c79836, ← c79837, k2, ← c79838] at e_1298_17
  have hr := e_1298_17
  simp only [e_1298_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1298 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12067)) := by
  have c79839 := (publicBatchWrapper86_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79840 := (publicBatchWrapper86_copies2495 a h).1
  have c79841 := (publicBatchWrapper86_copies2495 a h).2.1
  have c79842 := (publicBatchWrapper86_copies2495 a h).2.2.1
  have c79843 := (publicBatchWrapper86_copies2495 a h).2.2.2.1
  have c79844 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1298_18 := arithEq_of_rows h (row := 1298) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_18
  simp only [← c79839, ← c79840, ← c79841] at e_1298_18
  have e_1298_19 := arithEq_of_rows h (row := 1298) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_19
  simp only [← c79842, ← c79843, k2, ← c79844] at e_1298_19
  have hr := e_1298_19
  simp only [e_1298_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12068)) := by
  have c79845 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.1
  have c79846 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.1
  have c79847 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.1
  have c79848 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.1
  have c79849 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.1
  have c79850 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_0 := arithEq_of_rows h (row := 1299) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_0
  simp only [← c79845, ← c79846, ← c79847] at e_1299_0
  have e_1299_1 := arithEq_of_rows h (row := 1299) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_1
  simp only [← c79848, ← c79849, k2, ← c79850] at e_1299_1
  have hr := e_1299_1
  simp only [e_1299_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12069)) := by
  have c79851 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79852 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79853 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79854 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79855 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79856 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_2 := arithEq_of_rows h (row := 1299) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_2
  simp only [← c79851, ← c79852, ← c79853] at e_1299_2
  have e_1299_3 := arithEq_of_rows h (row := 1299) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_3
  simp only [← c79854, ← c79855, k2, ← c79856] at e_1299_3
  have hr := e_1299_3
  simp only [e_1299_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12070)) := by
  have c79857 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79858 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79859 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79860 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79861 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79862 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_4 := arithEq_of_rows h (row := 1299) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_4
  simp only [← c79857, ← c79858, ← c79859] at e_1299_4
  have e_1299_5 := arithEq_of_rows h (row := 1299) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_5
  simp only [← c79860, ← c79861, k2, ← c79862] at e_1299_5
  have hr := e_1299_5
  simp only [e_1299_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12071)) := by
  have c79863 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79864 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79865 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79866 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79867 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79868 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_6 := arithEq_of_rows h (row := 1299) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_6
  simp only [← c79863, ← c79864, ← c79865] at e_1299_6
  have e_1299_7 := arithEq_of_rows h (row := 1299) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_7
  simp only [← c79866, ← c79867, k2, ← c79868] at e_1299_7
  have hr := e_1299_7
  simp only [e_1299_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12072)) := by
  have c79869 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79870 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79871 := (publicBatchWrapper86_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79872 := (publicBatchWrapper86_copies2496 a h).1
  have c79873 := (publicBatchWrapper86_copies2496 a h).2.1
  have c79874 := (publicBatchWrapper86_copies2496 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_8 := arithEq_of_rows h (row := 1299) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_8
  simp only [← c79869, ← c79870, ← c79871] at e_1299_8
  have e_1299_9 := arithEq_of_rows h (row := 1299) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_9
  simp only [← c79872, ← c79873, k2, ← c79874] at e_1299_9
  have hr := e_1299_9
  simp only [e_1299_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12073)) := by
  have c79875 := (publicBatchWrapper86_copies2496 a h).2.2.2.1
  have c79876 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.1
  have c79877 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.1
  have c79878 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.1
  have c79879 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.1
  have c79880 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_10 := arithEq_of_rows h (row := 1299) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_10
  simp only [← c79875, ← c79876, ← c79877] at e_1299_10
  have e_1299_11 := arithEq_of_rows h (row := 1299) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_11
  simp only [← c79878, ← c79879, k2, ← c79880] at e_1299_11
  have hr := e_1299_11
  simp only [e_1299_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12074)) := by
  have c79881 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.1
  have c79882 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79883 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79884 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79885 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79886 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_12 := arithEq_of_rows h (row := 1299) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_12
  simp only [← c79881, ← c79882, ← c79883] at e_1299_12
  have e_1299_13 := arithEq_of_rows h (row := 1299) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_13
  simp only [← c79884, ← c79885, k2, ← c79886] at e_1299_13
  have hr := e_1299_13
  simp only [e_1299_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12075)) := by
  have c79887 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79888 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79889 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79890 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79891 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79892 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_14 := arithEq_of_rows h (row := 1299) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_14
  simp only [← c79887, ← c79888, ← c79889] at e_1299_14
  have e_1299_15 := arithEq_of_rows h (row := 1299) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_15
  simp only [← c79890, ← c79891, k2, ← c79892] at e_1299_15
  have hr := e_1299_15
  simp only [e_1299_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12076)) := by
  have c79893 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79894 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79895 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79896 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79897 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79898 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_16 := arithEq_of_rows h (row := 1299) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_16
  simp only [← c79893, ← c79894, ← c79895] at e_1299_16
  have e_1299_17 := arithEq_of_rows h (row := 1299) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_17
  simp only [← c79896, ← c79897, k2, ← c79898] at e_1299_17
  have hr := e_1299_17
  simp only [e_1299_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1299 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12077)) := by
  have c79899 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79900 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79901 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79902 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79903 := (publicBatchWrapper86_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79904 := (publicBatchWrapper86_copies2497 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1299_18 := arithEq_of_rows h (row := 1299) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_18
  simp only [← c79899, ← c79900, ← c79901] at e_1299_18
  have e_1299_19 := arithEq_of_rows h (row := 1299) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_19
  simp only [← c79902, ← c79903, k2, ← c79904] at e_1299_19
  have hr := e_1299_19
  simp only [e_1299_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12078)) := by
  have c79905 := (publicBatchWrapper86_copies2497 a h).2.1
  have c79906 := (publicBatchWrapper86_copies2497 a h).2.2.1
  have c79907 := (publicBatchWrapper86_copies2497 a h).2.2.2.1
  have c79908 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.1
  have c79909 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.1
  have c79910 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_0 := arithEq_of_rows h (row := 1300) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_0
  simp only [← c79905, ← c79906, ← c79907] at e_1300_0
  have e_1300_1 := arithEq_of_rows h (row := 1300) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_1
  simp only [← c79908, ← c79909, k2, ← c79910] at e_1300_1
  have hr := e_1300_1
  simp only [e_1300_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12079)) := by
  have c79911 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.1
  have c79912 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.1
  have c79913 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.1
  have c79914 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79915 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79916 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_2 := arithEq_of_rows h (row := 1300) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_2
  simp only [← c79911, ← c79912, ← c79913] at e_1300_2
  have e_1300_3 := arithEq_of_rows h (row := 1300) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_3
  simp only [← c79914, ← c79915, k2, ← c79916] at e_1300_3
  have hr := e_1300_3
  simp only [e_1300_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12080)) := by
  have c79917 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79918 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79919 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79920 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79921 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79922 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_4 := arithEq_of_rows h (row := 1300) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_4
  simp only [← c79917, ← c79918, ← c79919] at e_1300_4
  have e_1300_5 := arithEq_of_rows h (row := 1300) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_5
  simp only [← c79920, ← c79921, k2, ← c79922] at e_1300_5
  have hr := e_1300_5
  simp only [e_1300_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12081)) := by
  have c79923 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79924 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79925 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79926 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79927 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79928 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_6 := arithEq_of_rows h (row := 1300) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_6
  simp only [← c79923, ← c79924, ← c79925] at e_1300_6
  have e_1300_7 := arithEq_of_rows h (row := 1300) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_7
  simp only [← c79926, ← c79927, k2, ← c79928] at e_1300_7
  have hr := e_1300_7
  simp only [e_1300_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12082)) := by
  have c79929 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79930 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79931 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79932 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79933 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79934 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_8 := arithEq_of_rows h (row := 1300) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_8
  simp only [← c79929, ← c79930, ← c79931] at e_1300_8
  have e_1300_9 := arithEq_of_rows h (row := 1300) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_9
  simp only [← c79932, ← c79933, k2, ← c79934] at e_1300_9
  have hr := e_1300_9
  simp only [e_1300_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12083)) := by
  have c79935 := (publicBatchWrapper86_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79936 := (publicBatchWrapper86_copies2498 a h).1
  have c79937 := (publicBatchWrapper86_copies2498 a h).2.1
  have c79938 := (publicBatchWrapper86_copies2498 a h).2.2.1
  have c79939 := (publicBatchWrapper86_copies2498 a h).2.2.2.1
  have c79940 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_10 := arithEq_of_rows h (row := 1300) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_10
  simp only [← c79935, ← c79936, ← c79937] at e_1300_10
  have e_1300_11 := arithEq_of_rows h (row := 1300) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_11
  simp only [← c79938, ← c79939, k2, ← c79940] at e_1300_11
  have hr := e_1300_11
  simp only [e_1300_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12084)) := by
  have c79941 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.1
  have c79942 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.1
  have c79943 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.1
  have c79944 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.1
  have c79945 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.1
  have c79946 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_12 := arithEq_of_rows h (row := 1300) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_12
  simp only [← c79941, ← c79942, ← c79943] at e_1300_12
  have e_1300_13 := arithEq_of_rows h (row := 1300) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_13
  simp only [← c79944, ← c79945, k2, ← c79946] at e_1300_13
  have hr := e_1300_13
  simp only [e_1300_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12085)) := by
  have c79947 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79948 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79949 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79950 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79951 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79952 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_14 := arithEq_of_rows h (row := 1300) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_14
  simp only [← c79947, ← c79948, ← c79949] at e_1300_14
  have e_1300_15 := arithEq_of_rows h (row := 1300) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_15
  simp only [← c79950, ← c79951, k2, ← c79952] at e_1300_15
  have hr := e_1300_15
  simp only [e_1300_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12086)) := by
  have c79953 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79954 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79955 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79956 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79957 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79958 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_16 := arithEq_of_rows h (row := 1300) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_16
  simp only [← c79953, ← c79954, ← c79955] at e_1300_16
  have e_1300_17 := arithEq_of_rows h (row := 1300) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_17
  simp only [← c79956, ← c79957, k2, ← c79958] at e_1300_17
  have hr := e_1300_17
  simp only [e_1300_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1300 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12087)) := by
  have c79959 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79960 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79961 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79962 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79963 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79964 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1300_18 := arithEq_of_rows h (row := 1300) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_18
  simp only [← c79959, ← c79960, ← c79961] at e_1300_18
  have e_1300_19 := arithEq_of_rows h (row := 1300) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_19
  simp only [← c79962, ← c79963, k2, ← c79964] at e_1300_19
  have hr := e_1300_19
  simp only [e_1300_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12088)) := by
  have c79965 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79966 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79967 := (publicBatchWrapper86_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79968 := (publicBatchWrapper86_copies2499 a h).1
  have c79969 := (publicBatchWrapper86_copies2499 a h).2.1
  have c79970 := (publicBatchWrapper86_copies2499 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_0 := arithEq_of_rows h (row := 1301) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_0
  simp only [← c79965, ← c79966, ← c79967] at e_1301_0
  have e_1301_1 := arithEq_of_rows h (row := 1301) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_1
  simp only [← c79968, ← c79969, k2, ← c79970] at e_1301_1
  have hr := e_1301_1
  simp only [e_1301_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12089)) := by
  have c79971 := (publicBatchWrapper86_copies2499 a h).2.2.2.1
  have c79972 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.1
  have c79973 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.1
  have c79974 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.1
  have c79975 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.1
  have c79976 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_2 := arithEq_of_rows h (row := 1301) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_2
  simp only [← c79971, ← c79972, ← c79973] at e_1301_2
  have e_1301_3 := arithEq_of_rows h (row := 1301) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_3
  simp only [← c79974, ← c79975, k2, ← c79976] at e_1301_3
  have hr := e_1301_3
  simp only [e_1301_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12090)) := by
  have c79977 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.1
  have c79978 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79979 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79980 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79981 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79982 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_4 := arithEq_of_rows h (row := 1301) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_4
  simp only [← c79977, ← c79978, ← c79979] at e_1301_4
  have e_1301_5 := arithEq_of_rows h (row := 1301) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_5
  simp only [← c79980, ← c79981, k2, ← c79982] at e_1301_5
  have hr := e_1301_5
  simp only [e_1301_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12091)) := by
  have c79983 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79984 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79985 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79986 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79987 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79988 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_6 := arithEq_of_rows h (row := 1301) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_6
  simp only [← c79983, ← c79984, ← c79985] at e_1301_6
  have e_1301_7 := arithEq_of_rows h (row := 1301) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_7
  simp only [← c79986, ← c79987, k2, ← c79988] at e_1301_7
  have hr := e_1301_7
  simp only [e_1301_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12092)) := by
  have c79989 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79990 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79991 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79992 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79993 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79994 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_8 := arithEq_of_rows h (row := 1301) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_8
  simp only [← c79989, ← c79990, ← c79991] at e_1301_8
  have e_1301_9 := arithEq_of_rows h (row := 1301) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_9
  simp only [← c79992, ← c79993, k2, ← c79994] at e_1301_9
  have hr := e_1301_9
  simp only [e_1301_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12093)) := by
  have c79995 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79996 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79997 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79998 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79999 := (publicBatchWrapper86_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80000 := (publicBatchWrapper86_copies2500 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_10 := arithEq_of_rows h (row := 1301) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_10
  simp only [← c79995, ← c79996, ← c79997] at e_1301_10
  have e_1301_11 := arithEq_of_rows h (row := 1301) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_11
  simp only [← c79998, ← c79999, k2, ← c80000] at e_1301_11
  have hr := e_1301_11
  simp only [e_1301_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12094)) := by
  have c80001 := (publicBatchWrapper86_copies2500 a h).2.1
  have c80002 := (publicBatchWrapper86_copies2500 a h).2.2.1
  have c80003 := (publicBatchWrapper86_copies2500 a h).2.2.2.1
  have c80004 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.1
  have c80005 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.1
  have c80006 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_12 := arithEq_of_rows h (row := 1301) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_12
  simp only [← c80001, ← c80002, ← c80003] at e_1301_12
  have e_1301_13 := arithEq_of_rows h (row := 1301) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_13
  simp only [← c80004, ← c80005, k2, ← c80006] at e_1301_13
  have hr := e_1301_13
  simp only [e_1301_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12095)) := by
  have c80007 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.1
  have c80008 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.1
  have c80009 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.1
  have c80010 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80011 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80012 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_14 := arithEq_of_rows h (row := 1301) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_14
  simp only [← c80007, ← c80008, ← c80009] at e_1301_14
  have e_1301_15 := arithEq_of_rows h (row := 1301) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_15
  simp only [← c80010, ← c80011, k2, ← c80012] at e_1301_15
  have hr := e_1301_15
  simp only [e_1301_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12096)) := by
  have c80013 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80014 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80015 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80016 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80017 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80018 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_16 := arithEq_of_rows h (row := 1301) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_16
  simp only [← c80013, ← c80014, ← c80015] at e_1301_16
  have e_1301_17 := arithEq_of_rows h (row := 1301) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_17
  simp only [← c80016, ← c80017, k2, ← c80018] at e_1301_17
  have hr := e_1301_17
  simp only [e_1301_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1301 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12097)) := by
  have c80019 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80020 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80021 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80022 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80023 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80024 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1301_18 := arithEq_of_rows h (row := 1301) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_18
  simp only [← c80019, ← c80020, ← c80021] at e_1301_18
  have e_1301_19 := arithEq_of_rows h (row := 1301) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_19
  simp only [← c80022, ← c80023, k2, ← c80024] at e_1301_19
  have hr := e_1301_19
  simp only [e_1301_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12098)) := by
  have c80025 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80026 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80027 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80028 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80029 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80030 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_0 := arithEq_of_rows h (row := 1302) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_0
  simp only [← c80025, ← c80026, ← c80027] at e_1302_0
  have e_1302_1 := arithEq_of_rows h (row := 1302) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_1
  simp only [← c80028, ← c80029, k2, ← c80030] at e_1302_1
  have hr := e_1302_1
  simp only [e_1302_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12099)) := by
  have c80031 := (publicBatchWrapper86_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80032 := (publicBatchWrapper86_copies2501 a h).1
  have c80033 := (publicBatchWrapper86_copies2501 a h).2.1
  have c80034 := (publicBatchWrapper86_copies2501 a h).2.2.1
  have c80035 := (publicBatchWrapper86_copies2501 a h).2.2.2.1
  have c80036 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_2 := arithEq_of_rows h (row := 1302) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_2
  simp only [← c80031, ← c80032, ← c80033] at e_1302_2
  have e_1302_3 := arithEq_of_rows h (row := 1302) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_3
  simp only [← c80034, ← c80035, k2, ← c80036] at e_1302_3
  have hr := e_1302_3
  simp only [e_1302_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12100)) := by
  have c80037 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.1
  have c80038 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.1
  have c80039 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.1
  have c80040 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.1
  have c80041 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.1
  have c80042 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_4 := arithEq_of_rows h (row := 1302) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_4
  simp only [← c80037, ← c80038, ← c80039] at e_1302_4
  have e_1302_5 := arithEq_of_rows h (row := 1302) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_5
  simp only [← c80040, ← c80041, k2, ← c80042] at e_1302_5
  have hr := e_1302_5
  simp only [e_1302_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12101)) := by
  have c80043 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80044 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80045 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80046 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80047 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80048 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_6 := arithEq_of_rows h (row := 1302) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_6
  simp only [← c80043, ← c80044, ← c80045] at e_1302_6
  have e_1302_7 := arithEq_of_rows h (row := 1302) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_7
  simp only [← c80046, ← c80047, k2, ← c80048] at e_1302_7
  have hr := e_1302_7
  simp only [e_1302_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12102)) := by
  have c80049 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80050 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80051 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80052 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80053 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80054 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_8 := arithEq_of_rows h (row := 1302) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_8
  simp only [← c80049, ← c80050, ← c80051] at e_1302_8
  have e_1302_9 := arithEq_of_rows h (row := 1302) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_9
  simp only [← c80052, ← c80053, k2, ← c80054] at e_1302_9
  have hr := e_1302_9
  simp only [e_1302_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12103)) := by
  have c80055 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80056 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80057 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80058 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80059 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80060 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_10 := arithEq_of_rows h (row := 1302) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_10
  simp only [← c80055, ← c80056, ← c80057] at e_1302_10
  have e_1302_11 := arithEq_of_rows h (row := 1302) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_11
  simp only [← c80058, ← c80059, k2, ← c80060] at e_1302_11
  have hr := e_1302_11
  simp only [e_1302_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12104)) := by
  have c80061 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80062 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80063 := (publicBatchWrapper86_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80064 := (publicBatchWrapper86_copies2502 a h).1
  have c80065 := (publicBatchWrapper86_copies2502 a h).2.1
  have c80066 := (publicBatchWrapper86_copies2502 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_12 := arithEq_of_rows h (row := 1302) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_12
  simp only [← c80061, ← c80062, ← c80063] at e_1302_12
  have e_1302_13 := arithEq_of_rows h (row := 1302) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_13
  simp only [← c80064, ← c80065, k2, ← c80066] at e_1302_13
  have hr := e_1302_13
  simp only [e_1302_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12105)) := by
  have c80067 := (publicBatchWrapper86_copies2502 a h).2.2.2.1
  have c80068 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.1
  have c80069 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.1
  have c80070 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.1
  have c80071 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.1
  have c80072 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_14 := arithEq_of_rows h (row := 1302) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_14
  simp only [← c80067, ← c80068, ← c80069] at e_1302_14
  have e_1302_15 := arithEq_of_rows h (row := 1302) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_15
  simp only [← c80070, ← c80071, k2, ← c80072] at e_1302_15
  have hr := e_1302_15
  simp only [e_1302_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12106)) := by
  have c80073 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.1
  have c80074 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80075 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80076 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80077 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80078 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_16 := arithEq_of_rows h (row := 1302) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_16
  simp only [← c80073, ← c80074, ← c80075] at e_1302_16
  have e_1302_17 := arithEq_of_rows h (row := 1302) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_17
  simp only [← c80076, ← c80077, k2, ← c80078] at e_1302_17
  have hr := e_1302_17
  simp only [e_1302_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1302 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12107)) := by
  have c80079 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80080 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80081 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80082 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80083 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80084 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1302_18 := arithEq_of_rows h (row := 1302) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_18
  simp only [← c80079, ← c80080, ← c80081] at e_1302_18
  have e_1302_19 := arithEq_of_rows h (row := 1302) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_19
  simp only [← c80082, ← c80083, k2, ← c80084] at e_1302_19
  have hr := e_1302_19
  simp only [e_1302_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12108)) := by
  have c80085 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80086 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80087 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80088 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80089 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80090 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_0 := arithEq_of_rows h (row := 1303) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_0
  simp only [← c80085, ← c80086, ← c80087] at e_1303_0
  have e_1303_1 := arithEq_of_rows h (row := 1303) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_1
  simp only [← c80088, ← c80089, k2, ← c80090] at e_1303_1
  have hr := e_1303_1
  simp only [e_1303_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12109)) := by
  have c80091 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80092 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80093 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80094 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80095 := (publicBatchWrapper86_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80096 := (publicBatchWrapper86_copies2503 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_2 := arithEq_of_rows h (row := 1303) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_2
  simp only [← c80091, ← c80092, ← c80093] at e_1303_2
  have e_1303_3 := arithEq_of_rows h (row := 1303) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_3
  simp only [← c80094, ← c80095, k2, ← c80096] at e_1303_3
  have hr := e_1303_3
  simp only [e_1303_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12110)) := by
  have c80097 := (publicBatchWrapper86_copies2503 a h).2.1
  have c80098 := (publicBatchWrapper86_copies2503 a h).2.2.1
  have c80099 := (publicBatchWrapper86_copies2503 a h).2.2.2.1
  have c80100 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.1
  have c80101 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.1
  have c80102 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_4 := arithEq_of_rows h (row := 1303) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_4
  simp only [← c80097, ← c80098, ← c80099] at e_1303_4
  have e_1303_5 := arithEq_of_rows h (row := 1303) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_5
  simp only [← c80100, ← c80101, k2, ← c80102] at e_1303_5
  have hr := e_1303_5
  simp only [e_1303_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12111)) := by
  have c80103 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.1
  have c80104 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.1
  have c80105 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.1
  have c80106 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80107 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80108 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_6 := arithEq_of_rows h (row := 1303) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_6
  simp only [← c80103, ← c80104, ← c80105] at e_1303_6
  have e_1303_7 := arithEq_of_rows h (row := 1303) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_7
  simp only [← c80106, ← c80107, k2, ← c80108] at e_1303_7
  have hr := e_1303_7
  simp only [e_1303_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12112)) := by
  have c80109 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80110 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80111 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80112 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80113 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80114 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_8 := arithEq_of_rows h (row := 1303) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_8
  simp only [← c80109, ← c80110, ← c80111] at e_1303_8
  have e_1303_9 := arithEq_of_rows h (row := 1303) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_9
  simp only [← c80112, ← c80113, k2, ← c80114] at e_1303_9
  have hr := e_1303_9
  simp only [e_1303_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12113)) := by
  have c80115 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80116 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80117 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80118 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80119 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80120 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_10 := arithEq_of_rows h (row := 1303) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_10
  simp only [← c80115, ← c80116, ← c80117] at e_1303_10
  have e_1303_11 := arithEq_of_rows h (row := 1303) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_11
  simp only [← c80118, ← c80119, k2, ← c80120] at e_1303_11
  have hr := e_1303_11
  simp only [e_1303_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12114)) := by
  have c80121 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80122 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80123 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80124 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80125 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80126 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_12 := arithEq_of_rows h (row := 1303) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_12
  simp only [← c80121, ← c80122, ← c80123] at e_1303_12
  have e_1303_13 := arithEq_of_rows h (row := 1303) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_13
  simp only [← c80124, ← c80125, k2, ← c80126] at e_1303_13
  have hr := e_1303_13
  simp only [e_1303_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12115)) := by
  have c80127 := (publicBatchWrapper86_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80128 := (publicBatchWrapper86_copies2504 a h).1
  have c80129 := (publicBatchWrapper86_copies2504 a h).2.1
  have c80130 := (publicBatchWrapper86_copies2504 a h).2.2.1
  have c80131 := (publicBatchWrapper86_copies2504 a h).2.2.2.1
  have c80132 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_14 := arithEq_of_rows h (row := 1303) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_14
  simp only [← c80127, ← c80128, ← c80129] at e_1303_14
  have e_1303_15 := arithEq_of_rows h (row := 1303) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_15
  simp only [← c80130, ← c80131, k2, ← c80132] at e_1303_15
  have hr := e_1303_15
  simp only [e_1303_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12116)) := by
  have c80133 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.1
  have c80134 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.1
  have c80135 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.1
  have c80136 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.1
  have c80137 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.1
  have c80138 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_16 := arithEq_of_rows h (row := 1303) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_16
  simp only [← c80133, ← c80134, ← c80135] at e_1303_16
  have e_1303_17 := arithEq_of_rows h (row := 1303) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_17
  simp only [← c80136, ← c80137, k2, ← c80138] at e_1303_17
  have hr := e_1303_17
  simp only [e_1303_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1303 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12117)) := by
  have c80139 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80140 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80141 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80142 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80143 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80144 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1303_18 := arithEq_of_rows h (row := 1303) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_18
  simp only [← c80139, ← c80140, ← c80141] at e_1303_18
  have e_1303_19 := arithEq_of_rows h (row := 1303) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_19
  simp only [← c80142, ← c80143, k2, ← c80144] at e_1303_19
  have hr := e_1303_19
  simp only [e_1303_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12118)) := by
  have c80145 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80146 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80147 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80148 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80149 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80150 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_0 := arithEq_of_rows h (row := 1304) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_0
  simp only [← c80145, ← c80146, ← c80147] at e_1304_0
  have e_1304_1 := arithEq_of_rows h (row := 1304) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_1
  simp only [← c80148, ← c80149, k2, ← c80150] at e_1304_1
  have hr := e_1304_1
  simp only [e_1304_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12119)) := by
  have c80151 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80152 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80153 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80154 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80155 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80156 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_2 := arithEq_of_rows h (row := 1304) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_2
  simp only [← c80151, ← c80152, ← c80153] at e_1304_2
  have e_1304_3 := arithEq_of_rows h (row := 1304) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_3
  simp only [← c80154, ← c80155, k2, ← c80156] at e_1304_3
  have hr := e_1304_3
  simp only [e_1304_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12120)) := by
  have c80157 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80158 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80159 := (publicBatchWrapper86_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80160 := (publicBatchWrapper86_copies2505 a h).1
  have c80161 := (publicBatchWrapper86_copies2505 a h).2.1
  have c80162 := (publicBatchWrapper86_copies2505 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_4 := arithEq_of_rows h (row := 1304) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_4
  simp only [← c80157, ← c80158, ← c80159] at e_1304_4
  have e_1304_5 := arithEq_of_rows h (row := 1304) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_5
  simp only [← c80160, ← c80161, k2, ← c80162] at e_1304_5
  have hr := e_1304_5
  simp only [e_1304_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12121)) := by
  have c80163 := (publicBatchWrapper86_copies2505 a h).2.2.2.1
  have c80164 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.1
  have c80165 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.1
  have c80166 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.1
  have c80167 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.1
  have c80168 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_6 := arithEq_of_rows h (row := 1304) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_6
  simp only [← c80163, ← c80164, ← c80165] at e_1304_6
  have e_1304_7 := arithEq_of_rows h (row := 1304) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_7
  simp only [← c80166, ← c80167, k2, ← c80168] at e_1304_7
  have hr := e_1304_7
  simp only [e_1304_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12122)) := by
  have c80169 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.1
  have c80170 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80171 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80172 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80173 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80174 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_8 := arithEq_of_rows h (row := 1304) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_8
  simp only [← c80169, ← c80170, ← c80171] at e_1304_8
  have e_1304_9 := arithEq_of_rows h (row := 1304) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_9
  simp only [← c80172, ← c80173, k2, ← c80174] at e_1304_9
  have hr := e_1304_9
  simp only [e_1304_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12123)) := by
  have c80175 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80176 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80177 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80178 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80179 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80180 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_10 := arithEq_of_rows h (row := 1304) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_10
  simp only [← c80175, ← c80176, ← c80177] at e_1304_10
  have e_1304_11 := arithEq_of_rows h (row := 1304) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_11
  simp only [← c80178, ← c80179, k2, ← c80180] at e_1304_11
  have hr := e_1304_11
  simp only [e_1304_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12124)) := by
  have c80181 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80182 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80183 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80184 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80185 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80186 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_12 := arithEq_of_rows h (row := 1304) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_12
  simp only [← c80181, ← c80182, ← c80183] at e_1304_12
  have e_1304_13 := arithEq_of_rows h (row := 1304) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_13
  simp only [← c80184, ← c80185, k2, ← c80186] at e_1304_13
  have hr := e_1304_13
  simp only [e_1304_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12125)) := by
  have c80187 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80188 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80189 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80190 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80191 := (publicBatchWrapper86_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80192 := (publicBatchWrapper86_copies2506 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_14 := arithEq_of_rows h (row := 1304) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_14
  simp only [← c80187, ← c80188, ← c80189] at e_1304_14
  have e_1304_15 := arithEq_of_rows h (row := 1304) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_15
  simp only [← c80190, ← c80191, k2, ← c80192] at e_1304_15
  have hr := e_1304_15
  simp only [e_1304_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12126)) := by
  have c80193 := (publicBatchWrapper86_copies2506 a h).2.1
  have c80194 := (publicBatchWrapper86_copies2506 a h).2.2.1
  have c80195 := (publicBatchWrapper86_copies2506 a h).2.2.2.1
  have c80196 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.1
  have c80197 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.1
  have c80198 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_16 := arithEq_of_rows h (row := 1304) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_16
  simp only [← c80193, ← c80194, ← c80195] at e_1304_16
  have e_1304_17 := arithEq_of_rows h (row := 1304) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_17
  simp only [← c80196, ← c80197, k2, ← c80198] at e_1304_17
  have hr := e_1304_17
  simp only [e_1304_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1304 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12127)) := by
  have c80199 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.1
  have c80200 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.1
  have c80201 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.1
  have c80202 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80203 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80204 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1304_18 := arithEq_of_rows h (row := 1304) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_18
  simp only [← c80199, ← c80200, ← c80201] at e_1304_18
  have e_1304_19 := arithEq_of_rows h (row := 1304) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_19
  simp only [← c80202, ← c80203, k2, ← c80204] at e_1304_19
  have hr := e_1304_19
  simp only [e_1304_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12128)) := by
  have c80205 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80206 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80207 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80208 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80209 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80210 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_0 := arithEq_of_rows h (row := 1305) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_0
  simp only [← c80205, ← c80206, ← c80207] at e_1305_0
  have e_1305_1 := arithEq_of_rows h (row := 1305) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_1
  simp only [← c80208, ← c80209, k2, ← c80210] at e_1305_1
  have hr := e_1305_1
  simp only [e_1305_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12129)) := by
  have c80211 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80212 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80213 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80214 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80215 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80216 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_2 := arithEq_of_rows h (row := 1305) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_2
  simp only [← c80211, ← c80212, ← c80213] at e_1305_2
  have e_1305_3 := arithEq_of_rows h (row := 1305) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_3
  simp only [← c80214, ← c80215, k2, ← c80216] at e_1305_3
  have hr := e_1305_3
  simp only [e_1305_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12130)) := by
  have c80217 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80218 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80219 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80220 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80221 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80222 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_4 := arithEq_of_rows h (row := 1305) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_4
  simp only [← c80217, ← c80218, ← c80219] at e_1305_4
  have e_1305_5 := arithEq_of_rows h (row := 1305) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_5
  simp only [← c80220, ← c80221, k2, ← c80222] at e_1305_5
  have hr := e_1305_5
  simp only [e_1305_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12131)) := by
  have c80223 := (publicBatchWrapper86_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80224 := (publicBatchWrapper86_copies2507 a h).1
  have c80225 := (publicBatchWrapper86_copies2507 a h).2.1
  have c80226 := (publicBatchWrapper86_copies2507 a h).2.2.1
  have c80227 := (publicBatchWrapper86_copies2507 a h).2.2.2.1
  have c80228 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_6 := arithEq_of_rows h (row := 1305) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_6
  simp only [← c80223, ← c80224, ← c80225] at e_1305_6
  have e_1305_7 := arithEq_of_rows h (row := 1305) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_7
  simp only [← c80226, ← c80227, k2, ← c80228] at e_1305_7
  have hr := e_1305_7
  simp only [e_1305_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12132)) := by
  have c80229 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.1
  have c80230 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.1
  have c80231 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.1
  have c80232 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.1
  have c80233 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.1
  have c80234 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_8 := arithEq_of_rows h (row := 1305) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_8
  simp only [← c80229, ← c80230, ← c80231] at e_1305_8
  have e_1305_9 := arithEq_of_rows h (row := 1305) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_9
  simp only [← c80232, ← c80233, k2, ← c80234] at e_1305_9
  have hr := e_1305_9
  simp only [e_1305_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12133)) := by
  have c80235 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80236 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80237 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80238 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80239 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80240 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_10 := arithEq_of_rows h (row := 1305) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_10
  simp only [← c80235, ← c80236, ← c80237] at e_1305_10
  have e_1305_11 := arithEq_of_rows h (row := 1305) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_11
  simp only [← c80238, ← c80239, k2, ← c80240] at e_1305_11
  have hr := e_1305_11
  simp only [e_1305_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12134)) := by
  have c80241 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80242 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80243 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80244 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80245 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80246 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_12 := arithEq_of_rows h (row := 1305) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_12
  simp only [← c80241, ← c80242, ← c80243] at e_1305_12
  have e_1305_13 := arithEq_of_rows h (row := 1305) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_13
  simp only [← c80244, ← c80245, k2, ← c80246] at e_1305_13
  have hr := e_1305_13
  simp only [e_1305_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12135)) := by
  have c80247 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80248 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80249 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80250 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80251 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80252 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_14 := arithEq_of_rows h (row := 1305) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_14
  simp only [← c80247, ← c80248, ← c80249] at e_1305_14
  have e_1305_15 := arithEq_of_rows h (row := 1305) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_15
  simp only [← c80250, ← c80251, k2, ← c80252] at e_1305_15
  have hr := e_1305_15
  simp only [e_1305_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12136)) := by
  have c80253 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80254 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80255 := (publicBatchWrapper86_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80256 := (publicBatchWrapper86_copies2508 a h).1
  have c80257 := (publicBatchWrapper86_copies2508 a h).2.1
  have c80258 := (publicBatchWrapper86_copies2508 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_16 := arithEq_of_rows h (row := 1305) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_16
  simp only [← c80253, ← c80254, ← c80255] at e_1305_16
  have e_1305_17 := arithEq_of_rows h (row := 1305) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_17
  simp only [← c80256, ← c80257, k2, ← c80258] at e_1305_17
  have hr := e_1305_17
  simp only [e_1305_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1305 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12137)) := by
  have c80259 := (publicBatchWrapper86_copies2508 a h).2.2.2.1
  have c80260 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.1
  have c80261 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.1
  have c80262 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.1
  have c80263 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.1
  have c80264 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1305_18 := arithEq_of_rows h (row := 1305) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_18
  simp only [← c80259, ← c80260, ← c80261] at e_1305_18
  have e_1305_19 := arithEq_of_rows h (row := 1305) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_19
  simp only [← c80262, ← c80263, k2, ← c80264] at e_1305_19
  have hr := e_1305_19
  simp only [e_1305_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12138)) := by
  have c80265 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.1
  have c80266 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80267 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80268 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80269 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80270 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_0 := arithEq_of_rows h (row := 1306) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_0
  simp only [← c80265, ← c80266, ← c80267] at e_1306_0
  have e_1306_1 := arithEq_of_rows h (row := 1306) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_1
  simp only [← c80268, ← c80269, k2, ← c80270] at e_1306_1
  have hr := e_1306_1
  simp only [e_1306_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12139)) := by
  have c80271 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80272 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80273 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80274 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80275 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80276 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_2 := arithEq_of_rows h (row := 1306) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_2
  simp only [← c80271, ← c80272, ← c80273] at e_1306_2
  have e_1306_3 := arithEq_of_rows h (row := 1306) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_3
  simp only [← c80274, ← c80275, k2, ← c80276] at e_1306_3
  have hr := e_1306_3
  simp only [e_1306_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12140)) := by
  have c80277 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80278 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80279 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80280 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80281 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80282 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_4 := arithEq_of_rows h (row := 1306) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_4
  simp only [← c80277, ← c80278, ← c80279] at e_1306_4
  have e_1306_5 := arithEq_of_rows h (row := 1306) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_5
  simp only [← c80280, ← c80281, k2, ← c80282] at e_1306_5
  have hr := e_1306_5
  simp only [e_1306_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12141)) := by
  have c80283 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80284 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80285 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80286 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80287 := (publicBatchWrapper86_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80288 := (publicBatchWrapper86_copies2509 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_6 := arithEq_of_rows h (row := 1306) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_6
  simp only [← c80283, ← c80284, ← c80285] at e_1306_6
  have e_1306_7 := arithEq_of_rows h (row := 1306) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_7
  simp only [← c80286, ← c80287, k2, ← c80288] at e_1306_7
  have hr := e_1306_7
  simp only [e_1306_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12142)) := by
  have c80289 := (publicBatchWrapper86_copies2509 a h).2.1
  have c80290 := (publicBatchWrapper86_copies2509 a h).2.2.1
  have c80291 := (publicBatchWrapper86_copies2509 a h).2.2.2.1
  have c80292 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.1
  have c80293 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.1
  have c80294 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_8 := arithEq_of_rows h (row := 1306) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_8
  simp only [← c80289, ← c80290, ← c80291] at e_1306_8
  have e_1306_9 := arithEq_of_rows h (row := 1306) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_9
  simp only [← c80292, ← c80293, k2, ← c80294] at e_1306_9
  have hr := e_1306_9
  simp only [e_1306_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12143)) := by
  have c80295 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.1
  have c80296 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.1
  have c80297 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.1
  have c80298 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80299 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80300 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_10 := arithEq_of_rows h (row := 1306) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_10
  simp only [← c80295, ← c80296, ← c80297] at e_1306_10
  have e_1306_11 := arithEq_of_rows h (row := 1306) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_11
  simp only [← c80298, ← c80299, k2, ← c80300] at e_1306_11
  have hr := e_1306_11
  simp only [e_1306_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12144)) := by
  have c80301 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80302 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80303 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80304 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80305 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80306 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_12 := arithEq_of_rows h (row := 1306) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_12
  simp only [← c80301, ← c80302, ← c80303] at e_1306_12
  have e_1306_13 := arithEq_of_rows h (row := 1306) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_13
  simp only [← c80304, ← c80305, k2, ← c80306] at e_1306_13
  have hr := e_1306_13
  simp only [e_1306_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12145)) := by
  have c80307 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80308 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80309 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80310 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80311 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80312 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_14 := arithEq_of_rows h (row := 1306) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_14
  simp only [← c80307, ← c80308, ← c80309] at e_1306_14
  have e_1306_15 := arithEq_of_rows h (row := 1306) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_15
  simp only [← c80310, ← c80311, k2, ← c80312] at e_1306_15
  have hr := e_1306_15
  simp only [e_1306_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12146)) := by
  have c80313 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80314 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80315 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80316 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80317 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80318 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_16 := arithEq_of_rows h (row := 1306) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_16
  simp only [← c80313, ← c80314, ← c80315] at e_1306_16
  have e_1306_17 := arithEq_of_rows h (row := 1306) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_17
  simp only [← c80316, ← c80317, k2, ← c80318] at e_1306_17
  have hr := e_1306_17
  simp only [e_1306_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1306 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12147)) := by
  have c80319 := (publicBatchWrapper86_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80320 := (publicBatchWrapper86_copies2510 a h).1
  have c80321 := (publicBatchWrapper86_copies2510 a h).2.1
  have c80322 := (publicBatchWrapper86_copies2510 a h).2.2.1
  have c80323 := (publicBatchWrapper86_copies2510 a h).2.2.2.1
  have c80324 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1306_18 := arithEq_of_rows h (row := 1306) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_18
  simp only [← c80319, ← c80320, ← c80321] at e_1306_18
  have e_1306_19 := arithEq_of_rows h (row := 1306) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_19
  simp only [← c80322, ← c80323, k2, ← c80324] at e_1306_19
  have hr := e_1306_19
  simp only [e_1306_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12148)) := by
  have c80325 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.1
  have c80326 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.1
  have c80327 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.1
  have c80328 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.1
  have c80329 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.1
  have c80330 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_0 := arithEq_of_rows h (row := 1307) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_0
  simp only [← c80325, ← c80326, ← c80327] at e_1307_0
  have e_1307_1 := arithEq_of_rows h (row := 1307) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_1
  simp only [← c80328, ← c80329, k2, ← c80330] at e_1307_1
  have hr := e_1307_1
  simp only [e_1307_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12149)) := by
  have c80331 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80332 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80333 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80334 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80335 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80336 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_2 := arithEq_of_rows h (row := 1307) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_2
  simp only [← c80331, ← c80332, ← c80333] at e_1307_2
  have e_1307_3 := arithEq_of_rows h (row := 1307) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_3
  simp only [← c80334, ← c80335, k2, ← c80336] at e_1307_3
  have hr := e_1307_3
  simp only [e_1307_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12150)) := by
  have c80337 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80338 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80339 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80340 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80341 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80342 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_4 := arithEq_of_rows h (row := 1307) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_4
  simp only [← c80337, ← c80338, ← c80339] at e_1307_4
  have e_1307_5 := arithEq_of_rows h (row := 1307) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_5
  simp only [← c80340, ← c80341, k2, ← c80342] at e_1307_5
  have hr := e_1307_5
  simp only [e_1307_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12151)) := by
  have c80343 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80344 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80345 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80346 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80347 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80348 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_6 := arithEq_of_rows h (row := 1307) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_6
  simp only [← c80343, ← c80344, ← c80345] at e_1307_6
  have e_1307_7 := arithEq_of_rows h (row := 1307) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_7
  simp only [← c80346, ← c80347, k2, ← c80348] at e_1307_7
  have hr := e_1307_7
  simp only [e_1307_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12152)) := by
  have c80349 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80350 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80351 := (publicBatchWrapper86_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80352 := (publicBatchWrapper86_copies2511 a h).1
  have c80353 := (publicBatchWrapper86_copies2511 a h).2.1
  have c80354 := (publicBatchWrapper86_copies2511 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_8 := arithEq_of_rows h (row := 1307) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_8
  simp only [← c80349, ← c80350, ← c80351] at e_1307_8
  have e_1307_9 := arithEq_of_rows h (row := 1307) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_9
  simp only [← c80352, ← c80353, k2, ← c80354] at e_1307_9
  have hr := e_1307_9
  simp only [e_1307_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12153)) := by
  have c80355 := (publicBatchWrapper86_copies2511 a h).2.2.2.1
  have c80356 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.1
  have c80357 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.1
  have c80358 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.1
  have c80359 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.1
  have c80360 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_10 := arithEq_of_rows h (row := 1307) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_10
  simp only [← c80355, ← c80356, ← c80357] at e_1307_10
  have e_1307_11 := arithEq_of_rows h (row := 1307) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_11
  simp only [← c80358, ← c80359, k2, ← c80360] at e_1307_11
  have hr := e_1307_11
  simp only [e_1307_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12154)) := by
  have c80361 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.1
  have c80362 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80363 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80364 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80365 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80366 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_12 := arithEq_of_rows h (row := 1307) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_12
  simp only [← c80361, ← c80362, ← c80363] at e_1307_12
  have e_1307_13 := arithEq_of_rows h (row := 1307) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_13
  simp only [← c80364, ← c80365, k2, ← c80366] at e_1307_13
  have hr := e_1307_13
  simp only [e_1307_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12155)) := by
  have c80367 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80368 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80369 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80370 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80371 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80372 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_14 := arithEq_of_rows h (row := 1307) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_14
  simp only [← c80367, ← c80368, ← c80369] at e_1307_14
  have e_1307_15 := arithEq_of_rows h (row := 1307) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_15
  simp only [← c80370, ← c80371, k2, ← c80372] at e_1307_15
  have hr := e_1307_15
  simp only [e_1307_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12156)) := by
  have c80373 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80374 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80375 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80376 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80377 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80378 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_16 := arithEq_of_rows h (row := 1307) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_16
  simp only [← c80373, ← c80374, ← c80375] at e_1307_16
  have e_1307_17 := arithEq_of_rows h (row := 1307) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_17
  simp only [← c80376, ← c80377, k2, ← c80378] at e_1307_17
  have hr := e_1307_17
  simp only [e_1307_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1307 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12157)) := by
  have c80379 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80380 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80381 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80382 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80383 := (publicBatchWrapper86_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80384 := (publicBatchWrapper86_copies2512 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1307_18 := arithEq_of_rows h (row := 1307) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_18
  simp only [← c80379, ← c80380, ← c80381] at e_1307_18
  have e_1307_19 := arithEq_of_rows h (row := 1307) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_19
  simp only [← c80382, ← c80383, k2, ← c80384] at e_1307_19
  have hr := e_1307_19
  simp only [e_1307_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12158)) := by
  have c80385 := (publicBatchWrapper86_copies2512 a h).2.1
  have c80386 := (publicBatchWrapper86_copies2512 a h).2.2.1
  have c80387 := (publicBatchWrapper86_copies2512 a h).2.2.2.1
  have c80388 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.1
  have c80389 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.1
  have c80390 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_0 := arithEq_of_rows h (row := 1308) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_0
  simp only [← c80385, ← c80386, ← c80387] at e_1308_0
  have e_1308_1 := arithEq_of_rows h (row := 1308) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_1
  simp only [← c80388, ← c80389, k2, ← c80390] at e_1308_1
  have hr := e_1308_1
  simp only [e_1308_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12159)) := by
  have c80391 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.1
  have c80392 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.1
  have c80393 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.1
  have c80394 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80395 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80396 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_2 := arithEq_of_rows h (row := 1308) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_2
  simp only [← c80391, ← c80392, ← c80393] at e_1308_2
  have e_1308_3 := arithEq_of_rows h (row := 1308) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_3
  simp only [← c80394, ← c80395, k2, ← c80396] at e_1308_3
  have hr := e_1308_3
  simp only [e_1308_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12160)) := by
  have c80397 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80398 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80399 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80400 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80401 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80402 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_4 := arithEq_of_rows h (row := 1308) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_4
  simp only [← c80397, ← c80398, ← c80399] at e_1308_4
  have e_1308_5 := arithEq_of_rows h (row := 1308) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_5
  simp only [← c80400, ← c80401, k2, ← c80402] at e_1308_5
  have hr := e_1308_5
  simp only [e_1308_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12161)) := by
  have c80403 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80404 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80405 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80406 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80407 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80408 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_6 := arithEq_of_rows h (row := 1308) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_6
  simp only [← c80403, ← c80404, ← c80405] at e_1308_6
  have e_1308_7 := arithEq_of_rows h (row := 1308) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_7
  simp only [← c80406, ← c80407, k2, ← c80408] at e_1308_7
  have hr := e_1308_7
  simp only [e_1308_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12162)) := by
  have c80409 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80410 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80411 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80412 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80413 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80414 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_8 := arithEq_of_rows h (row := 1308) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_8
  simp only [← c80409, ← c80410, ← c80411] at e_1308_8
  have e_1308_9 := arithEq_of_rows h (row := 1308) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_9
  simp only [← c80412, ← c80413, k2, ← c80414] at e_1308_9
  have hr := e_1308_9
  simp only [e_1308_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12163)) := by
  have c80415 := (publicBatchWrapper86_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80416 := (publicBatchWrapper86_copies2513 a h).1
  have c80417 := (publicBatchWrapper86_copies2513 a h).2.1
  have c80418 := (publicBatchWrapper86_copies2513 a h).2.2.1
  have c80419 := (publicBatchWrapper86_copies2513 a h).2.2.2.1
  have c80420 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_10 := arithEq_of_rows h (row := 1308) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_10
  simp only [← c80415, ← c80416, ← c80417] at e_1308_10
  have e_1308_11 := arithEq_of_rows h (row := 1308) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_11
  simp only [← c80418, ← c80419, k2, ← c80420] at e_1308_11
  have hr := e_1308_11
  simp only [e_1308_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12164)) := by
  have c80421 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.1
  have c80422 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.1
  have c80423 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.1
  have c80424 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.1
  have c80425 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.1
  have c80426 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_12 := arithEq_of_rows h (row := 1308) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_12
  simp only [← c80421, ← c80422, ← c80423] at e_1308_12
  have e_1308_13 := arithEq_of_rows h (row := 1308) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_13
  simp only [← c80424, ← c80425, k2, ← c80426] at e_1308_13
  have hr := e_1308_13
  simp only [e_1308_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12165)) := by
  have c80427 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80428 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80429 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80430 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80431 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80432 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_14 := arithEq_of_rows h (row := 1308) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_14
  simp only [← c80427, ← c80428, ← c80429] at e_1308_14
  have e_1308_15 := arithEq_of_rows h (row := 1308) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_15
  simp only [← c80430, ← c80431, k2, ← c80432] at e_1308_15
  have hr := e_1308_15
  simp only [e_1308_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12166)) := by
  have c80433 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80434 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80435 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80436 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80437 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80438 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_16 := arithEq_of_rows h (row := 1308) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_16
  simp only [← c80433, ← c80434, ← c80435] at e_1308_16
  have e_1308_17 := arithEq_of_rows h (row := 1308) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_17
  simp only [← c80436, ← c80437, k2, ← c80438] at e_1308_17
  have hr := e_1308_17
  simp only [e_1308_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1308 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12167)) := by
  have c80439 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80440 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80441 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80442 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80443 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80444 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1308_18 := arithEq_of_rows h (row := 1308) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_18
  simp only [← c80439, ← c80440, ← c80441] at e_1308_18
  have e_1308_19 := arithEq_of_rows h (row := 1308) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_19
  simp only [← c80442, ← c80443, k2, ← c80444] at e_1308_19
  have hr := e_1308_19
  simp only [e_1308_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12168)) := by
  have c80445 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80446 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80447 := (publicBatchWrapper86_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80448 := (publicBatchWrapper86_copies2514 a h).1
  have c80449 := (publicBatchWrapper86_copies2514 a h).2.1
  have c80450 := (publicBatchWrapper86_copies2514 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_0 := arithEq_of_rows h (row := 1309) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_0
  simp only [← c80445, ← c80446, ← c80447] at e_1309_0
  have e_1309_1 := arithEq_of_rows h (row := 1309) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_1
  simp only [← c80448, ← c80449, k2, ← c80450] at e_1309_1
  have hr := e_1309_1
  simp only [e_1309_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12169)) := by
  have c80451 := (publicBatchWrapper86_copies2514 a h).2.2.2.1
  have c80452 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.1
  have c80453 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.1
  have c80454 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.1
  have c80455 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.1
  have c80456 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_2 := arithEq_of_rows h (row := 1309) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_2
  simp only [← c80451, ← c80452, ← c80453] at e_1309_2
  have e_1309_3 := arithEq_of_rows h (row := 1309) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_3
  simp only [← c80454, ← c80455, k2, ← c80456] at e_1309_3
  have hr := e_1309_3
  simp only [e_1309_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12210)) := by
  have c80457 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.1
  have c80458 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80459 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80460 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80461 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80462 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_4 := arithEq_of_rows h (row := 1309) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_4
  simp only [← c80457, ← c80458, ← c80459] at e_1309_4
  have e_1309_5 := arithEq_of_rows h (row := 1309) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_5
  simp only [← c80460, ← c80461, k2, ← c80462] at e_1309_5
  have hr := e_1309_5
  simp only [e_1309_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12211)) := by
  have c80463 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80464 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80465 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80466 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80467 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80468 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_6 := arithEq_of_rows h (row := 1309) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_6
  simp only [← c80463, ← c80464, ← c80465] at e_1309_6
  have e_1309_7 := arithEq_of_rows h (row := 1309) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_7
  simp only [← c80466, ← c80467, k2, ← c80468] at e_1309_7
  have hr := e_1309_7
  simp only [e_1309_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12212)) := by
  have c80469 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80470 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80471 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80472 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80473 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80474 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_8 := arithEq_of_rows h (row := 1309) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_8
  simp only [← c80469, ← c80470, ← c80471] at e_1309_8
  have e_1309_9 := arithEq_of_rows h (row := 1309) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_9
  simp only [← c80472, ← c80473, k2, ← c80474] at e_1309_9
  have hr := e_1309_9
  simp only [e_1309_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12213)) := by
  have c80475 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80476 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80477 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80478 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80479 := (publicBatchWrapper86_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80480 := (publicBatchWrapper86_copies2515 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_10 := arithEq_of_rows h (row := 1309) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_10
  simp only [← c80475, ← c80476, ← c80477] at e_1309_10
  have e_1309_11 := arithEq_of_rows h (row := 1309) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_11
  simp only [← c80478, ← c80479, k2, ← c80480] at e_1309_11
  have hr := e_1309_11
  simp only [e_1309_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12214)) := by
  have c80481 := (publicBatchWrapper86_copies2515 a h).2.1
  have c80482 := (publicBatchWrapper86_copies2515 a h).2.2.1
  have c80483 := (publicBatchWrapper86_copies2515 a h).2.2.2.1
  have c80484 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.1
  have c80485 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.1
  have c80486 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_12 := arithEq_of_rows h (row := 1309) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_12
  simp only [← c80481, ← c80482, ← c80483] at e_1309_12
  have e_1309_13 := arithEq_of_rows h (row := 1309) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_13
  simp only [← c80484, ← c80485, k2, ← c80486] at e_1309_13
  have hr := e_1309_13
  simp only [e_1309_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12215)) := by
  have c80487 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.1
  have c80488 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.1
  have c80489 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.1
  have c80490 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80491 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80492 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_14 := arithEq_of_rows h (row := 1309) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_14
  simp only [← c80487, ← c80488, ← c80489] at e_1309_14
  have e_1309_15 := arithEq_of_rows h (row := 1309) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_15
  simp only [← c80490, ← c80491, k2, ← c80492] at e_1309_15
  have hr := e_1309_15
  simp only [e_1309_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12216)) := by
  have c80493 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80494 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80495 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80496 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80497 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80498 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_16 := arithEq_of_rows h (row := 1309) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_16
  simp only [← c80493, ← c80494, ← c80495] at e_1309_16
  have e_1309_17 := arithEq_of_rows h (row := 1309) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_17
  simp only [← c80496, ← c80497, k2, ← c80498] at e_1309_17
  have hr := e_1309_17
  simp only [e_1309_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1309 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12217)) := by
  have c80499 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80500 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80501 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80502 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80503 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80504 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1309_18 := arithEq_of_rows h (row := 1309) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_18
  simp only [← c80499, ← c80500, ← c80501] at e_1309_18
  have e_1309_19 := arithEq_of_rows h (row := 1309) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_19
  simp only [← c80502, ← c80503, k2, ← c80504] at e_1309_19
  have hr := e_1309_19
  simp only [e_1309_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12218)) := by
  have c80505 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80506 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80507 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80508 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80509 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80510 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_0 := arithEq_of_rows h (row := 1310) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_0
  simp only [← c80505, ← c80506, ← c80507] at e_1310_0
  have e_1310_1 := arithEq_of_rows h (row := 1310) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_1
  simp only [← c80508, ← c80509, k2, ← c80510] at e_1310_1
  have hr := e_1310_1
  simp only [e_1310_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12219)) := by
  have c80511 := (publicBatchWrapper86_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80512 := (publicBatchWrapper86_copies2516 a h).1
  have c80513 := (publicBatchWrapper86_copies2516 a h).2.1
  have c80514 := (publicBatchWrapper86_copies2516 a h).2.2.1
  have c80515 := (publicBatchWrapper86_copies2516 a h).2.2.2.1
  have c80516 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_2 := arithEq_of_rows h (row := 1310) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_2
  simp only [← c80511, ← c80512, ← c80513] at e_1310_2
  have e_1310_3 := arithEq_of_rows h (row := 1310) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_3
  simp only [← c80514, ← c80515, k2, ← c80516] at e_1310_3
  have hr := e_1310_3
  simp only [e_1310_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12220)) := by
  have c80517 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.1
  have c80518 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.1
  have c80519 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.1
  have c80520 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.1
  have c80521 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.1
  have c80522 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_4 := arithEq_of_rows h (row := 1310) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_4
  simp only [← c80517, ← c80518, ← c80519] at e_1310_4
  have e_1310_5 := arithEq_of_rows h (row := 1310) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_5
  simp only [← c80520, ← c80521, k2, ← c80522] at e_1310_5
  have hr := e_1310_5
  simp only [e_1310_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12221)) := by
  have c80523 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80524 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80525 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80526 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80527 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80528 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_6 := arithEq_of_rows h (row := 1310) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_6
  simp only [← c80523, ← c80524, ← c80525] at e_1310_6
  have e_1310_7 := arithEq_of_rows h (row := 1310) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_7
  simp only [← c80526, ← c80527, k2, ← c80528] at e_1310_7
  have hr := e_1310_7
  simp only [e_1310_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12222)) := by
  have c80529 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80530 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80531 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80532 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80533 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80534 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_8 := arithEq_of_rows h (row := 1310) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_8
  simp only [← c80529, ← c80530, ← c80531] at e_1310_8
  have e_1310_9 := arithEq_of_rows h (row := 1310) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_9
  simp only [← c80532, ← c80533, k2, ← c80534] at e_1310_9
  have hr := e_1310_9
  simp only [e_1310_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12223)) := by
  have c80535 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80536 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80537 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80538 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80539 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80540 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_10 := arithEq_of_rows h (row := 1310) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_10
  simp only [← c80535, ← c80536, ← c80537] at e_1310_10
  have e_1310_11 := arithEq_of_rows h (row := 1310) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_11
  simp only [← c80538, ← c80539, k2, ← c80540] at e_1310_11
  have hr := e_1310_11
  simp only [e_1310_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12224)) := by
  have c80541 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80542 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80543 := (publicBatchWrapper86_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80544 := (publicBatchWrapper86_copies2517 a h).1
  have c80545 := (publicBatchWrapper86_copies2517 a h).2.1
  have c80546 := (publicBatchWrapper86_copies2517 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_12 := arithEq_of_rows h (row := 1310) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_12
  simp only [← c80541, ← c80542, ← c80543] at e_1310_12
  have e_1310_13 := arithEq_of_rows h (row := 1310) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_13
  simp only [← c80544, ← c80545, k2, ← c80546] at e_1310_13
  have hr := e_1310_13
  simp only [e_1310_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12225)) := by
  have c80547 := (publicBatchWrapper86_copies2517 a h).2.2.2.1
  have c80548 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.1
  have c80549 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.1
  have c80550 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.1
  have c80551 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.1
  have c80552 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_14 := arithEq_of_rows h (row := 1310) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_14
  simp only [← c80547, ← c80548, ← c80549] at e_1310_14
  have e_1310_15 := arithEq_of_rows h (row := 1310) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_15
  simp only [← c80550, ← c80551, k2, ← c80552] at e_1310_15
  have hr := e_1310_15
  simp only [e_1310_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12226)) := by
  have c80553 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.1
  have c80554 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80555 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80556 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80557 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80558 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_16 := arithEq_of_rows h (row := 1310) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_16
  simp only [← c80553, ← c80554, ← c80555] at e_1310_16
  have e_1310_17 := arithEq_of_rows h (row := 1310) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_17
  simp only [← c80556, ← c80557, k2, ← c80558] at e_1310_17
  have hr := e_1310_17
  simp only [e_1310_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1310 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12227)) := by
  have c80559 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80560 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80561 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80562 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80563 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80564 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1310_18 := arithEq_of_rows h (row := 1310) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_18
  simp only [← c80559, ← c80560, ← c80561] at e_1310_18
  have e_1310_19 := arithEq_of_rows h (row := 1310) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_19
  simp only [← c80562, ← c80563, k2, ← c80564] at e_1310_19
  have hr := e_1310_19
  simp only [e_1310_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12228)) := by
  have c80565 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80566 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80567 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80568 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80569 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80570 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_0 := arithEq_of_rows h (row := 1311) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_0
  simp only [← c80565, ← c80566, ← c80567] at e_1311_0
  have e_1311_1 := arithEq_of_rows h (row := 1311) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_1
  simp only [← c80568, ← c80569, k2, ← c80570] at e_1311_1
  have hr := e_1311_1
  simp only [e_1311_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12229)) := by
  have c80571 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80572 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80573 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80574 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80575 := (publicBatchWrapper86_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80576 := (publicBatchWrapper86_copies2518 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_2 := arithEq_of_rows h (row := 1311) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_2
  simp only [← c80571, ← c80572, ← c80573] at e_1311_2
  have e_1311_3 := arithEq_of_rows h (row := 1311) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_3
  simp only [← c80574, ← c80575, k2, ← c80576] at e_1311_3
  have hr := e_1311_3
  simp only [e_1311_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12230)) := by
  have c80577 := (publicBatchWrapper86_copies2518 a h).2.1
  have c80578 := (publicBatchWrapper86_copies2518 a h).2.2.1
  have c80579 := (publicBatchWrapper86_copies2518 a h).2.2.2.1
  have c80580 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.1
  have c80581 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.1
  have c80582 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_4 := arithEq_of_rows h (row := 1311) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_4
  simp only [← c80577, ← c80578, ← c80579] at e_1311_4
  have e_1311_5 := arithEq_of_rows h (row := 1311) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_5
  simp only [← c80580, ← c80581, k2, ← c80582] at e_1311_5
  have hr := e_1311_5
  simp only [e_1311_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12231)) := by
  have c80583 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.1
  have c80584 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.1
  have c80585 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.1
  have c80586 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80587 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80588 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_6 := arithEq_of_rows h (row := 1311) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_6
  simp only [← c80583, ← c80584, ← c80585] at e_1311_6
  have e_1311_7 := arithEq_of_rows h (row := 1311) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_7
  simp only [← c80586, ← c80587, k2, ← c80588] at e_1311_7
  have hr := e_1311_7
  simp only [e_1311_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12232)) := by
  have c80589 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80590 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80591 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80592 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80593 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80594 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_8 := arithEq_of_rows h (row := 1311) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_8
  simp only [← c80589, ← c80590, ← c80591] at e_1311_8
  have e_1311_9 := arithEq_of_rows h (row := 1311) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_9
  simp only [← c80592, ← c80593, k2, ← c80594] at e_1311_9
  have hr := e_1311_9
  simp only [e_1311_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12233)) := by
  have c80595 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80596 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80597 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80598 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80599 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80600 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_10 := arithEq_of_rows h (row := 1311) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_10
  simp only [← c80595, ← c80596, ← c80597] at e_1311_10
  have e_1311_11 := arithEq_of_rows h (row := 1311) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_11
  simp only [← c80598, ← c80599, k2, ← c80600] at e_1311_11
  have hr := e_1311_11
  simp only [e_1311_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12234)) := by
  have c80601 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80602 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80603 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80604 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80605 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80606 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_12 := arithEq_of_rows h (row := 1311) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_12
  simp only [← c80601, ← c80602, ← c80603] at e_1311_12
  have e_1311_13 := arithEq_of_rows h (row := 1311) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_13
  simp only [← c80604, ← c80605, k2, ← c80606] at e_1311_13
  have hr := e_1311_13
  simp only [e_1311_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12235)) := by
  have c80607 := (publicBatchWrapper86_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80608 := (publicBatchWrapper86_copies2519 a h).1
  have c80609 := (publicBatchWrapper86_copies2519 a h).2.1
  have c80610 := (publicBatchWrapper86_copies2519 a h).2.2.1
  have c80611 := (publicBatchWrapper86_copies2519 a h).2.2.2.1
  have c80612 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_14 := arithEq_of_rows h (row := 1311) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_14
  simp only [← c80607, ← c80608, ← c80609] at e_1311_14
  have e_1311_15 := arithEq_of_rows h (row := 1311) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_15
  simp only [← c80610, ← c80611, k2, ← c80612] at e_1311_15
  have hr := e_1311_15
  simp only [e_1311_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12236)) := by
  have c80613 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.1
  have c80614 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.1
  have c80615 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.1
  have c80616 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.1
  have c80617 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.1
  have c80618 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_16 := arithEq_of_rows h (row := 1311) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_16
  simp only [← c80613, ← c80614, ← c80615] at e_1311_16
  have e_1311_17 := arithEq_of_rows h (row := 1311) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_17
  simp only [← c80616, ← c80617, k2, ← c80618] at e_1311_17
  have hr := e_1311_17
  simp only [e_1311_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1311 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12237)) := by
  have c80619 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80620 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80621 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80622 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80623 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80624 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1311_18 := arithEq_of_rows h (row := 1311) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_18
  simp only [← c80619, ← c80620, ← c80621] at e_1311_18
  have e_1311_19 := arithEq_of_rows h (row := 1311) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_19
  simp only [← c80622, ← c80623, k2, ← c80624] at e_1311_19
  have hr := e_1311_19
  simp only [e_1311_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12238)) := by
  have c80625 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80626 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80627 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80628 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80629 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80630 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_0 := arithEq_of_rows h (row := 1312) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_0
  simp only [← c80625, ← c80626, ← c80627] at e_1312_0
  have e_1312_1 := arithEq_of_rows h (row := 1312) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_1
  simp only [← c80628, ← c80629, k2, ← c80630] at e_1312_1
  have hr := e_1312_1
  simp only [e_1312_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12239)) := by
  have c80631 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80632 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80633 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80634 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80635 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80636 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_2 := arithEq_of_rows h (row := 1312) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_2
  simp only [← c80631, ← c80632, ← c80633] at e_1312_2
  have e_1312_3 := arithEq_of_rows h (row := 1312) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_3
  simp only [← c80634, ← c80635, k2, ← c80636] at e_1312_3
  have hr := e_1312_3
  simp only [e_1312_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12240)) := by
  have c80637 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80638 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80639 := (publicBatchWrapper86_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80640 := (publicBatchWrapper86_copies2520 a h).1
  have c80641 := (publicBatchWrapper86_copies2520 a h).2.1
  have c80642 := (publicBatchWrapper86_copies2520 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_4 := arithEq_of_rows h (row := 1312) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_4
  simp only [← c80637, ← c80638, ← c80639] at e_1312_4
  have e_1312_5 := arithEq_of_rows h (row := 1312) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_5
  simp only [← c80640, ← c80641, k2, ← c80642] at e_1312_5
  have hr := e_1312_5
  simp only [e_1312_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12241)) := by
  have c80643 := (publicBatchWrapper86_copies2520 a h).2.2.2.1
  have c80644 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.1
  have c80645 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.1
  have c80646 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.1
  have c80647 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.1
  have c80648 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_6 := arithEq_of_rows h (row := 1312) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_6
  simp only [← c80643, ← c80644, ← c80645] at e_1312_6
  have e_1312_7 := arithEq_of_rows h (row := 1312) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_7
  simp only [← c80646, ← c80647, k2, ← c80648] at e_1312_7
  have hr := e_1312_7
  simp only [e_1312_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12242)) := by
  have c80649 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.1
  have c80650 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80651 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80652 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80653 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80654 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_8 := arithEq_of_rows h (row := 1312) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_8
  simp only [← c80649, ← c80650, ← c80651] at e_1312_8
  have e_1312_9 := arithEq_of_rows h (row := 1312) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_9
  simp only [← c80652, ← c80653, k2, ← c80654] at e_1312_9
  have hr := e_1312_9
  simp only [e_1312_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12243)) := by
  have c80655 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80656 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80657 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80658 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80659 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80660 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_10 := arithEq_of_rows h (row := 1312) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_10
  simp only [← c80655, ← c80656, ← c80657] at e_1312_10
  have e_1312_11 := arithEq_of_rows h (row := 1312) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_11
  simp only [← c80658, ← c80659, k2, ← c80660] at e_1312_11
  have hr := e_1312_11
  simp only [e_1312_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12244)) := by
  have c80661 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80662 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80663 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80664 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80665 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80666 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_12 := arithEq_of_rows h (row := 1312) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_12
  simp only [← c80661, ← c80662, ← c80663] at e_1312_12
  have e_1312_13 := arithEq_of_rows h (row := 1312) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_13
  simp only [← c80664, ← c80665, k2, ← c80666] at e_1312_13
  have hr := e_1312_13
  simp only [e_1312_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12245)) := by
  have c80667 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80668 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80669 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80670 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80671 := (publicBatchWrapper86_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80672 := (publicBatchWrapper86_copies2521 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_14 := arithEq_of_rows h (row := 1312) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_14
  simp only [← c80667, ← c80668, ← c80669] at e_1312_14
  have e_1312_15 := arithEq_of_rows h (row := 1312) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_15
  simp only [← c80670, ← c80671, k2, ← c80672] at e_1312_15
  have hr := e_1312_15
  simp only [e_1312_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12246)) := by
  have c80673 := (publicBatchWrapper86_copies2521 a h).2.1
  have c80674 := (publicBatchWrapper86_copies2521 a h).2.2.1
  have c80675 := (publicBatchWrapper86_copies2521 a h).2.2.2.1
  have c80676 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.1
  have c80677 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.1
  have c80678 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_16 := arithEq_of_rows h (row := 1312) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_16
  simp only [← c80673, ← c80674, ← c80675] at e_1312_16
  have e_1312_17 := arithEq_of_rows h (row := 1312) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_17
  simp only [← c80676, ← c80677, k2, ← c80678] at e_1312_17
  have hr := e_1312_17
  simp only [e_1312_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1312 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12247)) := by
  have c80679 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.1
  have c80680 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.1
  have c80681 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.1
  have c80682 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80683 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80684 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1312_18 := arithEq_of_rows h (row := 1312) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_18
  simp only [← c80679, ← c80680, ← c80681] at e_1312_18
  have e_1312_19 := arithEq_of_rows h (row := 1312) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_19
  simp only [← c80682, ← c80683, k2, ← c80684] at e_1312_19
  have hr := e_1312_19
  simp only [e_1312_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12248)) := by
  have c80685 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80686 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80687 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80688 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80689 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80690 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_0 := arithEq_of_rows h (row := 1313) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_0
  simp only [← c80685, ← c80686, ← c80687] at e_1313_0
  have e_1313_1 := arithEq_of_rows h (row := 1313) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_1
  simp only [← c80688, ← c80689, k2, ← c80690] at e_1313_1
  have hr := e_1313_1
  simp only [e_1313_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12249)) := by
  have c80691 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80692 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80693 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80694 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80695 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80696 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_2 := arithEq_of_rows h (row := 1313) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_2
  simp only [← c80691, ← c80692, ← c80693] at e_1313_2
  have e_1313_3 := arithEq_of_rows h (row := 1313) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_3
  simp only [← c80694, ← c80695, k2, ← c80696] at e_1313_3
  have hr := e_1313_3
  simp only [e_1313_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12250)) := by
  have c80697 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80698 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80699 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80700 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80701 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80702 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_4 := arithEq_of_rows h (row := 1313) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_4
  simp only [← c80697, ← c80698, ← c80699] at e_1313_4
  have e_1313_5 := arithEq_of_rows h (row := 1313) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_5
  simp only [← c80700, ← c80701, k2, ← c80702] at e_1313_5
  have hr := e_1313_5
  simp only [e_1313_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12251)) := by
  have c80703 := (publicBatchWrapper86_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80704 := (publicBatchWrapper86_copies2522 a h).1
  have c80705 := (publicBatchWrapper86_copies2522 a h).2.1
  have c80706 := (publicBatchWrapper86_copies2522 a h).2.2.1
  have c80707 := (publicBatchWrapper86_copies2522 a h).2.2.2.1
  have c80708 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_6 := arithEq_of_rows h (row := 1313) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_6
  simp only [← c80703, ← c80704, ← c80705] at e_1313_6
  have e_1313_7 := arithEq_of_rows h (row := 1313) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_7
  simp only [← c80706, ← c80707, k2, ← c80708] at e_1313_7
  have hr := e_1313_7
  simp only [e_1313_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12252)) := by
  have c80709 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.1
  have c80710 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.1
  have c80711 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.1
  have c80712 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.1
  have c80713 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.1
  have c80714 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_8 := arithEq_of_rows h (row := 1313) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_8
  simp only [← c80709, ← c80710, ← c80711] at e_1313_8
  have e_1313_9 := arithEq_of_rows h (row := 1313) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_9
  simp only [← c80712, ← c80713, k2, ← c80714] at e_1313_9
  have hr := e_1313_9
  simp only [e_1313_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12253)) := by
  have c80715 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80716 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80717 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80718 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80719 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80720 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_10 := arithEq_of_rows h (row := 1313) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_10
  simp only [← c80715, ← c80716, ← c80717] at e_1313_10
  have e_1313_11 := arithEq_of_rows h (row := 1313) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_11
  simp only [← c80718, ← c80719, k2, ← c80720] at e_1313_11
  have hr := e_1313_11
  simp only [e_1313_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12254)) := by
  have c80721 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80722 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80723 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80724 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80725 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80726 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_12 := arithEq_of_rows h (row := 1313) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_12
  simp only [← c80721, ← c80722, ← c80723] at e_1313_12
  have e_1313_13 := arithEq_of_rows h (row := 1313) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_13
  simp only [← c80724, ← c80725, k2, ← c80726] at e_1313_13
  have hr := e_1313_13
  simp only [e_1313_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12255)) := by
  have c80727 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80728 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80729 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80730 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80731 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80732 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_14 := arithEq_of_rows h (row := 1313) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_14
  simp only [← c80727, ← c80728, ← c80729] at e_1313_14
  have e_1313_15 := arithEq_of_rows h (row := 1313) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_15
  simp only [← c80730, ← c80731, k2, ← c80732] at e_1313_15
  have hr := e_1313_15
  simp only [e_1313_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12256)) := by
  have c80733 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80734 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80735 := (publicBatchWrapper86_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80736 := (publicBatchWrapper86_copies2523 a h).1
  have c80737 := (publicBatchWrapper86_copies2523 a h).2.1
  have c80738 := (publicBatchWrapper86_copies2523 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_16 := arithEq_of_rows h (row := 1313) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_16
  simp only [← c80733, ← c80734, ← c80735] at e_1313_16
  have e_1313_17 := arithEq_of_rows h (row := 1313) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_17
  simp only [← c80736, ← c80737, k2, ← c80738] at e_1313_17
  have hr := e_1313_17
  simp only [e_1313_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1313 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12257)) := by
  have c80739 := (publicBatchWrapper86_copies2523 a h).2.2.2.1
  have c80740 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.1
  have c80741 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.1
  have c80742 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.1
  have c80743 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.1
  have c80744 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1313_18 := arithEq_of_rows h (row := 1313) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_18
  simp only [← c80739, ← c80740, ← c80741] at e_1313_18
  have e_1313_19 := arithEq_of_rows h (row := 1313) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_19
  simp only [← c80742, ← c80743, k2, ← c80744] at e_1313_19
  have hr := e_1313_19
  simp only [e_1313_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12258)) := by
  have c80745 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.1
  have c80746 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80747 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80748 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80749 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80750 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_0 := arithEq_of_rows h (row := 1314) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_0
  simp only [← c80745, ← c80746, ← c80747] at e_1314_0
  have e_1314_1 := arithEq_of_rows h (row := 1314) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_1
  simp only [← c80748, ← c80749, k2, ← c80750] at e_1314_1
  have hr := e_1314_1
  simp only [e_1314_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12259)) := by
  have c80751 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80752 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80753 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80754 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80755 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80756 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_2 := arithEq_of_rows h (row := 1314) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_2
  simp only [← c80751, ← c80752, ← c80753] at e_1314_2
  have e_1314_3 := arithEq_of_rows h (row := 1314) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_3
  simp only [← c80754, ← c80755, k2, ← c80756] at e_1314_3
  have hr := e_1314_3
  simp only [e_1314_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12260)) := by
  have c80757 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80758 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80759 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80760 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80761 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80762 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_4 := arithEq_of_rows h (row := 1314) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_4
  simp only [← c80757, ← c80758, ← c80759] at e_1314_4
  have e_1314_5 := arithEq_of_rows h (row := 1314) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_5
  simp only [← c80760, ← c80761, k2, ← c80762] at e_1314_5
  have hr := e_1314_5
  simp only [e_1314_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12261)) := by
  have c80763 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80764 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80765 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80766 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80767 := (publicBatchWrapper86_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80768 := (publicBatchWrapper86_copies2524 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_6 := arithEq_of_rows h (row := 1314) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_6
  simp only [← c80763, ← c80764, ← c80765] at e_1314_6
  have e_1314_7 := arithEq_of_rows h (row := 1314) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_7
  simp only [← c80766, ← c80767, k2, ← c80768] at e_1314_7
  have hr := e_1314_7
  simp only [e_1314_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12262)) := by
  have c80769 := (publicBatchWrapper86_copies2524 a h).2.1
  have c80770 := (publicBatchWrapper86_copies2524 a h).2.2.1
  have c80771 := (publicBatchWrapper86_copies2524 a h).2.2.2.1
  have c80772 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.1
  have c80773 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.1
  have c80774 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_8 := arithEq_of_rows h (row := 1314) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_8
  simp only [← c80769, ← c80770, ← c80771] at e_1314_8
  have e_1314_9 := arithEq_of_rows h (row := 1314) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_9
  simp only [← c80772, ← c80773, k2, ← c80774] at e_1314_9
  have hr := e_1314_9
  simp only [e_1314_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12263)) := by
  have c80775 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.1
  have c80776 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.1
  have c80777 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.1
  have c80778 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80779 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80780 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_10 := arithEq_of_rows h (row := 1314) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_10
  simp only [← c80775, ← c80776, ← c80777] at e_1314_10
  have e_1314_11 := arithEq_of_rows h (row := 1314) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_11
  simp only [← c80778, ← c80779, k2, ← c80780] at e_1314_11
  have hr := e_1314_11
  simp only [e_1314_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12264)) := by
  have c80781 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80782 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80783 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80784 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80785 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80786 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_12 := arithEq_of_rows h (row := 1314) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_12
  simp only [← c80781, ← c80782, ← c80783] at e_1314_12
  have e_1314_13 := arithEq_of_rows h (row := 1314) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_13
  simp only [← c80784, ← c80785, k2, ← c80786] at e_1314_13
  have hr := e_1314_13
  simp only [e_1314_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12265)) := by
  have c80787 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80788 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80789 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80790 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80791 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80792 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_14 := arithEq_of_rows h (row := 1314) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_14
  simp only [← c80787, ← c80788, ← c80789] at e_1314_14
  have e_1314_15 := arithEq_of_rows h (row := 1314) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_15
  simp only [← c80790, ← c80791, k2, ← c80792] at e_1314_15
  have hr := e_1314_15
  simp only [e_1314_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12266)) := by
  have c80793 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80794 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80795 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80796 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80797 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80798 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_16 := arithEq_of_rows h (row := 1314) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_16
  simp only [← c80793, ← c80794, ← c80795] at e_1314_16
  have e_1314_17 := arithEq_of_rows h (row := 1314) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_17
  simp only [← c80796, ← c80797, k2, ← c80798] at e_1314_17
  have hr := e_1314_17
  simp only [e_1314_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1314 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12267)) := by
  have c80799 := (publicBatchWrapper86_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80800 := (publicBatchWrapper86_copies2525 a h).1
  have c80801 := (publicBatchWrapper86_copies2525 a h).2.1
  have c80802 := (publicBatchWrapper86_copies2525 a h).2.2.1
  have c80803 := (publicBatchWrapper86_copies2525 a h).2.2.2.1
  have c80804 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1314_18 := arithEq_of_rows h (row := 1314) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_18
  simp only [← c80799, ← c80800, ← c80801] at e_1314_18
  have e_1314_19 := arithEq_of_rows h (row := 1314) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_19
  simp only [← c80802, ← c80803, k2, ← c80804] at e_1314_19
  have hr := e_1314_19
  simp only [e_1314_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12268)) := by
  have c80805 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.1
  have c80806 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.1
  have c80807 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.1
  have c80808 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.1
  have c80809 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.1
  have c80810 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_0 := arithEq_of_rows h (row := 1315) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_0
  simp only [← c80805, ← c80806, ← c80807] at e_1315_0
  have e_1315_1 := arithEq_of_rows h (row := 1315) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_1
  simp only [← c80808, ← c80809, k2, ← c80810] at e_1315_1
  have hr := e_1315_1
  simp only [e_1315_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12269)) := by
  have c80811 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80812 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80813 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80814 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80815 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80816 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_2 := arithEq_of_rows h (row := 1315) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_2
  simp only [← c80811, ← c80812, ← c80813] at e_1315_2
  have e_1315_3 := arithEq_of_rows h (row := 1315) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_3
  simp only [← c80814, ← c80815, k2, ← c80816] at e_1315_3
  have hr := e_1315_3
  simp only [e_1315_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12270)) := by
  have c80817 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80818 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80819 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80820 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80821 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80822 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_4 := arithEq_of_rows h (row := 1315) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_4
  simp only [← c80817, ← c80818, ← c80819] at e_1315_4
  have e_1315_5 := arithEq_of_rows h (row := 1315) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_5
  simp only [← c80820, ← c80821, k2, ← c80822] at e_1315_5
  have hr := e_1315_5
  simp only [e_1315_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12271)) := by
  have c80823 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80824 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80825 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80826 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80827 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80828 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_6 := arithEq_of_rows h (row := 1315) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_6
  simp only [← c80823, ← c80824, ← c80825] at e_1315_6
  have e_1315_7 := arithEq_of_rows h (row := 1315) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_7
  simp only [← c80826, ← c80827, k2, ← c80828] at e_1315_7
  have hr := e_1315_7
  simp only [e_1315_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12272)) := by
  have c80829 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80830 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80831 := (publicBatchWrapper86_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80832 := (publicBatchWrapper86_copies2526 a h).1
  have c80833 := (publicBatchWrapper86_copies2526 a h).2.1
  have c80834 := (publicBatchWrapper86_copies2526 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_8 := arithEq_of_rows h (row := 1315) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_8
  simp only [← c80829, ← c80830, ← c80831] at e_1315_8
  have e_1315_9 := arithEq_of_rows h (row := 1315) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_9
  simp only [← c80832, ← c80833, k2, ← c80834] at e_1315_9
  have hr := e_1315_9
  simp only [e_1315_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12273)) := by
  have c80835 := (publicBatchWrapper86_copies2526 a h).2.2.2.1
  have c80836 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.1
  have c80837 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.1
  have c80838 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.1
  have c80839 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.1
  have c80840 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_10 := arithEq_of_rows h (row := 1315) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_10
  simp only [← c80835, ← c80836, ← c80837] at e_1315_10
  have e_1315_11 := arithEq_of_rows h (row := 1315) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_11
  simp only [← c80838, ← c80839, k2, ← c80840] at e_1315_11
  have hr := e_1315_11
  simp only [e_1315_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12274)) := by
  have c80841 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.1
  have c80842 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80843 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80844 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80845 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80846 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_12 := arithEq_of_rows h (row := 1315) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_12
  simp only [← c80841, ← c80842, ← c80843] at e_1315_12
  have e_1315_13 := arithEq_of_rows h (row := 1315) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_13
  simp only [← c80844, ← c80845, k2, ← c80846] at e_1315_13
  have hr := e_1315_13
  simp only [e_1315_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12275)) := by
  have c80847 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80848 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80849 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80850 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80851 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80852 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_14 := arithEq_of_rows h (row := 1315) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_14
  simp only [← c80847, ← c80848, ← c80849] at e_1315_14
  have e_1315_15 := arithEq_of_rows h (row := 1315) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_15
  simp only [← c80850, ← c80851, k2, ← c80852] at e_1315_15
  have hr := e_1315_15
  simp only [e_1315_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12276)) := by
  have c80853 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80854 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80855 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80856 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80857 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80858 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_16 := arithEq_of_rows h (row := 1315) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_16
  simp only [← c80853, ← c80854, ← c80855] at e_1315_16
  have e_1315_17 := arithEq_of_rows h (row := 1315) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_17
  simp only [← c80856, ← c80857, k2, ← c80858] at e_1315_17
  have hr := e_1315_17
  simp only [e_1315_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1315 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12277)) := by
  have c80859 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80860 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80861 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80862 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80863 := (publicBatchWrapper86_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80864 := (publicBatchWrapper86_copies2527 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1315_18 := arithEq_of_rows h (row := 1315) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_18
  simp only [← c80859, ← c80860, ← c80861] at e_1315_18
  have e_1315_19 := arithEq_of_rows h (row := 1315) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_19
  simp only [← c80862, ← c80863, k2, ← c80864] at e_1315_19
  have hr := e_1315_19
  simp only [e_1315_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12278)) := by
  have c80865 := (publicBatchWrapper86_copies2527 a h).2.1
  have c80866 := (publicBatchWrapper86_copies2527 a h).2.2.1
  have c80867 := (publicBatchWrapper86_copies2527 a h).2.2.2.1
  have c80868 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.1
  have c80869 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.1
  have c80870 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_0 := arithEq_of_rows h (row := 1316) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_0
  simp only [← c80865, ← c80866, ← c80867] at e_1316_0
  have e_1316_1 := arithEq_of_rows h (row := 1316) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_1
  simp only [← c80868, ← c80869, k2, ← c80870] at e_1316_1
  have hr := e_1316_1
  simp only [e_1316_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12279)) := by
  have c80871 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.1
  have c80872 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.1
  have c80873 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.1
  have c80874 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80875 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80876 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_2 := arithEq_of_rows h (row := 1316) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_2
  simp only [← c80871, ← c80872, ← c80873] at e_1316_2
  have e_1316_3 := arithEq_of_rows h (row := 1316) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_3
  simp only [← c80874, ← c80875, k2, ← c80876] at e_1316_3
  have hr := e_1316_3
  simp only [e_1316_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12280)) := by
  have c80877 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80878 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80879 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80880 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80881 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80882 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_4 := arithEq_of_rows h (row := 1316) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_4
  simp only [← c80877, ← c80878, ← c80879] at e_1316_4
  have e_1316_5 := arithEq_of_rows h (row := 1316) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_5
  simp only [← c80880, ← c80881, k2, ← c80882] at e_1316_5
  have hr := e_1316_5
  simp only [e_1316_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12281)) := by
  have c80883 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80884 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80885 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80886 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80887 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80888 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_6 := arithEq_of_rows h (row := 1316) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_6
  simp only [← c80883, ← c80884, ← c80885] at e_1316_6
  have e_1316_7 := arithEq_of_rows h (row := 1316) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_7
  simp only [← c80886, ← c80887, k2, ← c80888] at e_1316_7
  have hr := e_1316_7
  simp only [e_1316_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12282)) := by
  have c80889 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80890 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80891 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80892 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80893 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80894 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_8 := arithEq_of_rows h (row := 1316) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_8
  simp only [← c80889, ← c80890, ← c80891] at e_1316_8
  have e_1316_9 := arithEq_of_rows h (row := 1316) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_9
  simp only [← c80892, ← c80893, k2, ← c80894] at e_1316_9
  have hr := e_1316_9
  simp only [e_1316_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12283)) := by
  have c80895 := (publicBatchWrapper86_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80896 := (publicBatchWrapper86_copies2528 a h).1
  have c80897 := (publicBatchWrapper86_copies2528 a h).2.1
  have c80898 := (publicBatchWrapper86_copies2528 a h).2.2.1
  have c80899 := (publicBatchWrapper86_copies2528 a h).2.2.2.1
  have c80900 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_10 := arithEq_of_rows h (row := 1316) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_10
  simp only [← c80895, ← c80896, ← c80897] at e_1316_10
  have e_1316_11 := arithEq_of_rows h (row := 1316) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_11
  simp only [← c80898, ← c80899, k2, ← c80900] at e_1316_11
  have hr := e_1316_11
  simp only [e_1316_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12284)) := by
  have c80901 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.1
  have c80902 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.1
  have c80903 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.1
  have c80904 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.1
  have c80905 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.1
  have c80906 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_12 := arithEq_of_rows h (row := 1316) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_12
  simp only [← c80901, ← c80902, ← c80903] at e_1316_12
  have e_1316_13 := arithEq_of_rows h (row := 1316) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_13
  simp only [← c80904, ← c80905, k2, ← c80906] at e_1316_13
  have hr := e_1316_13
  simp only [e_1316_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12285)) := by
  have c80907 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80908 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80909 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80910 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80911 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80912 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_14 := arithEq_of_rows h (row := 1316) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_14
  simp only [← c80907, ← c80908, ← c80909] at e_1316_14
  have e_1316_15 := arithEq_of_rows h (row := 1316) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_15
  simp only [← c80910, ← c80911, k2, ← c80912] at e_1316_15
  have hr := e_1316_15
  simp only [e_1316_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12286)) := by
  have c80913 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80914 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80915 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80916 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80917 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80918 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_16 := arithEq_of_rows h (row := 1316) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_16
  simp only [← c80913, ← c80914, ← c80915] at e_1316_16
  have e_1316_17 := arithEq_of_rows h (row := 1316) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_17
  simp only [← c80916, ← c80917, k2, ← c80918] at e_1316_17
  have hr := e_1316_17
  simp only [e_1316_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1316 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12287)) := by
  have c80919 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80920 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80921 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80922 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80923 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80924 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1316_18 := arithEq_of_rows h (row := 1316) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_18
  simp only [← c80919, ← c80920, ← c80921] at e_1316_18
  have e_1316_19 := arithEq_of_rows h (row := 1316) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_19
  simp only [← c80922, ← c80923, k2, ← c80924] at e_1316_19
  have hr := e_1316_19
  simp only [e_1316_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12288)) := by
  have c80925 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80926 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80927 := (publicBatchWrapper86_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80928 := (publicBatchWrapper86_copies2529 a h).1
  have c80929 := (publicBatchWrapper86_copies2529 a h).2.1
  have c80930 := (publicBatchWrapper86_copies2529 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_0 := arithEq_of_rows h (row := 1317) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_0
  simp only [← c80925, ← c80926, ← c80927] at e_1317_0
  have e_1317_1 := arithEq_of_rows h (row := 1317) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_1
  simp only [← c80928, ← c80929, k2, ← c80930] at e_1317_1
  have hr := e_1317_1
  simp only [e_1317_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12289)) := by
  have c80931 := (publicBatchWrapper86_copies2529 a h).2.2.2.1
  have c80932 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.1
  have c80933 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.1
  have c80934 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.1
  have c80935 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.1
  have c80936 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_2 := arithEq_of_rows h (row := 1317) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_2
  simp only [← c80931, ← c80932, ← c80933] at e_1317_2
  have e_1317_3 := arithEq_of_rows h (row := 1317) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_3
  simp only [← c80934, ← c80935, k2, ← c80936] at e_1317_3
  have hr := e_1317_3
  simp only [e_1317_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12290)) := by
  have c80937 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.1
  have c80938 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80939 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80940 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80941 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80942 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_4 := arithEq_of_rows h (row := 1317) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_4
  simp only [← c80937, ← c80938, ← c80939] at e_1317_4
  have e_1317_5 := arithEq_of_rows h (row := 1317) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_5
  simp only [← c80940, ← c80941, k2, ← c80942] at e_1317_5
  have hr := e_1317_5
  simp only [e_1317_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12291)) := by
  have c80943 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80944 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80945 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80946 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80947 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80948 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_6 := arithEq_of_rows h (row := 1317) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_6
  simp only [← c80943, ← c80944, ← c80945] at e_1317_6
  have e_1317_7 := arithEq_of_rows h (row := 1317) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_7
  simp only [← c80946, ← c80947, k2, ← c80948] at e_1317_7
  have hr := e_1317_7
  simp only [e_1317_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12292)) := by
  have c80949 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80950 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80951 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80952 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80953 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80954 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_8 := arithEq_of_rows h (row := 1317) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_8
  simp only [← c80949, ← c80950, ← c80951] at e_1317_8
  have e_1317_9 := arithEq_of_rows h (row := 1317) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_9
  simp only [← c80952, ← c80953, k2, ← c80954] at e_1317_9
  have hr := e_1317_9
  simp only [e_1317_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12293)) := by
  have c80955 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80956 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80957 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80958 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80959 := (publicBatchWrapper86_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80960 := (publicBatchWrapper86_copies2530 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_10 := arithEq_of_rows h (row := 1317) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_10
  simp only [← c80955, ← c80956, ← c80957] at e_1317_10
  have e_1317_11 := arithEq_of_rows h (row := 1317) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_11
  simp only [← c80958, ← c80959, k2, ← c80960] at e_1317_11
  have hr := e_1317_11
  simp only [e_1317_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12294)) := by
  have c80961 := (publicBatchWrapper86_copies2530 a h).2.1
  have c80962 := (publicBatchWrapper86_copies2530 a h).2.2.1
  have c80963 := (publicBatchWrapper86_copies2530 a h).2.2.2.1
  have c80964 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.1
  have c80965 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.1
  have c80966 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_12 := arithEq_of_rows h (row := 1317) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_12
  simp only [← c80961, ← c80962, ← c80963] at e_1317_12
  have e_1317_13 := arithEq_of_rows h (row := 1317) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_13
  simp only [← c80964, ← c80965, k2, ← c80966] at e_1317_13
  have hr := e_1317_13
  simp only [e_1317_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12295)) := by
  have c80967 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.1
  have c80968 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.1
  have c80969 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.1
  have c80970 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80971 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80972 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_14 := arithEq_of_rows h (row := 1317) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_14
  simp only [← c80967, ← c80968, ← c80969] at e_1317_14
  have e_1317_15 := arithEq_of_rows h (row := 1317) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_15
  simp only [← c80970, ← c80971, k2, ← c80972] at e_1317_15
  have hr := e_1317_15
  simp only [e_1317_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12296)) := by
  have c80973 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80974 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80975 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80976 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80977 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80978 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_16 := arithEq_of_rows h (row := 1317) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_16
  simp only [← c80973, ← c80974, ← c80975] at e_1317_16
  have e_1317_17 := arithEq_of_rows h (row := 1317) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_17
  simp only [← c80976, ← c80977, k2, ← c80978] at e_1317_17
  have hr := e_1317_17
  simp only [e_1317_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1317 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12297)) := by
  have c80979 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80980 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80981 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80982 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80983 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80984 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1317_18 := arithEq_of_rows h (row := 1317) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_18
  simp only [← c80979, ← c80980, ← c80981] at e_1317_18
  have e_1317_19 := arithEq_of_rows h (row := 1317) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_19
  simp only [← c80982, ← c80983, k2, ← c80984] at e_1317_19
  have hr := e_1317_19
  simp only [e_1317_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12298)) := by
  have c80985 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80986 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80987 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80988 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80989 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80990 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_0 := arithEq_of_rows h (row := 1318) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_0
  simp only [← c80985, ← c80986, ← c80987] at e_1318_0
  have e_1318_1 := arithEq_of_rows h (row := 1318) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_1
  simp only [← c80988, ← c80989, k2, ← c80990] at e_1318_1
  have hr := e_1318_1
  simp only [e_1318_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12299)) := by
  have c80991 := (publicBatchWrapper86_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80992 := (publicBatchWrapper86_copies2531 a h).1
  have c80993 := (publicBatchWrapper86_copies2531 a h).2.1
  have c80994 := (publicBatchWrapper86_copies2531 a h).2.2.1
  have c80995 := (publicBatchWrapper86_copies2531 a h).2.2.2.1
  have c80996 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_2 := arithEq_of_rows h (row := 1318) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_2
  simp only [← c80991, ← c80992, ← c80993] at e_1318_2
  have e_1318_3 := arithEq_of_rows h (row := 1318) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_3
  simp only [← c80994, ← c80995, k2, ← c80996] at e_1318_3
  have hr := e_1318_3
  simp only [e_1318_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12300)) := by
  have c80997 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.1
  have c80998 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.1
  have c80999 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.1
  have c81000 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.1
  have c81001 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.1
  have c81002 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_4 := arithEq_of_rows h (row := 1318) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_4
  simp only [← c80997, ← c80998, ← c80999] at e_1318_4
  have e_1318_5 := arithEq_of_rows h (row := 1318) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_5
  simp only [← c81000, ← c81001, k2, ← c81002] at e_1318_5
  have hr := e_1318_5
  simp only [e_1318_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12301)) := by
  have c81003 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81004 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81005 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81006 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81007 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81008 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_6 := arithEq_of_rows h (row := 1318) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_6
  simp only [← c81003, ← c81004, ← c81005] at e_1318_6
  have e_1318_7 := arithEq_of_rows h (row := 1318) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_7
  simp only [← c81006, ← c81007, k2, ← c81008] at e_1318_7
  have hr := e_1318_7
  simp only [e_1318_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12302)) := by
  have c81009 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81010 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81011 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81012 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81013 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81014 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_8 := arithEq_of_rows h (row := 1318) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_8
  simp only [← c81009, ← c81010, ← c81011] at e_1318_8
  have e_1318_9 := arithEq_of_rows h (row := 1318) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_9
  simp only [← c81012, ← c81013, k2, ← c81014] at e_1318_9
  have hr := e_1318_9
  simp only [e_1318_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12303)) := by
  have c81015 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81016 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81017 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81018 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81019 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81020 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_10 := arithEq_of_rows h (row := 1318) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_10
  simp only [← c81015, ← c81016, ← c81017] at e_1318_10
  have e_1318_11 := arithEq_of_rows h (row := 1318) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_11
  simp only [← c81018, ← c81019, k2, ← c81020] at e_1318_11
  have hr := e_1318_11
  simp only [e_1318_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12304)) := by
  have c81021 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81022 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81023 := (publicBatchWrapper86_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81024 := (publicBatchWrapper86_copies2532 a h).1
  have c81025 := (publicBatchWrapper86_copies2532 a h).2.1
  have c81026 := (publicBatchWrapper86_copies2532 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_12 := arithEq_of_rows h (row := 1318) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_12
  simp only [← c81021, ← c81022, ← c81023] at e_1318_12
  have e_1318_13 := arithEq_of_rows h (row := 1318) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_13
  simp only [← c81024, ← c81025, k2, ← c81026] at e_1318_13
  have hr := e_1318_13
  simp only [e_1318_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12305)) := by
  have c81027 := (publicBatchWrapper86_copies2532 a h).2.2.2.1
  have c81028 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.1
  have c81029 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.1
  have c81030 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.1
  have c81031 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.1
  have c81032 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_14 := arithEq_of_rows h (row := 1318) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_14
  simp only [← c81027, ← c81028, ← c81029] at e_1318_14
  have e_1318_15 := arithEq_of_rows h (row := 1318) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_15
  simp only [← c81030, ← c81031, k2, ← c81032] at e_1318_15
  have hr := e_1318_15
  simp only [e_1318_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12306)) := by
  have c81033 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.1
  have c81034 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81035 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81036 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81037 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81038 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_16 := arithEq_of_rows h (row := 1318) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_16
  simp only [← c81033, ← c81034, ← c81035] at e_1318_16
  have e_1318_17 := arithEq_of_rows h (row := 1318) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_17
  simp only [← c81036, ← c81037, k2, ← c81038] at e_1318_17
  have hr := e_1318_17
  simp only [e_1318_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1318 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12307)) := by
  have c81039 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81040 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81041 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81042 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81043 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81044 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1318_18 := arithEq_of_rows h (row := 1318) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_18
  simp only [← c81039, ← c81040, ← c81041] at e_1318_18
  have e_1318_19 := arithEq_of_rows h (row := 1318) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_19
  simp only [← c81042, ← c81043, k2, ← c81044] at e_1318_19
  have hr := e_1318_19
  simp only [e_1318_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12308)) := by
  have c81045 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81046 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81047 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81048 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81049 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81050 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_0 := arithEq_of_rows h (row := 1319) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_0
  simp only [← c81045, ← c81046, ← c81047] at e_1319_0
  have e_1319_1 := arithEq_of_rows h (row := 1319) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_1
  simp only [← c81048, ← c81049, k2, ← c81050] at e_1319_1
  have hr := e_1319_1
  simp only [e_1319_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12309)) := by
  have c81051 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81052 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81053 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81054 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81055 := (publicBatchWrapper86_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81056 := (publicBatchWrapper86_copies2533 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_2 := arithEq_of_rows h (row := 1319) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_2
  simp only [← c81051, ← c81052, ← c81053] at e_1319_2
  have e_1319_3 := arithEq_of_rows h (row := 1319) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_3
  simp only [← c81054, ← c81055, k2, ← c81056] at e_1319_3
  have hr := e_1319_3
  simp only [e_1319_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12310)) := by
  have c81057 := (publicBatchWrapper86_copies2533 a h).2.1
  have c81058 := (publicBatchWrapper86_copies2533 a h).2.2.1
  have c81059 := (publicBatchWrapper86_copies2533 a h).2.2.2.1
  have c81060 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.1
  have c81061 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.1
  have c81062 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_4 := arithEq_of_rows h (row := 1319) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_4
  simp only [← c81057, ← c81058, ← c81059] at e_1319_4
  have e_1319_5 := arithEq_of_rows h (row := 1319) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_5
  simp only [← c81060, ← c81061, k2, ← c81062] at e_1319_5
  have hr := e_1319_5
  simp only [e_1319_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12311)) := by
  have c81063 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.1
  have c81064 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.1
  have c81065 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.1
  have c81066 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81067 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81068 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_6 := arithEq_of_rows h (row := 1319) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_6
  simp only [← c81063, ← c81064, ← c81065] at e_1319_6
  have e_1319_7 := arithEq_of_rows h (row := 1319) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_7
  simp only [← c81066, ← c81067, k2, ← c81068] at e_1319_7
  have hr := e_1319_7
  simp only [e_1319_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12312)) := by
  have c81069 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81070 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81071 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81072 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81073 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81074 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_8 := arithEq_of_rows h (row := 1319) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_8
  simp only [← c81069, ← c81070, ← c81071] at e_1319_8
  have e_1319_9 := arithEq_of_rows h (row := 1319) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_9
  simp only [← c81072, ← c81073, k2, ← c81074] at e_1319_9
  have hr := e_1319_9
  simp only [e_1319_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12313)) := by
  have c81075 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81076 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81077 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81078 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81079 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81080 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_10 := arithEq_of_rows h (row := 1319) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_10
  simp only [← c81075, ← c81076, ← c81077] at e_1319_10
  have e_1319_11 := arithEq_of_rows h (row := 1319) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_11
  simp only [← c81078, ← c81079, k2, ← c81080] at e_1319_11
  have hr := e_1319_11
  simp only [e_1319_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12314)) := by
  have c81081 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81082 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81083 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81084 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81085 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81086 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_12 := arithEq_of_rows h (row := 1319) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_12
  simp only [← c81081, ← c81082, ← c81083] at e_1319_12
  have e_1319_13 := arithEq_of_rows h (row := 1319) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_13
  simp only [← c81084, ← c81085, k2, ← c81086] at e_1319_13
  have hr := e_1319_13
  simp only [e_1319_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12315)) := by
  have c81087 := (publicBatchWrapper86_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81088 := (publicBatchWrapper86_copies2534 a h).1
  have c81089 := (publicBatchWrapper86_copies2534 a h).2.1
  have c81090 := (publicBatchWrapper86_copies2534 a h).2.2.1
  have c81091 := (publicBatchWrapper86_copies2534 a h).2.2.2.1
  have c81092 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_14 := arithEq_of_rows h (row := 1319) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_14
  simp only [← c81087, ← c81088, ← c81089] at e_1319_14
  have e_1319_15 := arithEq_of_rows h (row := 1319) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_15
  simp only [← c81090, ← c81091, k2, ← c81092] at e_1319_15
  have hr := e_1319_15
  simp only [e_1319_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12316)) := by
  have c81093 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.1
  have c81094 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.1
  have c81095 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.1
  have c81096 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.1
  have c81097 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.1
  have c81098 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_16 := arithEq_of_rows h (row := 1319) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_16
  simp only [← c81093, ← c81094, ← c81095] at e_1319_16
  have e_1319_17 := arithEq_of_rows h (row := 1319) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_17
  simp only [← c81096, ← c81097, k2, ← c81098] at e_1319_17
  have hr := e_1319_17
  simp only [e_1319_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1319 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12317)) := by
  have c81099 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81100 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81101 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81102 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81103 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81104 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1319_18 := arithEq_of_rows h (row := 1319) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_18
  simp only [← c81099, ← c81100, ← c81101] at e_1319_18
  have e_1319_19 := arithEq_of_rows h (row := 1319) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_19
  simp only [← c81102, ← c81103, k2, ← c81104] at e_1319_19
  have hr := e_1319_19
  simp only [e_1319_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12318)) := by
  have c81105 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81106 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81107 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81108 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81109 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81110 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_0 := arithEq_of_rows h (row := 1320) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_0
  simp only [← c81105, ← c81106, ← c81107] at e_1320_0
  have e_1320_1 := arithEq_of_rows h (row := 1320) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_1
  simp only [← c81108, ← c81109, k2, ← c81110] at e_1320_1
  have hr := e_1320_1
  simp only [e_1320_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12319)) := by
  have c81111 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81112 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81113 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81114 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81115 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81116 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_2 := arithEq_of_rows h (row := 1320) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_2
  simp only [← c81111, ← c81112, ← c81113] at e_1320_2
  have e_1320_3 := arithEq_of_rows h (row := 1320) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_3
  simp only [← c81114, ← c81115, k2, ← c81116] at e_1320_3
  have hr := e_1320_3
  simp only [e_1320_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12320)) := by
  have c81117 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81118 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81119 := (publicBatchWrapper86_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81120 := (publicBatchWrapper86_copies2535 a h).1
  have c81121 := (publicBatchWrapper86_copies2535 a h).2.1
  have c81122 := (publicBatchWrapper86_copies2535 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_4 := arithEq_of_rows h (row := 1320) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_4
  simp only [← c81117, ← c81118, ← c81119] at e_1320_4
  have e_1320_5 := arithEq_of_rows h (row := 1320) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_5
  simp only [← c81120, ← c81121, k2, ← c81122] at e_1320_5
  have hr := e_1320_5
  simp only [e_1320_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12321)) := by
  have c81123 := (publicBatchWrapper86_copies2535 a h).2.2.2.1
  have c81124 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.1
  have c81125 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.1
  have c81126 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.1
  have c81127 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.1
  have c81128 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_6 := arithEq_of_rows h (row := 1320) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_6
  simp only [← c81123, ← c81124, ← c81125] at e_1320_6
  have e_1320_7 := arithEq_of_rows h (row := 1320) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_7
  simp only [← c81126, ← c81127, k2, ← c81128] at e_1320_7
  have hr := e_1320_7
  simp only [e_1320_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12322)) := by
  have c81129 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.1
  have c81130 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81131 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81132 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81133 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81134 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_8 := arithEq_of_rows h (row := 1320) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_8
  simp only [← c81129, ← c81130, ← c81131] at e_1320_8
  have e_1320_9 := arithEq_of_rows h (row := 1320) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_9
  simp only [← c81132, ← c81133, k2, ← c81134] at e_1320_9
  have hr := e_1320_9
  simp only [e_1320_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12323)) := by
  have c81135 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81136 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81137 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81138 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81139 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81140 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_10 := arithEq_of_rows h (row := 1320) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_10
  simp only [← c81135, ← c81136, ← c81137] at e_1320_10
  have e_1320_11 := arithEq_of_rows h (row := 1320) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_11
  simp only [← c81138, ← c81139, k2, ← c81140] at e_1320_11
  have hr := e_1320_11
  simp only [e_1320_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12324)) := by
  have c81141 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81142 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81143 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81144 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81145 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81146 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_12 := arithEq_of_rows h (row := 1320) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_12
  simp only [← c81141, ← c81142, ← c81143] at e_1320_12
  have e_1320_13 := arithEq_of_rows h (row := 1320) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_13
  simp only [← c81144, ← c81145, k2, ← c81146] at e_1320_13
  have hr := e_1320_13
  simp only [e_1320_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12325)) := by
  have c81147 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81148 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81149 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81150 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81151 := (publicBatchWrapper86_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81152 := (publicBatchWrapper86_copies2536 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_14 := arithEq_of_rows h (row := 1320) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_14
  simp only [← c81147, ← c81148, ← c81149] at e_1320_14
  have e_1320_15 := arithEq_of_rows h (row := 1320) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_15
  simp only [← c81150, ← c81151, k2, ← c81152] at e_1320_15
  have hr := e_1320_15
  simp only [e_1320_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12326)) := by
  have c81153 := (publicBatchWrapper86_copies2536 a h).2.1
  have c81154 := (publicBatchWrapper86_copies2536 a h).2.2.1
  have c81155 := (publicBatchWrapper86_copies2536 a h).2.2.2.1
  have c81156 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.1
  have c81157 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.1
  have c81158 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_16 := arithEq_of_rows h (row := 1320) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_16
  simp only [← c81153, ← c81154, ← c81155] at e_1320_16
  have e_1320_17 := arithEq_of_rows h (row := 1320) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_17
  simp only [← c81156, ← c81157, k2, ← c81158] at e_1320_17
  have hr := e_1320_17
  simp only [e_1320_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1320 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12327)) := by
  have c81159 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.1
  have c81160 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.1
  have c81161 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.1
  have c81162 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81163 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81164 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1320_18 := arithEq_of_rows h (row := 1320) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_18
  simp only [← c81159, ← c81160, ← c81161] at e_1320_18
  have e_1320_19 := arithEq_of_rows h (row := 1320) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_19
  simp only [← c81162, ← c81163, k2, ← c81164] at e_1320_19
  have hr := e_1320_19
  simp only [e_1320_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12328)) := by
  have c81165 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81166 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81167 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81168 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81169 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81170 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_0 := arithEq_of_rows h (row := 1321) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_0
  simp only [← c81165, ← c81166, ← c81167] at e_1321_0
  have e_1321_1 := arithEq_of_rows h (row := 1321) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_1
  simp only [← c81168, ← c81169, k2, ← c81170] at e_1321_1
  have hr := e_1321_1
  simp only [e_1321_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12329)) := by
  have c81171 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81172 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81173 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81174 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81175 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81176 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_2 := arithEq_of_rows h (row := 1321) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_2
  simp only [← c81171, ← c81172, ← c81173] at e_1321_2
  have e_1321_3 := arithEq_of_rows h (row := 1321) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_3
  simp only [← c81174, ← c81175, k2, ← c81176] at e_1321_3
  have hr := e_1321_3
  simp only [e_1321_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12330)) := by
  have c81177 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81178 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81179 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81180 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81181 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81182 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_4 := arithEq_of_rows h (row := 1321) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_4
  simp only [← c81177, ← c81178, ← c81179] at e_1321_4
  have e_1321_5 := arithEq_of_rows h (row := 1321) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_5
  simp only [← c81180, ← c81181, k2, ← c81182] at e_1321_5
  have hr := e_1321_5
  simp only [e_1321_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12331)) := by
  have c81183 := (publicBatchWrapper86_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81184 := (publicBatchWrapper86_copies2537 a h).1
  have c81185 := (publicBatchWrapper86_copies2537 a h).2.1
  have c81186 := (publicBatchWrapper86_copies2537 a h).2.2.1
  have c81187 := (publicBatchWrapper86_copies2537 a h).2.2.2.1
  have c81188 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_6 := arithEq_of_rows h (row := 1321) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_6
  simp only [← c81183, ← c81184, ← c81185] at e_1321_6
  have e_1321_7 := arithEq_of_rows h (row := 1321) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_7
  simp only [← c81186, ← c81187, k2, ← c81188] at e_1321_7
  have hr := e_1321_7
  simp only [e_1321_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12332)) := by
  have c81189 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.1
  have c81190 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.1
  have c81191 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.1
  have c81192 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.1
  have c81193 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.1
  have c81194 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_8 := arithEq_of_rows h (row := 1321) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_8
  simp only [← c81189, ← c81190, ← c81191] at e_1321_8
  have e_1321_9 := arithEq_of_rows h (row := 1321) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_9
  simp only [← c81192, ← c81193, k2, ← c81194] at e_1321_9
  have hr := e_1321_9
  simp only [e_1321_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12333)) := by
  have c81195 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81196 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81197 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81198 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81199 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81200 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_10 := arithEq_of_rows h (row := 1321) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_10
  simp only [← c81195, ← c81196, ← c81197] at e_1321_10
  have e_1321_11 := arithEq_of_rows h (row := 1321) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_11
  simp only [← c81198, ← c81199, k2, ← c81200] at e_1321_11
  have hr := e_1321_11
  simp only [e_1321_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12334)) := by
  have c81201 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81202 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81203 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81204 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81205 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81206 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_12 := arithEq_of_rows h (row := 1321) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_12
  simp only [← c81201, ← c81202, ← c81203] at e_1321_12
  have e_1321_13 := arithEq_of_rows h (row := 1321) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_13
  simp only [← c81204, ← c81205, k2, ← c81206] at e_1321_13
  have hr := e_1321_13
  simp only [e_1321_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12335)) := by
  have c81207 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81208 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81209 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81210 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81211 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81212 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_14 := arithEq_of_rows h (row := 1321) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_14
  simp only [← c81207, ← c81208, ← c81209] at e_1321_14
  have e_1321_15 := arithEq_of_rows h (row := 1321) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_15
  simp only [← c81210, ← c81211, k2, ← c81212] at e_1321_15
  have hr := e_1321_15
  simp only [e_1321_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12336)) := by
  have c81213 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81214 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81215 := (publicBatchWrapper86_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81216 := (publicBatchWrapper86_copies2538 a h).1
  have c81217 := (publicBatchWrapper86_copies2538 a h).2.1
  have c81218 := (publicBatchWrapper86_copies2538 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_16 := arithEq_of_rows h (row := 1321) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_16
  simp only [← c81213, ← c81214, ← c81215] at e_1321_16
  have e_1321_17 := arithEq_of_rows h (row := 1321) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_17
  simp only [← c81216, ← c81217, k2, ← c81218] at e_1321_17
  have hr := e_1321_17
  simp only [e_1321_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1321 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12337)) := by
  have c81219 := (publicBatchWrapper86_copies2538 a h).2.2.2.1
  have c81220 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.1
  have c81221 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.1
  have c81222 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.1
  have c81223 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.1
  have c81224 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1321_18 := arithEq_of_rows h (row := 1321) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_18
  simp only [← c81219, ← c81220, ← c81221] at e_1321_18
  have e_1321_19 := arithEq_of_rows h (row := 1321) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_19
  simp only [← c81222, ← c81223, k2, ← c81224] at e_1321_19
  have hr := e_1321_19
  simp only [e_1321_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12338)) := by
  have c81225 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.1
  have c81226 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81227 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81228 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81229 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81230 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_0 := arithEq_of_rows h (row := 1322) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_0
  simp only [← c81225, ← c81226, ← c81227] at e_1322_0
  have e_1322_1 := arithEq_of_rows h (row := 1322) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_1
  simp only [← c81228, ← c81229, k2, ← c81230] at e_1322_1
  have hr := e_1322_1
  simp only [e_1322_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12339)) := by
  have c81231 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81232 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81233 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81234 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81235 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81236 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_2 := arithEq_of_rows h (row := 1322) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_2
  simp only [← c81231, ← c81232, ← c81233] at e_1322_2
  have e_1322_3 := arithEq_of_rows h (row := 1322) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_3
  simp only [← c81234, ← c81235, k2, ← c81236] at e_1322_3
  have hr := e_1322_3
  simp only [e_1322_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12340)) := by
  have c81237 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81238 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81239 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81240 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81241 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81242 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_4 := arithEq_of_rows h (row := 1322) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_4
  simp only [← c81237, ← c81238, ← c81239] at e_1322_4
  have e_1322_5 := arithEq_of_rows h (row := 1322) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_5
  simp only [← c81240, ← c81241, k2, ← c81242] at e_1322_5
  have hr := e_1322_5
  simp only [e_1322_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12341)) := by
  have c81243 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81244 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81245 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81246 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81247 := (publicBatchWrapper86_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81248 := (publicBatchWrapper86_copies2539 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_6 := arithEq_of_rows h (row := 1322) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_6
  simp only [← c81243, ← c81244, ← c81245] at e_1322_6
  have e_1322_7 := arithEq_of_rows h (row := 1322) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_7
  simp only [← c81246, ← c81247, k2, ← c81248] at e_1322_7
  have hr := e_1322_7
  simp only [e_1322_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12342)) := by
  have c81249 := (publicBatchWrapper86_copies2539 a h).2.1
  have c81250 := (publicBatchWrapper86_copies2539 a h).2.2.1
  have c81251 := (publicBatchWrapper86_copies2539 a h).2.2.2.1
  have c81252 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.1
  have c81253 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.1
  have c81254 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_8 := arithEq_of_rows h (row := 1322) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_8
  simp only [← c81249, ← c81250, ← c81251] at e_1322_8
  have e_1322_9 := arithEq_of_rows h (row := 1322) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_9
  simp only [← c81252, ← c81253, k2, ← c81254] at e_1322_9
  have hr := e_1322_9
  simp only [e_1322_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12343)) := by
  have c81255 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.1
  have c81256 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.1
  have c81257 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.1
  have c81258 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81259 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81260 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_10 := arithEq_of_rows h (row := 1322) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_10
  simp only [← c81255, ← c81256, ← c81257] at e_1322_10
  have e_1322_11 := arithEq_of_rows h (row := 1322) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_11
  simp only [← c81258, ← c81259, k2, ← c81260] at e_1322_11
  have hr := e_1322_11
  simp only [e_1322_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12344)) := by
  have c81261 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81262 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81263 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81264 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81265 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81266 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_12 := arithEq_of_rows h (row := 1322) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_12
  simp only [← c81261, ← c81262, ← c81263] at e_1322_12
  have e_1322_13 := arithEq_of_rows h (row := 1322) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_13
  simp only [← c81264, ← c81265, k2, ← c81266] at e_1322_13
  have hr := e_1322_13
  simp only [e_1322_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12345)) := by
  have c81267 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81268 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81269 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81270 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81271 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81272 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_14 := arithEq_of_rows h (row := 1322) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_14
  simp only [← c81267, ← c81268, ← c81269] at e_1322_14
  have e_1322_15 := arithEq_of_rows h (row := 1322) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_15
  simp only [← c81270, ← c81271, k2, ← c81272] at e_1322_15
  have hr := e_1322_15
  simp only [e_1322_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12346)) := by
  have c81273 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81274 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81275 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81276 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81277 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81278 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_16 := arithEq_of_rows h (row := 1322) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_16
  simp only [← c81273, ← c81274, ← c81275] at e_1322_16
  have e_1322_17 := arithEq_of_rows h (row := 1322) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_17
  simp only [← c81276, ← c81277, k2, ← c81278] at e_1322_17
  have hr := e_1322_17
  simp only [e_1322_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1322 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12347)) := by
  have c81279 := (publicBatchWrapper86_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81280 := (publicBatchWrapper86_copies2540 a h).1
  have c81281 := (publicBatchWrapper86_copies2540 a h).2.1
  have c81282 := (publicBatchWrapper86_copies2540 a h).2.2.1
  have c81283 := (publicBatchWrapper86_copies2540 a h).2.2.2.1
  have c81284 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1322_18 := arithEq_of_rows h (row := 1322) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_18
  simp only [← c81279, ← c81280, ← c81281] at e_1322_18
  have e_1322_19 := arithEq_of_rows h (row := 1322) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_19
  simp only [← c81282, ← c81283, k2, ← c81284] at e_1322_19
  have hr := e_1322_19
  simp only [e_1322_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12348)) := by
  have c81285 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.1
  have c81286 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.1
  have c81287 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.1
  have c81288 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.1
  have c81289 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.1
  have c81290 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_0 := arithEq_of_rows h (row := 1323) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_0
  simp only [← c81285, ← c81286, ← c81287] at e_1323_0
  have e_1323_1 := arithEq_of_rows h (row := 1323) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_1
  simp only [← c81288, ← c81289, k2, ← c81290] at e_1323_1
  have hr := e_1323_1
  simp only [e_1323_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12349)) := by
  have c81291 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81292 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81293 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81294 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81295 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81296 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_2 := arithEq_of_rows h (row := 1323) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_2
  simp only [← c81291, ← c81292, ← c81293] at e_1323_2
  have e_1323_3 := arithEq_of_rows h (row := 1323) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_3
  simp only [← c81294, ← c81295, k2, ← c81296] at e_1323_3
  have hr := e_1323_3
  simp only [e_1323_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12350)) := by
  have c81297 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81298 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81299 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81300 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81301 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81302 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_4 := arithEq_of_rows h (row := 1323) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_4
  simp only [← c81297, ← c81298, ← c81299] at e_1323_4
  have e_1323_5 := arithEq_of_rows h (row := 1323) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_5
  simp only [← c81300, ← c81301, k2, ← c81302] at e_1323_5
  have hr := e_1323_5
  simp only [e_1323_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12351)) := by
  have c81303 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81304 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81305 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81306 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81307 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81308 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_6 := arithEq_of_rows h (row := 1323) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_6
  simp only [← c81303, ← c81304, ← c81305] at e_1323_6
  have e_1323_7 := arithEq_of_rows h (row := 1323) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_7
  simp only [← c81306, ← c81307, k2, ← c81308] at e_1323_7
  have hr := e_1323_7
  simp only [e_1323_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12352)) := by
  have c81309 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81310 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81311 := (publicBatchWrapper86_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81312 := (publicBatchWrapper86_copies2541 a h).1
  have c81313 := (publicBatchWrapper86_copies2541 a h).2.1
  have c81314 := (publicBatchWrapper86_copies2541 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_8 := arithEq_of_rows h (row := 1323) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_8
  simp only [← c81309, ← c81310, ← c81311] at e_1323_8
  have e_1323_9 := arithEq_of_rows h (row := 1323) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_9
  simp only [← c81312, ← c81313, k2, ← c81314] at e_1323_9
  have hr := e_1323_9
  simp only [e_1323_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12353)) := by
  have c81315 := (publicBatchWrapper86_copies2541 a h).2.2.2.1
  have c81316 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.1
  have c81317 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.1
  have c81318 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.1
  have c81319 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.1
  have c81320 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_10 := arithEq_of_rows h (row := 1323) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_10
  simp only [← c81315, ← c81316, ← c81317] at e_1323_10
  have e_1323_11 := arithEq_of_rows h (row := 1323) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_11
  simp only [← c81318, ← c81319, k2, ← c81320] at e_1323_11
  have hr := e_1323_11
  simp only [e_1323_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12354)) := by
  have c81321 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.1
  have c81322 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81323 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81324 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81325 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81326 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_12 := arithEq_of_rows h (row := 1323) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_12
  simp only [← c81321, ← c81322, ← c81323] at e_1323_12
  have e_1323_13 := arithEq_of_rows h (row := 1323) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_13
  simp only [← c81324, ← c81325, k2, ← c81326] at e_1323_13
  have hr := e_1323_13
  simp only [e_1323_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12355)) := by
  have c81327 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81328 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81329 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81330 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81331 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81332 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_14 := arithEq_of_rows h (row := 1323) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_14
  simp only [← c81327, ← c81328, ← c81329] at e_1323_14
  have e_1323_15 := arithEq_of_rows h (row := 1323) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_15
  simp only [← c81330, ← c81331, k2, ← c81332] at e_1323_15
  have hr := e_1323_15
  simp only [e_1323_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12356)) := by
  have c81333 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81334 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81335 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81336 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81337 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81338 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_16 := arithEq_of_rows h (row := 1323) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_16
  simp only [← c81333, ← c81334, ← c81335] at e_1323_16
  have e_1323_17 := arithEq_of_rows h (row := 1323) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_17
  simp only [← c81336, ← c81337, k2, ← c81338] at e_1323_17
  have hr := e_1323_17
  simp only [e_1323_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1323 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12357)) := by
  have c81339 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81340 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81341 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81342 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81343 := (publicBatchWrapper86_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81344 := (publicBatchWrapper86_copies2542 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1323_18 := arithEq_of_rows h (row := 1323) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_18
  simp only [← c81339, ← c81340, ← c81341] at e_1323_18
  have e_1323_19 := arithEq_of_rows h (row := 1323) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_19
  simp only [← c81342, ← c81343, k2, ← c81344] at e_1323_19
  have hr := e_1323_19
  simp only [e_1323_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12358)) := by
  have c81345 := (publicBatchWrapper86_copies2542 a h).2.1
  have c81346 := (publicBatchWrapper86_copies2542 a h).2.2.1
  have c81347 := (publicBatchWrapper86_copies2542 a h).2.2.2.1
  have c81348 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.1
  have c81349 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.1
  have c81350 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_0 := arithEq_of_rows h (row := 1324) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_0
  simp only [← c81345, ← c81346, ← c81347] at e_1324_0
  have e_1324_1 := arithEq_of_rows h (row := 1324) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_1
  simp only [← c81348, ← c81349, k2, ← c81350] at e_1324_1
  have hr := e_1324_1
  simp only [e_1324_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12359)) := by
  have c81351 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.1
  have c81352 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.1
  have c81353 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.1
  have c81354 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81355 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81356 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_2 := arithEq_of_rows h (row := 1324) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_2
  simp only [← c81351, ← c81352, ← c81353] at e_1324_2
  have e_1324_3 := arithEq_of_rows h (row := 1324) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_3
  simp only [← c81354, ← c81355, k2, ← c81356] at e_1324_3
  have hr := e_1324_3
  simp only [e_1324_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12360)) := by
  have c81357 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81358 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81359 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81360 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81361 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81362 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_4 := arithEq_of_rows h (row := 1324) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_4
  simp only [← c81357, ← c81358, ← c81359] at e_1324_4
  have e_1324_5 := arithEq_of_rows h (row := 1324) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_5
  simp only [← c81360, ← c81361, k2, ← c81362] at e_1324_5
  have hr := e_1324_5
  simp only [e_1324_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12361)) := by
  have c81363 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81364 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81365 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81366 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81367 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81368 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_6 := arithEq_of_rows h (row := 1324) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_6
  simp only [← c81363, ← c81364, ← c81365] at e_1324_6
  have e_1324_7 := arithEq_of_rows h (row := 1324) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_7
  simp only [← c81366, ← c81367, k2, ← c81368] at e_1324_7
  have hr := e_1324_7
  simp only [e_1324_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12362)) := by
  have c81369 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81370 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81371 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81372 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81373 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81374 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_8 := arithEq_of_rows h (row := 1324) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_8
  simp only [← c81369, ← c81370, ← c81371] at e_1324_8
  have e_1324_9 := arithEq_of_rows h (row := 1324) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_9
  simp only [← c81372, ← c81373, k2, ← c81374] at e_1324_9
  have hr := e_1324_9
  simp only [e_1324_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12363)) := by
  have c81375 := (publicBatchWrapper86_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81376 := (publicBatchWrapper86_copies2543 a h).1
  have c81377 := (publicBatchWrapper86_copies2543 a h).2.1
  have c81378 := (publicBatchWrapper86_copies2543 a h).2.2.1
  have c81379 := (publicBatchWrapper86_copies2543 a h).2.2.2.1
  have c81380 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_10 := arithEq_of_rows h (row := 1324) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_10
  simp only [← c81375, ← c81376, ← c81377] at e_1324_10
  have e_1324_11 := arithEq_of_rows h (row := 1324) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_11
  simp only [← c81378, ← c81379, k2, ← c81380] at e_1324_11
  have hr := e_1324_11
  simp only [e_1324_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12364)) := by
  have c81381 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.1
  have c81382 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.1
  have c81383 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.1
  have c81384 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.1
  have c81385 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.1
  have c81386 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_12 := arithEq_of_rows h (row := 1324) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_12
  simp only [← c81381, ← c81382, ← c81383] at e_1324_12
  have e_1324_13 := arithEq_of_rows h (row := 1324) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_13
  simp only [← c81384, ← c81385, k2, ← c81386] at e_1324_13
  have hr := e_1324_13
  simp only [e_1324_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12365)) := by
  have c81387 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81388 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81389 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81390 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81391 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81392 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_14 := arithEq_of_rows h (row := 1324) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_14
  simp only [← c81387, ← c81388, ← c81389] at e_1324_14
  have e_1324_15 := arithEq_of_rows h (row := 1324) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_15
  simp only [← c81390, ← c81391, k2, ← c81392] at e_1324_15
  have hr := e_1324_15
  simp only [e_1324_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12366)) := by
  have c81393 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81394 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81395 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81396 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81397 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81398 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_16 := arithEq_of_rows h (row := 1324) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_16
  simp only [← c81393, ← c81394, ← c81395] at e_1324_16
  have e_1324_17 := arithEq_of_rows h (row := 1324) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_17
  simp only [← c81396, ← c81397, k2, ← c81398] at e_1324_17
  have hr := e_1324_17
  simp only [e_1324_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1324 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12367)) := by
  have c81399 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81400 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81401 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81402 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81403 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81404 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1324_18 := arithEq_of_rows h (row := 1324) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_18
  simp only [← c81399, ← c81400, ← c81401] at e_1324_18
  have e_1324_19 := arithEq_of_rows h (row := 1324) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_19
  simp only [← c81402, ← c81403, k2, ← c81404] at e_1324_19
  have hr := e_1324_19
  simp only [e_1324_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12368)) := by
  have c81405 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81406 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81407 := (publicBatchWrapper86_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81408 := (publicBatchWrapper86_copies2544 a h).1
  have c81409 := (publicBatchWrapper86_copies2544 a h).2.1
  have c81410 := (publicBatchWrapper86_copies2544 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_0 := arithEq_of_rows h (row := 1325) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_0
  simp only [← c81405, ← c81406, ← c81407] at e_1325_0
  have e_1325_1 := arithEq_of_rows h (row := 1325) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_1
  simp only [← c81408, ← c81409, k2, ← c81410] at e_1325_1
  have hr := e_1325_1
  simp only [e_1325_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12369)) := by
  have c81411 := (publicBatchWrapper86_copies2544 a h).2.2.2.1
  have c81412 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.1
  have c81413 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.1
  have c81414 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.1
  have c81415 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.1
  have c81416 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_2 := arithEq_of_rows h (row := 1325) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_2
  simp only [← c81411, ← c81412, ← c81413] at e_1325_2
  have e_1325_3 := arithEq_of_rows h (row := 1325) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_3
  simp only [← c81414, ← c81415, k2, ← c81416] at e_1325_3
  have hr := e_1325_3
  simp only [e_1325_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12410)) := by
  have c81417 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.1
  have c81418 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81419 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81420 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81421 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81422 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_4 := arithEq_of_rows h (row := 1325) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_4
  simp only [← c81417, ← c81418, ← c81419] at e_1325_4
  have e_1325_5 := arithEq_of_rows h (row := 1325) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_5
  simp only [← c81420, ← c81421, k2, ← c81422] at e_1325_5
  have hr := e_1325_5
  simp only [e_1325_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12411)) := by
  have c81423 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81424 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81425 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81426 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81427 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81428 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_6 := arithEq_of_rows h (row := 1325) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_6
  simp only [← c81423, ← c81424, ← c81425] at e_1325_6
  have e_1325_7 := arithEq_of_rows h (row := 1325) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_7
  simp only [← c81426, ← c81427, k2, ← c81428] at e_1325_7
  have hr := e_1325_7
  simp only [e_1325_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12412)) := by
  have c81429 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81430 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81431 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81432 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81433 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81434 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_8 := arithEq_of_rows h (row := 1325) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_8
  simp only [← c81429, ← c81430, ← c81431] at e_1325_8
  have e_1325_9 := arithEq_of_rows h (row := 1325) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_9
  simp only [← c81432, ← c81433, k2, ← c81434] at e_1325_9
  have hr := e_1325_9
  simp only [e_1325_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12413)) := by
  have c81435 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81436 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81437 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81438 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81439 := (publicBatchWrapper86_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81440 := (publicBatchWrapper86_copies2545 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_10 := arithEq_of_rows h (row := 1325) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_10
  simp only [← c81435, ← c81436, ← c81437] at e_1325_10
  have e_1325_11 := arithEq_of_rows h (row := 1325) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_11
  simp only [← c81438, ← c81439, k2, ← c81440] at e_1325_11
  have hr := e_1325_11
  simp only [e_1325_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12414)) := by
  have c81441 := (publicBatchWrapper86_copies2545 a h).2.1
  have c81442 := (publicBatchWrapper86_copies2545 a h).2.2.1
  have c81443 := (publicBatchWrapper86_copies2545 a h).2.2.2.1
  have c81444 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.1
  have c81445 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.1
  have c81446 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_12 := arithEq_of_rows h (row := 1325) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_12
  simp only [← c81441, ← c81442, ← c81443] at e_1325_12
  have e_1325_13 := arithEq_of_rows h (row := 1325) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_13
  simp only [← c81444, ← c81445, k2, ← c81446] at e_1325_13
  have hr := e_1325_13
  simp only [e_1325_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12415)) := by
  have c81447 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.1
  have c81448 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.1
  have c81449 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.1
  have c81450 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81451 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81452 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_14 := arithEq_of_rows h (row := 1325) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_14
  simp only [← c81447, ← c81448, ← c81449] at e_1325_14
  have e_1325_15 := arithEq_of_rows h (row := 1325) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_15
  simp only [← c81450, ← c81451, k2, ← c81452] at e_1325_15
  have hr := e_1325_15
  simp only [e_1325_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12416)) := by
  have c81453 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81454 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81455 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81456 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81457 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81458 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_16 := arithEq_of_rows h (row := 1325) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_16
  simp only [← c81453, ← c81454, ← c81455] at e_1325_16
  have e_1325_17 := arithEq_of_rows h (row := 1325) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_17
  simp only [← c81456, ← c81457, k2, ← c81458] at e_1325_17
  have hr := e_1325_17
  simp only [e_1325_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1325 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12417)) := by
  have c81459 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81460 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81461 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81462 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81463 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81464 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1325_18 := arithEq_of_rows h (row := 1325) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_18
  simp only [← c81459, ← c81460, ← c81461] at e_1325_18
  have e_1325_19 := arithEq_of_rows h (row := 1325) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_19
  simp only [← c81462, ← c81463, k2, ← c81464] at e_1325_19
  have hr := e_1325_19
  simp only [e_1325_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12418)) := by
  have c81465 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81466 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81467 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81468 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81469 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81470 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_0 := arithEq_of_rows h (row := 1326) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_0
  simp only [← c81465, ← c81466, ← c81467] at e_1326_0
  have e_1326_1 := arithEq_of_rows h (row := 1326) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_1
  simp only [← c81468, ← c81469, k2, ← c81470] at e_1326_1
  have hr := e_1326_1
  simp only [e_1326_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12419)) := by
  have c81471 := (publicBatchWrapper86_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81472 := (publicBatchWrapper86_copies2546 a h).1
  have c81473 := (publicBatchWrapper86_copies2546 a h).2.1
  have c81474 := (publicBatchWrapper86_copies2546 a h).2.2.1
  have c81475 := (publicBatchWrapper86_copies2546 a h).2.2.2.1
  have c81476 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_2 := arithEq_of_rows h (row := 1326) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_2
  simp only [← c81471, ← c81472, ← c81473] at e_1326_2
  have e_1326_3 := arithEq_of_rows h (row := 1326) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_3
  simp only [← c81474, ← c81475, k2, ← c81476] at e_1326_3
  have hr := e_1326_3
  simp only [e_1326_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12420)) := by
  have c81477 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.1
  have c81478 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.1
  have c81479 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.1
  have c81480 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.1
  have c81481 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.1
  have c81482 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_4 := arithEq_of_rows h (row := 1326) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_4
  simp only [← c81477, ← c81478, ← c81479] at e_1326_4
  have e_1326_5 := arithEq_of_rows h (row := 1326) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_5
  simp only [← c81480, ← c81481, k2, ← c81482] at e_1326_5
  have hr := e_1326_5
  simp only [e_1326_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12421)) := by
  have c81483 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81484 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81485 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81486 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81487 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81488 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_6 := arithEq_of_rows h (row := 1326) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_6
  simp only [← c81483, ← c81484, ← c81485] at e_1326_6
  have e_1326_7 := arithEq_of_rows h (row := 1326) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_7
  simp only [← c81486, ← c81487, k2, ← c81488] at e_1326_7
  have hr := e_1326_7
  simp only [e_1326_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12422)) := by
  have c81489 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81490 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81491 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81492 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81493 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81494 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_8 := arithEq_of_rows h (row := 1326) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_8
  simp only [← c81489, ← c81490, ← c81491] at e_1326_8
  have e_1326_9 := arithEq_of_rows h (row := 1326) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_9
  simp only [← c81492, ← c81493, k2, ← c81494] at e_1326_9
  have hr := e_1326_9
  simp only [e_1326_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12423)) := by
  have c81495 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81496 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81497 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81498 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81499 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81500 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_10 := arithEq_of_rows h (row := 1326) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_10
  simp only [← c81495, ← c81496, ← c81497] at e_1326_10
  have e_1326_11 := arithEq_of_rows h (row := 1326) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_11
  simp only [← c81498, ← c81499, k2, ← c81500] at e_1326_11
  have hr := e_1326_11
  simp only [e_1326_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12424)) := by
  have c81501 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81502 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81503 := (publicBatchWrapper86_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81504 := (publicBatchWrapper86_copies2547 a h).1
  have c81505 := (publicBatchWrapper86_copies2547 a h).2.1
  have c81506 := (publicBatchWrapper86_copies2547 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_12 := arithEq_of_rows h (row := 1326) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_12
  simp only [← c81501, ← c81502, ← c81503] at e_1326_12
  have e_1326_13 := arithEq_of_rows h (row := 1326) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_13
  simp only [← c81504, ← c81505, k2, ← c81506] at e_1326_13
  have hr := e_1326_13
  simp only [e_1326_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12425)) := by
  have c81507 := (publicBatchWrapper86_copies2547 a h).2.2.2.1
  have c81508 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.1
  have c81509 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.1
  have c81510 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.1
  have c81511 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.1
  have c81512 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_14 := arithEq_of_rows h (row := 1326) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_14
  simp only [← c81507, ← c81508, ← c81509] at e_1326_14
  have e_1326_15 := arithEq_of_rows h (row := 1326) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_15
  simp only [← c81510, ← c81511, k2, ← c81512] at e_1326_15
  have hr := e_1326_15
  simp only [e_1326_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12426)) := by
  have c81513 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.1
  have c81514 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81515 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81516 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81517 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81518 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_16 := arithEq_of_rows h (row := 1326) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_16
  simp only [← c81513, ← c81514, ← c81515] at e_1326_16
  have e_1326_17 := arithEq_of_rows h (row := 1326) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_17
  simp only [← c81516, ← c81517, k2, ← c81518] at e_1326_17
  have hr := e_1326_17
  simp only [e_1326_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1326 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12427)) := by
  have c81519 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81520 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81521 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81522 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81523 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81524 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1326_18 := arithEq_of_rows h (row := 1326) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_18
  simp only [← c81519, ← c81520, ← c81521] at e_1326_18
  have e_1326_19 := arithEq_of_rows h (row := 1326) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_19
  simp only [← c81522, ← c81523, k2, ← c81524] at e_1326_19
  have hr := e_1326_19
  simp only [e_1326_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12428)) := by
  have c81525 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81526 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81527 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81528 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81529 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81530 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_0 := arithEq_of_rows h (row := 1327) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_0
  simp only [← c81525, ← c81526, ← c81527] at e_1327_0
  have e_1327_1 := arithEq_of_rows h (row := 1327) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_1
  simp only [← c81528, ← c81529, k2, ← c81530] at e_1327_1
  have hr := e_1327_1
  simp only [e_1327_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12429)) := by
  have c81531 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81532 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81533 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81534 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81535 := (publicBatchWrapper86_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81536 := (publicBatchWrapper86_copies2548 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_2 := arithEq_of_rows h (row := 1327) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_2
  simp only [← c81531, ← c81532, ← c81533] at e_1327_2
  have e_1327_3 := arithEq_of_rows h (row := 1327) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_3
  simp only [← c81534, ← c81535, k2, ← c81536] at e_1327_3
  have hr := e_1327_3
  simp only [e_1327_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12430)) := by
  have c81537 := (publicBatchWrapper86_copies2548 a h).2.1
  have c81538 := (publicBatchWrapper86_copies2548 a h).2.2.1
  have c81539 := (publicBatchWrapper86_copies2548 a h).2.2.2.1
  have c81540 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.1
  have c81541 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.1
  have c81542 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_4 := arithEq_of_rows h (row := 1327) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_4
  simp only [← c81537, ← c81538, ← c81539] at e_1327_4
  have e_1327_5 := arithEq_of_rows h (row := 1327) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_5
  simp only [← c81540, ← c81541, k2, ← c81542] at e_1327_5
  have hr := e_1327_5
  simp only [e_1327_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12431)) := by
  have c81543 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.1
  have c81544 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.1
  have c81545 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.1
  have c81546 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81547 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81548 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_6 := arithEq_of_rows h (row := 1327) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_6
  simp only [← c81543, ← c81544, ← c81545] at e_1327_6
  have e_1327_7 := arithEq_of_rows h (row := 1327) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_7
  simp only [← c81546, ← c81547, k2, ← c81548] at e_1327_7
  have hr := e_1327_7
  simp only [e_1327_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12432)) := by
  have c81549 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81550 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81551 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81552 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81553 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81554 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_8 := arithEq_of_rows h (row := 1327) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_8
  simp only [← c81549, ← c81550, ← c81551] at e_1327_8
  have e_1327_9 := arithEq_of_rows h (row := 1327) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_9
  simp only [← c81552, ← c81553, k2, ← c81554] at e_1327_9
  have hr := e_1327_9
  simp only [e_1327_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12433)) := by
  have c81555 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81556 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81557 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81558 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81559 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81560 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_10 := arithEq_of_rows h (row := 1327) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_10
  simp only [← c81555, ← c81556, ← c81557] at e_1327_10
  have e_1327_11 := arithEq_of_rows h (row := 1327) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_11
  simp only [← c81558, ← c81559, k2, ← c81560] at e_1327_11
  have hr := e_1327_11
  simp only [e_1327_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12434)) := by
  have c81561 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81562 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81563 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81564 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81565 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81566 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_12 := arithEq_of_rows h (row := 1327) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_12
  simp only [← c81561, ← c81562, ← c81563] at e_1327_12
  have e_1327_13 := arithEq_of_rows h (row := 1327) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_13
  simp only [← c81564, ← c81565, k2, ← c81566] at e_1327_13
  have hr := e_1327_13
  simp only [e_1327_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12435)) := by
  have c81567 := (publicBatchWrapper86_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81568 := (publicBatchWrapper86_copies2549 a h).1
  have c81569 := (publicBatchWrapper86_copies2549 a h).2.1
  have c81570 := (publicBatchWrapper86_copies2549 a h).2.2.1
  have c81571 := (publicBatchWrapper86_copies2549 a h).2.2.2.1
  have c81572 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_14 := arithEq_of_rows h (row := 1327) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_14
  simp only [← c81567, ← c81568, ← c81569] at e_1327_14
  have e_1327_15 := arithEq_of_rows h (row := 1327) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_15
  simp only [← c81570, ← c81571, k2, ← c81572] at e_1327_15
  have hr := e_1327_15
  simp only [e_1327_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12436)) := by
  have c81573 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.1
  have c81574 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.1
  have c81575 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.1
  have c81576 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.1
  have c81577 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.1
  have c81578 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_16 := arithEq_of_rows h (row := 1327) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_16
  simp only [← c81573, ← c81574, ← c81575] at e_1327_16
  have e_1327_17 := arithEq_of_rows h (row := 1327) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_17
  simp only [← c81576, ← c81577, k2, ← c81578] at e_1327_17
  have hr := e_1327_17
  simp only [e_1327_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1327 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12437)) := by
  have c81579 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81580 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81581 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81582 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81583 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81584 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1327_18 := arithEq_of_rows h (row := 1327) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_18
  simp only [← c81579, ← c81580, ← c81581] at e_1327_18
  have e_1327_19 := arithEq_of_rows h (row := 1327) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_19
  simp only [← c81582, ← c81583, k2, ← c81584] at e_1327_19
  have hr := e_1327_19
  simp only [e_1327_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12438)) := by
  have c81585 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81586 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81587 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81588 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81589 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81590 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_0 := arithEq_of_rows h (row := 1328) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_0
  simp only [← c81585, ← c81586, ← c81587] at e_1328_0
  have e_1328_1 := arithEq_of_rows h (row := 1328) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_1
  simp only [← c81588, ← c81589, k2, ← c81590] at e_1328_1
  have hr := e_1328_1
  simp only [e_1328_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12439)) := by
  have c81591 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81592 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81593 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81594 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81595 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81596 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_2 := arithEq_of_rows h (row := 1328) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_2
  simp only [← c81591, ← c81592, ← c81593] at e_1328_2
  have e_1328_3 := arithEq_of_rows h (row := 1328) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_3
  simp only [← c81594, ← c81595, k2, ← c81596] at e_1328_3
  have hr := e_1328_3
  simp only [e_1328_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12440)) := by
  have c81597 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81598 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81599 := (publicBatchWrapper86_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81600 := (publicBatchWrapper86_copies2550 a h).1
  have c81601 := (publicBatchWrapper86_copies2550 a h).2.1
  have c81602 := (publicBatchWrapper86_copies2550 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_4 := arithEq_of_rows h (row := 1328) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_4
  simp only [← c81597, ← c81598, ← c81599] at e_1328_4
  have e_1328_5 := arithEq_of_rows h (row := 1328) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_5
  simp only [← c81600, ← c81601, k2, ← c81602] at e_1328_5
  have hr := e_1328_5
  simp only [e_1328_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12441)) := by
  have c81603 := (publicBatchWrapper86_copies2550 a h).2.2.2.1
  have c81604 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.1
  have c81605 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.1
  have c81606 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.1
  have c81607 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.1
  have c81608 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_6 := arithEq_of_rows h (row := 1328) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_6
  simp only [← c81603, ← c81604, ← c81605] at e_1328_6
  have e_1328_7 := arithEq_of_rows h (row := 1328) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_7
  simp only [← c81606, ← c81607, k2, ← c81608] at e_1328_7
  have hr := e_1328_7
  simp only [e_1328_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12442)) := by
  have c81609 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.1
  have c81610 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81611 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81612 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81613 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81614 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_8 := arithEq_of_rows h (row := 1328) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_8
  simp only [← c81609, ← c81610, ← c81611] at e_1328_8
  have e_1328_9 := arithEq_of_rows h (row := 1328) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_9
  simp only [← c81612, ← c81613, k2, ← c81614] at e_1328_9
  have hr := e_1328_9
  simp only [e_1328_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12443)) := by
  have c81615 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81616 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81617 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81618 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81619 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81620 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_10 := arithEq_of_rows h (row := 1328) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_10
  simp only [← c81615, ← c81616, ← c81617] at e_1328_10
  have e_1328_11 := arithEq_of_rows h (row := 1328) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_11
  simp only [← c81618, ← c81619, k2, ← c81620] at e_1328_11
  have hr := e_1328_11
  simp only [e_1328_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12444)) := by
  have c81621 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81622 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81623 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81624 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81625 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81626 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_12 := arithEq_of_rows h (row := 1328) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_12
  simp only [← c81621, ← c81622, ← c81623] at e_1328_12
  have e_1328_13 := arithEq_of_rows h (row := 1328) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_13
  simp only [← c81624, ← c81625, k2, ← c81626] at e_1328_13
  have hr := e_1328_13
  simp only [e_1328_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12445)) := by
  have c81627 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81628 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81629 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81630 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81631 := (publicBatchWrapper86_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81632 := (publicBatchWrapper86_copies2551 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_14 := arithEq_of_rows h (row := 1328) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_14
  simp only [← c81627, ← c81628, ← c81629] at e_1328_14
  have e_1328_15 := arithEq_of_rows h (row := 1328) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_15
  simp only [← c81630, ← c81631, k2, ← c81632] at e_1328_15
  have hr := e_1328_15
  simp only [e_1328_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12446)) := by
  have c81633 := (publicBatchWrapper86_copies2551 a h).2.1
  have c81634 := (publicBatchWrapper86_copies2551 a h).2.2.1
  have c81635 := (publicBatchWrapper86_copies2551 a h).2.2.2.1
  have c81636 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.1
  have c81637 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.1
  have c81638 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_16 := arithEq_of_rows h (row := 1328) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_16
  simp only [← c81633, ← c81634, ← c81635] at e_1328_16
  have e_1328_17 := arithEq_of_rows h (row := 1328) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_17
  simp only [← c81636, ← c81637, k2, ← c81638] at e_1328_17
  have hr := e_1328_17
  simp only [e_1328_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1328 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12447)) := by
  have c81639 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.1
  have c81640 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.1
  have c81641 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.1
  have c81642 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81643 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81644 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1328_18 := arithEq_of_rows h (row := 1328) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_18
  simp only [← c81639, ← c81640, ← c81641] at e_1328_18
  have e_1328_19 := arithEq_of_rows h (row := 1328) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_19
  simp only [← c81642, ← c81643, k2, ← c81644] at e_1328_19
  have hr := e_1328_19
  simp only [e_1328_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12448)) := by
  have c81645 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81646 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81647 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81648 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81649 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81650 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_0 := arithEq_of_rows h (row := 1329) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_0
  simp only [← c81645, ← c81646, ← c81647] at e_1329_0
  have e_1329_1 := arithEq_of_rows h (row := 1329) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_1
  simp only [← c81648, ← c81649, k2, ← c81650] at e_1329_1
  have hr := e_1329_1
  simp only [e_1329_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12449)) := by
  have c81651 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81652 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81653 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81654 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81655 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81656 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_2 := arithEq_of_rows h (row := 1329) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_2
  simp only [← c81651, ← c81652, ← c81653] at e_1329_2
  have e_1329_3 := arithEq_of_rows h (row := 1329) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_3
  simp only [← c81654, ← c81655, k2, ← c81656] at e_1329_3
  have hr := e_1329_3
  simp only [e_1329_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12450)) := by
  have c81657 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81658 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81659 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81660 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81661 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81662 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_4 := arithEq_of_rows h (row := 1329) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_4
  simp only [← c81657, ← c81658, ← c81659] at e_1329_4
  have e_1329_5 := arithEq_of_rows h (row := 1329) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_5
  simp only [← c81660, ← c81661, k2, ← c81662] at e_1329_5
  have hr := e_1329_5
  simp only [e_1329_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12451)) := by
  have c81663 := (publicBatchWrapper86_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81664 := (publicBatchWrapper86_copies2552 a h).1
  have c81665 := (publicBatchWrapper86_copies2552 a h).2.1
  have c81666 := (publicBatchWrapper86_copies2552 a h).2.2.1
  have c81667 := (publicBatchWrapper86_copies2552 a h).2.2.2.1
  have c81668 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_6 := arithEq_of_rows h (row := 1329) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_6
  simp only [← c81663, ← c81664, ← c81665] at e_1329_6
  have e_1329_7 := arithEq_of_rows h (row := 1329) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_7
  simp only [← c81666, ← c81667, k2, ← c81668] at e_1329_7
  have hr := e_1329_7
  simp only [e_1329_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12452)) := by
  have c81669 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.1
  have c81670 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.1
  have c81671 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.1
  have c81672 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.1
  have c81673 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.1
  have c81674 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_8 := arithEq_of_rows h (row := 1329) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_8
  simp only [← c81669, ← c81670, ← c81671] at e_1329_8
  have e_1329_9 := arithEq_of_rows h (row := 1329) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_9
  simp only [← c81672, ← c81673, k2, ← c81674] at e_1329_9
  have hr := e_1329_9
  simp only [e_1329_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12453)) := by
  have c81675 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81676 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81677 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81678 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81679 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81680 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_10 := arithEq_of_rows h (row := 1329) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_10
  simp only [← c81675, ← c81676, ← c81677] at e_1329_10
  have e_1329_11 := arithEq_of_rows h (row := 1329) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_11
  simp only [← c81678, ← c81679, k2, ← c81680] at e_1329_11
  have hr := e_1329_11
  simp only [e_1329_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12454)) := by
  have c81681 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81682 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81683 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81684 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81685 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81686 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_12 := arithEq_of_rows h (row := 1329) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_12
  simp only [← c81681, ← c81682, ← c81683] at e_1329_12
  have e_1329_13 := arithEq_of_rows h (row := 1329) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_13
  simp only [← c81684, ← c81685, k2, ← c81686] at e_1329_13
  have hr := e_1329_13
  simp only [e_1329_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12455)) := by
  have c81687 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81688 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81689 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81690 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81691 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81692 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_14 := arithEq_of_rows h (row := 1329) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_14
  simp only [← c81687, ← c81688, ← c81689] at e_1329_14
  have e_1329_15 := arithEq_of_rows h (row := 1329) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_15
  simp only [← c81690, ← c81691, k2, ← c81692] at e_1329_15
  have hr := e_1329_15
  simp only [e_1329_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12456)) := by
  have c81693 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81694 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81695 := (publicBatchWrapper86_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81696 := (publicBatchWrapper86_copies2553 a h).1
  have c81697 := (publicBatchWrapper86_copies2553 a h).2.1
  have c81698 := (publicBatchWrapper86_copies2553 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_16 := arithEq_of_rows h (row := 1329) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_16
  simp only [← c81693, ← c81694, ← c81695] at e_1329_16
  have e_1329_17 := arithEq_of_rows h (row := 1329) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_17
  simp only [← c81696, ← c81697, k2, ← c81698] at e_1329_17
  have hr := e_1329_17
  simp only [e_1329_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1329 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12457)) := by
  have c81699 := (publicBatchWrapper86_copies2553 a h).2.2.2.1
  have c81700 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.1
  have c81701 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.1
  have c81702 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.1
  have c81703 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.1
  have c81704 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1329_18 := arithEq_of_rows h (row := 1329) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_18
  simp only [← c81699, ← c81700, ← c81701] at e_1329_18
  have e_1329_19 := arithEq_of_rows h (row := 1329) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_19
  simp only [← c81702, ← c81703, k2, ← c81704] at e_1329_19
  have hr := e_1329_19
  simp only [e_1329_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12458)) := by
  have c81705 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.1
  have c81706 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81707 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81708 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81709 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81710 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_0 := arithEq_of_rows h (row := 1330) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_0
  simp only [← c81705, ← c81706, ← c81707] at e_1330_0
  have e_1330_1 := arithEq_of_rows h (row := 1330) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_1
  simp only [← c81708, ← c81709, k2, ← c81710] at e_1330_1
  have hr := e_1330_1
  simp only [e_1330_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12459)) := by
  have c81711 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81712 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81713 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81714 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81715 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81716 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_2 := arithEq_of_rows h (row := 1330) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_2
  simp only [← c81711, ← c81712, ← c81713] at e_1330_2
  have e_1330_3 := arithEq_of_rows h (row := 1330) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_3
  simp only [← c81714, ← c81715, k2, ← c81716] at e_1330_3
  have hr := e_1330_3
  simp only [e_1330_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12460)) := by
  have c81717 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81718 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81719 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81720 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81721 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81722 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_4 := arithEq_of_rows h (row := 1330) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_4
  simp only [← c81717, ← c81718, ← c81719] at e_1330_4
  have e_1330_5 := arithEq_of_rows h (row := 1330) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_5
  simp only [← c81720, ← c81721, k2, ← c81722] at e_1330_5
  have hr := e_1330_5
  simp only [e_1330_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12461)) := by
  have c81723 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81724 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81725 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81726 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81727 := (publicBatchWrapper86_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81728 := (publicBatchWrapper86_copies2554 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_6 := arithEq_of_rows h (row := 1330) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_6
  simp only [← c81723, ← c81724, ← c81725] at e_1330_6
  have e_1330_7 := arithEq_of_rows h (row := 1330) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_7
  simp only [← c81726, ← c81727, k2, ← c81728] at e_1330_7
  have hr := e_1330_7
  simp only [e_1330_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12462)) := by
  have c81729 := (publicBatchWrapper86_copies2554 a h).2.1
  have c81730 := (publicBatchWrapper86_copies2554 a h).2.2.1
  have c81731 := (publicBatchWrapper86_copies2554 a h).2.2.2.1
  have c81732 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.1
  have c81733 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.1
  have c81734 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_8 := arithEq_of_rows h (row := 1330) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_8
  simp only [← c81729, ← c81730, ← c81731] at e_1330_8
  have e_1330_9 := arithEq_of_rows h (row := 1330) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_9
  simp only [← c81732, ← c81733, k2, ← c81734] at e_1330_9
  have hr := e_1330_9
  simp only [e_1330_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12463)) := by
  have c81735 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.1
  have c81736 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.1
  have c81737 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.1
  have c81738 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81739 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81740 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_10 := arithEq_of_rows h (row := 1330) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_10
  simp only [← c81735, ← c81736, ← c81737] at e_1330_10
  have e_1330_11 := arithEq_of_rows h (row := 1330) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_11
  simp only [← c81738, ← c81739, k2, ← c81740] at e_1330_11
  have hr := e_1330_11
  simp only [e_1330_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12464)) := by
  have c81741 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81742 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81743 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81744 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81745 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81746 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_12 := arithEq_of_rows h (row := 1330) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_12
  simp only [← c81741, ← c81742, ← c81743] at e_1330_12
  have e_1330_13 := arithEq_of_rows h (row := 1330) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_13
  simp only [← c81744, ← c81745, k2, ← c81746] at e_1330_13
  have hr := e_1330_13
  simp only [e_1330_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12465)) := by
  have c81747 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81748 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81749 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81750 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81751 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81752 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_14 := arithEq_of_rows h (row := 1330) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_14
  simp only [← c81747, ← c81748, ← c81749] at e_1330_14
  have e_1330_15 := arithEq_of_rows h (row := 1330) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_15
  simp only [← c81750, ← c81751, k2, ← c81752] at e_1330_15
  have hr := e_1330_15
  simp only [e_1330_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12466)) := by
  have c81753 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81754 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81755 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81756 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81757 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81758 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_16 := arithEq_of_rows h (row := 1330) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_16
  simp only [← c81753, ← c81754, ← c81755] at e_1330_16
  have e_1330_17 := arithEq_of_rows h (row := 1330) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_17
  simp only [← c81756, ← c81757, k2, ← c81758] at e_1330_17
  have hr := e_1330_17
  simp only [e_1330_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1330 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12467)) := by
  have c81759 := (publicBatchWrapper86_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81760 := (publicBatchWrapper86_copies2555 a h).1
  have c81761 := (publicBatchWrapper86_copies2555 a h).2.1
  have c81762 := (publicBatchWrapper86_copies2555 a h).2.2.1
  have c81763 := (publicBatchWrapper86_copies2555 a h).2.2.2.1
  have c81764 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1330_18 := arithEq_of_rows h (row := 1330) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_18
  simp only [← c81759, ← c81760, ← c81761] at e_1330_18
  have e_1330_19 := arithEq_of_rows h (row := 1330) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_19
  simp only [← c81762, ← c81763, k2, ← c81764] at e_1330_19
  have hr := e_1330_19
  simp only [e_1330_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12468)) := by
  have c81765 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.1
  have c81766 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.1
  have c81767 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.1
  have c81768 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.1
  have c81769 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.1
  have c81770 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_0 := arithEq_of_rows h (row := 1331) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_0
  simp only [← c81765, ← c81766, ← c81767] at e_1331_0
  have e_1331_1 := arithEq_of_rows h (row := 1331) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_1
  simp only [← c81768, ← c81769, k2, ← c81770] at e_1331_1
  have hr := e_1331_1
  simp only [e_1331_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12469)) := by
  have c81771 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81772 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81773 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81774 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81775 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81776 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_2 := arithEq_of_rows h (row := 1331) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_2
  simp only [← c81771, ← c81772, ← c81773] at e_1331_2
  have e_1331_3 := arithEq_of_rows h (row := 1331) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_3
  simp only [← c81774, ← c81775, k2, ← c81776] at e_1331_3
  have hr := e_1331_3
  simp only [e_1331_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12470)) := by
  have c81777 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81778 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81779 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81780 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81781 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81782 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_4 := arithEq_of_rows h (row := 1331) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_4
  simp only [← c81777, ← c81778, ← c81779] at e_1331_4
  have e_1331_5 := arithEq_of_rows h (row := 1331) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_5
  simp only [← c81780, ← c81781, k2, ← c81782] at e_1331_5
  have hr := e_1331_5
  simp only [e_1331_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12471)) := by
  have c81783 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81784 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81785 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81786 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81787 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81788 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_6 := arithEq_of_rows h (row := 1331) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_6
  simp only [← c81783, ← c81784, ← c81785] at e_1331_6
  have e_1331_7 := arithEq_of_rows h (row := 1331) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_7
  simp only [← c81786, ← c81787, k2, ← c81788] at e_1331_7
  have hr := e_1331_7
  simp only [e_1331_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12472)) := by
  have c81789 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81790 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81791 := (publicBatchWrapper86_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81792 := (publicBatchWrapper86_copies2556 a h).1
  have c81793 := (publicBatchWrapper86_copies2556 a h).2.1
  have c81794 := (publicBatchWrapper86_copies2556 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_8 := arithEq_of_rows h (row := 1331) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_8
  simp only [← c81789, ← c81790, ← c81791] at e_1331_8
  have e_1331_9 := arithEq_of_rows h (row := 1331) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_9
  simp only [← c81792, ← c81793, k2, ← c81794] at e_1331_9
  have hr := e_1331_9
  simp only [e_1331_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12473)) := by
  have c81795 := (publicBatchWrapper86_copies2556 a h).2.2.2.1
  have c81796 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.1
  have c81797 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.1
  have c81798 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.1
  have c81799 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.1
  have c81800 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_10 := arithEq_of_rows h (row := 1331) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_10
  simp only [← c81795, ← c81796, ← c81797] at e_1331_10
  have e_1331_11 := arithEq_of_rows h (row := 1331) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_11
  simp only [← c81798, ← c81799, k2, ← c81800] at e_1331_11
  have hr := e_1331_11
  simp only [e_1331_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12474)) := by
  have c81801 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.1
  have c81802 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81803 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81804 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81805 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81806 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_12 := arithEq_of_rows h (row := 1331) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_12
  simp only [← c81801, ← c81802, ← c81803] at e_1331_12
  have e_1331_13 := arithEq_of_rows h (row := 1331) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_13
  simp only [← c81804, ← c81805, k2, ← c81806] at e_1331_13
  have hr := e_1331_13
  simp only [e_1331_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12475)) := by
  have c81807 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81808 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81809 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81810 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81811 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81812 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_14 := arithEq_of_rows h (row := 1331) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_14
  simp only [← c81807, ← c81808, ← c81809] at e_1331_14
  have e_1331_15 := arithEq_of_rows h (row := 1331) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_15
  simp only [← c81810, ← c81811, k2, ← c81812] at e_1331_15
  have hr := e_1331_15
  simp only [e_1331_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12476)) := by
  have c81813 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81814 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81815 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81816 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81817 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81818 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_16 := arithEq_of_rows h (row := 1331) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_16
  simp only [← c81813, ← c81814, ← c81815] at e_1331_16
  have e_1331_17 := arithEq_of_rows h (row := 1331) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_17
  simp only [← c81816, ← c81817, k2, ← c81818] at e_1331_17
  have hr := e_1331_17
  simp only [e_1331_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1331 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12477)) := by
  have c81819 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81820 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81821 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81822 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81823 := (publicBatchWrapper86_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81824 := (publicBatchWrapper86_copies2557 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1331_18 := arithEq_of_rows h (row := 1331) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_18
  simp only [← c81819, ← c81820, ← c81821] at e_1331_18
  have e_1331_19 := arithEq_of_rows h (row := 1331) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_19
  simp only [← c81822, ← c81823, k2, ← c81824] at e_1331_19
  have hr := e_1331_19
  simp only [e_1331_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12478)) := by
  have c81825 := (publicBatchWrapper86_copies2557 a h).2.1
  have c81826 := (publicBatchWrapper86_copies2557 a h).2.2.1
  have c81827 := (publicBatchWrapper86_copies2557 a h).2.2.2.1
  have c81828 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.1
  have c81829 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.1
  have c81830 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_0 := arithEq_of_rows h (row := 1332) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_0
  simp only [← c81825, ← c81826, ← c81827] at e_1332_0
  have e_1332_1 := arithEq_of_rows h (row := 1332) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_1
  simp only [← c81828, ← c81829, k2, ← c81830] at e_1332_1
  have hr := e_1332_1
  simp only [e_1332_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12479)) := by
  have c81831 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.1
  have c81832 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.1
  have c81833 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.1
  have c81834 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81835 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81836 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_2 := arithEq_of_rows h (row := 1332) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_2
  simp only [← c81831, ← c81832, ← c81833] at e_1332_2
  have e_1332_3 := arithEq_of_rows h (row := 1332) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_3
  simp only [← c81834, ← c81835, k2, ← c81836] at e_1332_3
  have hr := e_1332_3
  simp only [e_1332_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12480)) := by
  have c81837 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81838 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81839 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81840 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81841 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81842 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_4 := arithEq_of_rows h (row := 1332) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_4
  simp only [← c81837, ← c81838, ← c81839] at e_1332_4
  have e_1332_5 := arithEq_of_rows h (row := 1332) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_5
  simp only [← c81840, ← c81841, k2, ← c81842] at e_1332_5
  have hr := e_1332_5
  simp only [e_1332_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12481)) := by
  have c81843 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81844 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81845 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81846 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81847 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81848 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_6 := arithEq_of_rows h (row := 1332) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_6
  simp only [← c81843, ← c81844, ← c81845] at e_1332_6
  have e_1332_7 := arithEq_of_rows h (row := 1332) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_7
  simp only [← c81846, ← c81847, k2, ← c81848] at e_1332_7
  have hr := e_1332_7
  simp only [e_1332_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12482)) := by
  have c81849 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81850 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81851 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81852 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81853 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81854 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_8 := arithEq_of_rows h (row := 1332) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_8
  simp only [← c81849, ← c81850, ← c81851] at e_1332_8
  have e_1332_9 := arithEq_of_rows h (row := 1332) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_9
  simp only [← c81852, ← c81853, k2, ← c81854] at e_1332_9
  have hr := e_1332_9
  simp only [e_1332_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12483)) := by
  have c81855 := (publicBatchWrapper86_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81856 := (publicBatchWrapper86_copies2558 a h).1
  have c81857 := (publicBatchWrapper86_copies2558 a h).2.1
  have c81858 := (publicBatchWrapper86_copies2558 a h).2.2.1
  have c81859 := (publicBatchWrapper86_copies2558 a h).2.2.2.1
  have c81860 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_10 := arithEq_of_rows h (row := 1332) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_10
  simp only [← c81855, ← c81856, ← c81857] at e_1332_10
  have e_1332_11 := arithEq_of_rows h (row := 1332) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_11
  simp only [← c81858, ← c81859, k2, ← c81860] at e_1332_11
  have hr := e_1332_11
  simp only [e_1332_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12484)) := by
  have c81861 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.1
  have c81862 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.1
  have c81863 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.1
  have c81864 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.1
  have c81865 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.1
  have c81866 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_12 := arithEq_of_rows h (row := 1332) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_12
  simp only [← c81861, ← c81862, ← c81863] at e_1332_12
  have e_1332_13 := arithEq_of_rows h (row := 1332) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_13
  simp only [← c81864, ← c81865, k2, ← c81866] at e_1332_13
  have hr := e_1332_13
  simp only [e_1332_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12485)) := by
  have c81867 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81868 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81869 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81870 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81871 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81872 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_14 := arithEq_of_rows h (row := 1332) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_14
  simp only [← c81867, ← c81868, ← c81869] at e_1332_14
  have e_1332_15 := arithEq_of_rows h (row := 1332) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_15
  simp only [← c81870, ← c81871, k2, ← c81872] at e_1332_15
  have hr := e_1332_15
  simp only [e_1332_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12486)) := by
  have c81873 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81874 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81875 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81876 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81877 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81878 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_16 := arithEq_of_rows h (row := 1332) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_16
  simp only [← c81873, ← c81874, ← c81875] at e_1332_16
  have e_1332_17 := arithEq_of_rows h (row := 1332) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_17
  simp only [← c81876, ← c81877, k2, ← c81878] at e_1332_17
  have hr := e_1332_17
  simp only [e_1332_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1332 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12487)) := by
  have c81879 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81880 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81881 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81882 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81883 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81884 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1332_18 := arithEq_of_rows h (row := 1332) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_18
  simp only [← c81879, ← c81880, ← c81881] at e_1332_18
  have e_1332_19 := arithEq_of_rows h (row := 1332) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_19
  simp only [← c81882, ← c81883, k2, ← c81884] at e_1332_19
  have hr := e_1332_19
  simp only [e_1332_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12488)) := by
  have c81885 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81886 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81887 := (publicBatchWrapper86_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81888 := (publicBatchWrapper86_copies2559 a h).1
  have c81889 := (publicBatchWrapper86_copies2559 a h).2.1
  have c81890 := (publicBatchWrapper86_copies2559 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_0 := arithEq_of_rows h (row := 1333) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_0
  simp only [← c81885, ← c81886, ← c81887] at e_1333_0
  have e_1333_1 := arithEq_of_rows h (row := 1333) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_1
  simp only [← c81888, ← c81889, k2, ← c81890] at e_1333_1
  have hr := e_1333_1
  simp only [e_1333_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12489)) := by
  have c81891 := (publicBatchWrapper86_copies2559 a h).2.2.2.1
  have c81892 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.1
  have c81893 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.1
  have c81894 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.1
  have c81895 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.1
  have c81896 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_2 := arithEq_of_rows h (row := 1333) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_2
  simp only [← c81891, ← c81892, ← c81893] at e_1333_2
  have e_1333_3 := arithEq_of_rows h (row := 1333) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_3
  simp only [← c81894, ← c81895, k2, ← c81896] at e_1333_3
  have hr := e_1333_3
  simp only [e_1333_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12490)) := by
  have c81897 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.1
  have c81898 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81899 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81900 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81901 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81902 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_4 := arithEq_of_rows h (row := 1333) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_4
  simp only [← c81897, ← c81898, ← c81899] at e_1333_4
  have e_1333_5 := arithEq_of_rows h (row := 1333) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_5
  simp only [← c81900, ← c81901, k2, ← c81902] at e_1333_5
  have hr := e_1333_5
  simp only [e_1333_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12491)) := by
  have c81903 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81904 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81905 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81906 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81907 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81908 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_6 := arithEq_of_rows h (row := 1333) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_6
  simp only [← c81903, ← c81904, ← c81905] at e_1333_6
  have e_1333_7 := arithEq_of_rows h (row := 1333) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_7
  simp only [← c81906, ← c81907, k2, ← c81908] at e_1333_7
  have hr := e_1333_7
  simp only [e_1333_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12492)) := by
  have c81909 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81910 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81911 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81912 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81913 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81914 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_8 := arithEq_of_rows h (row := 1333) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_8
  simp only [← c81909, ← c81910, ← c81911] at e_1333_8
  have e_1333_9 := arithEq_of_rows h (row := 1333) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_9
  simp only [← c81912, ← c81913, k2, ← c81914] at e_1333_9
  have hr := e_1333_9
  simp only [e_1333_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12493)) := by
  have c81915 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81916 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81917 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81918 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81919 := (publicBatchWrapper86_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81920 := (publicBatchWrapper86_copies2560 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_10 := arithEq_of_rows h (row := 1333) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_10
  simp only [← c81915, ← c81916, ← c81917] at e_1333_10
  have e_1333_11 := arithEq_of_rows h (row := 1333) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_11
  simp only [← c81918, ← c81919, k2, ← c81920] at e_1333_11
  have hr := e_1333_11
  simp only [e_1333_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12494)) := by
  have c81921 := (publicBatchWrapper86_copies2560 a h).2.1
  have c81922 := (publicBatchWrapper86_copies2560 a h).2.2.1
  have c81923 := (publicBatchWrapper86_copies2560 a h).2.2.2.1
  have c81924 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.1
  have c81925 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.1
  have c81926 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_12 := arithEq_of_rows h (row := 1333) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_12
  simp only [← c81921, ← c81922, ← c81923] at e_1333_12
  have e_1333_13 := arithEq_of_rows h (row := 1333) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_13
  simp only [← c81924, ← c81925, k2, ← c81926] at e_1333_13
  have hr := e_1333_13
  simp only [e_1333_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12495)) := by
  have c81927 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.1
  have c81928 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.1
  have c81929 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.1
  have c81930 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81931 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81932 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_14 := arithEq_of_rows h (row := 1333) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_14
  simp only [← c81927, ← c81928, ← c81929] at e_1333_14
  have e_1333_15 := arithEq_of_rows h (row := 1333) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_15
  simp only [← c81930, ← c81931, k2, ← c81932] at e_1333_15
  have hr := e_1333_15
  simp only [e_1333_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12496)) := by
  have c81933 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81934 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81935 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81936 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81937 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81938 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_16 := arithEq_of_rows h (row := 1333) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_16
  simp only [← c81933, ← c81934, ← c81935] at e_1333_16
  have e_1333_17 := arithEq_of_rows h (row := 1333) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_17
  simp only [← c81936, ← c81937, k2, ← c81938] at e_1333_17
  have hr := e_1333_17
  simp only [e_1333_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1333 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12497)) := by
  have c81939 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81940 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81941 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81942 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81943 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81944 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1333_18 := arithEq_of_rows h (row := 1333) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_18
  simp only [← c81939, ← c81940, ← c81941] at e_1333_18
  have e_1333_19 := arithEq_of_rows h (row := 1333) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_19
  simp only [← c81942, ← c81943, k2, ← c81944] at e_1333_19
  have hr := e_1333_19
  simp only [e_1333_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12498)) := by
  have c81945 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81946 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81947 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81948 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81949 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81950 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_0 := arithEq_of_rows h (row := 1334) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_0
  simp only [← c81945, ← c81946, ← c81947] at e_1334_0
  have e_1334_1 := arithEq_of_rows h (row := 1334) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_1
  simp only [← c81948, ← c81949, k2, ← c81950] at e_1334_1
  have hr := e_1334_1
  simp only [e_1334_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12499)) := by
  have c81951 := (publicBatchWrapper86_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81952 := (publicBatchWrapper86_copies2561 a h).1
  have c81953 := (publicBatchWrapper86_copies2561 a h).2.1
  have c81954 := (publicBatchWrapper86_copies2561 a h).2.2.1
  have c81955 := (publicBatchWrapper86_copies2561 a h).2.2.2.1
  have c81956 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_2 := arithEq_of_rows h (row := 1334) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_2
  simp only [← c81951, ← c81952, ← c81953] at e_1334_2
  have e_1334_3 := arithEq_of_rows h (row := 1334) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_3
  simp only [← c81954, ← c81955, k2, ← c81956] at e_1334_3
  have hr := e_1334_3
  simp only [e_1334_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12500)) := by
  have c81957 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.1
  have c81958 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.1
  have c81959 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.1
  have c81960 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.1
  have c81961 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.1
  have c81962 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_4 := arithEq_of_rows h (row := 1334) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_4
  simp only [← c81957, ← c81958, ← c81959] at e_1334_4
  have e_1334_5 := arithEq_of_rows h (row := 1334) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_5
  simp only [← c81960, ← c81961, k2, ← c81962] at e_1334_5
  have hr := e_1334_5
  simp only [e_1334_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12501)) := by
  have c81963 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81964 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81965 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81966 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81967 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81968 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_6 := arithEq_of_rows h (row := 1334) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_6
  simp only [← c81963, ← c81964, ← c81965] at e_1334_6
  have e_1334_7 := arithEq_of_rows h (row := 1334) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_7
  simp only [← c81966, ← c81967, k2, ← c81968] at e_1334_7
  have hr := e_1334_7
  simp only [e_1334_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12502)) := by
  have c81969 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81970 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81971 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81972 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81973 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81974 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_8 := arithEq_of_rows h (row := 1334) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_8
  simp only [← c81969, ← c81970, ← c81971] at e_1334_8
  have e_1334_9 := arithEq_of_rows h (row := 1334) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_9
  simp only [← c81972, ← c81973, k2, ← c81974] at e_1334_9
  have hr := e_1334_9
  simp only [e_1334_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12503)) := by
  have c81975 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81976 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81977 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81978 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81979 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81980 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_10 := arithEq_of_rows h (row := 1334) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_10
  simp only [← c81975, ← c81976, ← c81977] at e_1334_10
  have e_1334_11 := arithEq_of_rows h (row := 1334) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_11
  simp only [← c81978, ← c81979, k2, ← c81980] at e_1334_11
  have hr := e_1334_11
  simp only [e_1334_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12504)) := by
  have c81981 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81982 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81983 := (publicBatchWrapper86_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81984 := (publicBatchWrapper86_copies2562 a h).1
  have c81985 := (publicBatchWrapper86_copies2562 a h).2.1
  have c81986 := (publicBatchWrapper86_copies2562 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_12 := arithEq_of_rows h (row := 1334) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_12
  simp only [← c81981, ← c81982, ← c81983] at e_1334_12
  have e_1334_13 := arithEq_of_rows h (row := 1334) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_13
  simp only [← c81984, ← c81985, k2, ← c81986] at e_1334_13
  have hr := e_1334_13
  simp only [e_1334_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12505)) := by
  have c81987 := (publicBatchWrapper86_copies2562 a h).2.2.2.1
  have c81988 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.1
  have c81989 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.1
  have c81990 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.1
  have c81991 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.1
  have c81992 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_14 := arithEq_of_rows h (row := 1334) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_14
  simp only [← c81987, ← c81988, ← c81989] at e_1334_14
  have e_1334_15 := arithEq_of_rows h (row := 1334) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_15
  simp only [← c81990, ← c81991, k2, ← c81992] at e_1334_15
  have hr := e_1334_15
  simp only [e_1334_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12506)) := by
  have c81993 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.1
  have c81994 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81995 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81996 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81997 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81998 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_16 := arithEq_of_rows h (row := 1334) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_16
  simp only [← c81993, ← c81994, ← c81995] at e_1334_16
  have e_1334_17 := arithEq_of_rows h (row := 1334) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_17
  simp only [← c81996, ← c81997, k2, ← c81998] at e_1334_17
  have hr := e_1334_17
  simp only [e_1334_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1334 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12507)) := by
  have c81999 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82000 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82001 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82002 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82003 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82004 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1334_18 := arithEq_of_rows h (row := 1334) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_18
  simp only [← c81999, ← c82000, ← c82001] at e_1334_18
  have e_1334_19 := arithEq_of_rows h (row := 1334) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_19
  simp only [← c82002, ← c82003, k2, ← c82004] at e_1334_19
  have hr := e_1334_19
  simp only [e_1334_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12508)) := by
  have c82005 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82006 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82007 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82008 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82009 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82010 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_0 := arithEq_of_rows h (row := 1335) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_0
  simp only [← c82005, ← c82006, ← c82007] at e_1335_0
  have e_1335_1 := arithEq_of_rows h (row := 1335) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_1
  simp only [← c82008, ← c82009, k2, ← c82010] at e_1335_1
  have hr := e_1335_1
  simp only [e_1335_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12509)) := by
  have c82011 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82012 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82013 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82014 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82015 := (publicBatchWrapper86_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82016 := (publicBatchWrapper86_copies2563 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_2 := arithEq_of_rows h (row := 1335) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_2
  simp only [← c82011, ← c82012, ← c82013] at e_1335_2
  have e_1335_3 := arithEq_of_rows h (row := 1335) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_3
  simp only [← c82014, ← c82015, k2, ← c82016] at e_1335_3
  have hr := e_1335_3
  simp only [e_1335_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12510)) := by
  have c82017 := (publicBatchWrapper86_copies2563 a h).2.1
  have c82018 := (publicBatchWrapper86_copies2563 a h).2.2.1
  have c82019 := (publicBatchWrapper86_copies2563 a h).2.2.2.1
  have c82020 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.1
  have c82021 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.1
  have c82022 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_4 := arithEq_of_rows h (row := 1335) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_4
  simp only [← c82017, ← c82018, ← c82019] at e_1335_4
  have e_1335_5 := arithEq_of_rows h (row := 1335) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_5
  simp only [← c82020, ← c82021, k2, ← c82022] at e_1335_5
  have hr := e_1335_5
  simp only [e_1335_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12511)) := by
  have c82023 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.1
  have c82024 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.1
  have c82025 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.1
  have c82026 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82027 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82028 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_6 := arithEq_of_rows h (row := 1335) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_6
  simp only [← c82023, ← c82024, ← c82025] at e_1335_6
  have e_1335_7 := arithEq_of_rows h (row := 1335) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_7
  simp only [← c82026, ← c82027, k2, ← c82028] at e_1335_7
  have hr := e_1335_7
  simp only [e_1335_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12512)) := by
  have c82029 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82030 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82031 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82032 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82033 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82034 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_8 := arithEq_of_rows h (row := 1335) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_8
  simp only [← c82029, ← c82030, ← c82031] at e_1335_8
  have e_1335_9 := arithEq_of_rows h (row := 1335) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_9
  simp only [← c82032, ← c82033, k2, ← c82034] at e_1335_9
  have hr := e_1335_9
  simp only [e_1335_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12513)) := by
  have c82035 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82036 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82037 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82038 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82039 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82040 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_10 := arithEq_of_rows h (row := 1335) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_10
  simp only [← c82035, ← c82036, ← c82037] at e_1335_10
  have e_1335_11 := arithEq_of_rows h (row := 1335) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_11
  simp only [← c82038, ← c82039, k2, ← c82040] at e_1335_11
  have hr := e_1335_11
  simp only [e_1335_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12514)) := by
  have c82041 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82042 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82043 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82044 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82045 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82046 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_12 := arithEq_of_rows h (row := 1335) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_12
  simp only [← c82041, ← c82042, ← c82043] at e_1335_12
  have e_1335_13 := arithEq_of_rows h (row := 1335) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_13
  simp only [← c82044, ← c82045, k2, ← c82046] at e_1335_13
  have hr := e_1335_13
  simp only [e_1335_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12515)) := by
  have c82047 := (publicBatchWrapper86_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82048 := (publicBatchWrapper86_copies2564 a h).1
  have c82049 := (publicBatchWrapper86_copies2564 a h).2.1
  have c82050 := (publicBatchWrapper86_copies2564 a h).2.2.1
  have c82051 := (publicBatchWrapper86_copies2564 a h).2.2.2.1
  have c82052 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_14 := arithEq_of_rows h (row := 1335) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_14
  simp only [← c82047, ← c82048, ← c82049] at e_1335_14
  have e_1335_15 := arithEq_of_rows h (row := 1335) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_15
  simp only [← c82050, ← c82051, k2, ← c82052] at e_1335_15
  have hr := e_1335_15
  simp only [e_1335_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12516)) := by
  have c82053 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.1
  have c82054 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.1
  have c82055 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.1
  have c82056 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.1
  have c82057 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.1
  have c82058 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_16 := arithEq_of_rows h (row := 1335) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_16
  simp only [← c82053, ← c82054, ← c82055] at e_1335_16
  have e_1335_17 := arithEq_of_rows h (row := 1335) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_17
  simp only [← c82056, ← c82057, k2, ← c82058] at e_1335_17
  have hr := e_1335_17
  simp only [e_1335_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1335 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12517)) := by
  have c82059 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82060 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82061 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82062 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82063 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82064 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1335_18 := arithEq_of_rows h (row := 1335) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_18
  simp only [← c82059, ← c82060, ← c82061] at e_1335_18
  have e_1335_19 := arithEq_of_rows h (row := 1335) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_19
  simp only [← c82062, ← c82063, k2, ← c82064] at e_1335_19
  have hr := e_1335_19
  simp only [e_1335_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12518)) := by
  have c82065 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82066 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82067 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82068 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82069 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82070 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_0 := arithEq_of_rows h (row := 1336) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_0
  simp only [← c82065, ← c82066, ← c82067] at e_1336_0
  have e_1336_1 := arithEq_of_rows h (row := 1336) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_1
  simp only [← c82068, ← c82069, k2, ← c82070] at e_1336_1
  have hr := e_1336_1
  simp only [e_1336_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12519)) := by
  have c82071 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82072 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82073 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82074 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82075 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82076 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_2 := arithEq_of_rows h (row := 1336) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_2
  simp only [← c82071, ← c82072, ← c82073] at e_1336_2
  have e_1336_3 := arithEq_of_rows h (row := 1336) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_3
  simp only [← c82074, ← c82075, k2, ← c82076] at e_1336_3
  have hr := e_1336_3
  simp only [e_1336_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12520)) := by
  have c82077 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82078 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82079 := (publicBatchWrapper86_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82080 := (publicBatchWrapper86_copies2565 a h).1
  have c82081 := (publicBatchWrapper86_copies2565 a h).2.1
  have c82082 := (publicBatchWrapper86_copies2565 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_4 := arithEq_of_rows h (row := 1336) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_4
  simp only [← c82077, ← c82078, ← c82079] at e_1336_4
  have e_1336_5 := arithEq_of_rows h (row := 1336) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_5
  simp only [← c82080, ← c82081, k2, ← c82082] at e_1336_5
  have hr := e_1336_5
  simp only [e_1336_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12521)) := by
  have c82083 := (publicBatchWrapper86_copies2565 a h).2.2.2.1
  have c82084 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.1
  have c82085 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.1
  have c82086 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.1
  have c82087 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.1
  have c82088 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_6 := arithEq_of_rows h (row := 1336) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_6
  simp only [← c82083, ← c82084, ← c82085] at e_1336_6
  have e_1336_7 := arithEq_of_rows h (row := 1336) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_7
  simp only [← c82086, ← c82087, k2, ← c82088] at e_1336_7
  have hr := e_1336_7
  simp only [e_1336_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12522)) := by
  have c82089 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.1
  have c82090 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82091 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82092 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82093 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82094 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_8 := arithEq_of_rows h (row := 1336) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_8
  simp only [← c82089, ← c82090, ← c82091] at e_1336_8
  have e_1336_9 := arithEq_of_rows h (row := 1336) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_9
  simp only [← c82092, ← c82093, k2, ← c82094] at e_1336_9
  have hr := e_1336_9
  simp only [e_1336_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12523)) := by
  have c82095 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82096 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82097 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82098 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82099 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82100 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_10 := arithEq_of_rows h (row := 1336) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_10
  simp only [← c82095, ← c82096, ← c82097] at e_1336_10
  have e_1336_11 := arithEq_of_rows h (row := 1336) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_11
  simp only [← c82098, ← c82099, k2, ← c82100] at e_1336_11
  have hr := e_1336_11
  simp only [e_1336_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12524)) := by
  have c82101 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82102 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82103 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82104 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82105 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82106 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_12 := arithEq_of_rows h (row := 1336) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_12
  simp only [← c82101, ← c82102, ← c82103] at e_1336_12
  have e_1336_13 := arithEq_of_rows h (row := 1336) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_13
  simp only [← c82104, ← c82105, k2, ← c82106] at e_1336_13
  have hr := e_1336_13
  simp only [e_1336_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12525)) := by
  have c82107 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82108 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82109 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82110 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82111 := (publicBatchWrapper86_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82112 := (publicBatchWrapper86_copies2566 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_14 := arithEq_of_rows h (row := 1336) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_14
  simp only [← c82107, ← c82108, ← c82109] at e_1336_14
  have e_1336_15 := arithEq_of_rows h (row := 1336) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_15
  simp only [← c82110, ← c82111, k2, ← c82112] at e_1336_15
  have hr := e_1336_15
  simp only [e_1336_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12526)) := by
  have c82113 := (publicBatchWrapper86_copies2566 a h).2.1
  have c82114 := (publicBatchWrapper86_copies2566 a h).2.2.1
  have c82115 := (publicBatchWrapper86_copies2566 a h).2.2.2.1
  have c82116 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.1
  have c82117 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.1
  have c82118 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_16 := arithEq_of_rows h (row := 1336) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_16
  simp only [← c82113, ← c82114, ← c82115] at e_1336_16
  have e_1336_17 := arithEq_of_rows h (row := 1336) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_17
  simp only [← c82116, ← c82117, k2, ← c82118] at e_1336_17
  have hr := e_1336_17
  simp only [e_1336_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1336 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12527)) := by
  have c82119 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.1
  have c82120 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.1
  have c82121 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.1
  have c82122 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82123 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82124 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1336_18 := arithEq_of_rows h (row := 1336) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_18
  simp only [← c82119, ← c82120, ← c82121] at e_1336_18
  have e_1336_19 := arithEq_of_rows h (row := 1336) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_19
  simp only [← c82122, ← c82123, k2, ← c82124] at e_1336_19
  have hr := e_1336_19
  simp only [e_1336_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12528)) := by
  have c82125 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82126 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82127 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82128 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82129 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82130 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_0 := arithEq_of_rows h (row := 1337) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_0
  simp only [← c82125, ← c82126, ← c82127] at e_1337_0
  have e_1337_1 := arithEq_of_rows h (row := 1337) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_1
  simp only [← c82128, ← c82129, k2, ← c82130] at e_1337_1
  have hr := e_1337_1
  simp only [e_1337_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12529)) := by
  have c82131 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82132 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82133 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82134 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82135 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82136 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_2 := arithEq_of_rows h (row := 1337) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_2
  simp only [← c82131, ← c82132, ← c82133] at e_1337_2
  have e_1337_3 := arithEq_of_rows h (row := 1337) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_3
  simp only [← c82134, ← c82135, k2, ← c82136] at e_1337_3
  have hr := e_1337_3
  simp only [e_1337_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12530)) := by
  have c82137 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82138 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82139 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82140 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82141 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82142 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_4 := arithEq_of_rows h (row := 1337) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_4
  simp only [← c82137, ← c82138, ← c82139] at e_1337_4
  have e_1337_5 := arithEq_of_rows h (row := 1337) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_5
  simp only [← c82140, ← c82141, k2, ← c82142] at e_1337_5
  have hr := e_1337_5
  simp only [e_1337_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12531)) := by
  have c82143 := (publicBatchWrapper86_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82144 := (publicBatchWrapper86_copies2567 a h).1
  have c82145 := (publicBatchWrapper86_copies2567 a h).2.1
  have c82146 := (publicBatchWrapper86_copies2567 a h).2.2.1
  have c82147 := (publicBatchWrapper86_copies2567 a h).2.2.2.1
  have c82148 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_6 := arithEq_of_rows h (row := 1337) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_6
  simp only [← c82143, ← c82144, ← c82145] at e_1337_6
  have e_1337_7 := arithEq_of_rows h (row := 1337) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_7
  simp only [← c82146, ← c82147, k2, ← c82148] at e_1337_7
  have hr := e_1337_7
  simp only [e_1337_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12532)) := by
  have c82149 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.1
  have c82150 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.1
  have c82151 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.1
  have c82152 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.1
  have c82153 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.1
  have c82154 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_8 := arithEq_of_rows h (row := 1337) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_8
  simp only [← c82149, ← c82150, ← c82151] at e_1337_8
  have e_1337_9 := arithEq_of_rows h (row := 1337) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_9
  simp only [← c82152, ← c82153, k2, ← c82154] at e_1337_9
  have hr := e_1337_9
  simp only [e_1337_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12533)) := by
  have c82155 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82156 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82157 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82158 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82159 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82160 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_10 := arithEq_of_rows h (row := 1337) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_10
  simp only [← c82155, ← c82156, ← c82157] at e_1337_10
  have e_1337_11 := arithEq_of_rows h (row := 1337) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_11
  simp only [← c82158, ← c82159, k2, ← c82160] at e_1337_11
  have hr := e_1337_11
  simp only [e_1337_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12534)) := by
  have c82161 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82162 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82163 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82164 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82165 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82166 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_12 := arithEq_of_rows h (row := 1337) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_12
  simp only [← c82161, ← c82162, ← c82163] at e_1337_12
  have e_1337_13 := arithEq_of_rows h (row := 1337) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_13
  simp only [← c82164, ← c82165, k2, ← c82166] at e_1337_13
  have hr := e_1337_13
  simp only [e_1337_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12535)) := by
  have c82167 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82168 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82169 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82170 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82171 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82172 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_14 := arithEq_of_rows h (row := 1337) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_14
  simp only [← c82167, ← c82168, ← c82169] at e_1337_14
  have e_1337_15 := arithEq_of_rows h (row := 1337) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_15
  simp only [← c82170, ← c82171, k2, ← c82172] at e_1337_15
  have hr := e_1337_15
  simp only [e_1337_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12536)) := by
  have c82173 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82174 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82175 := (publicBatchWrapper86_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82176 := (publicBatchWrapper86_copies2568 a h).1
  have c82177 := (publicBatchWrapper86_copies2568 a h).2.1
  have c82178 := (publicBatchWrapper86_copies2568 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_16 := arithEq_of_rows h (row := 1337) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_16
  simp only [← c82173, ← c82174, ← c82175] at e_1337_16
  have e_1337_17 := arithEq_of_rows h (row := 1337) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_17
  simp only [← c82176, ← c82177, k2, ← c82178] at e_1337_17
  have hr := e_1337_17
  simp only [e_1337_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1337 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12537)) := by
  have c82179 := (publicBatchWrapper86_copies2568 a h).2.2.2.1
  have c82180 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.1
  have c82181 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.1
  have c82182 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.1
  have c82183 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.1
  have c82184 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1337_18 := arithEq_of_rows h (row := 1337) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_18
  simp only [← c82179, ← c82180, ← c82181] at e_1337_18
  have e_1337_19 := arithEq_of_rows h (row := 1337) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_19
  simp only [← c82182, ← c82183, k2, ← c82184] at e_1337_19
  have hr := e_1337_19
  simp only [e_1337_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12538)) := by
  have c82185 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.1
  have c82186 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82187 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82188 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82189 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82190 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_0 := arithEq_of_rows h (row := 1338) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_0
  simp only [← c82185, ← c82186, ← c82187] at e_1338_0
  have e_1338_1 := arithEq_of_rows h (row := 1338) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_1
  simp only [← c82188, ← c82189, k2, ← c82190] at e_1338_1
  have hr := e_1338_1
  simp only [e_1338_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12539)) := by
  have c82191 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82192 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82193 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82194 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82195 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82196 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_2 := arithEq_of_rows h (row := 1338) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_2
  simp only [← c82191, ← c82192, ← c82193] at e_1338_2
  have e_1338_3 := arithEq_of_rows h (row := 1338) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_3
  simp only [← c82194, ← c82195, k2, ← c82196] at e_1338_3
  have hr := e_1338_3
  simp only [e_1338_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12540)) := by
  have c82197 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82198 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82199 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82200 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82201 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82202 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_4 := arithEq_of_rows h (row := 1338) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_4
  simp only [← c82197, ← c82198, ← c82199] at e_1338_4
  have e_1338_5 := arithEq_of_rows h (row := 1338) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_5
  simp only [← c82200, ← c82201, k2, ← c82202] at e_1338_5
  have hr := e_1338_5
  simp only [e_1338_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12541)) := by
  have c82203 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82204 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82205 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82206 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82207 := (publicBatchWrapper86_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82208 := (publicBatchWrapper86_copies2569 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_6 := arithEq_of_rows h (row := 1338) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_6
  simp only [← c82203, ← c82204, ← c82205] at e_1338_6
  have e_1338_7 := arithEq_of_rows h (row := 1338) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_7
  simp only [← c82206, ← c82207, k2, ← c82208] at e_1338_7
  have hr := e_1338_7
  simp only [e_1338_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12542)) := by
  have c82209 := (publicBatchWrapper86_copies2569 a h).2.1
  have c82210 := (publicBatchWrapper86_copies2569 a h).2.2.1
  have c82211 := (publicBatchWrapper86_copies2569 a h).2.2.2.1
  have c82212 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.1
  have c82213 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.1
  have c82214 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_8 := arithEq_of_rows h (row := 1338) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_8
  simp only [← c82209, ← c82210, ← c82211] at e_1338_8
  have e_1338_9 := arithEq_of_rows h (row := 1338) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_9
  simp only [← c82212, ← c82213, k2, ← c82214] at e_1338_9
  have hr := e_1338_9
  simp only [e_1338_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12543)) := by
  have c82215 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.1
  have c82216 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.1
  have c82217 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.1
  have c82218 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82219 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82220 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_10 := arithEq_of_rows h (row := 1338) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_10
  simp only [← c82215, ← c82216, ← c82217] at e_1338_10
  have e_1338_11 := arithEq_of_rows h (row := 1338) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_11
  simp only [← c82218, ← c82219, k2, ← c82220] at e_1338_11
  have hr := e_1338_11
  simp only [e_1338_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12544)) := by
  have c82221 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82222 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82223 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82224 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82225 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82226 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_12 := arithEq_of_rows h (row := 1338) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_12
  simp only [← c82221, ← c82222, ← c82223] at e_1338_12
  have e_1338_13 := arithEq_of_rows h (row := 1338) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_13
  simp only [← c82224, ← c82225, k2, ← c82226] at e_1338_13
  have hr := e_1338_13
  simp only [e_1338_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12545)) := by
  have c82227 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82228 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82229 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82230 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82231 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82232 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_14 := arithEq_of_rows h (row := 1338) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_14
  simp only [← c82227, ← c82228, ← c82229] at e_1338_14
  have e_1338_15 := arithEq_of_rows h (row := 1338) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_15
  simp only [← c82230, ← c82231, k2, ← c82232] at e_1338_15
  have hr := e_1338_15
  simp only [e_1338_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12546)) := by
  have c82233 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82234 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82235 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82236 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82237 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82238 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_16 := arithEq_of_rows h (row := 1338) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_16
  simp only [← c82233, ← c82234, ← c82235] at e_1338_16
  have e_1338_17 := arithEq_of_rows h (row := 1338) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_17
  simp only [← c82236, ← c82237, k2, ← c82238] at e_1338_17
  have hr := e_1338_17
  simp only [e_1338_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1338 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12547)) := by
  have c82239 := (publicBatchWrapper86_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82240 := (publicBatchWrapper86_copies2570 a h).1
  have c82241 := (publicBatchWrapper86_copies2570 a h).2.1
  have c82242 := (publicBatchWrapper86_copies2570 a h).2.2.1
  have c82243 := (publicBatchWrapper86_copies2570 a h).2.2.2.1
  have c82244 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1338_18 := arithEq_of_rows h (row := 1338) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_18
  simp only [← c82239, ← c82240, ← c82241] at e_1338_18
  have e_1338_19 := arithEq_of_rows h (row := 1338) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_19
  simp only [← c82242, ← c82243, k2, ← c82244] at e_1338_19
  have hr := e_1338_19
  simp only [e_1338_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12548)) := by
  have c82245 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.1
  have c82246 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.1
  have c82247 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.1
  have c82248 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.1
  have c82249 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.1
  have c82250 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_0 := arithEq_of_rows h (row := 1339) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_0
  simp only [← c82245, ← c82246, ← c82247] at e_1339_0
  have e_1339_1 := arithEq_of_rows h (row := 1339) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_1
  simp only [← c82248, ← c82249, k2, ← c82250] at e_1339_1
  have hr := e_1339_1
  simp only [e_1339_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12549)) := by
  have c82251 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82252 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82253 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82254 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82255 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82256 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_2 := arithEq_of_rows h (row := 1339) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_2
  simp only [← c82251, ← c82252, ← c82253] at e_1339_2
  have e_1339_3 := arithEq_of_rows h (row := 1339) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_3
  simp only [← c82254, ← c82255, k2, ← c82256] at e_1339_3
  have hr := e_1339_3
  simp only [e_1339_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12550)) := by
  have c82257 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82258 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82259 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82260 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82261 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82262 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_4 := arithEq_of_rows h (row := 1339) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_4
  simp only [← c82257, ← c82258, ← c82259] at e_1339_4
  have e_1339_5 := arithEq_of_rows h (row := 1339) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_5
  simp only [← c82260, ← c82261, k2, ← c82262] at e_1339_5
  have hr := e_1339_5
  simp only [e_1339_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12551)) := by
  have c82263 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82264 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82265 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82266 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82267 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82268 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_6 := arithEq_of_rows h (row := 1339) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_6
  simp only [← c82263, ← c82264, ← c82265] at e_1339_6
  have e_1339_7 := arithEq_of_rows h (row := 1339) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_7
  simp only [← c82266, ← c82267, k2, ← c82268] at e_1339_7
  have hr := e_1339_7
  simp only [e_1339_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12552)) := by
  have c82269 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82270 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82271 := (publicBatchWrapper86_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82272 := (publicBatchWrapper86_copies2571 a h).1
  have c82273 := (publicBatchWrapper86_copies2571 a h).2.1
  have c82274 := (publicBatchWrapper86_copies2571 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_8 := arithEq_of_rows h (row := 1339) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_8
  simp only [← c82269, ← c82270, ← c82271] at e_1339_8
  have e_1339_9 := arithEq_of_rows h (row := 1339) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_9
  simp only [← c82272, ← c82273, k2, ← c82274] at e_1339_9
  have hr := e_1339_9
  simp only [e_1339_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12553)) := by
  have c82275 := (publicBatchWrapper86_copies2571 a h).2.2.2.1
  have c82276 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.1
  have c82277 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.1
  have c82278 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.1
  have c82279 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.1
  have c82280 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_10 := arithEq_of_rows h (row := 1339) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_10
  simp only [← c82275, ← c82276, ← c82277] at e_1339_10
  have e_1339_11 := arithEq_of_rows h (row := 1339) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_11
  simp only [← c82278, ← c82279, k2, ← c82280] at e_1339_11
  have hr := e_1339_11
  simp only [e_1339_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12554)) := by
  have c82281 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.1
  have c82282 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82283 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82284 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82285 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82286 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_12 := arithEq_of_rows h (row := 1339) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_12
  simp only [← c82281, ← c82282, ← c82283] at e_1339_12
  have e_1339_13 := arithEq_of_rows h (row := 1339) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_13
  simp only [← c82284, ← c82285, k2, ← c82286] at e_1339_13
  have hr := e_1339_13
  simp only [e_1339_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12555)) := by
  have c82287 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82288 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82289 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82290 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82291 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82292 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_14 := arithEq_of_rows h (row := 1339) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_14
  simp only [← c82287, ← c82288, ← c82289] at e_1339_14
  have e_1339_15 := arithEq_of_rows h (row := 1339) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_15
  simp only [← c82290, ← c82291, k2, ← c82292] at e_1339_15
  have hr := e_1339_15
  simp only [e_1339_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12556)) := by
  have c82293 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82294 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82295 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82296 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82297 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82298 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_16 := arithEq_of_rows h (row := 1339) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_16
  simp only [← c82293, ← c82294, ← c82295] at e_1339_16
  have e_1339_17 := arithEq_of_rows h (row := 1339) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_17
  simp only [← c82296, ← c82297, k2, ← c82298] at e_1339_17
  have hr := e_1339_17
  simp only [e_1339_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1339 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12557)) := by
  have c82299 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82300 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82301 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82302 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82303 := (publicBatchWrapper86_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82304 := (publicBatchWrapper86_copies2572 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1339_18 := arithEq_of_rows h (row := 1339) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_18
  simp only [← c82299, ← c82300, ← c82301] at e_1339_18
  have e_1339_19 := arithEq_of_rows h (row := 1339) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_19
  simp only [← c82302, ← c82303, k2, ← c82304] at e_1339_19
  have hr := e_1339_19
  simp only [e_1339_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12558)) := by
  have c82305 := (publicBatchWrapper86_copies2572 a h).2.1
  have c82306 := (publicBatchWrapper86_copies2572 a h).2.2.1
  have c82307 := (publicBatchWrapper86_copies2572 a h).2.2.2.1
  have c82308 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.1
  have c82309 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.1
  have c82310 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_0 := arithEq_of_rows h (row := 1340) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_0
  simp only [← c82305, ← c82306, ← c82307] at e_1340_0
  have e_1340_1 := arithEq_of_rows h (row := 1340) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_1
  simp only [← c82308, ← c82309, k2, ← c82310] at e_1340_1
  have hr := e_1340_1
  simp only [e_1340_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12559)) := by
  have c82311 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.1
  have c82312 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.1
  have c82313 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.1
  have c82314 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82315 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82316 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_2 := arithEq_of_rows h (row := 1340) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_2
  simp only [← c82311, ← c82312, ← c82313] at e_1340_2
  have e_1340_3 := arithEq_of_rows h (row := 1340) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_3
  simp only [← c82314, ← c82315, k2, ← c82316] at e_1340_3
  have hr := e_1340_3
  simp only [e_1340_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12560)) := by
  have c82317 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82318 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82319 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82320 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82321 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82322 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_4 := arithEq_of_rows h (row := 1340) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_4
  simp only [← c82317, ← c82318, ← c82319] at e_1340_4
  have e_1340_5 := arithEq_of_rows h (row := 1340) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_5
  simp only [← c82320, ← c82321, k2, ← c82322] at e_1340_5
  have hr := e_1340_5
  simp only [e_1340_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12561)) := by
  have c82323 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82324 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82325 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82326 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82327 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82328 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_6 := arithEq_of_rows h (row := 1340) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_6
  simp only [← c82323, ← c82324, ← c82325] at e_1340_6
  have e_1340_7 := arithEq_of_rows h (row := 1340) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_7
  simp only [← c82326, ← c82327, k2, ← c82328] at e_1340_7
  have hr := e_1340_7
  simp only [e_1340_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12562)) := by
  have c82329 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82330 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82331 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82332 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82333 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82334 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_8 := arithEq_of_rows h (row := 1340) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_8
  simp only [← c82329, ← c82330, ← c82331] at e_1340_8
  have e_1340_9 := arithEq_of_rows h (row := 1340) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_9
  simp only [← c82332, ← c82333, k2, ← c82334] at e_1340_9
  have hr := e_1340_9
  simp only [e_1340_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12563)) := by
  have c82335 := (publicBatchWrapper86_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82336 := (publicBatchWrapper86_copies2573 a h).1
  have c82337 := (publicBatchWrapper86_copies2573 a h).2.1
  have c82338 := (publicBatchWrapper86_copies2573 a h).2.2.1
  have c82339 := (publicBatchWrapper86_copies2573 a h).2.2.2.1
  have c82340 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_10 := arithEq_of_rows h (row := 1340) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_10
  simp only [← c82335, ← c82336, ← c82337] at e_1340_10
  have e_1340_11 := arithEq_of_rows h (row := 1340) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_11
  simp only [← c82338, ← c82339, k2, ← c82340] at e_1340_11
  have hr := e_1340_11
  simp only [e_1340_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12564)) := by
  have c82341 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.1
  have c82342 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.1
  have c82343 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.1
  have c82344 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.1
  have c82345 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.1
  have c82346 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_12 := arithEq_of_rows h (row := 1340) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_12
  simp only [← c82341, ← c82342, ← c82343] at e_1340_12
  have e_1340_13 := arithEq_of_rows h (row := 1340) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_13
  simp only [← c82344, ← c82345, k2, ← c82346] at e_1340_13
  have hr := e_1340_13
  simp only [e_1340_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12565)) := by
  have c82347 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82348 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82349 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82350 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82351 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82352 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_14 := arithEq_of_rows h (row := 1340) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_14
  simp only [← c82347, ← c82348, ← c82349] at e_1340_14
  have e_1340_15 := arithEq_of_rows h (row := 1340) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_15
  simp only [← c82350, ← c82351, k2, ← c82352] at e_1340_15
  have hr := e_1340_15
  simp only [e_1340_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12566)) := by
  have c82353 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82354 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82355 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82356 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82357 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82358 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_16 := arithEq_of_rows h (row := 1340) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_16
  simp only [← c82353, ← c82354, ← c82355] at e_1340_16
  have e_1340_17 := arithEq_of_rows h (row := 1340) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_17
  simp only [← c82356, ← c82357, k2, ← c82358] at e_1340_17
  have hr := e_1340_17
  simp only [e_1340_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1340 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12567)) := by
  have c82359 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82360 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82361 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82362 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82363 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82364 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1340_18 := arithEq_of_rows h (row := 1340) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_18
  simp only [← c82359, ← c82360, ← c82361] at e_1340_18
  have e_1340_19 := arithEq_of_rows h (row := 1340) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_19
  simp only [← c82362, ← c82363, k2, ← c82364] at e_1340_19
  have hr := e_1340_19
  simp only [e_1340_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12568)) := by
  have c82365 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82366 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82367 := (publicBatchWrapper86_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82368 := (publicBatchWrapper86_copies2574 a h).1
  have c82369 := (publicBatchWrapper86_copies2574 a h).2.1
  have c82370 := (publicBatchWrapper86_copies2574 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_0 := arithEq_of_rows h (row := 1341) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_0
  simp only [← c82365, ← c82366, ← c82367] at e_1341_0
  have e_1341_1 := arithEq_of_rows h (row := 1341) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_1
  simp only [← c82368, ← c82369, k2, ← c82370] at e_1341_1
  have hr := e_1341_1
  simp only [e_1341_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12569)) := by
  have c82371 := (publicBatchWrapper86_copies2574 a h).2.2.2.1
  have c82372 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.1
  have c82373 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.1
  have c82374 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.1
  have c82375 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.1
  have c82376 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_2 := arithEq_of_rows h (row := 1341) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_2
  simp only [← c82371, ← c82372, ← c82373] at e_1341_2
  have e_1341_3 := arithEq_of_rows h (row := 1341) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_3
  simp only [← c82374, ← c82375, k2, ← c82376] at e_1341_3
  have hr := e_1341_3
  simp only [e_1341_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12610)) := by
  have c82377 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.1
  have c82378 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82379 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82380 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82381 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82382 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_4 := arithEq_of_rows h (row := 1341) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_4
  simp only [← c82377, ← c82378, ← c82379] at e_1341_4
  have e_1341_5 := arithEq_of_rows h (row := 1341) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_5
  simp only [← c82380, ← c82381, k2, ← c82382] at e_1341_5
  have hr := e_1341_5
  simp only [e_1341_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12611)) := by
  have c82383 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82384 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82385 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82386 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82387 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82388 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_6 := arithEq_of_rows h (row := 1341) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_6
  simp only [← c82383, ← c82384, ← c82385] at e_1341_6
  have e_1341_7 := arithEq_of_rows h (row := 1341) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_7
  simp only [← c82386, ← c82387, k2, ← c82388] at e_1341_7
  have hr := e_1341_7
  simp only [e_1341_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12612)) := by
  have c82389 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82390 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82391 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82392 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82393 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82394 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_8 := arithEq_of_rows h (row := 1341) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_8
  simp only [← c82389, ← c82390, ← c82391] at e_1341_8
  have e_1341_9 := arithEq_of_rows h (row := 1341) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_9
  simp only [← c82392, ← c82393, k2, ← c82394] at e_1341_9
  have hr := e_1341_9
  simp only [e_1341_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12613)) := by
  have c82395 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82396 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82397 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82398 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82399 := (publicBatchWrapper86_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82400 := (publicBatchWrapper86_copies2575 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_10 := arithEq_of_rows h (row := 1341) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_10
  simp only [← c82395, ← c82396, ← c82397] at e_1341_10
  have e_1341_11 := arithEq_of_rows h (row := 1341) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_11
  simp only [← c82398, ← c82399, k2, ← c82400] at e_1341_11
  have hr := e_1341_11
  simp only [e_1341_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12614)) := by
  have c82401 := (publicBatchWrapper86_copies2575 a h).2.1
  have c82402 := (publicBatchWrapper86_copies2575 a h).2.2.1
  have c82403 := (publicBatchWrapper86_copies2575 a h).2.2.2.1
  have c82404 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.1
  have c82405 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.1
  have c82406 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_12 := arithEq_of_rows h (row := 1341) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_12
  simp only [← c82401, ← c82402, ← c82403] at e_1341_12
  have e_1341_13 := arithEq_of_rows h (row := 1341) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_13
  simp only [← c82404, ← c82405, k2, ← c82406] at e_1341_13
  have hr := e_1341_13
  simp only [e_1341_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12615)) := by
  have c82407 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.1
  have c82408 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.1
  have c82409 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.1
  have c82410 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82411 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82412 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_14 := arithEq_of_rows h (row := 1341) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_14
  simp only [← c82407, ← c82408, ← c82409] at e_1341_14
  have e_1341_15 := arithEq_of_rows h (row := 1341) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_15
  simp only [← c82410, ← c82411, k2, ← c82412] at e_1341_15
  have hr := e_1341_15
  simp only [e_1341_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12616)) := by
  have c82413 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82414 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82415 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82416 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82417 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82418 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_16 := arithEq_of_rows h (row := 1341) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_16
  simp only [← c82413, ← c82414, ← c82415] at e_1341_16
  have e_1341_17 := arithEq_of_rows h (row := 1341) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_17
  simp only [← c82416, ← c82417, k2, ← c82418] at e_1341_17
  have hr := e_1341_17
  simp only [e_1341_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1341 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12617)) := by
  have c82419 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82420 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82421 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82422 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82423 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82424 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1341_18 := arithEq_of_rows h (row := 1341) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_18
  simp only [← c82419, ← c82420, ← c82421] at e_1341_18
  have e_1341_19 := arithEq_of_rows h (row := 1341) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_19
  simp only [← c82422, ← c82423, k2, ← c82424] at e_1341_19
  have hr := e_1341_19
  simp only [e_1341_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12618)) := by
  have c82425 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82426 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82427 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82428 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82429 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82430 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_0 := arithEq_of_rows h (row := 1342) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_0
  simp only [← c82425, ← c82426, ← c82427] at e_1342_0
  have e_1342_1 := arithEq_of_rows h (row := 1342) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_1
  simp only [← c82428, ← c82429, k2, ← c82430] at e_1342_1
  have hr := e_1342_1
  simp only [e_1342_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12619)) := by
  have c82431 := (publicBatchWrapper86_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82432 := (publicBatchWrapper86_copies2576 a h).1
  have c82433 := (publicBatchWrapper86_copies2576 a h).2.1
  have c82434 := (publicBatchWrapper86_copies2576 a h).2.2.1
  have c82435 := (publicBatchWrapper86_copies2576 a h).2.2.2.1
  have c82436 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_2 := arithEq_of_rows h (row := 1342) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_2
  simp only [← c82431, ← c82432, ← c82433] at e_1342_2
  have e_1342_3 := arithEq_of_rows h (row := 1342) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_3
  simp only [← c82434, ← c82435, k2, ← c82436] at e_1342_3
  have hr := e_1342_3
  simp only [e_1342_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12620)) := by
  have c82437 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.1
  have c82438 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.1
  have c82439 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.1
  have c82440 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.1
  have c82441 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.1
  have c82442 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_4 := arithEq_of_rows h (row := 1342) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_4
  simp only [← c82437, ← c82438, ← c82439] at e_1342_4
  have e_1342_5 := arithEq_of_rows h (row := 1342) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_5
  simp only [← c82440, ← c82441, k2, ← c82442] at e_1342_5
  have hr := e_1342_5
  simp only [e_1342_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12621)) := by
  have c82443 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82444 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82445 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82446 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82447 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82448 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_6 := arithEq_of_rows h (row := 1342) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_6
  simp only [← c82443, ← c82444, ← c82445] at e_1342_6
  have e_1342_7 := arithEq_of_rows h (row := 1342) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_7
  simp only [← c82446, ← c82447, k2, ← c82448] at e_1342_7
  have hr := e_1342_7
  simp only [e_1342_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12622)) := by
  have c82449 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82450 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82451 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82452 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82453 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82454 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_8 := arithEq_of_rows h (row := 1342) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_8
  simp only [← c82449, ← c82450, ← c82451] at e_1342_8
  have e_1342_9 := arithEq_of_rows h (row := 1342) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_9
  simp only [← c82452, ← c82453, k2, ← c82454] at e_1342_9
  have hr := e_1342_9
  simp only [e_1342_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12623)) := by
  have c82455 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82456 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82457 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82458 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82459 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82460 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_10 := arithEq_of_rows h (row := 1342) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_10
  simp only [← c82455, ← c82456, ← c82457] at e_1342_10
  have e_1342_11 := arithEq_of_rows h (row := 1342) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_11
  simp only [← c82458, ← c82459, k2, ← c82460] at e_1342_11
  have hr := e_1342_11
  simp only [e_1342_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12624)) := by
  have c82461 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82462 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82463 := (publicBatchWrapper86_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82464 := (publicBatchWrapper86_copies2577 a h).1
  have c82465 := (publicBatchWrapper86_copies2577 a h).2.1
  have c82466 := (publicBatchWrapper86_copies2577 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_12 := arithEq_of_rows h (row := 1342) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_12
  simp only [← c82461, ← c82462, ← c82463] at e_1342_12
  have e_1342_13 := arithEq_of_rows h (row := 1342) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_13
  simp only [← c82464, ← c82465, k2, ← c82466] at e_1342_13
  have hr := e_1342_13
  simp only [e_1342_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12625)) := by
  have c82467 := (publicBatchWrapper86_copies2577 a h).2.2.2.1
  have c82468 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.1
  have c82469 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.1
  have c82470 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.1
  have c82471 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.1
  have c82472 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_14 := arithEq_of_rows h (row := 1342) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_14
  simp only [← c82467, ← c82468, ← c82469] at e_1342_14
  have e_1342_15 := arithEq_of_rows h (row := 1342) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_15
  simp only [← c82470, ← c82471, k2, ← c82472] at e_1342_15
  have hr := e_1342_15
  simp only [e_1342_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12626)) := by
  have c82473 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.1
  have c82474 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82475 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82476 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82477 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82478 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_16 := arithEq_of_rows h (row := 1342) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_16
  simp only [← c82473, ← c82474, ← c82475] at e_1342_16
  have e_1342_17 := arithEq_of_rows h (row := 1342) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_17
  simp only [← c82476, ← c82477, k2, ← c82478] at e_1342_17
  have hr := e_1342_17
  simp only [e_1342_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1342 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12627)) := by
  have c82479 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82480 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82481 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82482 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82483 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82484 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1342_18 := arithEq_of_rows h (row := 1342) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_18
  simp only [← c82479, ← c82480, ← c82481] at e_1342_18
  have e_1342_19 := arithEq_of_rows h (row := 1342) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_19
  simp only [← c82482, ← c82483, k2, ← c82484] at e_1342_19
  have hr := e_1342_19
  simp only [e_1342_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12628)) := by
  have c82485 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82486 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82487 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82488 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82489 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82490 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_0 := arithEq_of_rows h (row := 1343) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_0
  simp only [← c82485, ← c82486, ← c82487] at e_1343_0
  have e_1343_1 := arithEq_of_rows h (row := 1343) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_1
  simp only [← c82488, ← c82489, k2, ← c82490] at e_1343_1
  have hr := e_1343_1
  simp only [e_1343_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12629)) := by
  have c82491 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82492 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82493 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82494 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82495 := (publicBatchWrapper86_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82496 := (publicBatchWrapper86_copies2578 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_2 := arithEq_of_rows h (row := 1343) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_2
  simp only [← c82491, ← c82492, ← c82493] at e_1343_2
  have e_1343_3 := arithEq_of_rows h (row := 1343) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_3
  simp only [← c82494, ← c82495, k2, ← c82496] at e_1343_3
  have hr := e_1343_3
  simp only [e_1343_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12630)) := by
  have c82497 := (publicBatchWrapper86_copies2578 a h).2.1
  have c82498 := (publicBatchWrapper86_copies2578 a h).2.2.1
  have c82499 := (publicBatchWrapper86_copies2578 a h).2.2.2.1
  have c82500 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.1
  have c82501 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.1
  have c82502 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_4 := arithEq_of_rows h (row := 1343) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_4
  simp only [← c82497, ← c82498, ← c82499] at e_1343_4
  have e_1343_5 := arithEq_of_rows h (row := 1343) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_5
  simp only [← c82500, ← c82501, k2, ← c82502] at e_1343_5
  have hr := e_1343_5
  simp only [e_1343_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12631)) := by
  have c82503 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.1
  have c82504 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.1
  have c82505 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.1
  have c82506 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82507 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82508 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_6 := arithEq_of_rows h (row := 1343) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_6
  simp only [← c82503, ← c82504, ← c82505] at e_1343_6
  have e_1343_7 := arithEq_of_rows h (row := 1343) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_7
  simp only [← c82506, ← c82507, k2, ← c82508] at e_1343_7
  have hr := e_1343_7
  simp only [e_1343_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12632)) := by
  have c82509 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82510 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82511 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82512 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82513 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82514 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_8 := arithEq_of_rows h (row := 1343) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_8
  simp only [← c82509, ← c82510, ← c82511] at e_1343_8
  have e_1343_9 := arithEq_of_rows h (row := 1343) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_9
  simp only [← c82512, ← c82513, k2, ← c82514] at e_1343_9
  have hr := e_1343_9
  simp only [e_1343_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12633)) := by
  have c82515 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82516 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82517 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82518 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82519 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82520 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_10 := arithEq_of_rows h (row := 1343) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_10
  simp only [← c82515, ← c82516, ← c82517] at e_1343_10
  have e_1343_11 := arithEq_of_rows h (row := 1343) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_11
  simp only [← c82518, ← c82519, k2, ← c82520] at e_1343_11
  have hr := e_1343_11
  simp only [e_1343_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12634)) := by
  have c82521 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82522 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82523 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82524 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82525 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82526 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_12 := arithEq_of_rows h (row := 1343) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_12
  simp only [← c82521, ← c82522, ← c82523] at e_1343_12
  have e_1343_13 := arithEq_of_rows h (row := 1343) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_13
  simp only [← c82524, ← c82525, k2, ← c82526] at e_1343_13
  have hr := e_1343_13
  simp only [e_1343_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12635)) := by
  have c82527 := (publicBatchWrapper86_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82528 := (publicBatchWrapper86_copies2579 a h).1
  have c82529 := (publicBatchWrapper86_copies2579 a h).2.1
  have c82530 := (publicBatchWrapper86_copies2579 a h).2.2.1
  have c82531 := (publicBatchWrapper86_copies2579 a h).2.2.2.1
  have c82532 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_14 := arithEq_of_rows h (row := 1343) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_14
  simp only [← c82527, ← c82528, ← c82529] at e_1343_14
  have e_1343_15 := arithEq_of_rows h (row := 1343) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_15
  simp only [← c82530, ← c82531, k2, ← c82532] at e_1343_15
  have hr := e_1343_15
  simp only [e_1343_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12636)) := by
  have c82533 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.1
  have c82534 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.1
  have c82535 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.1
  have c82536 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.1
  have c82537 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.1
  have c82538 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_16 := arithEq_of_rows h (row := 1343) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_16
  simp only [← c82533, ← c82534, ← c82535] at e_1343_16
  have e_1343_17 := arithEq_of_rows h (row := 1343) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_17
  simp only [← c82536, ← c82537, k2, ← c82538] at e_1343_17
  have hr := e_1343_17
  simp only [e_1343_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1343 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12637)) := by
  have c82539 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82540 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82541 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82542 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82543 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82544 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1343_18 := arithEq_of_rows h (row := 1343) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_18
  simp only [← c82539, ← c82540, ← c82541] at e_1343_18
  have e_1343_19 := arithEq_of_rows h (row := 1343) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_19
  simp only [← c82542, ← c82543, k2, ← c82544] at e_1343_19
  have hr := e_1343_19
  simp only [e_1343_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12638)) := by
  have c82545 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82546 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82547 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82548 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82549 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82550 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_0 := arithEq_of_rows h (row := 1344) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_0
  simp only [← c82545, ← c82546, ← c82547] at e_1344_0
  have e_1344_1 := arithEq_of_rows h (row := 1344) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_1
  simp only [← c82548, ← c82549, k2, ← c82550] at e_1344_1
  have hr := e_1344_1
  simp only [e_1344_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12639)) := by
  have c82551 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82552 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82553 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82554 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82555 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82556 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_2 := arithEq_of_rows h (row := 1344) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_2
  simp only [← c82551, ← c82552, ← c82553] at e_1344_2
  have e_1344_3 := arithEq_of_rows h (row := 1344) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_3
  simp only [← c82554, ← c82555, k2, ← c82556] at e_1344_3
  have hr := e_1344_3
  simp only [e_1344_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12640)) := by
  have c82557 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82558 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82559 := (publicBatchWrapper86_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82560 := (publicBatchWrapper86_copies2580 a h).1
  have c82561 := (publicBatchWrapper86_copies2580 a h).2.1
  have c82562 := (publicBatchWrapper86_copies2580 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_4 := arithEq_of_rows h (row := 1344) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_4
  simp only [← c82557, ← c82558, ← c82559] at e_1344_4
  have e_1344_5 := arithEq_of_rows h (row := 1344) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_5
  simp only [← c82560, ← c82561, k2, ← c82562] at e_1344_5
  have hr := e_1344_5
  simp only [e_1344_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12641)) := by
  have c82563 := (publicBatchWrapper86_copies2580 a h).2.2.2.1
  have c82564 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.1
  have c82565 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.1
  have c82566 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.1
  have c82567 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.1
  have c82568 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_6 := arithEq_of_rows h (row := 1344) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_6
  simp only [← c82563, ← c82564, ← c82565] at e_1344_6
  have e_1344_7 := arithEq_of_rows h (row := 1344) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_7
  simp only [← c82566, ← c82567, k2, ← c82568] at e_1344_7
  have hr := e_1344_7
  simp only [e_1344_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12642)) := by
  have c82569 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.1
  have c82570 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82571 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82572 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82573 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82574 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_8 := arithEq_of_rows h (row := 1344) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_8
  simp only [← c82569, ← c82570, ← c82571] at e_1344_8
  have e_1344_9 := arithEq_of_rows h (row := 1344) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_9
  simp only [← c82572, ← c82573, k2, ← c82574] at e_1344_9
  have hr := e_1344_9
  simp only [e_1344_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12643)) := by
  have c82575 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82576 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82577 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82578 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82579 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82580 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_10 := arithEq_of_rows h (row := 1344) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_10
  simp only [← c82575, ← c82576, ← c82577] at e_1344_10
  have e_1344_11 := arithEq_of_rows h (row := 1344) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_11
  simp only [← c82578, ← c82579, k2, ← c82580] at e_1344_11
  have hr := e_1344_11
  simp only [e_1344_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12644)) := by
  have c82581 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82582 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82583 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82584 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82585 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82586 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_12 := arithEq_of_rows h (row := 1344) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_12
  simp only [← c82581, ← c82582, ← c82583] at e_1344_12
  have e_1344_13 := arithEq_of_rows h (row := 1344) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_13
  simp only [← c82584, ← c82585, k2, ← c82586] at e_1344_13
  have hr := e_1344_13
  simp only [e_1344_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12645)) := by
  have c82587 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82588 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82589 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82590 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82591 := (publicBatchWrapper86_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82592 := (publicBatchWrapper86_copies2581 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_14 := arithEq_of_rows h (row := 1344) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_14
  simp only [← c82587, ← c82588, ← c82589] at e_1344_14
  have e_1344_15 := arithEq_of_rows h (row := 1344) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_15
  simp only [← c82590, ← c82591, k2, ← c82592] at e_1344_15
  have hr := e_1344_15
  simp only [e_1344_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12646)) := by
  have c82593 := (publicBatchWrapper86_copies2581 a h).2.1
  have c82594 := (publicBatchWrapper86_copies2581 a h).2.2.1
  have c82595 := (publicBatchWrapper86_copies2581 a h).2.2.2.1
  have c82596 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.1
  have c82597 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.1
  have c82598 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_16 := arithEq_of_rows h (row := 1344) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_16
  simp only [← c82593, ← c82594, ← c82595] at e_1344_16
  have e_1344_17 := arithEq_of_rows h (row := 1344) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_17
  simp only [← c82596, ← c82597, k2, ← c82598] at e_1344_17
  have hr := e_1344_17
  simp only [e_1344_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1344 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12647)) := by
  have c82599 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.1
  have c82600 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.1
  have c82601 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.1
  have c82602 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82603 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82604 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1344_18 := arithEq_of_rows h (row := 1344) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_18
  simp only [← c82599, ← c82600, ← c82601] at e_1344_18
  have e_1344_19 := arithEq_of_rows h (row := 1344) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_19
  simp only [← c82602, ← c82603, k2, ← c82604] at e_1344_19
  have hr := e_1344_19
  simp only [e_1344_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12648)) := by
  have c82605 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82606 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82607 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82608 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82609 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82610 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_0 := arithEq_of_rows h (row := 1345) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_0
  simp only [← c82605, ← c82606, ← c82607] at e_1345_0
  have e_1345_1 := arithEq_of_rows h (row := 1345) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_1
  simp only [← c82608, ← c82609, k2, ← c82610] at e_1345_1
  have hr := e_1345_1
  simp only [e_1345_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12649)) := by
  have c82611 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82612 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82613 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82614 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82615 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82616 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_2 := arithEq_of_rows h (row := 1345) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_2
  simp only [← c82611, ← c82612, ← c82613] at e_1345_2
  have e_1345_3 := arithEq_of_rows h (row := 1345) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_3
  simp only [← c82614, ← c82615, k2, ← c82616] at e_1345_3
  have hr := e_1345_3
  simp only [e_1345_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12650)) := by
  have c82617 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82618 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82619 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82620 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82621 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82622 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_4 := arithEq_of_rows h (row := 1345) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_4
  simp only [← c82617, ← c82618, ← c82619] at e_1345_4
  have e_1345_5 := arithEq_of_rows h (row := 1345) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_5
  simp only [← c82620, ← c82621, k2, ← c82622] at e_1345_5
  have hr := e_1345_5
  simp only [e_1345_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12651)) := by
  have c82623 := (publicBatchWrapper86_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82624 := (publicBatchWrapper86_copies2582 a h).1
  have c82625 := (publicBatchWrapper86_copies2582 a h).2.1
  have c82626 := (publicBatchWrapper86_copies2582 a h).2.2.1
  have c82627 := (publicBatchWrapper86_copies2582 a h).2.2.2.1
  have c82628 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_6 := arithEq_of_rows h (row := 1345) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_6
  simp only [← c82623, ← c82624, ← c82625] at e_1345_6
  have e_1345_7 := arithEq_of_rows h (row := 1345) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_7
  simp only [← c82626, ← c82627, k2, ← c82628] at e_1345_7
  have hr := e_1345_7
  simp only [e_1345_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12652)) := by
  have c82629 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.1
  have c82630 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.1
  have c82631 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.1
  have c82632 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.1
  have c82633 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.1
  have c82634 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_8 := arithEq_of_rows h (row := 1345) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_8
  simp only [← c82629, ← c82630, ← c82631] at e_1345_8
  have e_1345_9 := arithEq_of_rows h (row := 1345) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_9
  simp only [← c82632, ← c82633, k2, ← c82634] at e_1345_9
  have hr := e_1345_9
  simp only [e_1345_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12653)) := by
  have c82635 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82636 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82637 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82638 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82639 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82640 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_10 := arithEq_of_rows h (row := 1345) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_10
  simp only [← c82635, ← c82636, ← c82637] at e_1345_10
  have e_1345_11 := arithEq_of_rows h (row := 1345) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_11
  simp only [← c82638, ← c82639, k2, ← c82640] at e_1345_11
  have hr := e_1345_11
  simp only [e_1345_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12654)) := by
  have c82641 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82642 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82643 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82644 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82645 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82646 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_12 := arithEq_of_rows h (row := 1345) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_12
  simp only [← c82641, ← c82642, ← c82643] at e_1345_12
  have e_1345_13 := arithEq_of_rows h (row := 1345) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_13
  simp only [← c82644, ← c82645, k2, ← c82646] at e_1345_13
  have hr := e_1345_13
  simp only [e_1345_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12655)) := by
  have c82647 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82648 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82649 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82650 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82651 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82652 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_14 := arithEq_of_rows h (row := 1345) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_14
  simp only [← c82647, ← c82648, ← c82649] at e_1345_14
  have e_1345_15 := arithEq_of_rows h (row := 1345) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_15
  simp only [← c82650, ← c82651, k2, ← c82652] at e_1345_15
  have hr := e_1345_15
  simp only [e_1345_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12656)) := by
  have c82653 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82654 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82655 := (publicBatchWrapper86_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82656 := (publicBatchWrapper86_copies2583 a h).1
  have c82657 := (publicBatchWrapper86_copies2583 a h).2.1
  have c82658 := (publicBatchWrapper86_copies2583 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_16 := arithEq_of_rows h (row := 1345) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_16
  simp only [← c82653, ← c82654, ← c82655] at e_1345_16
  have e_1345_17 := arithEq_of_rows h (row := 1345) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_17
  simp only [← c82656, ← c82657, k2, ← c82658] at e_1345_17
  have hr := e_1345_17
  simp only [e_1345_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1345 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12657)) := by
  have c82659 := (publicBatchWrapper86_copies2583 a h).2.2.2.1
  have c82660 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.1
  have c82661 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.1
  have c82662 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.1
  have c82663 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.1
  have c82664 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1345_18 := arithEq_of_rows h (row := 1345) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_18
  simp only [← c82659, ← c82660, ← c82661] at e_1345_18
  have e_1345_19 := arithEq_of_rows h (row := 1345) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_19
  simp only [← c82662, ← c82663, k2, ← c82664] at e_1345_19
  have hr := e_1345_19
  simp only [e_1345_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12658)) := by
  have c82665 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.1
  have c82666 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82667 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82668 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82669 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82670 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_0 := arithEq_of_rows h (row := 1346) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_0
  simp only [← c82665, ← c82666, ← c82667] at e_1346_0
  have e_1346_1 := arithEq_of_rows h (row := 1346) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_1
  simp only [← c82668, ← c82669, k2, ← c82670] at e_1346_1
  have hr := e_1346_1
  simp only [e_1346_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12659)) := by
  have c82671 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82672 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82673 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82674 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82675 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82676 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_2 := arithEq_of_rows h (row := 1346) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_2
  simp only [← c82671, ← c82672, ← c82673] at e_1346_2
  have e_1346_3 := arithEq_of_rows h (row := 1346) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_3
  simp only [← c82674, ← c82675, k2, ← c82676] at e_1346_3
  have hr := e_1346_3
  simp only [e_1346_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12660)) := by
  have c82677 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82678 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82679 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82680 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82681 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82682 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_4 := arithEq_of_rows h (row := 1346) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_4
  simp only [← c82677, ← c82678, ← c82679] at e_1346_4
  have e_1346_5 := arithEq_of_rows h (row := 1346) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_5
  simp only [← c82680, ← c82681, k2, ← c82682] at e_1346_5
  have hr := e_1346_5
  simp only [e_1346_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12661)) := by
  have c82683 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82684 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82685 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82686 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82687 := (publicBatchWrapper86_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82688 := (publicBatchWrapper86_copies2584 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_6 := arithEq_of_rows h (row := 1346) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_6
  simp only [← c82683, ← c82684, ← c82685] at e_1346_6
  have e_1346_7 := arithEq_of_rows h (row := 1346) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_7
  simp only [← c82686, ← c82687, k2, ← c82688] at e_1346_7
  have hr := e_1346_7
  simp only [e_1346_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12662)) := by
  have c82689 := (publicBatchWrapper86_copies2584 a h).2.1
  have c82690 := (publicBatchWrapper86_copies2584 a h).2.2.1
  have c82691 := (publicBatchWrapper86_copies2584 a h).2.2.2.1
  have c82692 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.1
  have c82693 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.1
  have c82694 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_8 := arithEq_of_rows h (row := 1346) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_8
  simp only [← c82689, ← c82690, ← c82691] at e_1346_8
  have e_1346_9 := arithEq_of_rows h (row := 1346) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_9
  simp only [← c82692, ← c82693, k2, ← c82694] at e_1346_9
  have hr := e_1346_9
  simp only [e_1346_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12663)) := by
  have c82695 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.1
  have c82696 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.1
  have c82697 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.1
  have c82698 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82699 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82700 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_10 := arithEq_of_rows h (row := 1346) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_10
  simp only [← c82695, ← c82696, ← c82697] at e_1346_10
  have e_1346_11 := arithEq_of_rows h (row := 1346) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_11
  simp only [← c82698, ← c82699, k2, ← c82700] at e_1346_11
  have hr := e_1346_11
  simp only [e_1346_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12664)) := by
  have c82701 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82702 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82703 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82704 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82705 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82706 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_12 := arithEq_of_rows h (row := 1346) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_12
  simp only [← c82701, ← c82702, ← c82703] at e_1346_12
  have e_1346_13 := arithEq_of_rows h (row := 1346) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_13
  simp only [← c82704, ← c82705, k2, ← c82706] at e_1346_13
  have hr := e_1346_13
  simp only [e_1346_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12665)) := by
  have c82707 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82708 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82709 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82710 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82711 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82712 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_14 := arithEq_of_rows h (row := 1346) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_14
  simp only [← c82707, ← c82708, ← c82709] at e_1346_14
  have e_1346_15 := arithEq_of_rows h (row := 1346) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_15
  simp only [← c82710, ← c82711, k2, ← c82712] at e_1346_15
  have hr := e_1346_15
  simp only [e_1346_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12666)) := by
  have c82713 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82714 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82715 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82716 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82717 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82718 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_16 := arithEq_of_rows h (row := 1346) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_16
  simp only [← c82713, ← c82714, ← c82715] at e_1346_16
  have e_1346_17 := arithEq_of_rows h (row := 1346) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_17
  simp only [← c82716, ← c82717, k2, ← c82718] at e_1346_17
  have hr := e_1346_17
  simp only [e_1346_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1346 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12667)) := by
  have c82719 := (publicBatchWrapper86_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82720 := (publicBatchWrapper86_copies2585 a h).1
  have c82721 := (publicBatchWrapper86_copies2585 a h).2.1
  have c82722 := (publicBatchWrapper86_copies2585 a h).2.2.1
  have c82723 := (publicBatchWrapper86_copies2585 a h).2.2.2.1
  have c82724 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1346_18 := arithEq_of_rows h (row := 1346) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_18
  simp only [← c82719, ← c82720, ← c82721] at e_1346_18
  have e_1346_19 := arithEq_of_rows h (row := 1346) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_19
  simp only [← c82722, ← c82723, k2, ← c82724] at e_1346_19
  have hr := e_1346_19
  simp only [e_1346_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12668)) := by
  have c82725 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.1
  have c82726 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.1
  have c82727 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.1
  have c82728 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.1
  have c82729 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.1
  have c82730 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_0 := arithEq_of_rows h (row := 1347) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_0
  simp only [← c82725, ← c82726, ← c82727] at e_1347_0
  have e_1347_1 := arithEq_of_rows h (row := 1347) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_1
  simp only [← c82728, ← c82729, k2, ← c82730] at e_1347_1
  have hr := e_1347_1
  simp only [e_1347_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12669)) := by
  have c82731 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82732 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82733 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82734 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82735 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82736 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_2 := arithEq_of_rows h (row := 1347) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_2
  simp only [← c82731, ← c82732, ← c82733] at e_1347_2
  have e_1347_3 := arithEq_of_rows h (row := 1347) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_3
  simp only [← c82734, ← c82735, k2, ← c82736] at e_1347_3
  have hr := e_1347_3
  simp only [e_1347_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12670)) := by
  have c82737 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82738 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82739 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82740 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82741 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82742 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_4 := arithEq_of_rows h (row := 1347) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_4
  simp only [← c82737, ← c82738, ← c82739] at e_1347_4
  have e_1347_5 := arithEq_of_rows h (row := 1347) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_5
  simp only [← c82740, ← c82741, k2, ← c82742] at e_1347_5
  have hr := e_1347_5
  simp only [e_1347_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12671)) := by
  have c82743 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82744 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82745 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82746 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82747 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82748 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_6 := arithEq_of_rows h (row := 1347) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_6
  simp only [← c82743, ← c82744, ← c82745] at e_1347_6
  have e_1347_7 := arithEq_of_rows h (row := 1347) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_7
  simp only [← c82746, ← c82747, k2, ← c82748] at e_1347_7
  have hr := e_1347_7
  simp only [e_1347_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12672)) := by
  have c82749 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82750 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82751 := (publicBatchWrapper86_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82752 := (publicBatchWrapper86_copies2586 a h).1
  have c82753 := (publicBatchWrapper86_copies2586 a h).2.1
  have c82754 := (publicBatchWrapper86_copies2586 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_8 := arithEq_of_rows h (row := 1347) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_8
  simp only [← c82749, ← c82750, ← c82751] at e_1347_8
  have e_1347_9 := arithEq_of_rows h (row := 1347) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_9
  simp only [← c82752, ← c82753, k2, ← c82754] at e_1347_9
  have hr := e_1347_9
  simp only [e_1347_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12673)) := by
  have c82755 := (publicBatchWrapper86_copies2586 a h).2.2.2.1
  have c82756 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.1
  have c82757 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.1
  have c82758 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.1
  have c82759 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.1
  have c82760 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_10 := arithEq_of_rows h (row := 1347) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_10
  simp only [← c82755, ← c82756, ← c82757] at e_1347_10
  have e_1347_11 := arithEq_of_rows h (row := 1347) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_11
  simp only [← c82758, ← c82759, k2, ← c82760] at e_1347_11
  have hr := e_1347_11
  simp only [e_1347_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12674)) := by
  have c82761 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.1
  have c82762 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82763 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82764 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82765 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82766 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_12 := arithEq_of_rows h (row := 1347) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_12
  simp only [← c82761, ← c82762, ← c82763] at e_1347_12
  have e_1347_13 := arithEq_of_rows h (row := 1347) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_13
  simp only [← c82764, ← c82765, k2, ← c82766] at e_1347_13
  have hr := e_1347_13
  simp only [e_1347_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12675)) := by
  have c82767 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82768 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82769 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82770 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82771 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82772 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_14 := arithEq_of_rows h (row := 1347) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_14
  simp only [← c82767, ← c82768, ← c82769] at e_1347_14
  have e_1347_15 := arithEq_of_rows h (row := 1347) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_15
  simp only [← c82770, ← c82771, k2, ← c82772] at e_1347_15
  have hr := e_1347_15
  simp only [e_1347_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12676)) := by
  have c82773 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82774 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82775 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82776 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82777 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82778 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_16 := arithEq_of_rows h (row := 1347) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_16
  simp only [← c82773, ← c82774, ← c82775] at e_1347_16
  have e_1347_17 := arithEq_of_rows h (row := 1347) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_17
  simp only [← c82776, ← c82777, k2, ← c82778] at e_1347_17
  have hr := e_1347_17
  simp only [e_1347_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1347 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12677)) := by
  have c82779 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82780 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82781 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82782 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82783 := (publicBatchWrapper86_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82784 := (publicBatchWrapper86_copies2587 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1347_18 := arithEq_of_rows h (row := 1347) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_18
  simp only [← c82779, ← c82780, ← c82781] at e_1347_18
  have e_1347_19 := arithEq_of_rows h (row := 1347) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_19
  simp only [← c82782, ← c82783, k2, ← c82784] at e_1347_19
  have hr := e_1347_19
  simp only [e_1347_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12678)) := by
  have c82785 := (publicBatchWrapper86_copies2587 a h).2.1
  have c82786 := (publicBatchWrapper86_copies2587 a h).2.2.1
  have c82787 := (publicBatchWrapper86_copies2587 a h).2.2.2.1
  have c82788 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.1
  have c82789 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.1
  have c82790 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_0 := arithEq_of_rows h (row := 1348) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_0
  simp only [← c82785, ← c82786, ← c82787] at e_1348_0
  have e_1348_1 := arithEq_of_rows h (row := 1348) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_1
  simp only [← c82788, ← c82789, k2, ← c82790] at e_1348_1
  have hr := e_1348_1
  simp only [e_1348_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12679)) := by
  have c82791 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.1
  have c82792 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.1
  have c82793 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.1
  have c82794 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82795 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82796 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_2 := arithEq_of_rows h (row := 1348) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_2
  simp only [← c82791, ← c82792, ← c82793] at e_1348_2
  have e_1348_3 := arithEq_of_rows h (row := 1348) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_3
  simp only [← c82794, ← c82795, k2, ← c82796] at e_1348_3
  have hr := e_1348_3
  simp only [e_1348_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12680)) := by
  have c82797 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82798 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82799 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82800 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82801 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82802 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_4 := arithEq_of_rows h (row := 1348) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_4
  simp only [← c82797, ← c82798, ← c82799] at e_1348_4
  have e_1348_5 := arithEq_of_rows h (row := 1348) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_5
  simp only [← c82800, ← c82801, k2, ← c82802] at e_1348_5
  have hr := e_1348_5
  simp only [e_1348_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12681)) := by
  have c82803 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82804 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82805 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82806 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82807 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82808 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_6 := arithEq_of_rows h (row := 1348) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_6
  simp only [← c82803, ← c82804, ← c82805] at e_1348_6
  have e_1348_7 := arithEq_of_rows h (row := 1348) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_7
  simp only [← c82806, ← c82807, k2, ← c82808] at e_1348_7
  have hr := e_1348_7
  simp only [e_1348_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12682)) := by
  have c82809 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82810 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82811 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82812 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82813 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82814 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_8 := arithEq_of_rows h (row := 1348) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_8
  simp only [← c82809, ← c82810, ← c82811] at e_1348_8
  have e_1348_9 := arithEq_of_rows h (row := 1348) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_9
  simp only [← c82812, ← c82813, k2, ← c82814] at e_1348_9
  have hr := e_1348_9
  simp only [e_1348_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12683)) := by
  have c82815 := (publicBatchWrapper86_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82816 := (publicBatchWrapper86_copies2588 a h).1
  have c82817 := (publicBatchWrapper86_copies2588 a h).2.1
  have c82818 := (publicBatchWrapper86_copies2588 a h).2.2.1
  have c82819 := (publicBatchWrapper86_copies2588 a h).2.2.2.1
  have c82820 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_10 := arithEq_of_rows h (row := 1348) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_10
  simp only [← c82815, ← c82816, ← c82817] at e_1348_10
  have e_1348_11 := arithEq_of_rows h (row := 1348) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_11
  simp only [← c82818, ← c82819, k2, ← c82820] at e_1348_11
  have hr := e_1348_11
  simp only [e_1348_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12684)) := by
  have c82821 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.1
  have c82822 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.1
  have c82823 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.1
  have c82824 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.1
  have c82825 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.1
  have c82826 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_12 := arithEq_of_rows h (row := 1348) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_12
  simp only [← c82821, ← c82822, ← c82823] at e_1348_12
  have e_1348_13 := arithEq_of_rows h (row := 1348) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_13
  simp only [← c82824, ← c82825, k2, ← c82826] at e_1348_13
  have hr := e_1348_13
  simp only [e_1348_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12685)) := by
  have c82827 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82828 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82829 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82830 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82831 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82832 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_14 := arithEq_of_rows h (row := 1348) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_14
  simp only [← c82827, ← c82828, ← c82829] at e_1348_14
  have e_1348_15 := arithEq_of_rows h (row := 1348) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_15
  simp only [← c82830, ← c82831, k2, ← c82832] at e_1348_15
  have hr := e_1348_15
  simp only [e_1348_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12686)) := by
  have c82833 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82834 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82835 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82836 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82837 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82838 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_16 := arithEq_of_rows h (row := 1348) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_16
  simp only [← c82833, ← c82834, ← c82835] at e_1348_16
  have e_1348_17 := arithEq_of_rows h (row := 1348) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_17
  simp only [← c82836, ← c82837, k2, ← c82838] at e_1348_17
  have hr := e_1348_17
  simp only [e_1348_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1348 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12687)) := by
  have c82839 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82840 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82841 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82842 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82843 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82844 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1348_18 := arithEq_of_rows h (row := 1348) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_18
  simp only [← c82839, ← c82840, ← c82841] at e_1348_18
  have e_1348_19 := arithEq_of_rows h (row := 1348) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_19
  simp only [← c82842, ← c82843, k2, ← c82844] at e_1348_19
  have hr := e_1348_19
  simp only [e_1348_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12688)) := by
  have c82845 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82846 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82847 := (publicBatchWrapper86_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82848 := (publicBatchWrapper86_copies2589 a h).1
  have c82849 := (publicBatchWrapper86_copies2589 a h).2.1
  have c82850 := (publicBatchWrapper86_copies2589 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_0 := arithEq_of_rows h (row := 1349) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_0
  simp only [← c82845, ← c82846, ← c82847] at e_1349_0
  have e_1349_1 := arithEq_of_rows h (row := 1349) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_1
  simp only [← c82848, ← c82849, k2, ← c82850] at e_1349_1
  have hr := e_1349_1
  simp only [e_1349_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12689)) := by
  have c82851 := (publicBatchWrapper86_copies2589 a h).2.2.2.1
  have c82852 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.1
  have c82853 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.1
  have c82854 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.1
  have c82855 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.1
  have c82856 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_2 := arithEq_of_rows h (row := 1349) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_2
  simp only [← c82851, ← c82852, ← c82853] at e_1349_2
  have e_1349_3 := arithEq_of_rows h (row := 1349) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_3
  simp only [← c82854, ← c82855, k2, ← c82856] at e_1349_3
  have hr := e_1349_3
  simp only [e_1349_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12690)) := by
  have c82857 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.1
  have c82858 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82859 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82860 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82861 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82862 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_4 := arithEq_of_rows h (row := 1349) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_4
  simp only [← c82857, ← c82858, ← c82859] at e_1349_4
  have e_1349_5 := arithEq_of_rows h (row := 1349) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_5
  simp only [← c82860, ← c82861, k2, ← c82862] at e_1349_5
  have hr := e_1349_5
  simp only [e_1349_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12691)) := by
  have c82863 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82864 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82865 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82866 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82867 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82868 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_6 := arithEq_of_rows h (row := 1349) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_6
  simp only [← c82863, ← c82864, ← c82865] at e_1349_6
  have e_1349_7 := arithEq_of_rows h (row := 1349) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_7
  simp only [← c82866, ← c82867, k2, ← c82868] at e_1349_7
  have hr := e_1349_7
  simp only [e_1349_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12692)) := by
  have c82869 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82870 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82871 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82872 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82873 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82874 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_8 := arithEq_of_rows h (row := 1349) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_8
  simp only [← c82869, ← c82870, ← c82871] at e_1349_8
  have e_1349_9 := arithEq_of_rows h (row := 1349) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_9
  simp only [← c82872, ← c82873, k2, ← c82874] at e_1349_9
  have hr := e_1349_9
  simp only [e_1349_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12693)) := by
  have c82875 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82876 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82877 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82878 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82879 := (publicBatchWrapper86_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82880 := (publicBatchWrapper86_copies2590 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_10 := arithEq_of_rows h (row := 1349) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_10
  simp only [← c82875, ← c82876, ← c82877] at e_1349_10
  have e_1349_11 := arithEq_of_rows h (row := 1349) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_11
  simp only [← c82878, ← c82879, k2, ← c82880] at e_1349_11
  have hr := e_1349_11
  simp only [e_1349_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12694)) := by
  have c82881 := (publicBatchWrapper86_copies2590 a h).2.1
  have c82882 := (publicBatchWrapper86_copies2590 a h).2.2.1
  have c82883 := (publicBatchWrapper86_copies2590 a h).2.2.2.1
  have c82884 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.1
  have c82885 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.1
  have c82886 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_12 := arithEq_of_rows h (row := 1349) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_12
  simp only [← c82881, ← c82882, ← c82883] at e_1349_12
  have e_1349_13 := arithEq_of_rows h (row := 1349) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_13
  simp only [← c82884, ← c82885, k2, ← c82886] at e_1349_13
  have hr := e_1349_13
  simp only [e_1349_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12695)) := by
  have c82887 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.1
  have c82888 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.1
  have c82889 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.1
  have c82890 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82891 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82892 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_14 := arithEq_of_rows h (row := 1349) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_14
  simp only [← c82887, ← c82888, ← c82889] at e_1349_14
  have e_1349_15 := arithEq_of_rows h (row := 1349) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_15
  simp only [← c82890, ← c82891, k2, ← c82892] at e_1349_15
  have hr := e_1349_15
  simp only [e_1349_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12696)) := by
  have c82893 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82894 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82895 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82896 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82897 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82898 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_16 := arithEq_of_rows h (row := 1349) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_16
  simp only [← c82893, ← c82894, ← c82895] at e_1349_16
  have e_1349_17 := arithEq_of_rows h (row := 1349) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_17
  simp only [← c82896, ← c82897, k2, ← c82898] at e_1349_17
  have hr := e_1349_17
  simp only [e_1349_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1349 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12697)) := by
  have c82899 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82900 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82901 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82902 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82903 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82904 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1349_18 := arithEq_of_rows h (row := 1349) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_18
  simp only [← c82899, ← c82900, ← c82901] at e_1349_18
  have e_1349_19 := arithEq_of_rows h (row := 1349) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_19
  simp only [← c82902, ← c82903, k2, ← c82904] at e_1349_19
  have hr := e_1349_19
  simp only [e_1349_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12698)) := by
  have c82905 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82906 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82907 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82908 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82909 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82910 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_0 := arithEq_of_rows h (row := 1350) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_0
  simp only [← c82905, ← c82906, ← c82907] at e_1350_0
  have e_1350_1 := arithEq_of_rows h (row := 1350) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_1
  simp only [← c82908, ← c82909, k2, ← c82910] at e_1350_1
  have hr := e_1350_1
  simp only [e_1350_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12699)) := by
  have c82911 := (publicBatchWrapper86_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82912 := (publicBatchWrapper86_copies2591 a h).1
  have c82913 := (publicBatchWrapper86_copies2591 a h).2.1
  have c82914 := (publicBatchWrapper86_copies2591 a h).2.2.1
  have c82915 := (publicBatchWrapper86_copies2591 a h).2.2.2.1
  have c82916 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_2 := arithEq_of_rows h (row := 1350) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_2
  simp only [← c82911, ← c82912, ← c82913] at e_1350_2
  have e_1350_3 := arithEq_of_rows h (row := 1350) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_3
  simp only [← c82914, ← c82915, k2, ← c82916] at e_1350_3
  have hr := e_1350_3
  simp only [e_1350_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12700)) := by
  have c82917 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.1
  have c82918 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.1
  have c82919 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.1
  have c82920 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.1
  have c82921 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.1
  have c82922 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_4 := arithEq_of_rows h (row := 1350) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_4
  simp only [← c82917, ← c82918, ← c82919] at e_1350_4
  have e_1350_5 := arithEq_of_rows h (row := 1350) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_5
  simp only [← c82920, ← c82921, k2, ← c82922] at e_1350_5
  have hr := e_1350_5
  simp only [e_1350_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12701)) := by
  have c82923 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82924 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82925 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82926 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82927 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82928 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_6 := arithEq_of_rows h (row := 1350) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_6
  simp only [← c82923, ← c82924, ← c82925] at e_1350_6
  have e_1350_7 := arithEq_of_rows h (row := 1350) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_7
  simp only [← c82926, ← c82927, k2, ← c82928] at e_1350_7
  have hr := e_1350_7
  simp only [e_1350_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12702)) := by
  have c82929 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82930 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82931 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82932 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82933 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82934 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_8 := arithEq_of_rows h (row := 1350) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_8
  simp only [← c82929, ← c82930, ← c82931] at e_1350_8
  have e_1350_9 := arithEq_of_rows h (row := 1350) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_9
  simp only [← c82932, ← c82933, k2, ← c82934] at e_1350_9
  have hr := e_1350_9
  simp only [e_1350_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12703)) := by
  have c82935 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82936 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82937 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82938 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82939 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82940 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_10 := arithEq_of_rows h (row := 1350) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_10
  simp only [← c82935, ← c82936, ← c82937] at e_1350_10
  have e_1350_11 := arithEq_of_rows h (row := 1350) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_11
  simp only [← c82938, ← c82939, k2, ← c82940] at e_1350_11
  have hr := e_1350_11
  simp only [e_1350_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12704)) := by
  have c82941 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82942 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82943 := (publicBatchWrapper86_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82944 := (publicBatchWrapper86_copies2592 a h).1
  have c82945 := (publicBatchWrapper86_copies2592 a h).2.1
  have c82946 := (publicBatchWrapper86_copies2592 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_12 := arithEq_of_rows h (row := 1350) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_12
  simp only [← c82941, ← c82942, ← c82943] at e_1350_12
  have e_1350_13 := arithEq_of_rows h (row := 1350) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_13
  simp only [← c82944, ← c82945, k2, ← c82946] at e_1350_13
  have hr := e_1350_13
  simp only [e_1350_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12705)) := by
  have c82947 := (publicBatchWrapper86_copies2592 a h).2.2.2.1
  have c82948 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.1
  have c82949 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.1
  have c82950 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.1
  have c82951 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.1
  have c82952 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_14 := arithEq_of_rows h (row := 1350) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_14
  simp only [← c82947, ← c82948, ← c82949] at e_1350_14
  have e_1350_15 := arithEq_of_rows h (row := 1350) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_15
  simp only [← c82950, ← c82951, k2, ← c82952] at e_1350_15
  have hr := e_1350_15
  simp only [e_1350_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12706)) := by
  have c82953 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.1
  have c82954 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82955 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82956 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82957 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82958 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_16 := arithEq_of_rows h (row := 1350) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_16
  simp only [← c82953, ← c82954, ← c82955] at e_1350_16
  have e_1350_17 := arithEq_of_rows h (row := 1350) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_17
  simp only [← c82956, ← c82957, k2, ← c82958] at e_1350_17
  have hr := e_1350_17
  simp only [e_1350_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1350 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12707)) := by
  have c82959 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82960 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82961 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82962 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82963 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82964 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1350_18 := arithEq_of_rows h (row := 1350) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_18
  simp only [← c82959, ← c82960, ← c82961] at e_1350_18
  have e_1350_19 := arithEq_of_rows h (row := 1350) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_19
  simp only [← c82962, ← c82963, k2, ← c82964] at e_1350_19
  have hr := e_1350_19
  simp only [e_1350_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1351 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12708)) := by
  have c82965 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82966 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82967 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82968 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82969 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82970 := (publicBatchWrapper86_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1351_0 := arithEq_of_rows h (row := 1351) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_0
  simp only [← c82965, ← c82966, ← c82967] at e_1351_0
  have e_1351_1 := arithEq_of_rows h (row := 1351) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1351_1
  simp only [← c82968, ← c82969, k2, ← c82970] at e_1351_1
  have hr := e_1351_1
  simp only [e_1351_0] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
