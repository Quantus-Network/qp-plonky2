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

theorem publicBatchWrapper89_f2882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15805)) (a (.wire 169 63)) (a (.virt 19471)) (a (.virt 19472)) := by
  have c21390 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21391 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21392 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21393 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21394 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21395 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21396 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21397 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21398 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21399 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21400 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21401 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21402 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21403 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21404 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21405 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21406 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_324_16 := arithEq_of_rows h (row := 324) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_16
  simp only [← c21390, k3, ← c21391, k3, ← c21392] at e_324_16
  have e_324_17 := arithEq_of_rows h (row := 324) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_17
  simp only [← c21393, ← c21394, k3, ← c21395] at e_324_17
  have e_324_18 := arithEq_of_rows h (row := 324) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_18
  simp only [← c21402, ← c21403, k3, ← c21404] at e_324_18
  have e_325_0 := arithEq_of_rows h (row := 325) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_0
  simp only [← c21396, ← c21397, ← c21398] at e_325_0
  have e_325_1 := arithEq_of_rows h (row := 325) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_1
  simp only [← c21399, ← c21400, ← c21401] at e_325_1
  refine ⟨?_, ?_⟩
  · have hc := e_325_0
    simp only [e_324_17] at hc
    linear_combination c21405.trans k2 - hc
  · have hc := e_324_18
    simp only [e_324_16, e_325_1, e_324_17] at hc
    linear_combination c21406.trans k2 - hc

theorem publicBatchWrapper89_f2883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15806)) (a (.wire 169 71)) (a (.virt 19473)) (a (.virt 19474)) := by
  have c21407 := (publicBatchWrapper89_copies668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21408 := (publicBatchWrapper89_copies669 a h).1
  have c21409 := (publicBatchWrapper89_copies669 a h).2.1
  have c21410 := (publicBatchWrapper89_copies669 a h).2.2.1
  have c21411 := (publicBatchWrapper89_copies669 a h).2.2.2.1
  have c21412 := (publicBatchWrapper89_copies669 a h).2.2.2.2.1
  have c21413 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.1
  have c21414 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.1
  have c21415 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.1
  have c21416 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.1
  have c21417 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.1
  have c21418 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21419 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21420 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21421 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21422 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21423 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_324_19 := arithEq_of_rows h (row := 324) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_19
  simp only [← c21407, k3, ← c21408, k3, ← c21409] at e_324_19
  have e_325_2 := arithEq_of_rows h (row := 325) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_2
  simp only [← c21413, ← c21414, ← c21415] at e_325_2
  have e_325_3 := arithEq_of_rows h (row := 325) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_3
  simp only [← c21416, ← c21417, ← c21418] at e_325_3
  have e_326_0 := arithEq_of_rows h (row := 326) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_0
  simp only [← c21410, ← c21411, k3, ← c21412] at e_326_0
  have e_326_1 := arithEq_of_rows h (row := 326) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_1
  simp only [← c21419, ← c21420, k3, ← c21421] at e_326_1
  refine ⟨?_, ?_⟩
  · have hc := e_325_2
    simp only [e_326_0] at hc
    linear_combination c21422.trans k2 - hc
  · have hc := e_326_1
    simp only [e_324_19, e_325_3, e_326_0] at hc
    linear_combination c21423.trans k2 - hc

theorem publicBatchWrapper89_f2884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15807)) (a (.wire 169 79)) (a (.virt 19475)) (a (.virt 19476)) := by
  have c21424 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21425 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21426 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21427 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21428 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21429 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21430 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21431 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21432 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21433 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21434 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21435 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21436 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21437 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21438 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21439 := (publicBatchWrapper89_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21440 := (publicBatchWrapper89_copies670 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_4 := arithEq_of_rows h (row := 325) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_4
  simp only [← c21430, ← c21431, ← c21432] at e_325_4
  have e_325_5 := arithEq_of_rows h (row := 325) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_5
  simp only [← c21433, ← c21434, ← c21435] at e_325_5
  have e_326_2 := arithEq_of_rows h (row := 326) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_2
  simp only [← c21424, k3, ← c21425, k3, ← c21426] at e_326_2
  have e_326_3 := arithEq_of_rows h (row := 326) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_3
  simp only [← c21427, ← c21428, k3, ← c21429] at e_326_3
  have e_326_4 := arithEq_of_rows h (row := 326) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_4
  simp only [← c21436, ← c21437, k3, ← c21438] at e_326_4
  refine ⟨?_, ?_⟩
  · have hc := e_325_4
    simp only [e_326_3] at hc
    linear_combination c21439.trans k2 - hc
  · have hc := e_326_4
    simp only [e_326_2, e_325_5, e_326_3] at hc
    linear_combination c21440.trans k2 - hc

theorem publicBatchWrapper89_f2885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15808)) (a (.wire 170 7)) (a (.virt 19477)) (a (.virt 19478)) := by
  have c21441 := (publicBatchWrapper89_copies670 a h).2.1
  have c21442 := (publicBatchWrapper89_copies670 a h).2.2.1
  have c21443 := (publicBatchWrapper89_copies670 a h).2.2.2.1
  have c21444 := (publicBatchWrapper89_copies670 a h).2.2.2.2.1
  have c21445 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.1
  have c21446 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.1
  have c21447 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.1
  have c21448 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.1
  have c21449 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.1
  have c21450 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21451 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21452 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21453 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21454 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21455 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21456 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21457 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_6 := arithEq_of_rows h (row := 325) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_6
  simp only [← c21447, ← c21448, ← c21449] at e_325_6
  have e_325_7 := arithEq_of_rows h (row := 325) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_7
  simp only [← c21450, ← c21451, ← c21452] at e_325_7
  have e_326_5 := arithEq_of_rows h (row := 326) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_5
  simp only [← c21441, k3, ← c21442, k3, ← c21443] at e_326_5
  have e_326_6 := arithEq_of_rows h (row := 326) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_6
  simp only [← c21444, ← c21445, k3, ← c21446] at e_326_6
  have e_326_7 := arithEq_of_rows h (row := 326) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_7
  simp only [← c21453, ← c21454, k3, ← c21455] at e_326_7
  refine ⟨?_, ?_⟩
  · have hc := e_325_6
    simp only [e_326_6] at hc
    linear_combination c21456.trans k2 - hc
  · have hc := e_326_7
    simp only [e_326_5, e_325_7, e_326_6] at hc
    linear_combination c21457.trans k2 - hc

theorem publicBatchWrapper89_f2886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 325 35) = band (a (.virt 19471)) (a (.virt 19473)) := by
  have c21458 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21459 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21460 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_8 := arithEq_of_rows h (row := 325) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_8
  simp only [← c21458, ← c21459, ← c21460] at e_325_8
  have hr := e_325_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 325 39) = band (a (.virt 19475)) (a (.virt 19477)) := by
  have c21461 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21462 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21463 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_9 := arithEq_of_rows h (row := 325) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_9
  simp only [← c21461, ← c21462, ← c21463] at e_325_9
  have hr := e_325_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 325 43) = band (a (.wire 325 35)) (a (.wire 325 39)) := by
  have c21464 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21465 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21466 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_10 := arithEq_of_rows h (row := 325) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_10
  simp only [← c21464, ← c21465, ← c21466] at e_325_10
  have hr := e_325_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 79) = bor (a (.wire 77 15)) (a (.wire 325 43)) := by
  have c21467 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21468 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21469 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21470 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21471 := (publicBatchWrapper89_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21472 := (publicBatchWrapper89_copies671 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_314_19 := arithEq_of_rows h (row := 314) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_314_19
  simp only [← c21470, ← c21471, k3, ← c21472] at e_314_19
  have e_321_7 := arithEq_of_rows h (row := 321) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_7
  simp only [← c21467, ← c21468, ← c21469] at e_321_7
  have hr := e_314_19
  simp only [e_321_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 314 79) = a (.virt 201) := by
  have c21473 := (publicBatchWrapper89_copies671 a h).2.1
  exact c21473

theorem publicBatchWrapper89_f2891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16003)) (a (.wire 170 23)) (a (.virt 19479)) (a (.virt 19480)) := by
  have c21474 := (publicBatchWrapper89_copies671 a h).2.2.1
  have c21475 := (publicBatchWrapper89_copies671 a h).2.2.2.1
  have c21476 := (publicBatchWrapper89_copies671 a h).2.2.2.2.1
  have c21477 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.1
  have c21478 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.1
  have c21479 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.1
  have c21480 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.1
  have c21481 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.1
  have c21482 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21483 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21484 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21485 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21486 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21487 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21488 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21489 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21490 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_11 := arithEq_of_rows h (row := 325) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_11
  simp only [← c21480, ← c21481, ← c21482] at e_325_11
  have e_325_12 := arithEq_of_rows h (row := 325) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_12
  simp only [← c21483, ← c21484, ← c21485] at e_325_12
  have e_326_8 := arithEq_of_rows h (row := 326) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_8
  simp only [← c21474, k3, ← c21475, k3, ← c21476] at e_326_8
  have e_326_9 := arithEq_of_rows h (row := 326) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_9
  simp only [← c21477, ← c21478, k3, ← c21479] at e_326_9
  have e_326_10 := arithEq_of_rows h (row := 326) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_10
  simp only [← c21486, ← c21487, k3, ← c21488] at e_326_10
  refine ⟨?_, ?_⟩
  · have hc := e_325_11
    simp only [e_326_9] at hc
    linear_combination c21489.trans k2 - hc
  · have hc := e_326_10
    simp only [e_326_8, e_325_12, e_326_9] at hc
    linear_combination c21490.trans k2 - hc

theorem publicBatchWrapper89_f2892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 3) = bor (a (.wire 77 59)) (a (.virt 19479)) := by
  have c21491 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21492 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21493 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21494 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21495 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21496 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_8 := arithEq_of_rows h (row := 321) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_8
  simp only [← c21491, ← c21492, ← c21493] at e_321_8
  have e_327_0 := arithEq_of_rows h (row := 327) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_0
  simp only [← c21494, ← c21495, k3, ← c21496] at e_327_0
  have hr := e_327_0
  simp only [e_321_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 3) = a (.virt 201) := by
  have c21497 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21497

theorem publicBatchWrapper89_f2894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16004)) (a (.wire 170 31)) (a (.virt 19481)) (a (.virt 19482)) := by
  have c21498 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21499 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21500 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21501 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21502 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21503 := (publicBatchWrapper89_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21504 := (publicBatchWrapper89_copies672 a h).1
  have c21505 := (publicBatchWrapper89_copies672 a h).2.1
  have c21506 := (publicBatchWrapper89_copies672 a h).2.2.1
  have c21507 := (publicBatchWrapper89_copies672 a h).2.2.2.1
  have c21508 := (publicBatchWrapper89_copies672 a h).2.2.2.2.1
  have c21509 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.1
  have c21510 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.1
  have c21511 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.1
  have c21512 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.1
  have c21513 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.1
  have c21514 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_13 := arithEq_of_rows h (row := 325) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_13
  simp only [← c21504, ← c21505, ← c21506] at e_325_13
  have e_325_14 := arithEq_of_rows h (row := 325) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_14
  simp only [← c21507, ← c21508, ← c21509] at e_325_14
  have e_326_11 := arithEq_of_rows h (row := 326) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_11
  simp only [← c21498, k3, ← c21499, k3, ← c21500] at e_326_11
  have e_326_12 := arithEq_of_rows h (row := 326) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_12
  simp only [← c21501, ← c21502, k3, ← c21503] at e_326_12
  have e_326_13 := arithEq_of_rows h (row := 326) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_13
  simp only [← c21510, ← c21511, k3, ← c21512] at e_326_13
  refine ⟨?_, ?_⟩
  · have hc := e_325_13
    simp only [e_326_12] at hc
    linear_combination c21513.trans k2 - hc
  · have hc := e_326_13
    simp only [e_326_11, e_325_14, e_326_12] at hc
    linear_combination c21514.trans k2 - hc

theorem publicBatchWrapper89_f2895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 7) = bor (a (.wire 77 59)) (a (.virt 19481)) := by
  have c21515 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21516 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21517 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21518 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21519 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21520 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_9 := arithEq_of_rows h (row := 321) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_9
  simp only [← c21515, ← c21516, ← c21517] at e_321_9
  have e_327_1 := arithEq_of_rows h (row := 327) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_1
  simp only [← c21518, ← c21519, k3, ← c21520] at e_327_1
  have hr := e_327_1
  simp only [e_321_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 7) = a (.virt 201) := by
  have c21521 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21521

theorem publicBatchWrapper89_f2897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16005)) (a (.wire 169 63)) (a (.virt 19483)) (a (.virt 19484)) := by
  have c21522 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21523 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21524 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21525 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21526 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21527 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21528 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21529 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21530 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21531 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21532 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21533 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21534 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21535 := (publicBatchWrapper89_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21536 := (publicBatchWrapper89_copies673 a h).1
  have c21537 := (publicBatchWrapper89_copies673 a h).2.1
  have c21538 := (publicBatchWrapper89_copies673 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_15 := arithEq_of_rows h (row := 325) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_15
  simp only [← c21528, ← c21529, ← c21530] at e_325_15
  have e_325_16 := arithEq_of_rows h (row := 325) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_16
  simp only [← c21531, ← c21532, ← c21533] at e_325_16
  have e_326_14 := arithEq_of_rows h (row := 326) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_14
  simp only [← c21522, k3, ← c21523, k3, ← c21524] at e_326_14
  have e_326_15 := arithEq_of_rows h (row := 326) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_15
  simp only [← c21525, ← c21526, k3, ← c21527] at e_326_15
  have e_326_16 := arithEq_of_rows h (row := 326) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_16
  simp only [← c21534, ← c21535, k3, ← c21536] at e_326_16
  refine ⟨?_, ?_⟩
  · have hc := e_325_15
    simp only [e_326_15] at hc
    linear_combination c21537.trans k2 - hc
  · have hc := e_326_16
    simp only [e_326_14, e_325_16, e_326_15] at hc
    linear_combination c21538.trans k2 - hc

theorem publicBatchWrapper89_f2898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16006)) (a (.wire 169 71)) (a (.virt 19485)) (a (.virt 19486)) := by
  have c21539 := (publicBatchWrapper89_copies673 a h).2.2.2.1
  have c21540 := (publicBatchWrapper89_copies673 a h).2.2.2.2.1
  have c21541 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.1
  have c21542 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.1
  have c21543 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.1
  have c21544 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.1
  have c21545 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.1
  have c21546 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21547 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21548 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21549 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21550 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21551 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21552 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21553 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21554 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21555 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_17 := arithEq_of_rows h (row := 325) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_17
  simp only [← c21545, ← c21546, ← c21547] at e_325_17
  have e_325_18 := arithEq_of_rows h (row := 325) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_18
  simp only [← c21548, ← c21549, ← c21550] at e_325_18
  have e_326_17 := arithEq_of_rows h (row := 326) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_17
  simp only [← c21539, k3, ← c21540, k3, ← c21541] at e_326_17
  have e_326_18 := arithEq_of_rows h (row := 326) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_18
  simp only [← c21542, ← c21543, k3, ← c21544] at e_326_18
  have e_326_19 := arithEq_of_rows h (row := 326) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_19
  simp only [← c21551, ← c21552, k3, ← c21553] at e_326_19
  refine ⟨?_, ?_⟩
  · have hc := e_325_17
    simp only [e_326_18] at hc
    linear_combination c21554.trans k2 - hc
  · have hc := e_326_19
    simp only [e_326_17, e_325_18, e_326_18] at hc
    linear_combination c21555.trans k2 - hc

theorem publicBatchWrapper89_f2899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16007)) (a (.wire 169 79)) (a (.virt 19487)) (a (.virt 19488)) := by
  have c21556 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21557 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21558 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21559 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21560 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21561 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21562 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21563 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21564 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21565 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21566 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21567 := (publicBatchWrapper89_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21568 := (publicBatchWrapper89_copies674 a h).1
  have c21569 := (publicBatchWrapper89_copies674 a h).2.1
  have c21570 := (publicBatchWrapper89_copies674 a h).2.2.1
  have c21571 := (publicBatchWrapper89_copies674 a h).2.2.2.1
  have c21572 := (publicBatchWrapper89_copies674 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_325_19 := arithEq_of_rows h (row := 325) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_19
  simp only [← c21562, ← c21563, ← c21564] at e_325_19
  have e_328_0 := arithEq_of_rows h (row := 328) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_0
  simp only [← c21556, k3, ← c21557, k3, ← c21558] at e_328_0
  have e_328_1 := arithEq_of_rows h (row := 328) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_1
  simp only [← c21559, ← c21560, k3, ← c21561] at e_328_1
  have e_328_2 := arithEq_of_rows h (row := 328) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_2
  simp only [← c21568, ← c21569, k3, ← c21570] at e_328_2
  have e_329_0 := arithEq_of_rows h (row := 329) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_0
  simp only [← c21565, ← c21566, ← c21567] at e_329_0
  refine ⟨?_, ?_⟩
  · have hc := e_325_19
    simp only [e_328_1] at hc
    linear_combination c21571.trans k2 - hc
  · have hc := e_328_2
    simp only [e_328_0, e_329_0, e_328_1] at hc
    linear_combination c21572.trans k2 - hc

theorem publicBatchWrapper89_f2900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16008)) (a (.wire 170 7)) (a (.virt 19489)) (a (.virt 19490)) := by
  have c21573 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.1
  have c21574 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.1
  have c21575 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.1
  have c21576 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.1
  have c21577 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.1
  have c21578 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21579 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21580 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21581 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21582 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21583 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21584 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21585 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21586 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21587 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21588 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21589 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_3 := arithEq_of_rows h (row := 328) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_3
  simp only [← c21573, k3, ← c21574, k3, ← c21575] at e_328_3
  have e_328_4 := arithEq_of_rows h (row := 328) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_4
  simp only [← c21576, ← c21577, k3, ← c21578] at e_328_4
  have e_328_5 := arithEq_of_rows h (row := 328) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_5
  simp only [← c21585, ← c21586, k3, ← c21587] at e_328_5
  have e_329_1 := arithEq_of_rows h (row := 329) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_1
  simp only [← c21579, ← c21580, ← c21581] at e_329_1
  have e_329_2 := arithEq_of_rows h (row := 329) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_2
  simp only [← c21582, ← c21583, ← c21584] at e_329_2
  refine ⟨?_, ?_⟩
  · have hc := e_329_1
    simp only [e_328_4] at hc
    linear_combination c21588.trans k2 - hc
  · have hc := e_328_5
    simp only [e_328_3, e_329_2, e_328_4] at hc
    linear_combination c21589.trans k2 - hc

theorem publicBatchWrapper89_f2901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 329 15) = band (a (.virt 19483)) (a (.virt 19485)) := by
  have c21590 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21591 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21592 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_3 := arithEq_of_rows h (row := 329) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_3
  simp only [← c21590, ← c21591, ← c21592] at e_329_3
  have hr := e_329_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 329 19) = band (a (.virt 19487)) (a (.virt 19489)) := by
  have c21593 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21594 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21595 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_4 := arithEq_of_rows h (row := 329) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_4
  simp only [← c21593, ← c21594, ← c21595] at e_329_4
  have hr := e_329_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 329 23) = band (a (.wire 329 15)) (a (.wire 329 19)) := by
  have c21596 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21597 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21598 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_5 := arithEq_of_rows h (row := 329) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_5
  simp only [← c21596, ← c21597, ← c21598] at e_329_5
  have hr := e_329_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 11) = bor (a (.wire 77 59)) (a (.wire 329 23)) := by
  have c21599 := (publicBatchWrapper89_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21600 := (publicBatchWrapper89_copies675 a h).1
  have c21601 := (publicBatchWrapper89_copies675 a h).2.1
  have c21602 := (publicBatchWrapper89_copies675 a h).2.2.1
  have c21603 := (publicBatchWrapper89_copies675 a h).2.2.2.1
  have c21604 := (publicBatchWrapper89_copies675 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_10 := arithEq_of_rows h (row := 321) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_10
  simp only [← c21599, ← c21600, ← c21601] at e_321_10
  have e_327_2 := arithEq_of_rows h (row := 327) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_2
  simp only [← c21602, ← c21603, k3, ← c21604] at e_327_2
  have hr := e_327_2
  simp only [e_321_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 11) = a (.virt 201) := by
  have c21605 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.1
  exact c21605

theorem publicBatchWrapper89_f2906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16203)) (a (.wire 170 23)) (a (.virt 19491)) (a (.virt 19492)) := by
  have c21606 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.1
  have c21607 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.1
  have c21608 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.1
  have c21609 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.1
  have c21610 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21611 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21612 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21613 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21614 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21615 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21616 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21617 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21618 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21619 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21620 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21621 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21622 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_6 := arithEq_of_rows h (row := 328) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_6
  simp only [← c21606, k3, ← c21607, k3, ← c21608] at e_328_6
  have e_328_7 := arithEq_of_rows h (row := 328) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_7
  simp only [← c21609, ← c21610, k3, ← c21611] at e_328_7
  have e_328_8 := arithEq_of_rows h (row := 328) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_8
  simp only [← c21618, ← c21619, k3, ← c21620] at e_328_8
  have e_329_6 := arithEq_of_rows h (row := 329) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_6
  simp only [← c21612, ← c21613, ← c21614] at e_329_6
  have e_329_7 := arithEq_of_rows h (row := 329) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_7
  simp only [← c21615, ← c21616, ← c21617] at e_329_7
  refine ⟨?_, ?_⟩
  · have hc := e_329_6
    simp only [e_328_7] at hc
    linear_combination c21621.trans k2 - hc
  · have hc := e_328_8
    simp only [e_328_6, e_329_7, e_328_7] at hc
    linear_combination c21622.trans k2 - hc

theorem publicBatchWrapper89_f2907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 15) = bor (a (.wire 79 23)) (a (.virt 19491)) := by
  have c21623 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21624 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21625 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21626 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21627 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21628 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_11 := arithEq_of_rows h (row := 321) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_11
  simp only [← c21623, ← c21624, ← c21625] at e_321_11
  have e_327_3 := arithEq_of_rows h (row := 327) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_3
  simp only [← c21626, ← c21627, k3, ← c21628] at e_327_3
  have hr := e_327_3
  simp only [e_321_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 15) = a (.virt 201) := by
  have c21629 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21629

theorem publicBatchWrapper89_f2909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16204)) (a (.wire 170 31)) (a (.virt 19493)) (a (.virt 19494)) := by
  have c21630 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21631 := (publicBatchWrapper89_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21632 := (publicBatchWrapper89_copies676 a h).1
  have c21633 := (publicBatchWrapper89_copies676 a h).2.1
  have c21634 := (publicBatchWrapper89_copies676 a h).2.2.1
  have c21635 := (publicBatchWrapper89_copies676 a h).2.2.2.1
  have c21636 := (publicBatchWrapper89_copies676 a h).2.2.2.2.1
  have c21637 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.1
  have c21638 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.1
  have c21639 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.1
  have c21640 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.1
  have c21641 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.1
  have c21642 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21643 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21644 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21645 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21646 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_9 := arithEq_of_rows h (row := 328) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_9
  simp only [← c21630, k3, ← c21631, k3, ← c21632] at e_328_9
  have e_328_10 := arithEq_of_rows h (row := 328) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_10
  simp only [← c21633, ← c21634, k3, ← c21635] at e_328_10
  have e_328_11 := arithEq_of_rows h (row := 328) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_11
  simp only [← c21642, ← c21643, k3, ← c21644] at e_328_11
  have e_329_8 := arithEq_of_rows h (row := 329) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_8
  simp only [← c21636, ← c21637, ← c21638] at e_329_8
  have e_329_9 := arithEq_of_rows h (row := 329) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_9
  simp only [← c21639, ← c21640, ← c21641] at e_329_9
  refine ⟨?_, ?_⟩
  · have hc := e_329_8
    simp only [e_328_10] at hc
    linear_combination c21645.trans k2 - hc
  · have hc := e_328_11
    simp only [e_328_9, e_329_9, e_328_10] at hc
    linear_combination c21646.trans k2 - hc

theorem publicBatchWrapper89_f2910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 19) = bor (a (.wire 79 23)) (a (.virt 19493)) := by
  have c21647 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21648 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21649 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21650 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21651 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21652 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_12 := arithEq_of_rows h (row := 321) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_12
  simp only [← c21647, ← c21648, ← c21649] at e_321_12
  have e_327_4 := arithEq_of_rows h (row := 327) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_4
  simp only [← c21650, ← c21651, k3, ← c21652] at e_327_4
  have hr := e_327_4
  simp only [e_321_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 19) = a (.virt 201) := by
  have c21653 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21653

theorem publicBatchWrapper89_f2912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16205)) (a (.wire 169 63)) (a (.virt 19495)) (a (.virt 19496)) := by
  have c21654 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21655 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21656 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21657 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21658 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21659 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21660 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21661 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21662 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21663 := (publicBatchWrapper89_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21664 := (publicBatchWrapper89_copies677 a h).1
  have c21665 := (publicBatchWrapper89_copies677 a h).2.1
  have c21666 := (publicBatchWrapper89_copies677 a h).2.2.1
  have c21667 := (publicBatchWrapper89_copies677 a h).2.2.2.1
  have c21668 := (publicBatchWrapper89_copies677 a h).2.2.2.2.1
  have c21669 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.1
  have c21670 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_12 := arithEq_of_rows h (row := 328) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_12
  simp only [← c21654, k3, ← c21655, k3, ← c21656] at e_328_12
  have e_328_13 := arithEq_of_rows h (row := 328) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_13
  simp only [← c21657, ← c21658, k3, ← c21659] at e_328_13
  have e_328_14 := arithEq_of_rows h (row := 328) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_14
  simp only [← c21666, ← c21667, k3, ← c21668] at e_328_14
  have e_329_10 := arithEq_of_rows h (row := 329) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_10
  simp only [← c21660, ← c21661, ← c21662] at e_329_10
  have e_329_11 := arithEq_of_rows h (row := 329) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_11
  simp only [← c21663, ← c21664, ← c21665] at e_329_11
  refine ⟨?_, ?_⟩
  · have hc := e_329_10
    simp only [e_328_13] at hc
    linear_combination c21669.trans k2 - hc
  · have hc := e_328_14
    simp only [e_328_12, e_329_11, e_328_13] at hc
    linear_combination c21670.trans k2 - hc

theorem publicBatchWrapper89_f2913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16206)) (a (.wire 169 71)) (a (.virt 19497)) (a (.virt 19498)) := by
  have c21671 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.1
  have c21672 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.1
  have c21673 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.1
  have c21674 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21675 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21676 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21677 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21678 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21679 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21680 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21681 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21682 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21683 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21684 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21685 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21686 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21687 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_15 := arithEq_of_rows h (row := 328) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_15
  simp only [← c21671, k3, ← c21672, k3, ← c21673] at e_328_15
  have e_328_16 := arithEq_of_rows h (row := 328) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_16
  simp only [← c21674, ← c21675, k3, ← c21676] at e_328_16
  have e_328_17 := arithEq_of_rows h (row := 328) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_17
  simp only [← c21683, ← c21684, k3, ← c21685] at e_328_17
  have e_329_12 := arithEq_of_rows h (row := 329) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_12
  simp only [← c21677, ← c21678, ← c21679] at e_329_12
  have e_329_13 := arithEq_of_rows h (row := 329) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_13
  simp only [← c21680, ← c21681, ← c21682] at e_329_13
  refine ⟨?_, ?_⟩
  · have hc := e_329_12
    simp only [e_328_16] at hc
    linear_combination c21686.trans k2 - hc
  · have hc := e_328_17
    simp only [e_328_15, e_329_13, e_328_16] at hc
    linear_combination c21687.trans k2 - hc

theorem publicBatchWrapper89_f2914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16207)) (a (.wire 169 79)) (a (.virt 19499)) (a (.virt 19500)) := by
  have c21688 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21689 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21690 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21691 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21692 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21693 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21694 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21695 := (publicBatchWrapper89_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21696 := (publicBatchWrapper89_copies678 a h).1
  have c21697 := (publicBatchWrapper89_copies678 a h).2.1
  have c21698 := (publicBatchWrapper89_copies678 a h).2.2.1
  have c21699 := (publicBatchWrapper89_copies678 a h).2.2.2.1
  have c21700 := (publicBatchWrapper89_copies678 a h).2.2.2.2.1
  have c21701 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.1
  have c21702 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.1
  have c21703 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.1
  have c21704 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_328_18 := arithEq_of_rows h (row := 328) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_18
  simp only [← c21688, k3, ← c21689, k3, ← c21690] at e_328_18
  have e_328_19 := arithEq_of_rows h (row := 328) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_19
  simp only [← c21691, ← c21692, k3, ← c21693] at e_328_19
  have e_329_14 := arithEq_of_rows h (row := 329) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_14
  simp only [← c21694, ← c21695, ← c21696] at e_329_14
  have e_329_15 := arithEq_of_rows h (row := 329) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_15
  simp only [← c21697, ← c21698, ← c21699] at e_329_15
  have e_330_0 := arithEq_of_rows h (row := 330) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_0
  simp only [← c21700, ← c21701, k3, ← c21702] at e_330_0
  refine ⟨?_, ?_⟩
  · have hc := e_329_14
    simp only [e_328_19] at hc
    linear_combination c21703.trans k2 - hc
  · have hc := e_330_0
    simp only [e_328_18, e_329_15, e_328_19] at hc
    linear_combination c21704.trans k2 - hc

theorem publicBatchWrapper89_f2915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16208)) (a (.wire 170 7)) (a (.virt 19501)) (a (.virt 19502)) := by
  have c21705 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.1
  have c21706 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21707 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21708 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21709 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21710 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21711 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21712 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21713 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21714 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21715 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21716 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21717 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21718 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21719 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21720 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21721 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_329_16 := arithEq_of_rows h (row := 329) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_16
  simp only [← c21711, ← c21712, ← c21713] at e_329_16
  have e_329_17 := arithEq_of_rows h (row := 329) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_17
  simp only [← c21714, ← c21715, ← c21716] at e_329_17
  have e_330_1 := arithEq_of_rows h (row := 330) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_1
  simp only [← c21705, k3, ← c21706, k3, ← c21707] at e_330_1
  have e_330_2 := arithEq_of_rows h (row := 330) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_2
  simp only [← c21708, ← c21709, k3, ← c21710] at e_330_2
  have e_330_3 := arithEq_of_rows h (row := 330) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_3
  simp only [← c21717, ← c21718, k3, ← c21719] at e_330_3
  refine ⟨?_, ?_⟩
  · have hc := e_329_16
    simp only [e_330_2] at hc
    linear_combination c21720.trans k2 - hc
  · have hc := e_330_3
    simp only [e_330_1, e_329_17, e_330_2] at hc
    linear_combination c21721.trans k2 - hc

theorem publicBatchWrapper89_f2916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 329 75) = band (a (.virt 19495)) (a (.virt 19497)) := by
  have c21722 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21723 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21724 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_18 := arithEq_of_rows h (row := 329) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_18
  simp only [← c21722, ← c21723, ← c21724] at e_329_18
  have hr := e_329_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 329 79) = band (a (.virt 19499)) (a (.virt 19501)) := by
  have c21725 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21726 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21727 := (publicBatchWrapper89_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_329_19 := arithEq_of_rows h (row := 329) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_19
  simp only [← c21725, ← c21726, ← c21727] at e_329_19
  have hr := e_329_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 331 3) = band (a (.wire 329 75)) (a (.wire 329 79)) := by
  have c21728 := (publicBatchWrapper89_copies679 a h).1
  have c21729 := (publicBatchWrapper89_copies679 a h).2.1
  have c21730 := (publicBatchWrapper89_copies679 a h).2.2.1
  have e_331_0 := arithEq_of_rows h (row := 331) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_0
  simp only [← c21728, ← c21729, ← c21730] at e_331_0
  have hr := e_331_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 23) = bor (a (.wire 79 23)) (a (.wire 331 3)) := by
  have c21731 := (publicBatchWrapper89_copies679 a h).2.2.2.1
  have c21732 := (publicBatchWrapper89_copies679 a h).2.2.2.2.1
  have c21733 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.1
  have c21734 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.1
  have c21735 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.1
  have c21736 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_13 := arithEq_of_rows h (row := 321) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_13
  simp only [← c21731, ← c21732, ← c21733] at e_321_13
  have e_327_5 := arithEq_of_rows h (row := 327) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_5
  simp only [← c21734, ← c21735, k3, ← c21736] at e_327_5
  have hr := e_327_5
  simp only [e_321_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 23) = a (.virt 201) := by
  have c21737 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.1
  exact c21737

theorem publicBatchWrapper89_f2921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16403)) (a (.wire 170 23)) (a (.virt 19503)) (a (.virt 19504)) := by
  have c21738 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21739 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21740 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21741 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21742 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21743 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21744 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21745 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21746 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21747 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21748 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21749 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21750 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21751 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21752 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21753 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21754 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_4 := arithEq_of_rows h (row := 330) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_4
  simp only [← c21738, k3, ← c21739, k3, ← c21740] at e_330_4
  have e_330_5 := arithEq_of_rows h (row := 330) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_5
  simp only [← c21741, ← c21742, k3, ← c21743] at e_330_5
  have e_330_6 := arithEq_of_rows h (row := 330) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_6
  simp only [← c21750, ← c21751, k3, ← c21752] at e_330_6
  have e_331_1 := arithEq_of_rows h (row := 331) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_1
  simp only [← c21744, ← c21745, ← c21746] at e_331_1
  have e_331_2 := arithEq_of_rows h (row := 331) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_2
  simp only [← c21747, ← c21748, ← c21749] at e_331_2
  refine ⟨?_, ?_⟩
  · have hc := e_331_1
    simp only [e_330_5] at hc
    linear_combination c21753.trans k2 - hc
  · have hc := e_330_6
    simp only [e_330_4, e_331_2, e_330_5] at hc
    linear_combination c21754.trans k2 - hc

theorem publicBatchWrapper89_f2922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 27) = bor (a (.wire 79 67)) (a (.virt 19503)) := by
  have c21755 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21756 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21757 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21758 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21759 := (publicBatchWrapper89_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21760 := (publicBatchWrapper89_copies680 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_14 := arithEq_of_rows h (row := 321) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_14
  simp only [← c21755, ← c21756, ← c21757] at e_321_14
  have e_327_6 := arithEq_of_rows h (row := 327) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_6
  simp only [← c21758, ← c21759, k3, ← c21760] at e_327_6
  have hr := e_327_6
  simp only [e_321_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 27) = a (.virt 201) := by
  have c21761 := (publicBatchWrapper89_copies680 a h).2.1
  exact c21761

theorem publicBatchWrapper89_f2924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16404)) (a (.wire 170 31)) (a (.virt 19505)) (a (.virt 19506)) := by
  have c21762 := (publicBatchWrapper89_copies680 a h).2.2.1
  have c21763 := (publicBatchWrapper89_copies680 a h).2.2.2.1
  have c21764 := (publicBatchWrapper89_copies680 a h).2.2.2.2.1
  have c21765 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.1
  have c21766 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.1
  have c21767 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.1
  have c21768 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.1
  have c21769 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.1
  have c21770 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21771 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21772 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21773 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21774 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21775 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21776 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21777 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21778 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_7 := arithEq_of_rows h (row := 330) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_7
  simp only [← c21762, k3, ← c21763, k3, ← c21764] at e_330_7
  have e_330_8 := arithEq_of_rows h (row := 330) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_8
  simp only [← c21765, ← c21766, k3, ← c21767] at e_330_8
  have e_330_9 := arithEq_of_rows h (row := 330) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_9
  simp only [← c21774, ← c21775, k3, ← c21776] at e_330_9
  have e_331_3 := arithEq_of_rows h (row := 331) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_3
  simp only [← c21768, ← c21769, ← c21770] at e_331_3
  have e_331_4 := arithEq_of_rows h (row := 331) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_4
  simp only [← c21771, ← c21772, ← c21773] at e_331_4
  refine ⟨?_, ?_⟩
  · have hc := e_331_3
    simp only [e_330_8] at hc
    linear_combination c21777.trans k2 - hc
  · have hc := e_330_9
    simp only [e_330_7, e_331_4, e_330_8] at hc
    linear_combination c21778.trans k2 - hc

theorem publicBatchWrapper89_f2925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 31) = bor (a (.wire 79 67)) (a (.virt 19505)) := by
  have c21779 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21780 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21781 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21782 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21783 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21784 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_15 := arithEq_of_rows h (row := 321) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_15
  simp only [← c21779, ← c21780, ← c21781] at e_321_15
  have e_327_7 := arithEq_of_rows h (row := 327) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_7
  simp only [← c21782, ← c21783, k3, ← c21784] at e_327_7
  have hr := e_327_7
  simp only [e_321_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 31) = a (.virt 201) := by
  have c21785 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21785

theorem publicBatchWrapper89_f2927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16405)) (a (.wire 169 63)) (a (.virt 19507)) (a (.virt 19508)) := by
  have c21786 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21787 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21788 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21789 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21790 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21791 := (publicBatchWrapper89_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21792 := (publicBatchWrapper89_copies681 a h).1
  have c21793 := (publicBatchWrapper89_copies681 a h).2.1
  have c21794 := (publicBatchWrapper89_copies681 a h).2.2.1
  have c21795 := (publicBatchWrapper89_copies681 a h).2.2.2.1
  have c21796 := (publicBatchWrapper89_copies681 a h).2.2.2.2.1
  have c21797 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.1
  have c21798 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.1
  have c21799 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.1
  have c21800 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.1
  have c21801 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.1
  have c21802 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_10 := arithEq_of_rows h (row := 330) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_10
  simp only [← c21786, k3, ← c21787, k3, ← c21788] at e_330_10
  have e_330_11 := arithEq_of_rows h (row := 330) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_11
  simp only [← c21789, ← c21790, k3, ← c21791] at e_330_11
  have e_330_12 := arithEq_of_rows h (row := 330) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_12
  simp only [← c21798, ← c21799, k3, ← c21800] at e_330_12
  have e_331_5 := arithEq_of_rows h (row := 331) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_5
  simp only [← c21792, ← c21793, ← c21794] at e_331_5
  have e_331_6 := arithEq_of_rows h (row := 331) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_6
  simp only [← c21795, ← c21796, ← c21797] at e_331_6
  refine ⟨?_, ?_⟩
  · have hc := e_331_5
    simp only [e_330_11] at hc
    linear_combination c21801.trans k2 - hc
  · have hc := e_330_12
    simp only [e_330_10, e_331_6, e_330_11] at hc
    linear_combination c21802.trans k2 - hc

theorem publicBatchWrapper89_f2928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16406)) (a (.wire 169 71)) (a (.virt 19509)) (a (.virt 19510)) := by
  have c21803 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21804 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21805 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21806 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21807 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21808 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21809 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21810 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21811 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21812 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21813 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21814 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21815 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21816 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21817 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21818 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21819 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_13 := arithEq_of_rows h (row := 330) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_13
  simp only [← c21803, k3, ← c21804, k3, ← c21805] at e_330_13
  have e_330_14 := arithEq_of_rows h (row := 330) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_14
  simp only [← c21806, ← c21807, k3, ← c21808] at e_330_14
  have e_330_15 := arithEq_of_rows h (row := 330) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_15
  simp only [← c21815, ← c21816, k3, ← c21817] at e_330_15
  have e_331_7 := arithEq_of_rows h (row := 331) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_7
  simp only [← c21809, ← c21810, ← c21811] at e_331_7
  have e_331_8 := arithEq_of_rows h (row := 331) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_8
  simp only [← c21812, ← c21813, ← c21814] at e_331_8
  refine ⟨?_, ?_⟩
  · have hc := e_331_7
    simp only [e_330_14] at hc
    linear_combination c21818.trans k2 - hc
  · have hc := e_330_15
    simp only [e_330_13, e_331_8, e_330_14] at hc
    linear_combination c21819.trans k2 - hc

theorem publicBatchWrapper89_f2929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16407)) (a (.wire 169 79)) (a (.virt 19511)) (a (.virt 19512)) := by
  have c21820 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21821 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21822 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21823 := (publicBatchWrapper89_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21824 := (publicBatchWrapper89_copies682 a h).1
  have c21825 := (publicBatchWrapper89_copies682 a h).2.1
  have c21826 := (publicBatchWrapper89_copies682 a h).2.2.1
  have c21827 := (publicBatchWrapper89_copies682 a h).2.2.2.1
  have c21828 := (publicBatchWrapper89_copies682 a h).2.2.2.2.1
  have c21829 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.1
  have c21830 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.1
  have c21831 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.1
  have c21832 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.1
  have c21833 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.1
  have c21834 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21835 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21836 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_16 := arithEq_of_rows h (row := 330) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_16
  simp only [← c21820, k3, ← c21821, k3, ← c21822] at e_330_16
  have e_330_17 := arithEq_of_rows h (row := 330) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_17
  simp only [← c21823, ← c21824, k3, ← c21825] at e_330_17
  have e_330_18 := arithEq_of_rows h (row := 330) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_18
  simp only [← c21832, ← c21833, k3, ← c21834] at e_330_18
  have e_331_9 := arithEq_of_rows h (row := 331) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_9
  simp only [← c21826, ← c21827, ← c21828] at e_331_9
  have e_331_10 := arithEq_of_rows h (row := 331) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_10
  simp only [← c21829, ← c21830, ← c21831] at e_331_10
  refine ⟨?_, ?_⟩
  · have hc := e_331_9
    simp only [e_330_17] at hc
    linear_combination c21835.trans k2 - hc
  · have hc := e_330_18
    simp only [e_330_16, e_331_10, e_330_17] at hc
    linear_combination c21836.trans k2 - hc

theorem publicBatchWrapper89_f2930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16408)) (a (.wire 170 7)) (a (.virt 19513)) (a (.virt 19514)) := by
  have c21837 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21838 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21839 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21840 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21841 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21842 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21843 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21844 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21845 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21846 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21847 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21848 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21849 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21850 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21851 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21852 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21853 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_330_19 := arithEq_of_rows h (row := 330) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_19
  simp only [← c21837, k3, ← c21838, k3, ← c21839] at e_330_19
  have e_331_11 := arithEq_of_rows h (row := 331) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_11
  simp only [← c21843, ← c21844, ← c21845] at e_331_11
  have e_331_12 := arithEq_of_rows h (row := 331) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_12
  simp only [← c21846, ← c21847, ← c21848] at e_331_12
  have e_332_0 := arithEq_of_rows h (row := 332) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_0
  simp only [← c21840, ← c21841, k3, ← c21842] at e_332_0
  have e_332_1 := arithEq_of_rows h (row := 332) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_1
  simp only [← c21849, ← c21850, k3, ← c21851] at e_332_1
  refine ⟨?_, ?_⟩
  · have hc := e_331_11
    simp only [e_332_0] at hc
    linear_combination c21852.trans k2 - hc
  · have hc := e_332_1
    simp only [e_330_19, e_331_12, e_332_0] at hc
    linear_combination c21853.trans k2 - hc

theorem publicBatchWrapper89_f2931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 331 55) = band (a (.virt 19507)) (a (.virt 19509)) := by
  have c21854 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21855 := (publicBatchWrapper89_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21856 := (publicBatchWrapper89_copies683 a h).1
  have e_331_13 := arithEq_of_rows h (row := 331) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_13
  simp only [← c21854, ← c21855, ← c21856] at e_331_13
  have hr := e_331_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 331 59) = band (a (.virt 19511)) (a (.virt 19513)) := by
  have c21857 := (publicBatchWrapper89_copies683 a h).2.1
  have c21858 := (publicBatchWrapper89_copies683 a h).2.2.1
  have c21859 := (publicBatchWrapper89_copies683 a h).2.2.2.1
  have e_331_14 := arithEq_of_rows h (row := 331) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_14
  simp only [← c21857, ← c21858, ← c21859] at e_331_14
  have hr := e_331_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 331 63) = band (a (.wire 331 55)) (a (.wire 331 59)) := by
  have c21860 := (publicBatchWrapper89_copies683 a h).2.2.2.2.1
  have c21861 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.1
  have c21862 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.1
  have e_331_15 := arithEq_of_rows h (row := 331) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_15
  simp only [← c21860, ← c21861, ← c21862] at e_331_15
  have hr := e_331_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 35) = bor (a (.wire 79 67)) (a (.wire 331 63)) := by
  have c21863 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.1
  have c21864 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.1
  have c21865 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.1
  have c21866 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21867 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21868 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_16 := arithEq_of_rows h (row := 321) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_16
  simp only [← c21863, ← c21864, ← c21865] at e_321_16
  have e_327_8 := arithEq_of_rows h (row := 327) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_8
  simp only [← c21866, ← c21867, k3, ← c21868] at e_327_8
  have hr := e_327_8
  simp only [e_321_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 35) = a (.virt 201) := by
  have c21869 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21869

theorem publicBatchWrapper89_f2936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16603)) (a (.wire 170 23)) (a (.virt 19515)) (a (.virt 19516)) := by
  have c21870 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21871 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21872 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21873 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21874 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21875 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21876 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21877 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21878 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21879 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21880 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21881 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21882 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21883 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21884 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21885 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21886 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_331_16 := arithEq_of_rows h (row := 331) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_16
  simp only [← c21876, ← c21877, ← c21878] at e_331_16
  have e_331_17 := arithEq_of_rows h (row := 331) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_17
  simp only [← c21879, ← c21880, ← c21881] at e_331_17
  have e_332_2 := arithEq_of_rows h (row := 332) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_2
  simp only [← c21870, k3, ← c21871, k3, ← c21872] at e_332_2
  have e_332_3 := arithEq_of_rows h (row := 332) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_3
  simp only [← c21873, ← c21874, k3, ← c21875] at e_332_3
  have e_332_4 := arithEq_of_rows h (row := 332) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_4
  simp only [← c21882, ← c21883, k3, ← c21884] at e_332_4
  refine ⟨?_, ?_⟩
  · have hc := e_331_16
    simp only [e_332_3] at hc
    linear_combination c21885.trans k2 - hc
  · have hc := e_332_4
    simp only [e_332_2, e_331_17, e_332_3] at hc
    linear_combination c21886.trans k2 - hc

theorem publicBatchWrapper89_f2937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 39) = bor (a (.wire 81 31)) (a (.virt 19515)) := by
  have c21887 := (publicBatchWrapper89_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21888 := (publicBatchWrapper89_copies684 a h).1
  have c21889 := (publicBatchWrapper89_copies684 a h).2.1
  have c21890 := (publicBatchWrapper89_copies684 a h).2.2.1
  have c21891 := (publicBatchWrapper89_copies684 a h).2.2.2.1
  have c21892 := (publicBatchWrapper89_copies684 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_17 := arithEq_of_rows h (row := 321) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_17
  simp only [← c21887, ← c21888, ← c21889] at e_321_17
  have e_327_9 := arithEq_of_rows h (row := 327) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_9
  simp only [← c21890, ← c21891, k3, ← c21892] at e_327_9
  have hr := e_327_9
  simp only [e_321_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 39) = a (.virt 201) := by
  have c21893 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.1
  exact c21893

theorem publicBatchWrapper89_f2939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16604)) (a (.wire 170 31)) (a (.virt 19517)) (a (.virt 19518)) := by
  have c21894 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.1
  have c21895 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.1
  have c21896 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.1
  have c21897 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.1
  have c21898 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21899 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21900 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21901 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21902 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21903 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21904 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21905 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21906 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21907 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21908 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21909 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21910 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_331_18 := arithEq_of_rows h (row := 331) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_18
  simp only [← c21900, ← c21901, ← c21902] at e_331_18
  have e_331_19 := arithEq_of_rows h (row := 331) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_19
  simp only [← c21903, ← c21904, ← c21905] at e_331_19
  have e_332_5 := arithEq_of_rows h (row := 332) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_5
  simp only [← c21894, k3, ← c21895, k3, ← c21896] at e_332_5
  have e_332_6 := arithEq_of_rows h (row := 332) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_6
  simp only [← c21897, ← c21898, k3, ← c21899] at e_332_6
  have e_332_7 := arithEq_of_rows h (row := 332) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_7
  simp only [← c21906, ← c21907, k3, ← c21908] at e_332_7
  refine ⟨?_, ?_⟩
  · have hc := e_331_18
    simp only [e_332_6] at hc
    linear_combination c21909.trans k2 - hc
  · have hc := e_332_7
    simp only [e_332_5, e_331_19, e_332_6] at hc
    linear_combination c21910.trans k2 - hc

theorem publicBatchWrapper89_f2940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 43) = bor (a (.wire 81 31)) (a (.virt 19517)) := by
  have c21911 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21912 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21913 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21914 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21915 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21916 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_18 := arithEq_of_rows h (row := 321) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_18
  simp only [← c21911, ← c21912, ← c21913] at e_321_18
  have e_327_10 := arithEq_of_rows h (row := 327) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_10
  simp only [← c21914, ← c21915, k3, ← c21916] at e_327_10
  have hr := e_327_10
  simp only [e_321_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 43) = a (.virt 201) := by
  have c21917 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21917

theorem publicBatchWrapper89_f2942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16605)) (a (.wire 169 63)) (a (.virt 19519)) (a (.virt 19520)) := by
  have c21918 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21919 := (publicBatchWrapper89_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21920 := (publicBatchWrapper89_copies685 a h).1
  have c21921 := (publicBatchWrapper89_copies685 a h).2.1
  have c21922 := (publicBatchWrapper89_copies685 a h).2.2.1
  have c21923 := (publicBatchWrapper89_copies685 a h).2.2.2.1
  have c21924 := (publicBatchWrapper89_copies685 a h).2.2.2.2.1
  have c21925 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.1
  have c21926 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.1
  have c21927 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.1
  have c21928 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.1
  have c21929 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.1
  have c21930 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21931 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21932 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21933 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21934 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_332_8 := arithEq_of_rows h (row := 332) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_8
  simp only [← c21918, k3, ← c21919, k3, ← c21920] at e_332_8
  have e_332_9 := arithEq_of_rows h (row := 332) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_9
  simp only [← c21921, ← c21922, k3, ← c21923] at e_332_9
  have e_332_10 := arithEq_of_rows h (row := 332) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_10
  simp only [← c21930, ← c21931, k3, ← c21932] at e_332_10
  have e_333_0 := arithEq_of_rows h (row := 333) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_0
  simp only [← c21924, ← c21925, ← c21926] at e_333_0
  have e_333_1 := arithEq_of_rows h (row := 333) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_1
  simp only [← c21927, ← c21928, ← c21929] at e_333_1
  refine ⟨?_, ?_⟩
  · have hc := e_333_0
    simp only [e_332_9] at hc
    linear_combination c21933.trans k2 - hc
  · have hc := e_332_10
    simp only [e_332_8, e_333_1, e_332_9] at hc
    linear_combination c21934.trans k2 - hc

theorem publicBatchWrapper89_f2943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16606)) (a (.wire 169 71)) (a (.virt 19521)) (a (.virt 19522)) := by
  have c21935 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21936 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21937 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21938 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21939 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21940 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21941 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21942 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21943 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21944 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21945 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21946 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21947 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21948 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21949 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21950 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21951 := (publicBatchWrapper89_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_332_11 := arithEq_of_rows h (row := 332) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_11
  simp only [← c21935, k3, ← c21936, k3, ← c21937] at e_332_11
  have e_332_12 := arithEq_of_rows h (row := 332) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_12
  simp only [← c21938, ← c21939, k3, ← c21940] at e_332_12
  have e_332_13 := arithEq_of_rows h (row := 332) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_13
  simp only [← c21947, ← c21948, k3, ← c21949] at e_332_13
  have e_333_2 := arithEq_of_rows h (row := 333) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_2
  simp only [← c21941, ← c21942, ← c21943] at e_333_2
  have e_333_3 := arithEq_of_rows h (row := 333) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_3
  simp only [← c21944, ← c21945, ← c21946] at e_333_3
  refine ⟨?_, ?_⟩
  · have hc := e_333_2
    simp only [e_332_12] at hc
    linear_combination c21950.trans k2 - hc
  · have hc := e_332_13
    simp only [e_332_11, e_333_3, e_332_12] at hc
    linear_combination c21951.trans k2 - hc

theorem publicBatchWrapper89_f2944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16607)) (a (.wire 169 79)) (a (.virt 19523)) (a (.virt 19524)) := by
  have c21952 := (publicBatchWrapper89_copies686 a h).1
  have c21953 := (publicBatchWrapper89_copies686 a h).2.1
  have c21954 := (publicBatchWrapper89_copies686 a h).2.2.1
  have c21955 := (publicBatchWrapper89_copies686 a h).2.2.2.1
  have c21956 := (publicBatchWrapper89_copies686 a h).2.2.2.2.1
  have c21957 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.1
  have c21958 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.1
  have c21959 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.1
  have c21960 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.1
  have c21961 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.1
  have c21962 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21963 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21964 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21965 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21966 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21967 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21968 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_332_14 := arithEq_of_rows h (row := 332) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_14
  simp only [← c21952, k3, ← c21953, k3, ← c21954] at e_332_14
  have e_332_15 := arithEq_of_rows h (row := 332) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_15
  simp only [← c21955, ← c21956, k3, ← c21957] at e_332_15
  have e_332_16 := arithEq_of_rows h (row := 332) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_16
  simp only [← c21964, ← c21965, k3, ← c21966] at e_332_16
  have e_333_4 := arithEq_of_rows h (row := 333) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_4
  simp only [← c21958, ← c21959, ← c21960] at e_333_4
  have e_333_5 := arithEq_of_rows h (row := 333) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_5
  simp only [← c21961, ← c21962, ← c21963] at e_333_5
  refine ⟨?_, ?_⟩
  · have hc := e_333_4
    simp only [e_332_15] at hc
    linear_combination c21967.trans k2 - hc
  · have hc := e_332_16
    simp only [e_332_14, e_333_5, e_332_15] at hc
    linear_combination c21968.trans k2 - hc

theorem publicBatchWrapper89_f2945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16608)) (a (.wire 170 7)) (a (.virt 19525)) (a (.virt 19526)) := by
  have c21969 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21970 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21971 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21972 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21973 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21974 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21975 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21976 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21977 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21978 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21979 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21980 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21981 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21982 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21983 := (publicBatchWrapper89_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21984 := (publicBatchWrapper89_copies687 a h).1
  have c21985 := (publicBatchWrapper89_copies687 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_332_17 := arithEq_of_rows h (row := 332) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_17
  simp only [← c21969, k3, ← c21970, k3, ← c21971] at e_332_17
  have e_332_18 := arithEq_of_rows h (row := 332) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_18
  simp only [← c21972, ← c21973, k3, ← c21974] at e_332_18
  have e_332_19 := arithEq_of_rows h (row := 332) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_19
  simp only [← c21981, ← c21982, k3, ← c21983] at e_332_19
  have e_333_6 := arithEq_of_rows h (row := 333) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_6
  simp only [← c21975, ← c21976, ← c21977] at e_333_6
  have e_333_7 := arithEq_of_rows h (row := 333) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_7
  simp only [← c21978, ← c21979, ← c21980] at e_333_7
  refine ⟨?_, ?_⟩
  · have hc := e_333_6
    simp only [e_332_18] at hc
    linear_combination c21984.trans k2 - hc
  · have hc := e_332_19
    simp only [e_332_17, e_333_7, e_332_18] at hc
    linear_combination c21985.trans k2 - hc

theorem publicBatchWrapper89_f2946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 333 35) = band (a (.virt 19519)) (a (.virt 19521)) := by
  have c21986 := (publicBatchWrapper89_copies687 a h).2.2.1
  have c21987 := (publicBatchWrapper89_copies687 a h).2.2.2.1
  have c21988 := (publicBatchWrapper89_copies687 a h).2.2.2.2.1
  have e_333_8 := arithEq_of_rows h (row := 333) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_8
  simp only [← c21986, ← c21987, ← c21988] at e_333_8
  have hr := e_333_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 333 39) = band (a (.virt 19523)) (a (.virt 19525)) := by
  have c21989 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.1
  have c21990 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.1
  have c21991 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.1
  have e_333_9 := arithEq_of_rows h (row := 333) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_9
  simp only [← c21989, ← c21990, ← c21991] at e_333_9
  have hr := e_333_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 333 43) = band (a (.wire 333 35)) (a (.wire 333 39)) := by
  have c21992 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.1
  have c21993 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.1
  have c21994 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_333_10 := arithEq_of_rows h (row := 333) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_10
  simp only [← c21992, ← c21993, ← c21994] at e_333_10
  have hr := e_333_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 47) = bor (a (.wire 81 31)) (a (.wire 333 43)) := by
  have c21995 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21996 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21997 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21998 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21999 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22000 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_321_19 := arithEq_of_rows h (row := 321) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_321_19
  simp only [← c21995, ← c21996, ← c21997] at e_321_19
  have e_327_11 := arithEq_of_rows h (row := 327) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_11
  simp only [← c21998, ← c21999, k3, ← c22000] at e_327_11
  have hr := e_327_11
  simp only [e_321_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 47) = a (.virt 201) := by
  have c22001 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22001

theorem publicBatchWrapper89_f2951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16803)) (a (.wire 170 23)) (a (.virt 19527)) (a (.virt 19528)) := by
  have c22002 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22003 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22004 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22005 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22006 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22007 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22008 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22009 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22010 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22011 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22012 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22013 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22014 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22015 := (publicBatchWrapper89_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22016 := (publicBatchWrapper89_copies688 a h).1
  have c22017 := (publicBatchWrapper89_copies688 a h).2.1
  have c22018 := (publicBatchWrapper89_copies688 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_333_11 := arithEq_of_rows h (row := 333) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_11
  simp only [← c22008, ← c22009, ← c22010] at e_333_11
  have e_333_12 := arithEq_of_rows h (row := 333) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_12
  simp only [← c22011, ← c22012, ← c22013] at e_333_12
  have e_334_0 := arithEq_of_rows h (row := 334) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_0
  simp only [← c22002, k3, ← c22003, k3, ← c22004] at e_334_0
  have e_334_1 := arithEq_of_rows h (row := 334) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_1
  simp only [← c22005, ← c22006, k3, ← c22007] at e_334_1
  have e_334_2 := arithEq_of_rows h (row := 334) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_2
  simp only [← c22014, ← c22015, k3, ← c22016] at e_334_2
  refine ⟨?_, ?_⟩
  · have hc := e_333_11
    simp only [e_334_1] at hc
    linear_combination c22017.trans k2 - hc
  · have hc := e_334_2
    simp only [e_334_0, e_333_12, e_334_1] at hc
    linear_combination c22018.trans k2 - hc

theorem publicBatchWrapper89_f2952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 51) = bor (a (.wire 81 75)) (a (.virt 19527)) := by
  have c22019 := (publicBatchWrapper89_copies688 a h).2.2.2.1
  have c22020 := (publicBatchWrapper89_copies688 a h).2.2.2.2.1
  have c22021 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.1
  have c22022 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.1
  have c22023 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.1
  have c22024 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_12 := arithEq_of_rows h (row := 327) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_12
  simp only [← c22022, ← c22023, k3, ← c22024] at e_327_12
  have e_335_0 := arithEq_of_rows h (row := 335) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_0
  simp only [← c22019, ← c22020, ← c22021] at e_335_0
  have hr := e_327_12
  simp only [e_335_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 51) = a (.virt 201) := by
  have c22025 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.1
  exact c22025

theorem publicBatchWrapper89_f2954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16804)) (a (.wire 170 31)) (a (.virt 19529)) (a (.virt 19530)) := by
  have c22026 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22027 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22028 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22029 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22030 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22031 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22032 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22033 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22034 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22035 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22036 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22037 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22038 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22039 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22040 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22041 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22042 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_333_13 := arithEq_of_rows h (row := 333) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_13
  simp only [← c22032, ← c22033, ← c22034] at e_333_13
  have e_333_14 := arithEq_of_rows h (row := 333) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_14
  simp only [← c22035, ← c22036, ← c22037] at e_333_14
  have e_334_3 := arithEq_of_rows h (row := 334) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_3
  simp only [← c22026, k3, ← c22027, k3, ← c22028] at e_334_3
  have e_334_4 := arithEq_of_rows h (row := 334) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_4
  simp only [← c22029, ← c22030, k3, ← c22031] at e_334_4
  have e_334_5 := arithEq_of_rows h (row := 334) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_5
  simp only [← c22038, ← c22039, k3, ← c22040] at e_334_5
  refine ⟨?_, ?_⟩
  · have hc := e_333_13
    simp only [e_334_4] at hc
    linear_combination c22041.trans k2 - hc
  · have hc := e_334_5
    simp only [e_334_3, e_333_14, e_334_4] at hc
    linear_combination c22042.trans k2 - hc

theorem publicBatchWrapper89_f2955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 55) = bor (a (.wire 81 75)) (a (.virt 19529)) := by
  have c22043 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22044 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22045 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22046 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22047 := (publicBatchWrapper89_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22048 := (publicBatchWrapper89_copies689 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_13 := arithEq_of_rows h (row := 327) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_13
  simp only [← c22046, ← c22047, k3, ← c22048] at e_327_13
  have e_335_1 := arithEq_of_rows h (row := 335) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_1
  simp only [← c22043, ← c22044, ← c22045] at e_335_1
  have hr := e_327_13
  simp only [e_335_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 55) = a (.virt 201) := by
  have c22049 := (publicBatchWrapper89_copies689 a h).2.1
  exact c22049

theorem publicBatchWrapper89_f2957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16805)) (a (.wire 169 63)) (a (.virt 19531)) (a (.virt 19532)) := by
  have c22050 := (publicBatchWrapper89_copies689 a h).2.2.1
  have c22051 := (publicBatchWrapper89_copies689 a h).2.2.2.1
  have c22052 := (publicBatchWrapper89_copies689 a h).2.2.2.2.1
  have c22053 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.1
  have c22054 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.1
  have c22055 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.1
  have c22056 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.1
  have c22057 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.1
  have c22058 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22059 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22060 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22061 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22062 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22063 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22064 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22065 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22066 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_333_15 := arithEq_of_rows h (row := 333) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_15
  simp only [← c22056, ← c22057, ← c22058] at e_333_15
  have e_333_16 := arithEq_of_rows h (row := 333) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_16
  simp only [← c22059, ← c22060, ← c22061] at e_333_16
  have e_334_6 := arithEq_of_rows h (row := 334) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_6
  simp only [← c22050, k3, ← c22051, k3, ← c22052] at e_334_6
  have e_334_7 := arithEq_of_rows h (row := 334) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_7
  simp only [← c22053, ← c22054, k3, ← c22055] at e_334_7
  have e_334_8 := arithEq_of_rows h (row := 334) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_8
  simp only [← c22062, ← c22063, k3, ← c22064] at e_334_8
  refine ⟨?_, ?_⟩
  · have hc := e_333_15
    simp only [e_334_7] at hc
    linear_combination c22065.trans k2 - hc
  · have hc := e_334_8
    simp only [e_334_6, e_333_16, e_334_7] at hc
    linear_combination c22066.trans k2 - hc

theorem publicBatchWrapper89_f2958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16806)) (a (.wire 169 71)) (a (.virt 19533)) (a (.virt 19534)) := by
  have c22067 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22068 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22069 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22070 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22071 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22072 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22073 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22074 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22075 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22076 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22077 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22078 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22079 := (publicBatchWrapper89_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22080 := (publicBatchWrapper89_copies690 a h).1
  have c22081 := (publicBatchWrapper89_copies690 a h).2.1
  have c22082 := (publicBatchWrapper89_copies690 a h).2.2.1
  have c22083 := (publicBatchWrapper89_copies690 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_333_17 := arithEq_of_rows h (row := 333) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_17
  simp only [← c22073, ← c22074, ← c22075] at e_333_17
  have e_333_18 := arithEq_of_rows h (row := 333) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_18
  simp only [← c22076, ← c22077, ← c22078] at e_333_18
  have e_334_9 := arithEq_of_rows h (row := 334) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_9
  simp only [← c22067, k3, ← c22068, k3, ← c22069] at e_334_9
  have e_334_10 := arithEq_of_rows h (row := 334) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_10
  simp only [← c22070, ← c22071, k3, ← c22072] at e_334_10
  have e_334_11 := arithEq_of_rows h (row := 334) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_11
  simp only [← c22079, ← c22080, k3, ← c22081] at e_334_11
  refine ⟨?_, ?_⟩
  · have hc := e_333_17
    simp only [e_334_10] at hc
    linear_combination c22082.trans k2 - hc
  · have hc := e_334_11
    simp only [e_334_9, e_333_18, e_334_10] at hc
    linear_combination c22083.trans k2 - hc

theorem publicBatchWrapper89_f2959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16807)) (a (.wire 169 79)) (a (.virt 19535)) (a (.virt 19536)) := by
  have c22084 := (publicBatchWrapper89_copies690 a h).2.2.2.2.1
  have c22085 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.1
  have c22086 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.1
  have c22087 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.1
  have c22088 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.1
  have c22089 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.1
  have c22090 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22091 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22092 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22093 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22094 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22095 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22096 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22097 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22098 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22099 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22100 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_333_19 := arithEq_of_rows h (row := 333) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_19
  simp only [← c22090, ← c22091, ← c22092] at e_333_19
  have e_334_12 := arithEq_of_rows h (row := 334) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_12
  simp only [← c22084, k3, ← c22085, k3, ← c22086] at e_334_12
  have e_334_13 := arithEq_of_rows h (row := 334) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_13
  simp only [← c22087, ← c22088, k3, ← c22089] at e_334_13
  have e_334_14 := arithEq_of_rows h (row := 334) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_14
  simp only [← c22096, ← c22097, k3, ← c22098] at e_334_14
  have e_336_0 := arithEq_of_rows h (row := 336) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_0
  simp only [← c22093, ← c22094, ← c22095] at e_336_0
  refine ⟨?_, ?_⟩
  · have hc := e_333_19
    simp only [e_334_13] at hc
    linear_combination c22099.trans k2 - hc
  · have hc := e_334_14
    simp only [e_334_12, e_336_0, e_334_13] at hc
    linear_combination c22100.trans k2 - hc

theorem publicBatchWrapper89_f2960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16808)) (a (.wire 170 7)) (a (.virt 19537)) (a (.virt 19538)) := by
  have c22101 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22102 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22103 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22104 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22105 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22106 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22107 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22108 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22109 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22110 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22111 := (publicBatchWrapper89_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22112 := (publicBatchWrapper89_copies691 a h).1
  have c22113 := (publicBatchWrapper89_copies691 a h).2.1
  have c22114 := (publicBatchWrapper89_copies691 a h).2.2.1
  have c22115 := (publicBatchWrapper89_copies691 a h).2.2.2.1
  have c22116 := (publicBatchWrapper89_copies691 a h).2.2.2.2.1
  have c22117 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_334_15 := arithEq_of_rows h (row := 334) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_15
  simp only [← c22101, k3, ← c22102, k3, ← c22103] at e_334_15
  have e_334_16 := arithEq_of_rows h (row := 334) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_16
  simp only [← c22104, ← c22105, k3, ← c22106] at e_334_16
  have e_334_17 := arithEq_of_rows h (row := 334) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_17
  simp only [← c22113, ← c22114, k3, ← c22115] at e_334_17
  have e_336_1 := arithEq_of_rows h (row := 336) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_1
  simp only [← c22107, ← c22108, ← c22109] at e_336_1
  have e_336_2 := arithEq_of_rows h (row := 336) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_2
  simp only [← c22110, ← c22111, ← c22112] at e_336_2
  refine ⟨?_, ?_⟩
  · have hc := e_336_1
    simp only [e_334_16] at hc
    linear_combination c22116.trans k2 - hc
  · have hc := e_334_17
    simp only [e_334_15, e_336_2, e_334_16] at hc
    linear_combination c22117.trans k2 - hc

theorem publicBatchWrapper89_f2961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 336 15) = band (a (.virt 19531)) (a (.virt 19533)) := by
  have c22118 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.1
  have c22119 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.1
  have c22120 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.1
  have e_336_3 := arithEq_of_rows h (row := 336) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_3
  simp only [← c22118, ← c22119, ← c22120] at e_336_3
  have hr := e_336_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 336 19) = band (a (.virt 19535)) (a (.virt 19537)) := by
  have c22121 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.1
  have c22122 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22123 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_336_4 := arithEq_of_rows h (row := 336) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_4
  simp only [← c22121, ← c22122, ← c22123] at e_336_4
  have hr := e_336_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 336 23) = band (a (.wire 336 15)) (a (.wire 336 19)) := by
  have c22124 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22125 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22126 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_336_5 := arithEq_of_rows h (row := 336) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_5
  simp only [← c22124, ← c22125, ← c22126] at e_336_5
  have hr := e_336_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 59) = bor (a (.wire 81 75)) (a (.wire 336 23)) := by
  have c22127 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22128 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22129 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22130 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22131 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22132 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_14 := arithEq_of_rows h (row := 327) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_14
  simp only [← c22130, ← c22131, k3, ← c22132] at e_327_14
  have e_335_2 := arithEq_of_rows h (row := 335) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_2
  simp only [← c22127, ← c22128, ← c22129] at e_335_2
  have hr := e_327_14
  simp only [e_335_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 59) = a (.virt 201) := by
  have c22133 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22133

theorem publicBatchWrapper89_f2966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17003)) (a (.wire 170 23)) (a (.virt 19539)) (a (.virt 19540)) := by
  have c22134 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22135 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22136 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22137 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22138 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22139 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22140 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22141 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22142 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22143 := (publicBatchWrapper89_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22144 := (publicBatchWrapper89_copies692 a h).1
  have c22145 := (publicBatchWrapper89_copies692 a h).2.1
  have c22146 := (publicBatchWrapper89_copies692 a h).2.2.1
  have c22147 := (publicBatchWrapper89_copies692 a h).2.2.2.1
  have c22148 := (publicBatchWrapper89_copies692 a h).2.2.2.2.1
  have c22149 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.1
  have c22150 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_334_18 := arithEq_of_rows h (row := 334) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_18
  simp only [← c22134, k3, ← c22135, k3, ← c22136] at e_334_18
  have e_334_19 := arithEq_of_rows h (row := 334) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_19
  simp only [← c22137, ← c22138, k3, ← c22139] at e_334_19
  have e_336_6 := arithEq_of_rows h (row := 336) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_6
  simp only [← c22140, ← c22141, ← c22142] at e_336_6
  have e_336_7 := arithEq_of_rows h (row := 336) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_7
  simp only [← c22143, ← c22144, ← c22145] at e_336_7
  have e_337_0 := arithEq_of_rows h (row := 337) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_0
  simp only [← c22146, ← c22147, k3, ← c22148] at e_337_0
  refine ⟨?_, ?_⟩
  · have hc := e_336_6
    simp only [e_334_19] at hc
    linear_combination c22149.trans k2 - hc
  · have hc := e_337_0
    simp only [e_334_18, e_336_7, e_334_19] at hc
    linear_combination c22150.trans k2 - hc

theorem publicBatchWrapper89_f2967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 63) = bor (a (.wire 83 39)) (a (.virt 19539)) := by
  have c22151 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.1
  have c22152 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.1
  have c22153 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.1
  have c22154 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22155 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22156 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_15 := arithEq_of_rows h (row := 327) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_15
  simp only [← c22154, ← c22155, k3, ← c22156] at e_327_15
  have e_335_3 := arithEq_of_rows h (row := 335) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_3
  simp only [← c22151, ← c22152, ← c22153] at e_335_3
  have hr := e_327_15
  simp only [e_335_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 63) = a (.virt 201) := by
  have c22157 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22157

theorem publicBatchWrapper89_f2969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17004)) (a (.wire 170 31)) (a (.virt 19541)) (a (.virt 19542)) := by
  have c22158 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22159 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22160 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22161 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22162 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22163 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22164 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22165 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22166 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22167 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22168 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22169 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22170 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22171 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22172 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22173 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22174 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_336_8 := arithEq_of_rows h (row := 336) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_8
  simp only [← c22164, ← c22165, ← c22166] at e_336_8
  have e_336_9 := arithEq_of_rows h (row := 336) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_9
  simp only [← c22167, ← c22168, ← c22169] at e_336_9
  have e_337_1 := arithEq_of_rows h (row := 337) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_1
  simp only [← c22158, k3, ← c22159, k3, ← c22160] at e_337_1
  have e_337_2 := arithEq_of_rows h (row := 337) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_2
  simp only [← c22161, ← c22162, k3, ← c22163] at e_337_2
  have e_337_3 := arithEq_of_rows h (row := 337) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_3
  simp only [← c22170, ← c22171, k3, ← c22172] at e_337_3
  refine ⟨?_, ?_⟩
  · have hc := e_336_8
    simp only [e_337_2] at hc
    linear_combination c22173.trans k2 - hc
  · have hc := e_337_3
    simp only [e_337_1, e_336_9, e_337_2] at hc
    linear_combination c22174.trans k2 - hc

theorem publicBatchWrapper89_f2970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 67) = bor (a (.wire 83 39)) (a (.virt 19541)) := by
  have c22175 := (publicBatchWrapper89_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22176 := (publicBatchWrapper89_copies693 a h).1
  have c22177 := (publicBatchWrapper89_copies693 a h).2.1
  have c22178 := (publicBatchWrapper89_copies693 a h).2.2.1
  have c22179 := (publicBatchWrapper89_copies693 a h).2.2.2.1
  have c22180 := (publicBatchWrapper89_copies693 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_16 := arithEq_of_rows h (row := 327) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_16
  simp only [← c22178, ← c22179, k3, ← c22180] at e_327_16
  have e_335_4 := arithEq_of_rows h (row := 335) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_4
  simp only [← c22175, ← c22176, ← c22177] at e_335_4
  have hr := e_327_16
  simp only [e_335_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 67) = a (.virt 201) := by
  have c22181 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.1
  exact c22181

theorem publicBatchWrapper89_f2972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17005)) (a (.wire 169 63)) (a (.virt 19543)) (a (.virt 19544)) := by
  have c22182 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.1
  have c22183 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.1
  have c22184 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.1
  have c22185 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.1
  have c22186 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22187 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22188 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22189 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22190 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22191 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22192 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22193 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22194 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22195 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22196 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22197 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22198 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_336_10 := arithEq_of_rows h (row := 336) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_10
  simp only [← c22188, ← c22189, ← c22190] at e_336_10
  have e_336_11 := arithEq_of_rows h (row := 336) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_11
  simp only [← c22191, ← c22192, ← c22193] at e_336_11
  have e_337_4 := arithEq_of_rows h (row := 337) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_4
  simp only [← c22182, k3, ← c22183, k3, ← c22184] at e_337_4
  have e_337_5 := arithEq_of_rows h (row := 337) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_5
  simp only [← c22185, ← c22186, k3, ← c22187] at e_337_5
  have e_337_6 := arithEq_of_rows h (row := 337) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_6
  simp only [← c22194, ← c22195, k3, ← c22196] at e_337_6
  refine ⟨?_, ?_⟩
  · have hc := e_336_10
    simp only [e_337_5] at hc
    linear_combination c22197.trans k2 - hc
  · have hc := e_337_6
    simp only [e_337_4, e_336_11, e_337_5] at hc
    linear_combination c22198.trans k2 - hc

theorem publicBatchWrapper89_f2973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17006)) (a (.wire 169 71)) (a (.virt 19545)) (a (.virt 19546)) := by
  have c22199 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22200 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22201 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22202 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22203 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22204 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22205 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22206 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22207 := (publicBatchWrapper89_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22208 := (publicBatchWrapper89_copies694 a h).1
  have c22209 := (publicBatchWrapper89_copies694 a h).2.1
  have c22210 := (publicBatchWrapper89_copies694 a h).2.2.1
  have c22211 := (publicBatchWrapper89_copies694 a h).2.2.2.1
  have c22212 := (publicBatchWrapper89_copies694 a h).2.2.2.2.1
  have c22213 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.1
  have c22214 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.1
  have c22215 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_336_12 := arithEq_of_rows h (row := 336) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_12
  simp only [← c22205, ← c22206, ← c22207] at e_336_12
  have e_336_13 := arithEq_of_rows h (row := 336) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_13
  simp only [← c22208, ← c22209, ← c22210] at e_336_13
  have e_337_7 := arithEq_of_rows h (row := 337) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_7
  simp only [← c22199, k3, ← c22200, k3, ← c22201] at e_337_7
  have e_337_8 := arithEq_of_rows h (row := 337) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_8
  simp only [← c22202, ← c22203, k3, ← c22204] at e_337_8
  have e_337_9 := arithEq_of_rows h (row := 337) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_9
  simp only [← c22211, ← c22212, k3, ← c22213] at e_337_9
  refine ⟨?_, ?_⟩
  · have hc := e_336_12
    simp only [e_337_8] at hc
    linear_combination c22214.trans k2 - hc
  · have hc := e_337_9
    simp only [e_337_7, e_336_13, e_337_8] at hc
    linear_combination c22215.trans k2 - hc

theorem publicBatchWrapper89_f2974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17007)) (a (.wire 169 79)) (a (.virt 19547)) (a (.virt 19548)) := by
  have c22216 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.1
  have c22217 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.1
  have c22218 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22219 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22220 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22221 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22222 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22223 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22224 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22225 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22226 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22227 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22228 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22229 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22230 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22231 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22232 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_336_14 := arithEq_of_rows h (row := 336) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_14
  simp only [← c22222, ← c22223, ← c22224] at e_336_14
  have e_336_15 := arithEq_of_rows h (row := 336) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_15
  simp only [← c22225, ← c22226, ← c22227] at e_336_15
  have e_337_10 := arithEq_of_rows h (row := 337) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_10
  simp only [← c22216, k3, ← c22217, k3, ← c22218] at e_337_10
  have e_337_11 := arithEq_of_rows h (row := 337) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_11
  simp only [← c22219, ← c22220, k3, ← c22221] at e_337_11
  have e_337_12 := arithEq_of_rows h (row := 337) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_12
  simp only [← c22228, ← c22229, k3, ← c22230] at e_337_12
  refine ⟨?_, ?_⟩
  · have hc := e_336_14
    simp only [e_337_11] at hc
    linear_combination c22231.trans k2 - hc
  · have hc := e_337_12
    simp only [e_337_10, e_336_15, e_337_11] at hc
    linear_combination c22232.trans k2 - hc

theorem publicBatchWrapper89_f2975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17008)) (a (.wire 170 7)) (a (.virt 19549)) (a (.virt 19550)) := by
  have c22233 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22234 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22235 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22236 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22237 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22238 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22239 := (publicBatchWrapper89_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22240 := (publicBatchWrapper89_copies695 a h).1
  have c22241 := (publicBatchWrapper89_copies695 a h).2.1
  have c22242 := (publicBatchWrapper89_copies695 a h).2.2.1
  have c22243 := (publicBatchWrapper89_copies695 a h).2.2.2.1
  have c22244 := (publicBatchWrapper89_copies695 a h).2.2.2.2.1
  have c22245 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.1
  have c22246 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.1
  have c22247 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.1
  have c22248 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.1
  have c22249 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_336_16 := arithEq_of_rows h (row := 336) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_16
  simp only [← c22239, ← c22240, ← c22241] at e_336_16
  have e_336_17 := arithEq_of_rows h (row := 336) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_17
  simp only [← c22242, ← c22243, ← c22244] at e_336_17
  have e_337_13 := arithEq_of_rows h (row := 337) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_13
  simp only [← c22233, k3, ← c22234, k3, ← c22235] at e_337_13
  have e_337_14 := arithEq_of_rows h (row := 337) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_14
  simp only [← c22236, ← c22237, k3, ← c22238] at e_337_14
  have e_337_15 := arithEq_of_rows h (row := 337) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_15
  simp only [← c22245, ← c22246, k3, ← c22247] at e_337_15
  refine ⟨?_, ?_⟩
  · have hc := e_336_16
    simp only [e_337_14] at hc
    linear_combination c22248.trans k2 - hc
  · have hc := e_337_15
    simp only [e_337_13, e_336_17, e_337_14] at hc
    linear_combination c22249.trans k2 - hc

theorem publicBatchWrapper89_f2976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 336 75) = band (a (.virt 19543)) (a (.virt 19545)) := by
  have c22250 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22251 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22252 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_336_18 := arithEq_of_rows h (row := 336) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_18
  simp only [← c22250, ← c22251, ← c22252] at e_336_18
  have hr := e_336_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 336 79) = band (a (.virt 19547)) (a (.virt 19549)) := by
  have c22253 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22254 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22255 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_336_19 := arithEq_of_rows h (row := 336) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_19
  simp only [← c22253, ← c22254, ← c22255] at e_336_19
  have hr := e_336_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 338 3) = band (a (.wire 336 75)) (a (.wire 336 79)) := by
  have c22256 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22257 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22258 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_338_0 := arithEq_of_rows h (row := 338) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_0
  simp only [← c22256, ← c22257, ← c22258] at e_338_0
  have hr := e_338_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 71) = bor (a (.wire 83 39)) (a (.wire 338 3)) := by
  have c22259 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22260 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22261 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22262 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22263 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22264 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_17 := arithEq_of_rows h (row := 327) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_17
  simp only [← c22262, ← c22263, k3, ← c22264] at e_327_17
  have e_335_5 := arithEq_of_rows h (row := 335) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_5
  simp only [← c22259, ← c22260, ← c22261] at e_335_5
  have hr := e_327_17
  simp only [e_335_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 71) = a (.virt 201) := by
  have c22265 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22265

theorem publicBatchWrapper89_f2981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17203)) (a (.wire 170 23)) (a (.virt 19551)) (a (.virt 19552)) := by
  have c22266 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22267 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22268 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22269 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22270 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22271 := (publicBatchWrapper89_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22272 := (publicBatchWrapper89_copies696 a h).1
  have c22273 := (publicBatchWrapper89_copies696 a h).2.1
  have c22274 := (publicBatchWrapper89_copies696 a h).2.2.1
  have c22275 := (publicBatchWrapper89_copies696 a h).2.2.2.1
  have c22276 := (publicBatchWrapper89_copies696 a h).2.2.2.2.1
  have c22277 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.1
  have c22278 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.1
  have c22279 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.1
  have c22280 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.1
  have c22281 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.1
  have c22282 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_337_16 := arithEq_of_rows h (row := 337) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_16
  simp only [← c22266, k3, ← c22267, k3, ← c22268] at e_337_16
  have e_337_17 := arithEq_of_rows h (row := 337) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_17
  simp only [← c22269, ← c22270, k3, ← c22271] at e_337_17
  have e_337_18 := arithEq_of_rows h (row := 337) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_18
  simp only [← c22278, ← c22279, k3, ← c22280] at e_337_18
  have e_338_1 := arithEq_of_rows h (row := 338) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_1
  simp only [← c22272, ← c22273, ← c22274] at e_338_1
  have e_338_2 := arithEq_of_rows h (row := 338) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_2
  simp only [← c22275, ← c22276, ← c22277] at e_338_2
  refine ⟨?_, ?_⟩
  · have hc := e_338_1
    simp only [e_337_17] at hc
    linear_combination c22281.trans k2 - hc
  · have hc := e_337_18
    simp only [e_337_16, e_338_2, e_337_17] at hc
    linear_combination c22282.trans k2 - hc

theorem publicBatchWrapper89_f2982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 75) = bor (a (.wire 84 3)) (a (.virt 19551)) := by
  have c22283 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22284 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22285 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22286 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22287 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22288 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_18 := arithEq_of_rows h (row := 327) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_18
  simp only [← c22286, ← c22287, k3, ← c22288] at e_327_18
  have e_335_6 := arithEq_of_rows h (row := 335) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_6
  simp only [← c22283, ← c22284, ← c22285] at e_335_6
  have hr := e_327_18
  simp only [e_335_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 75) = a (.virt 201) := by
  have c22289 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22289

theorem publicBatchWrapper89_f2984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17204)) (a (.wire 170 31)) (a (.virt 19553)) (a (.virt 19554)) := by
  have c22290 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22291 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22292 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22293 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22294 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22295 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22296 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22297 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22298 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22299 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22300 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22301 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22302 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22303 := (publicBatchWrapper89_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22304 := (publicBatchWrapper89_copies697 a h).1
  have c22305 := (publicBatchWrapper89_copies697 a h).2.1
  have c22306 := (publicBatchWrapper89_copies697 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_337_19 := arithEq_of_rows h (row := 337) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_19
  simp only [← c22290, k3, ← c22291, k3, ← c22292] at e_337_19
  have e_338_3 := arithEq_of_rows h (row := 338) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_3
  simp only [← c22296, ← c22297, ← c22298] at e_338_3
  have e_338_4 := arithEq_of_rows h (row := 338) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_4
  simp only [← c22299, ← c22300, ← c22301] at e_338_4
  have e_339_0 := arithEq_of_rows h (row := 339) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_0
  simp only [← c22293, ← c22294, k3, ← c22295] at e_339_0
  have e_339_1 := arithEq_of_rows h (row := 339) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_1
  simp only [← c22302, ← c22303, k3, ← c22304] at e_339_1
  refine ⟨?_, ?_⟩
  · have hc := e_338_3
    simp only [e_339_0] at hc
    linear_combination c22305.trans k2 - hc
  · have hc := e_339_1
    simp only [e_337_19, e_338_4, e_339_0] at hc
    linear_combination c22306.trans k2 - hc

theorem publicBatchWrapper89_f2985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 79) = bor (a (.wire 84 3)) (a (.virt 19553)) := by
  have c22307 := (publicBatchWrapper89_copies697 a h).2.2.2.1
  have c22308 := (publicBatchWrapper89_copies697 a h).2.2.2.2.1
  have c22309 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.1
  have c22310 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.1
  have c22311 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.1
  have c22312 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_327_19 := arithEq_of_rows h (row := 327) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_19
  simp only [← c22310, ← c22311, k3, ← c22312] at e_327_19
  have e_335_7 := arithEq_of_rows h (row := 335) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_7
  simp only [← c22307, ← c22308, ← c22309] at e_335_7
  have hr := e_327_19
  simp only [e_335_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 327 79) = a (.virt 201) := by
  have c22313 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.1
  exact c22313

theorem publicBatchWrapper89_f2987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17205)) (a (.wire 169 63)) (a (.virt 19555)) (a (.virt 19556)) := by
  have c22314 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22315 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22316 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22317 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22318 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22319 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22320 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22321 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22322 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22323 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22324 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22325 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22326 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22327 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22328 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22329 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22330 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_5 := arithEq_of_rows h (row := 338) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_5
  simp only [← c22320, ← c22321, ← c22322] at e_338_5
  have e_338_6 := arithEq_of_rows h (row := 338) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_6
  simp only [← c22323, ← c22324, ← c22325] at e_338_6
  have e_339_2 := arithEq_of_rows h (row := 339) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_2
  simp only [← c22314, k3, ← c22315, k3, ← c22316] at e_339_2
  have e_339_3 := arithEq_of_rows h (row := 339) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_3
  simp only [← c22317, ← c22318, k3, ← c22319] at e_339_3
  have e_339_4 := arithEq_of_rows h (row := 339) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_4
  simp only [← c22326, ← c22327, k3, ← c22328] at e_339_4
  refine ⟨?_, ?_⟩
  · have hc := e_338_5
    simp only [e_339_3] at hc
    linear_combination c22329.trans k2 - hc
  · have hc := e_339_4
    simp only [e_339_2, e_338_6, e_339_3] at hc
    linear_combination c22330.trans k2 - hc

theorem publicBatchWrapper89_f2988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17206)) (a (.wire 169 71)) (a (.virt 19557)) (a (.virt 19558)) := by
  have c22331 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22332 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22333 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22334 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22335 := (publicBatchWrapper89_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22336 := (publicBatchWrapper89_copies698 a h).1
  have c22337 := (publicBatchWrapper89_copies698 a h).2.1
  have c22338 := (publicBatchWrapper89_copies698 a h).2.2.1
  have c22339 := (publicBatchWrapper89_copies698 a h).2.2.2.1
  have c22340 := (publicBatchWrapper89_copies698 a h).2.2.2.2.1
  have c22341 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.1
  have c22342 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.1
  have c22343 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.1
  have c22344 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.1
  have c22345 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.1
  have c22346 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22347 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_7 := arithEq_of_rows h (row := 338) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_7
  simp only [← c22337, ← c22338, ← c22339] at e_338_7
  have e_338_8 := arithEq_of_rows h (row := 338) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_8
  simp only [← c22340, ← c22341, ← c22342] at e_338_8
  have e_339_5 := arithEq_of_rows h (row := 339) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_5
  simp only [← c22331, k3, ← c22332, k3, ← c22333] at e_339_5
  have e_339_6 := arithEq_of_rows h (row := 339) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_6
  simp only [← c22334, ← c22335, k3, ← c22336] at e_339_6
  have e_339_7 := arithEq_of_rows h (row := 339) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_7
  simp only [← c22343, ← c22344, k3, ← c22345] at e_339_7
  refine ⟨?_, ?_⟩
  · have hc := e_338_7
    simp only [e_339_6] at hc
    linear_combination c22346.trans k2 - hc
  · have hc := e_339_7
    simp only [e_339_5, e_338_8, e_339_6] at hc
    linear_combination c22347.trans k2 - hc

theorem publicBatchWrapper89_f2989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17207)) (a (.wire 169 79)) (a (.virt 19559)) (a (.virt 19560)) := by
  have c22348 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22349 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22350 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22351 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22352 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22353 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22354 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22355 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22356 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22357 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22358 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22359 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22360 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22361 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22362 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22363 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22364 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_9 := arithEq_of_rows h (row := 338) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_9
  simp only [← c22354, ← c22355, ← c22356] at e_338_9
  have e_338_10 := arithEq_of_rows h (row := 338) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_10
  simp only [← c22357, ← c22358, ← c22359] at e_338_10
  have e_339_8 := arithEq_of_rows h (row := 339) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_8
  simp only [← c22348, k3, ← c22349, k3, ← c22350] at e_339_8
  have e_339_9 := arithEq_of_rows h (row := 339) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_9
  simp only [← c22351, ← c22352, k3, ← c22353] at e_339_9
  have e_339_10 := arithEq_of_rows h (row := 339) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_10
  simp only [← c22360, ← c22361, k3, ← c22362] at e_339_10
  refine ⟨?_, ?_⟩
  · have hc := e_338_9
    simp only [e_339_9] at hc
    linear_combination c22363.trans k2 - hc
  · have hc := e_339_10
    simp only [e_339_8, e_338_10, e_339_9] at hc
    linear_combination c22364.trans k2 - hc

theorem publicBatchWrapper89_f2990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17208)) (a (.wire 170 7)) (a (.virt 19561)) (a (.virt 19562)) := by
  have c22365 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22366 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22367 := (publicBatchWrapper89_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22368 := (publicBatchWrapper89_copies699 a h).1
  have c22369 := (publicBatchWrapper89_copies699 a h).2.1
  have c22370 := (publicBatchWrapper89_copies699 a h).2.2.1
  have c22371 := (publicBatchWrapper89_copies699 a h).2.2.2.1
  have c22372 := (publicBatchWrapper89_copies699 a h).2.2.2.2.1
  have c22373 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.1
  have c22374 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.1
  have c22375 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.1
  have c22376 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.1
  have c22377 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.1
  have c22378 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22379 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22380 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22381 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_11 := arithEq_of_rows h (row := 338) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_11
  simp only [← c22371, ← c22372, ← c22373] at e_338_11
  have e_338_12 := arithEq_of_rows h (row := 338) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_12
  simp only [← c22374, ← c22375, ← c22376] at e_338_12
  have e_339_11 := arithEq_of_rows h (row := 339) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_11
  simp only [← c22365, k3, ← c22366, k3, ← c22367] at e_339_11
  have e_339_12 := arithEq_of_rows h (row := 339) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_12
  simp only [← c22368, ← c22369, k3, ← c22370] at e_339_12
  have e_339_13 := arithEq_of_rows h (row := 339) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_13
  simp only [← c22377, ← c22378, k3, ← c22379] at e_339_13
  refine ⟨?_, ?_⟩
  · have hc := e_338_11
    simp only [e_339_12] at hc
    linear_combination c22380.trans k2 - hc
  · have hc := e_339_13
    simp only [e_339_11, e_338_12, e_339_12] at hc
    linear_combination c22381.trans k2 - hc

theorem publicBatchWrapper89_f2991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 338 55) = band (a (.virt 19555)) (a (.virt 19557)) := by
  have c22382 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22383 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22384 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_338_13 := arithEq_of_rows h (row := 338) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_13
  simp only [← c22382, ← c22383, ← c22384] at e_338_13
  have hr := e_338_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 338 59) = band (a (.virt 19559)) (a (.virt 19561)) := by
  have c22385 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22386 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22387 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_338_14 := arithEq_of_rows h (row := 338) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_14
  simp only [← c22385, ← c22386, ← c22387] at e_338_14
  have hr := e_338_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 338 63) = band (a (.wire 338 55)) (a (.wire 338 59)) := by
  have c22388 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22389 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22390 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_338_15 := arithEq_of_rows h (row := 338) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_15
  simp only [← c22388, ← c22389, ← c22390] at e_338_15
  have hr := e_338_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f2994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 3) = bor (a (.wire 84 3)) (a (.wire 338 63)) := by
  have c22391 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22392 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22393 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22394 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22395 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22396 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_8 := arithEq_of_rows h (row := 335) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_8
  simp only [← c22391, ← c22392, ← c22393] at e_335_8
  have e_340_0 := arithEq_of_rows h (row := 340) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_0
  simp only [← c22394, ← c22395, k3, ← c22396] at e_340_0
  have hr := e_340_0
  simp only [e_335_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 3) = a (.virt 201) := by
  have c22397 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22397

theorem publicBatchWrapper89_f2996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17403)) (a (.wire 170 23)) (a (.virt 19563)) (a (.virt 19564)) := by
  have c22398 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22399 := (publicBatchWrapper89_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22400 := (publicBatchWrapper89_copies700 a h).1
  have c22401 := (publicBatchWrapper89_copies700 a h).2.1
  have c22402 := (publicBatchWrapper89_copies700 a h).2.2.1
  have c22403 := (publicBatchWrapper89_copies700 a h).2.2.2.1
  have c22404 := (publicBatchWrapper89_copies700 a h).2.2.2.2.1
  have c22405 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.1
  have c22406 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.1
  have c22407 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.1
  have c22408 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.1
  have c22409 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.1
  have c22410 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22411 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22412 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22413 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22414 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_16 := arithEq_of_rows h (row := 338) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_16
  simp only [← c22404, ← c22405, ← c22406] at e_338_16
  have e_338_17 := arithEq_of_rows h (row := 338) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_17
  simp only [← c22407, ← c22408, ← c22409] at e_338_17
  have e_339_14 := arithEq_of_rows h (row := 339) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_14
  simp only [← c22398, k3, ← c22399, k3, ← c22400] at e_339_14
  have e_339_15 := arithEq_of_rows h (row := 339) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_15
  simp only [← c22401, ← c22402, k3, ← c22403] at e_339_15
  have e_339_16 := arithEq_of_rows h (row := 339) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_16
  simp only [← c22410, ← c22411, k3, ← c22412] at e_339_16
  refine ⟨?_, ?_⟩
  · have hc := e_338_16
    simp only [e_339_15] at hc
    linear_combination c22413.trans k2 - hc
  · have hc := e_339_16
    simp only [e_339_14, e_338_17, e_339_15] at hc
    linear_combination c22414.trans k2 - hc

theorem publicBatchWrapper89_f2997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 7) = bor (a (.wire 84 47)) (a (.virt 19563)) := by
  have c22415 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22416 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22417 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22418 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22419 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22420 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_9 := arithEq_of_rows h (row := 335) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_9
  simp only [← c22415, ← c22416, ← c22417] at e_335_9
  have e_340_1 := arithEq_of_rows h (row := 340) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_1
  simp only [← c22418, ← c22419, k3, ← c22420] at e_340_1
  have hr := e_340_1
  simp only [e_335_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f2998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 7) = a (.virt 201) := by
  have c22421 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22421

theorem publicBatchWrapper89_f2999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17404)) (a (.wire 170 31)) (a (.virt 19565)) (a (.virt 19566)) := by
  have c22422 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22423 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22424 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22425 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22426 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22427 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22428 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22429 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22430 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22431 := (publicBatchWrapper89_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22432 := (publicBatchWrapper89_copies701 a h).1
  have c22433 := (publicBatchWrapper89_copies701 a h).2.1
  have c22434 := (publicBatchWrapper89_copies701 a h).2.2.1
  have c22435 := (publicBatchWrapper89_copies701 a h).2.2.2.1
  have c22436 := (publicBatchWrapper89_copies701 a h).2.2.2.2.1
  have c22437 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.1
  have c22438 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_338_18 := arithEq_of_rows h (row := 338) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_18
  simp only [← c22428, ← c22429, ← c22430] at e_338_18
  have e_338_19 := arithEq_of_rows h (row := 338) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_19
  simp only [← c22431, ← c22432, ← c22433] at e_338_19
  have e_339_17 := arithEq_of_rows h (row := 339) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_17
  simp only [← c22422, k3, ← c22423, k3, ← c22424] at e_339_17
  have e_339_18 := arithEq_of_rows h (row := 339) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_18
  simp only [← c22425, ← c22426, k3, ← c22427] at e_339_18
  have e_339_19 := arithEq_of_rows h (row := 339) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_19
  simp only [← c22434, ← c22435, k3, ← c22436] at e_339_19
  refine ⟨?_, ?_⟩
  · have hc := e_338_18
    simp only [e_339_18] at hc
    linear_combination c22437.trans k2 - hc
  · have hc := e_339_19
    simp only [e_339_17, e_338_19, e_339_18] at hc
    linear_combination c22438.trans k2 - hc

theorem publicBatchWrapper89_f3000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 11) = bor (a (.wire 84 47)) (a (.virt 19565)) := by
  have c22439 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.1
  have c22440 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.1
  have c22441 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.1
  have c22442 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22443 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22444 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_10 := arithEq_of_rows h (row := 335) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_10
  simp only [← c22439, ← c22440, ← c22441] at e_335_10
  have e_340_2 := arithEq_of_rows h (row := 340) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_2
  simp only [← c22442, ← c22443, k3, ← c22444] at e_340_2
  have hr := e_340_2
  simp only [e_335_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f3001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 11) = a (.virt 201) := by
  have c22445 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22445

theorem publicBatchWrapper89_f3002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17405)) (a (.wire 169 63)) (a (.virt 19567)) (a (.virt 19568)) := by
  have c22446 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22447 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22448 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22449 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22450 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22451 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22452 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22453 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22454 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22455 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22456 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22457 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22458 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22459 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22460 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22461 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22462 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_0 := arithEq_of_rows h (row := 341) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_0
  simp only [← c22446, k3, ← c22447, k3, ← c22448] at e_341_0
  have e_341_1 := arithEq_of_rows h (row := 341) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_1
  simp only [← c22449, ← c22450, k3, ← c22451] at e_341_1
  have e_341_2 := arithEq_of_rows h (row := 341) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_2
  simp only [← c22458, ← c22459, k3, ← c22460] at e_341_2
  have e_342_0 := arithEq_of_rows h (row := 342) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_0
  simp only [← c22452, ← c22453, ← c22454] at e_342_0
  have e_342_1 := arithEq_of_rows h (row := 342) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_1
  simp only [← c22455, ← c22456, ← c22457] at e_342_1
  refine ⟨?_, ?_⟩
  · have hc := e_342_0
    simp only [e_341_1] at hc
    linear_combination c22461.trans k2 - hc
  · have hc := e_341_2
    simp only [e_341_0, e_342_1, e_341_1] at hc
    linear_combination c22462.trans k2 - hc

theorem publicBatchWrapper89_f3003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17406)) (a (.wire 169 71)) (a (.virt 19569)) (a (.virt 19570)) := by
  have c22463 := (publicBatchWrapper89_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22464 := (publicBatchWrapper89_copies702 a h).1
  have c22465 := (publicBatchWrapper89_copies702 a h).2.1
  have c22466 := (publicBatchWrapper89_copies702 a h).2.2.1
  have c22467 := (publicBatchWrapper89_copies702 a h).2.2.2.1
  have c22468 := (publicBatchWrapper89_copies702 a h).2.2.2.2.1
  have c22469 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.1
  have c22470 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.1
  have c22471 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.1
  have c22472 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.1
  have c22473 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.1
  have c22474 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22475 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22476 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22477 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22478 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22479 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_3 := arithEq_of_rows h (row := 341) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_3
  simp only [← c22463, k3, ← c22464, k3, ← c22465] at e_341_3
  have e_341_4 := arithEq_of_rows h (row := 341) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_4
  simp only [← c22466, ← c22467, k3, ← c22468] at e_341_4
  have e_341_5 := arithEq_of_rows h (row := 341) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_5
  simp only [← c22475, ← c22476, k3, ← c22477] at e_341_5
  have e_342_2 := arithEq_of_rows h (row := 342) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_2
  simp only [← c22469, ← c22470, ← c22471] at e_342_2
  have e_342_3 := arithEq_of_rows h (row := 342) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_3
  simp only [← c22472, ← c22473, ← c22474] at e_342_3
  refine ⟨?_, ?_⟩
  · have hc := e_342_2
    simp only [e_341_4] at hc
    linear_combination c22478.trans k2 - hc
  · have hc := e_341_5
    simp only [e_341_3, e_342_3, e_341_4] at hc
    linear_combination c22479.trans k2 - hc

theorem publicBatchWrapper89_f3004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17407)) (a (.wire 169 79)) (a (.virt 19571)) (a (.virt 19572)) := by
  have c22480 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22481 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22482 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22483 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22484 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22485 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22486 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22487 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22488 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22489 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22490 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22491 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22492 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22493 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22494 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22495 := (publicBatchWrapper89_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22496 := (publicBatchWrapper89_copies703 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_6 := arithEq_of_rows h (row := 341) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_6
  simp only [← c22480, k3, ← c22481, k3, ← c22482] at e_341_6
  have e_341_7 := arithEq_of_rows h (row := 341) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_7
  simp only [← c22483, ← c22484, k3, ← c22485] at e_341_7
  have e_341_8 := arithEq_of_rows h (row := 341) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_8
  simp only [← c22492, ← c22493, k3, ← c22494] at e_341_8
  have e_342_4 := arithEq_of_rows h (row := 342) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_4
  simp only [← c22486, ← c22487, ← c22488] at e_342_4
  have e_342_5 := arithEq_of_rows h (row := 342) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_5
  simp only [← c22489, ← c22490, ← c22491] at e_342_5
  refine ⟨?_, ?_⟩
  · have hc := e_342_4
    simp only [e_341_7] at hc
    linear_combination c22495.trans k2 - hc
  · have hc := e_341_8
    simp only [e_341_6, e_342_5, e_341_7] at hc
    linear_combination c22496.trans k2 - hc

theorem publicBatchWrapper89_f3005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17408)) (a (.wire 170 7)) (a (.virt 19573)) (a (.virt 19574)) := by
  have c22497 := (publicBatchWrapper89_copies703 a h).2.1
  have c22498 := (publicBatchWrapper89_copies703 a h).2.2.1
  have c22499 := (publicBatchWrapper89_copies703 a h).2.2.2.1
  have c22500 := (publicBatchWrapper89_copies703 a h).2.2.2.2.1
  have c22501 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.1
  have c22502 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.1
  have c22503 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.1
  have c22504 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.1
  have c22505 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.1
  have c22506 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22507 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22508 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22509 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22510 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22511 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22512 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22513 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_9 := arithEq_of_rows h (row := 341) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_9
  simp only [← c22497, k3, ← c22498, k3, ← c22499] at e_341_9
  have e_341_10 := arithEq_of_rows h (row := 341) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_10
  simp only [← c22500, ← c22501, k3, ← c22502] at e_341_10
  have e_341_11 := arithEq_of_rows h (row := 341) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_11
  simp only [← c22509, ← c22510, k3, ← c22511] at e_341_11
  have e_342_6 := arithEq_of_rows h (row := 342) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_6
  simp only [← c22503, ← c22504, ← c22505] at e_342_6
  have e_342_7 := arithEq_of_rows h (row := 342) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_7
  simp only [← c22506, ← c22507, ← c22508] at e_342_7
  refine ⟨?_, ?_⟩
  · have hc := e_342_6
    simp only [e_341_10] at hc
    linear_combination c22512.trans k2 - hc
  · have hc := e_341_11
    simp only [e_341_9, e_342_7, e_341_10] at hc
    linear_combination c22513.trans k2 - hc

theorem publicBatchWrapper89_f3006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 342 35) = band (a (.virt 19567)) (a (.virt 19569)) := by
  have c22514 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22515 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22516 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_342_8 := arithEq_of_rows h (row := 342) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_8
  simp only [← c22514, ← c22515, ← c22516] at e_342_8
  have hr := e_342_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 342 39) = band (a (.virt 19571)) (a (.virt 19573)) := by
  have c22517 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22518 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22519 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_342_9 := arithEq_of_rows h (row := 342) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_9
  simp only [← c22517, ← c22518, ← c22519] at e_342_9
  have hr := e_342_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 342 43) = band (a (.wire 342 35)) (a (.wire 342 39)) := by
  have c22520 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22521 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22522 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_342_10 := arithEq_of_rows h (row := 342) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_10
  simp only [← c22520, ← c22521, ← c22522] at e_342_10
  have hr := e_342_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 15) = bor (a (.wire 84 47)) (a (.wire 342 43)) := by
  have c22523 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22524 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22525 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22526 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22527 := (publicBatchWrapper89_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22528 := (publicBatchWrapper89_copies704 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_11 := arithEq_of_rows h (row := 335) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_11
  simp only [← c22523, ← c22524, ← c22525] at e_335_11
  have e_340_3 := arithEq_of_rows h (row := 340) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_3
  simp only [← c22526, ← c22527, k3, ← c22528] at e_340_3
  have hr := e_340_3
  simp only [e_335_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f3010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 15) = a (.virt 201) := by
  have c22529 := (publicBatchWrapper89_copies704 a h).2.1
  exact c22529

theorem publicBatchWrapper89_f3011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17603)) (a (.wire 170 23)) (a (.virt 19575)) (a (.virt 19576)) := by
  have c22530 := (publicBatchWrapper89_copies704 a h).2.2.1
  have c22531 := (publicBatchWrapper89_copies704 a h).2.2.2.1
  have c22532 := (publicBatchWrapper89_copies704 a h).2.2.2.2.1
  have c22533 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.1
  have c22534 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.1
  have c22535 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.1
  have c22536 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.1
  have c22537 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.1
  have c22538 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22539 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22540 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22541 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22542 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22543 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22544 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22545 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22546 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_12 := arithEq_of_rows h (row := 341) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_12
  simp only [← c22530, k3, ← c22531, k3, ← c22532] at e_341_12
  have e_341_13 := arithEq_of_rows h (row := 341) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_13
  simp only [← c22533, ← c22534, k3, ← c22535] at e_341_13
  have e_341_14 := arithEq_of_rows h (row := 341) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_14
  simp only [← c22542, ← c22543, k3, ← c22544] at e_341_14
  have e_342_11 := arithEq_of_rows h (row := 342) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_11
  simp only [← c22536, ← c22537, ← c22538] at e_342_11
  have e_342_12 := arithEq_of_rows h (row := 342) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_12
  simp only [← c22539, ← c22540, ← c22541] at e_342_12
  refine ⟨?_, ?_⟩
  · have hc := e_342_11
    simp only [e_341_13] at hc
    linear_combination c22545.trans k2 - hc
  · have hc := e_341_14
    simp only [e_341_12, e_342_12, e_341_13] at hc
    linear_combination c22546.trans k2 - hc

theorem publicBatchWrapper89_f3012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 19) = bor (a (.wire 86 11)) (a (.virt 19575)) := by
  have c22547 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22548 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22549 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22550 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22551 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22552 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_12 := arithEq_of_rows h (row := 335) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_12
  simp only [← c22547, ← c22548, ← c22549] at e_335_12
  have e_340_4 := arithEq_of_rows h (row := 340) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_4
  simp only [← c22550, ← c22551, k3, ← c22552] at e_340_4
  have hr := e_340_4
  simp only [e_335_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f3013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 19) = a (.virt 201) := by
  have c22553 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22553

theorem publicBatchWrapper89_f3014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17604)) (a (.wire 170 31)) (a (.virt 19577)) (a (.virt 19578)) := by
  have c22554 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22555 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22556 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22557 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22558 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22559 := (publicBatchWrapper89_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22560 := (publicBatchWrapper89_copies705 a h).1
  have c22561 := (publicBatchWrapper89_copies705 a h).2.1
  have c22562 := (publicBatchWrapper89_copies705 a h).2.2.1
  have c22563 := (publicBatchWrapper89_copies705 a h).2.2.2.1
  have c22564 := (publicBatchWrapper89_copies705 a h).2.2.2.2.1
  have c22565 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.1
  have c22566 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.1
  have c22567 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.1
  have c22568 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.1
  have c22569 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.1
  have c22570 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_15 := arithEq_of_rows h (row := 341) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_15
  simp only [← c22554, k3, ← c22555, k3, ← c22556] at e_341_15
  have e_341_16 := arithEq_of_rows h (row := 341) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_16
  simp only [← c22557, ← c22558, k3, ← c22559] at e_341_16
  have e_341_17 := arithEq_of_rows h (row := 341) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_17
  simp only [← c22566, ← c22567, k3, ← c22568] at e_341_17
  have e_342_13 := arithEq_of_rows h (row := 342) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_13
  simp only [← c22560, ← c22561, ← c22562] at e_342_13
  have e_342_14 := arithEq_of_rows h (row := 342) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_14
  simp only [← c22563, ← c22564, ← c22565] at e_342_14
  refine ⟨?_, ?_⟩
  · have hc := e_342_13
    simp only [e_341_16] at hc
    linear_combination c22569.trans k2 - hc
  · have hc := e_341_17
    simp only [e_341_15, e_342_14, e_341_16] at hc
    linear_combination c22570.trans k2 - hc

theorem publicBatchWrapper89_f3015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 23) = bor (a (.wire 86 11)) (a (.virt 19577)) := by
  have c22571 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22572 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22573 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22574 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22575 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22576 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_13 := arithEq_of_rows h (row := 335) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_13
  simp only [← c22571, ← c22572, ← c22573] at e_335_13
  have e_340_5 := arithEq_of_rows h (row := 340) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_5
  simp only [← c22574, ← c22575, k3, ← c22576] at e_340_5
  have hr := e_340_5
  simp only [e_335_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f3016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 23) = a (.virt 201) := by
  have c22577 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22577

theorem publicBatchWrapper89_f3017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17605)) (a (.wire 169 63)) (a (.virt 19579)) (a (.virt 19580)) := by
  have c22578 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22579 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22580 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22581 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22582 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22583 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22584 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22585 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22586 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22587 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22588 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22589 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22590 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22591 := (publicBatchWrapper89_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22592 := (publicBatchWrapper89_copies706 a h).1
  have c22593 := (publicBatchWrapper89_copies706 a h).2.1
  have c22594 := (publicBatchWrapper89_copies706 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_341_18 := arithEq_of_rows h (row := 341) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_18
  simp only [← c22578, k3, ← c22579, k3, ← c22580] at e_341_18
  have e_341_19 := arithEq_of_rows h (row := 341) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_19
  simp only [← c22581, ← c22582, k3, ← c22583] at e_341_19
  have e_342_15 := arithEq_of_rows h (row := 342) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_15
  simp only [← c22584, ← c22585, ← c22586] at e_342_15
  have e_342_16 := arithEq_of_rows h (row := 342) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_16
  simp only [← c22587, ← c22588, ← c22589] at e_342_16
  have e_343_0 := arithEq_of_rows h (row := 343) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_0
  simp only [← c22590, ← c22591, k3, ← c22592] at e_343_0
  refine ⟨?_, ?_⟩
  · have hc := e_342_15
    simp only [e_341_19] at hc
    linear_combination c22593.trans k2 - hc
  · have hc := e_343_0
    simp only [e_341_18, e_342_16, e_341_19] at hc
    linear_combination c22594.trans k2 - hc

theorem publicBatchWrapper89_f3018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17606)) (a (.wire 169 71)) (a (.virt 19581)) (a (.virt 19582)) := by
  have c22595 := (publicBatchWrapper89_copies706 a h).2.2.2.1
  have c22596 := (publicBatchWrapper89_copies706 a h).2.2.2.2.1
  have c22597 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.1
  have c22598 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.1
  have c22599 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.1
  have c22600 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.1
  have c22601 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.1
  have c22602 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22603 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22604 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22605 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22606 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22607 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22608 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22609 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22610 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22611 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_342_17 := arithEq_of_rows h (row := 342) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_17
  simp only [← c22601, ← c22602, ← c22603] at e_342_17
  have e_342_18 := arithEq_of_rows h (row := 342) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_18
  simp only [← c22604, ← c22605, ← c22606] at e_342_18
  have e_343_1 := arithEq_of_rows h (row := 343) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_1
  simp only [← c22595, k3, ← c22596, k3, ← c22597] at e_343_1
  have e_343_2 := arithEq_of_rows h (row := 343) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_2
  simp only [← c22598, ← c22599, k3, ← c22600] at e_343_2
  have e_343_3 := arithEq_of_rows h (row := 343) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_3
  simp only [← c22607, ← c22608, k3, ← c22609] at e_343_3
  refine ⟨?_, ?_⟩
  · have hc := e_342_17
    simp only [e_343_2] at hc
    linear_combination c22610.trans k2 - hc
  · have hc := e_343_3
    simp only [e_343_1, e_342_18, e_343_2] at hc
    linear_combination c22611.trans k2 - hc

theorem publicBatchWrapper89_f3019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17607)) (a (.wire 169 79)) (a (.virt 19583)) (a (.virt 19584)) := by
  have c22612 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22613 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22614 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22615 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22616 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22617 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22618 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22619 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22620 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22621 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22622 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22623 := (publicBatchWrapper89_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22624 := (publicBatchWrapper89_copies707 a h).1
  have c22625 := (publicBatchWrapper89_copies707 a h).2.1
  have c22626 := (publicBatchWrapper89_copies707 a h).2.2.1
  have c22627 := (publicBatchWrapper89_copies707 a h).2.2.2.1
  have c22628 := (publicBatchWrapper89_copies707 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_342_19 := arithEq_of_rows h (row := 342) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_19
  simp only [← c22618, ← c22619, ← c22620] at e_342_19
  have e_343_4 := arithEq_of_rows h (row := 343) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_4
  simp only [← c22612, k3, ← c22613, k3, ← c22614] at e_343_4
  have e_343_5 := arithEq_of_rows h (row := 343) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_5
  simp only [← c22615, ← c22616, k3, ← c22617] at e_343_5
  have e_343_6 := arithEq_of_rows h (row := 343) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_6
  simp only [← c22624, ← c22625, k3, ← c22626] at e_343_6
  have e_344_0 := arithEq_of_rows h (row := 344) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_0
  simp only [← c22621, ← c22622, ← c22623] at e_344_0
  refine ⟨?_, ?_⟩
  · have hc := e_342_19
    simp only [e_343_5] at hc
    linear_combination c22627.trans k2 - hc
  · have hc := e_343_6
    simp only [e_343_4, e_344_0, e_343_5] at hc
    linear_combination c22628.trans k2 - hc

theorem publicBatchWrapper89_f3020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 17608)) (a (.wire 170 7)) (a (.virt 19585)) (a (.virt 19586)) := by
  have c22629 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.1
  have c22630 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.1
  have c22631 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.1
  have c22632 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.1
  have c22633 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.1
  have c22634 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22635 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22636 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22637 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22638 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22639 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22640 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22641 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22642 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22643 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22644 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22645 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_7 := arithEq_of_rows h (row := 343) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_7
  simp only [← c22629, k3, ← c22630, k3, ← c22631] at e_343_7
  have e_343_8 := arithEq_of_rows h (row := 343) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_8
  simp only [← c22632, ← c22633, k3, ← c22634] at e_343_8
  have e_343_9 := arithEq_of_rows h (row := 343) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_9
  simp only [← c22641, ← c22642, k3, ← c22643] at e_343_9
  have e_344_1 := arithEq_of_rows h (row := 344) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_1
  simp only [← c22635, ← c22636, ← c22637] at e_344_1
  have e_344_2 := arithEq_of_rows h (row := 344) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_2
  simp only [← c22638, ← c22639, ← c22640] at e_344_2
  refine ⟨?_, ?_⟩
  · have hc := e_344_1
    simp only [e_343_8] at hc
    linear_combination c22644.trans k2 - hc
  · have hc := e_343_9
    simp only [e_343_7, e_344_2, e_343_8] at hc
    linear_combination c22645.trans k2 - hc

theorem publicBatchWrapper89_f3021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 344 15) = band (a (.virt 19579)) (a (.virt 19581)) := by
  have c22646 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22647 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22648 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_344_3 := arithEq_of_rows h (row := 344) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_3
  simp only [← c22646, ← c22647, ← c22648] at e_344_3
  have hr := e_344_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 344 19) = band (a (.virt 19583)) (a (.virt 19585)) := by
  have c22649 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22650 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22651 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_344_4 := arithEq_of_rows h (row := 344) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_4
  simp only [← c22649, ← c22650, ← c22651] at e_344_4
  have hr := e_344_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 344 23) = band (a (.wire 344 15)) (a (.wire 344 19)) := by
  have c22652 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22653 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22654 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_344_5 := arithEq_of_rows h (row := 344) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_5
  simp only [← c22652, ← c22653, ← c22654] at e_344_5
  have hr := e_344_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f3024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 27) = bor (a (.wire 86 11)) (a (.wire 344 23)) := by
  have c22655 := (publicBatchWrapper89_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22656 := (publicBatchWrapper89_copies708 a h).1
  have c22657 := (publicBatchWrapper89_copies708 a h).2.1
  have c22658 := (publicBatchWrapper89_copies708 a h).2.2.1
  have c22659 := (publicBatchWrapper89_copies708 a h).2.2.2.1
  have c22660 := (publicBatchWrapper89_copies708 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_335_14 := arithEq_of_rows h (row := 335) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_14
  simp only [← c22655, ← c22656, ← c22657] at e_335_14
  have e_340_6 := arithEq_of_rows h (row := 340) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_6
  simp only [← c22658, ← c22659, k3, ← c22660] at e_340_6
  have hr := e_340_6
  simp only [e_335_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper89_f3025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 340 27) = a (.virt 201) := by
  have c22661 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.1
  exact c22661

theorem publicBatchWrapper89_f3026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 343 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 8)) := by
  have c22662 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.1
  have c22663 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.1
  have c22664 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.1
  have c22665 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.1
  have c22666 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22667 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_10 := arithEq_of_rows h (row := 343) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_10
  simp only [← c22662, ← c22663, ← c22664] at e_343_10
  have e_343_11 := arithEq_of_rows h (row := 343) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_11
  simp only [← c22665, ← c22666, k2, ← c22667] at e_343_11
  have hr := e_343_11
  simp only [e_343_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 343 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 9)) := by
  have c22668 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22669 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22670 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22671 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22672 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22673 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_12 := arithEq_of_rows h (row := 343) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_12
  simp only [← c22668, ← c22669, ← c22670] at e_343_12
  have e_343_13 := arithEq_of_rows h (row := 343) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_13
  simp only [← c22671, ← c22672, k2, ← c22673] at e_343_13
  have hr := e_343_13
  simp only [e_343_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 343 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 10)) := by
  have c22674 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22675 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22676 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22677 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22678 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22679 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_14 := arithEq_of_rows h (row := 343) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_14
  simp only [← c22674, ← c22675, ← c22676] at e_343_14
  have e_343_15 := arithEq_of_rows h (row := 343) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_15
  simp only [← c22677, ← c22678, k2, ← c22679] at e_343_15
  have hr := e_343_15
  simp only [e_343_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 343 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 11)) := by
  have c22680 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22681 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22682 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22683 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22684 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22685 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_16 := arithEq_of_rows h (row := 343) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_16
  simp only [← c22680, ← c22681, ← c22682] at e_343_16
  have e_343_17 := arithEq_of_rows h (row := 343) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_17
  simp only [← c22683, ← c22684, k2, ← c22685] at e_343_17
  have hr := e_343_17
  simp only [e_343_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 343 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 12)) := by
  have c22686 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22687 := (publicBatchWrapper89_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22688 := (publicBatchWrapper89_copies709 a h).1
  have c22689 := (publicBatchWrapper89_copies709 a h).2.1
  have c22690 := (publicBatchWrapper89_copies709 a h).2.2.1
  have c22691 := (publicBatchWrapper89_copies709 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_343_18 := arithEq_of_rows h (row := 343) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_18
  simp only [← c22686, ← c22687, ← c22688] at e_343_18
  have e_343_19 := arithEq_of_rows h (row := 343) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_19
  simp only [← c22689, ← c22690, k2, ← c22691] at e_343_19
  have hr := e_343_19
  simp only [e_343_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 13)) := by
  have c22692 := (publicBatchWrapper89_copies709 a h).2.2.2.2.1
  have c22693 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.1
  have c22694 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.1
  have c22695 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.1
  have c22696 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.1
  have c22697 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_0 := arithEq_of_rows h (row := 345) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_0
  simp only [← c22692, ← c22693, ← c22694] at e_345_0
  have e_345_1 := arithEq_of_rows h (row := 345) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_1
  simp only [← c22695, ← c22696, k2, ← c22697] at e_345_1
  have hr := e_345_1
  simp only [e_345_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 14)) := by
  have c22698 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22699 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22700 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22701 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22702 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22703 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_2 := arithEq_of_rows h (row := 345) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_2
  simp only [← c22698, ← c22699, ← c22700] at e_345_2
  have e_345_3 := arithEq_of_rows h (row := 345) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_3
  simp only [← c22701, ← c22702, k2, ← c22703] at e_345_3
  have hr := e_345_3
  simp only [e_345_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 15)) := by
  have c22704 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22705 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22706 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22707 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22708 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22709 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_4 := arithEq_of_rows h (row := 345) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_4
  simp only [← c22704, ← c22705, ← c22706] at e_345_4
  have e_345_5 := arithEq_of_rows h (row := 345) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_5
  simp only [← c22707, ← c22708, k2, ← c22709] at e_345_5
  have hr := e_345_5
  simp only [e_345_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 16)) := by
  have c22710 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22711 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22712 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22713 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22714 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22715 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_6 := arithEq_of_rows h (row := 345) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_6
  simp only [← c22710, ← c22711, ← c22712] at e_345_6
  have e_345_7 := arithEq_of_rows h (row := 345) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_7
  simp only [← c22713, ← c22714, k2, ← c22715] at e_345_7
  have hr := e_345_7
  simp only [e_345_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 17)) := by
  have c22716 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22717 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22718 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22719 := (publicBatchWrapper89_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22720 := (publicBatchWrapper89_copies710 a h).1
  have c22721 := (publicBatchWrapper89_copies710 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_8 := arithEq_of_rows h (row := 345) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_8
  simp only [← c22716, ← c22717, ← c22718] at e_345_8
  have e_345_9 := arithEq_of_rows h (row := 345) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_9
  simp only [← c22719, ← c22720, k2, ← c22721] at e_345_9
  have hr := e_345_9
  simp only [e_345_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 18)) := by
  have c22722 := (publicBatchWrapper89_copies710 a h).2.2.1
  have c22723 := (publicBatchWrapper89_copies710 a h).2.2.2.1
  have c22724 := (publicBatchWrapper89_copies710 a h).2.2.2.2.1
  have c22725 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.1
  have c22726 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.1
  have c22727 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_10 := arithEq_of_rows h (row := 345) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_10
  simp only [← c22722, ← c22723, ← c22724] at e_345_10
  have e_345_11 := arithEq_of_rows h (row := 345) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_11
  simp only [← c22725, ← c22726, k2, ← c22727] at e_345_11
  have hr := e_345_11
  simp only [e_345_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 19)) := by
  have c22728 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.1
  have c22729 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.1
  have c22730 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22731 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22732 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22733 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_12 := arithEq_of_rows h (row := 345) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_12
  simp only [← c22728, ← c22729, ← c22730] at e_345_12
  have e_345_13 := arithEq_of_rows h (row := 345) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_13
  simp only [← c22731, ← c22732, k2, ← c22733] at e_345_13
  have hr := e_345_13
  simp only [e_345_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 20)) := by
  have c22734 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22735 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22736 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22737 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22738 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22739 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_14 := arithEq_of_rows h (row := 345) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_14
  simp only [← c22734, ← c22735, ← c22736] at e_345_14
  have e_345_15 := arithEq_of_rows h (row := 345) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_15
  simp only [← c22737, ← c22738, k2, ← c22739] at e_345_15
  have hr := e_345_15
  simp only [e_345_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 21)) := by
  have c22740 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22741 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22742 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22743 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22744 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22745 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_16 := arithEq_of_rows h (row := 345) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_16
  simp only [← c22740, ← c22741, ← c22742] at e_345_16
  have e_345_17 := arithEq_of_rows h (row := 345) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_17
  simp only [← c22743, ← c22744, k2, ← c22745] at e_345_17
  have hr := e_345_17
  simp only [e_345_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 345 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 22)) := by
  have c22746 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22747 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22748 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22749 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22750 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22751 := (publicBatchWrapper89_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_345_18 := arithEq_of_rows h (row := 345) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_18
  simp only [← c22746, ← c22747, ← c22748] at e_345_18
  have e_345_19 := arithEq_of_rows h (row := 345) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_19
  simp only [← c22749, ← c22750, k2, ← c22751] at e_345_19
  have hr := e_345_19
  simp only [e_345_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 23)) := by
  have c22752 := (publicBatchWrapper89_copies711 a h).1
  have c22753 := (publicBatchWrapper89_copies711 a h).2.1
  have c22754 := (publicBatchWrapper89_copies711 a h).2.2.1
  have c22755 := (publicBatchWrapper89_copies711 a h).2.2.2.1
  have c22756 := (publicBatchWrapper89_copies711 a h).2.2.2.2.1
  have c22757 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_0 := arithEq_of_rows h (row := 346) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_0
  simp only [← c22752, ← c22753, ← c22754] at e_346_0
  have e_346_1 := arithEq_of_rows h (row := 346) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_1
  simp only [← c22755, ← c22756, k2, ← c22757] at e_346_1
  have hr := e_346_1
  simp only [e_346_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 24)) := by
  have c22758 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.1
  have c22759 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.1
  have c22760 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.1
  have c22761 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.1
  have c22762 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22763 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_2 := arithEq_of_rows h (row := 346) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_2
  simp only [← c22758, ← c22759, ← c22760] at e_346_2
  have e_346_3 := arithEq_of_rows h (row := 346) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_3
  simp only [← c22761, ← c22762, k2, ← c22763] at e_346_3
  have hr := e_346_3
  simp only [e_346_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 25)) := by
  have c22764 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22765 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22766 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22767 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22768 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22769 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_4 := arithEq_of_rows h (row := 346) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_4
  simp only [← c22764, ← c22765, ← c22766] at e_346_4
  have e_346_5 := arithEq_of_rows h (row := 346) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_5
  simp only [← c22767, ← c22768, k2, ← c22769] at e_346_5
  have hr := e_346_5
  simp only [e_346_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 26)) := by
  have c22770 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22771 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22772 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22773 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22774 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22775 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_6 := arithEq_of_rows h (row := 346) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_6
  simp only [← c22770, ← c22771, ← c22772] at e_346_6
  have e_346_7 := arithEq_of_rows h (row := 346) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_7
  simp only [← c22773, ← c22774, k2, ← c22775] at e_346_7
  have hr := e_346_7
  simp only [e_346_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 27)) := by
  have c22776 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22777 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22778 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22779 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22780 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22781 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_8 := arithEq_of_rows h (row := 346) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_8
  simp only [← c22776, ← c22777, ← c22778] at e_346_8
  have e_346_9 := arithEq_of_rows h (row := 346) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_9
  simp only [← c22779, ← c22780, k2, ← c22781] at e_346_9
  have hr := e_346_9
  simp only [e_346_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 28)) := by
  have c22782 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22783 := (publicBatchWrapper89_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22784 := (publicBatchWrapper89_copies712 a h).1
  have c22785 := (publicBatchWrapper89_copies712 a h).2.1
  have c22786 := (publicBatchWrapper89_copies712 a h).2.2.1
  have c22787 := (publicBatchWrapper89_copies712 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_10 := arithEq_of_rows h (row := 346) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_10
  simp only [← c22782, ← c22783, ← c22784] at e_346_10
  have e_346_11 := arithEq_of_rows h (row := 346) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_11
  simp only [← c22785, ← c22786, k2, ← c22787] at e_346_11
  have hr := e_346_11
  simp only [e_346_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 29)) := by
  have c22788 := (publicBatchWrapper89_copies712 a h).2.2.2.2.1
  have c22789 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.1
  have c22790 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.1
  have c22791 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.1
  have c22792 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.1
  have c22793 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_12 := arithEq_of_rows h (row := 346) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_12
  simp only [← c22788, ← c22789, ← c22790] at e_346_12
  have e_346_13 := arithEq_of_rows h (row := 346) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_13
  simp only [← c22791, ← c22792, k2, ← c22793] at e_346_13
  have hr := e_346_13
  simp only [e_346_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 30)) := by
  have c22794 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22795 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22796 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22797 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22798 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22799 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_14 := arithEq_of_rows h (row := 346) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_14
  simp only [← c22794, ← c22795, ← c22796] at e_346_14
  have e_346_15 := arithEq_of_rows h (row := 346) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_15
  simp only [← c22797, ← c22798, k2, ← c22799] at e_346_15
  have hr := e_346_15
  simp only [e_346_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 31)) := by
  have c22800 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22801 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22802 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22803 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22804 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22805 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_16 := arithEq_of_rows h (row := 346) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_16
  simp only [← c22800, ← c22801, ← c22802] at e_346_16
  have e_346_17 := arithEq_of_rows h (row := 346) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_17
  simp only [← c22803, ← c22804, k2, ← c22805] at e_346_17
  have hr := e_346_17
  simp only [e_346_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 346 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 32)) := by
  have c22806 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22807 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22808 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22809 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22810 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22811 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_346_18 := arithEq_of_rows h (row := 346) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_18
  simp only [← c22806, ← c22807, ← c22808] at e_346_18
  have e_346_19 := arithEq_of_rows h (row := 346) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_19
  simp only [← c22809, ← c22810, k2, ← c22811] at e_346_19
  have hr := e_346_19
  simp only [e_346_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 33)) := by
  have c22812 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22813 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22814 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22815 := (publicBatchWrapper89_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22816 := (publicBatchWrapper89_copies713 a h).1
  have c22817 := (publicBatchWrapper89_copies713 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_0 := arithEq_of_rows h (row := 347) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_0
  simp only [← c22812, ← c22813, ← c22814] at e_347_0
  have e_347_1 := arithEq_of_rows h (row := 347) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_1
  simp only [← c22815, ← c22816, k2, ← c22817] at e_347_1
  have hr := e_347_1
  simp only [e_347_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 34)) := by
  have c22818 := (publicBatchWrapper89_copies713 a h).2.2.1
  have c22819 := (publicBatchWrapper89_copies713 a h).2.2.2.1
  have c22820 := (publicBatchWrapper89_copies713 a h).2.2.2.2.1
  have c22821 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.1
  have c22822 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.1
  have c22823 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_2 := arithEq_of_rows h (row := 347) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_2
  simp only [← c22818, ← c22819, ← c22820] at e_347_2
  have e_347_3 := arithEq_of_rows h (row := 347) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_3
  simp only [← c22821, ← c22822, k2, ← c22823] at e_347_3
  have hr := e_347_3
  simp only [e_347_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 35)) := by
  have c22824 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.1
  have c22825 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.1
  have c22826 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22827 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22828 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22829 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_4 := arithEq_of_rows h (row := 347) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_4
  simp only [← c22824, ← c22825, ← c22826] at e_347_4
  have e_347_5 := arithEq_of_rows h (row := 347) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_5
  simp only [← c22827, ← c22828, k2, ← c22829] at e_347_5
  have hr := e_347_5
  simp only [e_347_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 36)) := by
  have c22830 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22831 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22832 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22833 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22834 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22835 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_6 := arithEq_of_rows h (row := 347) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_6
  simp only [← c22830, ← c22831, ← c22832] at e_347_6
  have e_347_7 := arithEq_of_rows h (row := 347) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_7
  simp only [← c22833, ← c22834, k2, ← c22835] at e_347_7
  have hr := e_347_7
  simp only [e_347_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 37)) := by
  have c22836 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22837 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22838 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22839 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22840 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22841 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_8 := arithEq_of_rows h (row := 347) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_8
  simp only [← c22836, ← c22837, ← c22838] at e_347_8
  have e_347_9 := arithEq_of_rows h (row := 347) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_9
  simp only [← c22839, ← c22840, k2, ← c22841] at e_347_9
  have hr := e_347_9
  simp only [e_347_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 38)) := by
  have c22842 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22843 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22844 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22845 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22846 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22847 := (publicBatchWrapper89_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_10 := arithEq_of_rows h (row := 347) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_10
  simp only [← c22842, ← c22843, ← c22844] at e_347_10
  have e_347_11 := arithEq_of_rows h (row := 347) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_11
  simp only [← c22845, ← c22846, k2, ← c22847] at e_347_11
  have hr := e_347_11
  simp only [e_347_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 39)) := by
  have c22848 := (publicBatchWrapper89_copies714 a h).1
  have c22849 := (publicBatchWrapper89_copies714 a h).2.1
  have c22850 := (publicBatchWrapper89_copies714 a h).2.2.1
  have c22851 := (publicBatchWrapper89_copies714 a h).2.2.2.1
  have c22852 := (publicBatchWrapper89_copies714 a h).2.2.2.2.1
  have c22853 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_12 := arithEq_of_rows h (row := 347) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_12
  simp only [← c22848, ← c22849, ← c22850] at e_347_12
  have e_347_13 := arithEq_of_rows h (row := 347) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_13
  simp only [← c22851, ← c22852, k2, ← c22853] at e_347_13
  have hr := e_347_13
  simp only [e_347_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 40)) := by
  have c22854 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.1
  have c22855 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.1
  have c22856 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.1
  have c22857 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.1
  have c22858 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22859 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_14 := arithEq_of_rows h (row := 347) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_14
  simp only [← c22854, ← c22855, ← c22856] at e_347_14
  have e_347_15 := arithEq_of_rows h (row := 347) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_15
  simp only [← c22857, ← c22858, k2, ← c22859] at e_347_15
  have hr := e_347_15
  simp only [e_347_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 41)) := by
  have c22860 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22861 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22862 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22863 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22864 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22865 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_16 := arithEq_of_rows h (row := 347) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_16
  simp only [← c22860, ← c22861, ← c22862] at e_347_16
  have e_347_17 := arithEq_of_rows h (row := 347) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_17
  simp only [← c22863, ← c22864, k2, ← c22865] at e_347_17
  have hr := e_347_17
  simp only [e_347_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 347 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 42)) := by
  have c22866 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22867 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22868 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22869 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22870 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22871 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_347_18 := arithEq_of_rows h (row := 347) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_18
  simp only [← c22866, ← c22867, ← c22868] at e_347_18
  have e_347_19 := arithEq_of_rows h (row := 347) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_19
  simp only [← c22869, ← c22870, k2, ← c22871] at e_347_19
  have hr := e_347_19
  simp only [e_347_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 43)) := by
  have c22872 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22873 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22874 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22875 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22876 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22877 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_0 := arithEq_of_rows h (row := 348) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_0
  simp only [← c22872, ← c22873, ← c22874] at e_348_0
  have e_348_1 := arithEq_of_rows h (row := 348) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_1
  simp only [← c22875, ← c22876, k2, ← c22877] at e_348_1
  have hr := e_348_1
  simp only [e_348_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 44)) := by
  have c22878 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22879 := (publicBatchWrapper89_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22880 := (publicBatchWrapper89_copies715 a h).1
  have c22881 := (publicBatchWrapper89_copies715 a h).2.1
  have c22882 := (publicBatchWrapper89_copies715 a h).2.2.1
  have c22883 := (publicBatchWrapper89_copies715 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_2 := arithEq_of_rows h (row := 348) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_2
  simp only [← c22878, ← c22879, ← c22880] at e_348_2
  have e_348_3 := arithEq_of_rows h (row := 348) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_3
  simp only [← c22881, ← c22882, k2, ← c22883] at e_348_3
  have hr := e_348_3
  simp only [e_348_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 45)) := by
  have c22884 := (publicBatchWrapper89_copies715 a h).2.2.2.2.1
  have c22885 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.1
  have c22886 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.1
  have c22887 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.1
  have c22888 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.1
  have c22889 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_4 := arithEq_of_rows h (row := 348) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_4
  simp only [← c22884, ← c22885, ← c22886] at e_348_4
  have e_348_5 := arithEq_of_rows h (row := 348) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_5
  simp only [← c22887, ← c22888, k2, ← c22889] at e_348_5
  have hr := e_348_5
  simp only [e_348_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 46)) := by
  have c22890 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22891 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22892 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22893 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22894 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22895 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_6 := arithEq_of_rows h (row := 348) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_6
  simp only [← c22890, ← c22891, ← c22892] at e_348_6
  have e_348_7 := arithEq_of_rows h (row := 348) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_7
  simp only [← c22893, ← c22894, k2, ← c22895] at e_348_7
  have hr := e_348_7
  simp only [e_348_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 47)) := by
  have c22896 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22897 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22898 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22899 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22900 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22901 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_8 := arithEq_of_rows h (row := 348) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_8
  simp only [← c22896, ← c22897, ← c22898] at e_348_8
  have e_348_9 := arithEq_of_rows h (row := 348) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_9
  simp only [← c22899, ← c22900, k2, ← c22901] at e_348_9
  have hr := e_348_9
  simp only [e_348_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 48)) := by
  have c22902 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22903 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22904 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22905 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22906 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22907 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_10 := arithEq_of_rows h (row := 348) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_10
  simp only [← c22902, ← c22903, ← c22904] at e_348_10
  have e_348_11 := arithEq_of_rows h (row := 348) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_11
  simp only [← c22905, ← c22906, k2, ← c22907] at e_348_11
  have hr := e_348_11
  simp only [e_348_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 49)) := by
  have c22908 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22909 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22910 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22911 := (publicBatchWrapper89_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22912 := (publicBatchWrapper89_copies716 a h).1
  have c22913 := (publicBatchWrapper89_copies716 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_12 := arithEq_of_rows h (row := 348) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_12
  simp only [← c22908, ← c22909, ← c22910] at e_348_12
  have e_348_13 := arithEq_of_rows h (row := 348) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_13
  simp only [← c22911, ← c22912, k2, ← c22913] at e_348_13
  have hr := e_348_13
  simp only [e_348_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 50)) := by
  have c22914 := (publicBatchWrapper89_copies716 a h).2.2.1
  have c22915 := (publicBatchWrapper89_copies716 a h).2.2.2.1
  have c22916 := (publicBatchWrapper89_copies716 a h).2.2.2.2.1
  have c22917 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.1
  have c22918 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.1
  have c22919 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_14 := arithEq_of_rows h (row := 348) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_14
  simp only [← c22914, ← c22915, ← c22916] at e_348_14
  have e_348_15 := arithEq_of_rows h (row := 348) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_15
  simp only [← c22917, ← c22918, k2, ← c22919] at e_348_15
  have hr := e_348_15
  simp only [e_348_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 51)) := by
  have c22920 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.1
  have c22921 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.1
  have c22922 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22923 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22924 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22925 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_16 := arithEq_of_rows h (row := 348) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_16
  simp only [← c22920, ← c22921, ← c22922] at e_348_16
  have e_348_17 := arithEq_of_rows h (row := 348) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_17
  simp only [← c22923, ← c22924, k2, ← c22925] at e_348_17
  have hr := e_348_17
  simp only [e_348_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 348 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 52)) := by
  have c22926 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22927 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22928 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22929 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22930 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22931 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_348_18 := arithEq_of_rows h (row := 348) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_18
  simp only [← c22926, ← c22927, ← c22928] at e_348_18
  have e_348_19 := arithEq_of_rows h (row := 348) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_19
  simp only [← c22929, ← c22930, k2, ← c22931] at e_348_19
  have hr := e_348_19
  simp only [e_348_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 53)) := by
  have c22932 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22933 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22934 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22935 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22936 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22937 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_0 := arithEq_of_rows h (row := 349) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_0
  simp only [← c22932, ← c22933, ← c22934] at e_349_0
  have e_349_1 := arithEq_of_rows h (row := 349) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_1
  simp only [← c22935, ← c22936, k2, ← c22937] at e_349_1
  have hr := e_349_1
  simp only [e_349_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 54)) := by
  have c22938 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22939 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22940 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22941 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22942 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22943 := (publicBatchWrapper89_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_2 := arithEq_of_rows h (row := 349) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_2
  simp only [← c22938, ← c22939, ← c22940] at e_349_2
  have e_349_3 := arithEq_of_rows h (row := 349) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_3
  simp only [← c22941, ← c22942, k2, ← c22943] at e_349_3
  have hr := e_349_3
  simp only [e_349_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 55)) := by
  have c22944 := (publicBatchWrapper89_copies717 a h).1
  have c22945 := (publicBatchWrapper89_copies717 a h).2.1
  have c22946 := (publicBatchWrapper89_copies717 a h).2.2.1
  have c22947 := (publicBatchWrapper89_copies717 a h).2.2.2.1
  have c22948 := (publicBatchWrapper89_copies717 a h).2.2.2.2.1
  have c22949 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_4 := arithEq_of_rows h (row := 349) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_4
  simp only [← c22944, ← c22945, ← c22946] at e_349_4
  have e_349_5 := arithEq_of_rows h (row := 349) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_5
  simp only [← c22947, ← c22948, k2, ← c22949] at e_349_5
  have hr := e_349_5
  simp only [e_349_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 56)) := by
  have c22950 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.1
  have c22951 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.1
  have c22952 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.1
  have c22953 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.1
  have c22954 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22955 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_6 := arithEq_of_rows h (row := 349) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_6
  simp only [← c22950, ← c22951, ← c22952] at e_349_6
  have e_349_7 := arithEq_of_rows h (row := 349) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_7
  simp only [← c22953, ← c22954, k2, ← c22955] at e_349_7
  have hr := e_349_7
  simp only [e_349_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 57)) := by
  have c22956 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22957 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22958 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22959 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22960 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22961 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_8 := arithEq_of_rows h (row := 349) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_8
  simp only [← c22956, ← c22957, ← c22958] at e_349_8
  have e_349_9 := arithEq_of_rows h (row := 349) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_9
  simp only [← c22959, ← c22960, k2, ← c22961] at e_349_9
  have hr := e_349_9
  simp only [e_349_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 58)) := by
  have c22962 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22963 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22964 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22965 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22966 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22967 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_10 := arithEq_of_rows h (row := 349) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_10
  simp only [← c22962, ← c22963, ← c22964] at e_349_10
  have e_349_11 := arithEq_of_rows h (row := 349) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_11
  simp only [← c22965, ← c22966, k2, ← c22967] at e_349_11
  have hr := e_349_11
  simp only [e_349_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 59)) := by
  have c22968 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22969 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22970 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22971 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22972 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22973 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_12 := arithEq_of_rows h (row := 349) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_12
  simp only [← c22968, ← c22969, ← c22970] at e_349_12
  have e_349_13 := arithEq_of_rows h (row := 349) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_13
  simp only [← c22971, ← c22972, k2, ← c22973] at e_349_13
  have hr := e_349_13
  simp only [e_349_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 60)) := by
  have c22974 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22975 := (publicBatchWrapper89_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22976 := (publicBatchWrapper89_copies718 a h).1
  have c22977 := (publicBatchWrapper89_copies718 a h).2.1
  have c22978 := (publicBatchWrapper89_copies718 a h).2.2.1
  have c22979 := (publicBatchWrapper89_copies718 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_14 := arithEq_of_rows h (row := 349) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_14
  simp only [← c22974, ← c22975, ← c22976] at e_349_14
  have e_349_15 := arithEq_of_rows h (row := 349) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_15
  simp only [← c22977, ← c22978, k2, ← c22979] at e_349_15
  have hr := e_349_15
  simp only [e_349_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 61)) := by
  have c22980 := (publicBatchWrapper89_copies718 a h).2.2.2.2.1
  have c22981 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.1
  have c22982 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.1
  have c22983 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.1
  have c22984 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.1
  have c22985 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_16 := arithEq_of_rows h (row := 349) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_16
  simp only [← c22980, ← c22981, ← c22982] at e_349_16
  have e_349_17 := arithEq_of_rows h (row := 349) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_17
  simp only [← c22983, ← c22984, k2, ← c22985] at e_349_17
  have hr := e_349_17
  simp only [e_349_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 349 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 62)) := by
  have c22986 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22987 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22988 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22989 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22990 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22991 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_349_18 := arithEq_of_rows h (row := 349) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_18
  simp only [← c22986, ← c22987, ← c22988] at e_349_18
  have e_349_19 := arithEq_of_rows h (row := 349) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_19
  simp only [← c22989, ← c22990, k2, ← c22991] at e_349_19
  have hr := e_349_19
  simp only [e_349_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 63)) := by
  have c22992 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22993 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22994 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22995 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22996 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22997 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_0 := arithEq_of_rows h (row := 350) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_0
  simp only [← c22992, ← c22993, ← c22994] at e_350_0
  have e_350_1 := arithEq_of_rows h (row := 350) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_1
  simp only [← c22995, ← c22996, k2, ← c22997] at e_350_1
  have hr := e_350_1
  simp only [e_350_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 64)) := by
  have c22998 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22999 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23000 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23001 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23002 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23003 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_2 := arithEq_of_rows h (row := 350) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_2
  simp only [← c22998, ← c22999, ← c23000] at e_350_2
  have e_350_3 := arithEq_of_rows h (row := 350) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_3
  simp only [← c23001, ← c23002, k2, ← c23003] at e_350_3
  have hr := e_350_3
  simp only [e_350_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 65)) := by
  have c23004 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23005 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23006 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23007 := (publicBatchWrapper89_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23008 := (publicBatchWrapper89_copies719 a h).1
  have c23009 := (publicBatchWrapper89_copies719 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_4 := arithEq_of_rows h (row := 350) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_4
  simp only [← c23004, ← c23005, ← c23006] at e_350_4
  have e_350_5 := arithEq_of_rows h (row := 350) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_5
  simp only [← c23007, ← c23008, k2, ← c23009] at e_350_5
  have hr := e_350_5
  simp only [e_350_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 66)) := by
  have c23010 := (publicBatchWrapper89_copies719 a h).2.2.1
  have c23011 := (publicBatchWrapper89_copies719 a h).2.2.2.1
  have c23012 := (publicBatchWrapper89_copies719 a h).2.2.2.2.1
  have c23013 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.1
  have c23014 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.1
  have c23015 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_6 := arithEq_of_rows h (row := 350) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_6
  simp only [← c23010, ← c23011, ← c23012] at e_350_6
  have e_350_7 := arithEq_of_rows h (row := 350) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_7
  simp only [← c23013, ← c23014, k2, ← c23015] at e_350_7
  have hr := e_350_7
  simp only [e_350_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 67)) := by
  have c23016 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.1
  have c23017 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.1
  have c23018 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23019 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23020 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23021 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_8 := arithEq_of_rows h (row := 350) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_8
  simp only [← c23016, ← c23017, ← c23018] at e_350_8
  have e_350_9 := arithEq_of_rows h (row := 350) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_9
  simp only [← c23019, ← c23020, k2, ← c23021] at e_350_9
  have hr := e_350_9
  simp only [e_350_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 68)) := by
  have c23022 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23023 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23024 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23025 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23026 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23027 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_10 := arithEq_of_rows h (row := 350) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_10
  simp only [← c23022, ← c23023, ← c23024] at e_350_10
  have e_350_11 := arithEq_of_rows h (row := 350) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_11
  simp only [← c23025, ← c23026, k2, ← c23027] at e_350_11
  have hr := e_350_11
  simp only [e_350_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 69)) := by
  have c23028 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23029 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23030 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23031 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23032 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23033 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_12 := arithEq_of_rows h (row := 350) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_12
  simp only [← c23028, ← c23029, ← c23030] at e_350_12
  have e_350_13 := arithEq_of_rows h (row := 350) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_13
  simp only [← c23031, ← c23032, k2, ← c23033] at e_350_13
  have hr := e_350_13
  simp only [e_350_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 70)) := by
  have c23034 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23035 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23036 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23037 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23038 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23039 := (publicBatchWrapper89_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_14 := arithEq_of_rows h (row := 350) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_14
  simp only [← c23034, ← c23035, ← c23036] at e_350_14
  have e_350_15 := arithEq_of_rows h (row := 350) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_15
  simp only [← c23037, ← c23038, k2, ← c23039] at e_350_15
  have hr := e_350_15
  simp only [e_350_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 71)) := by
  have c23040 := (publicBatchWrapper89_copies720 a h).1
  have c23041 := (publicBatchWrapper89_copies720 a h).2.1
  have c23042 := (publicBatchWrapper89_copies720 a h).2.2.1
  have c23043 := (publicBatchWrapper89_copies720 a h).2.2.2.1
  have c23044 := (publicBatchWrapper89_copies720 a h).2.2.2.2.1
  have c23045 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_16 := arithEq_of_rows h (row := 350) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_16
  simp only [← c23040, ← c23041, ← c23042] at e_350_16
  have e_350_17 := arithEq_of_rows h (row := 350) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_17
  simp only [← c23043, ← c23044, k2, ← c23045] at e_350_17
  have hr := e_350_17
  simp only [e_350_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 350 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 72)) := by
  have c23046 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.1
  have c23047 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.1
  have c23048 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.1
  have c23049 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.1
  have c23050 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23051 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_350_18 := arithEq_of_rows h (row := 350) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_18
  simp only [← c23046, ← c23047, ← c23048] at e_350_18
  have e_350_19 := arithEq_of_rows h (row := 350) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_19
  simp only [← c23049, ← c23050, k2, ← c23051] at e_350_19
  have hr := e_350_19
  simp only [e_350_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 73)) := by
  have c23052 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23053 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23054 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23055 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23056 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23057 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_0 := arithEq_of_rows h (row := 351) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_0
  simp only [← c23052, ← c23053, ← c23054] at e_351_0
  have e_351_1 := arithEq_of_rows h (row := 351) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_1
  simp only [← c23055, ← c23056, k2, ← c23057] at e_351_1
  have hr := e_351_1
  simp only [e_351_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 74)) := by
  have c23058 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23059 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23060 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23061 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23062 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23063 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_2 := arithEq_of_rows h (row := 351) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_2
  simp only [← c23058, ← c23059, ← c23060] at e_351_2
  have e_351_3 := arithEq_of_rows h (row := 351) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_3
  simp only [← c23061, ← c23062, k2, ← c23063] at e_351_3
  have hr := e_351_3
  simp only [e_351_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 75)) := by
  have c23064 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23065 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23066 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23067 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23068 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23069 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_4 := arithEq_of_rows h (row := 351) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_4
  simp only [← c23064, ← c23065, ← c23066] at e_351_4
  have e_351_5 := arithEq_of_rows h (row := 351) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_5
  simp only [← c23067, ← c23068, k2, ← c23069] at e_351_5
  have hr := e_351_5
  simp only [e_351_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 76)) := by
  have c23070 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23071 := (publicBatchWrapper89_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23072 := (publicBatchWrapper89_copies721 a h).1
  have c23073 := (publicBatchWrapper89_copies721 a h).2.1
  have c23074 := (publicBatchWrapper89_copies721 a h).2.2.1
  have c23075 := (publicBatchWrapper89_copies721 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_6 := arithEq_of_rows h (row := 351) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_6
  simp only [← c23070, ← c23071, ← c23072] at e_351_6
  have e_351_7 := arithEq_of_rows h (row := 351) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_7
  simp only [← c23073, ← c23074, k2, ← c23075] at e_351_7
  have hr := e_351_7
  simp only [e_351_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 77)) := by
  have c23076 := (publicBatchWrapper89_copies721 a h).2.2.2.2.1
  have c23077 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.1
  have c23078 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.1
  have c23079 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.1
  have c23080 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.1
  have c23081 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_8 := arithEq_of_rows h (row := 351) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_8
  simp only [← c23076, ← c23077, ← c23078] at e_351_8
  have e_351_9 := arithEq_of_rows h (row := 351) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_9
  simp only [← c23079, ← c23080, k2, ← c23081] at e_351_9
  have hr := e_351_9
  simp only [e_351_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 78)) := by
  have c23082 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23083 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23084 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23085 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23086 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23087 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_10 := arithEq_of_rows h (row := 351) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_10
  simp only [← c23082, ← c23083, ← c23084] at e_351_10
  have e_351_11 := arithEq_of_rows h (row := 351) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_11
  simp only [← c23085, ← c23086, k2, ← c23087] at e_351_11
  have hr := e_351_11
  simp only [e_351_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 79)) := by
  have c23088 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23089 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23090 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23091 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23092 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23093 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_12 := arithEq_of_rows h (row := 351) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_12
  simp only [← c23088, ← c23089, ← c23090] at e_351_12
  have e_351_13 := arithEq_of_rows h (row := 351) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_13
  simp only [← c23091, ← c23092, k2, ← c23093] at e_351_13
  have hr := e_351_13
  simp only [e_351_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 80)) := by
  have c23094 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23095 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23096 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23097 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23098 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23099 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_14 := arithEq_of_rows h (row := 351) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_14
  simp only [← c23094, ← c23095, ← c23096] at e_351_14
  have e_351_15 := arithEq_of_rows h (row := 351) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_15
  simp only [← c23097, ← c23098, k2, ← c23099] at e_351_15
  have hr := e_351_15
  simp only [e_351_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 81)) := by
  have c23100 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23101 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23102 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23103 := (publicBatchWrapper89_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23104 := (publicBatchWrapper89_copies722 a h).1
  have c23105 := (publicBatchWrapper89_copies722 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_16 := arithEq_of_rows h (row := 351) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_16
  simp only [← c23100, ← c23101, ← c23102] at e_351_16
  have e_351_17 := arithEq_of_rows h (row := 351) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_17
  simp only [← c23103, ← c23104, k2, ← c23105] at e_351_17
  have hr := e_351_17
  simp only [e_351_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 351 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 82)) := by
  have c23106 := (publicBatchWrapper89_copies722 a h).2.2.1
  have c23107 := (publicBatchWrapper89_copies722 a h).2.2.2.1
  have c23108 := (publicBatchWrapper89_copies722 a h).2.2.2.2.1
  have c23109 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.1
  have c23110 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.1
  have c23111 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_351_18 := arithEq_of_rows h (row := 351) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_18
  simp only [← c23106, ← c23107, ← c23108] at e_351_18
  have e_351_19 := arithEq_of_rows h (row := 351) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_19
  simp only [← c23109, ← c23110, k2, ← c23111] at e_351_19
  have hr := e_351_19
  simp only [e_351_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 83)) := by
  have c23112 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.1
  have c23113 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.1
  have c23114 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23115 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23116 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23117 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_0 := arithEq_of_rows h (row := 352) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_0
  simp only [← c23112, ← c23113, ← c23114] at e_352_0
  have e_352_1 := arithEq_of_rows h (row := 352) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_1
  simp only [← c23115, ← c23116, k2, ← c23117] at e_352_1
  have hr := e_352_1
  simp only [e_352_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 84)) := by
  have c23118 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23119 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23120 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23121 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23122 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23123 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_2 := arithEq_of_rows h (row := 352) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_2
  simp only [← c23118, ← c23119, ← c23120] at e_352_2
  have e_352_3 := arithEq_of_rows h (row := 352) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_3
  simp only [← c23121, ← c23122, k2, ← c23123] at e_352_3
  have hr := e_352_3
  simp only [e_352_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 85)) := by
  have c23124 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23125 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23126 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23127 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23128 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23129 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_4 := arithEq_of_rows h (row := 352) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_4
  simp only [← c23124, ← c23125, ← c23126] at e_352_4
  have e_352_5 := arithEq_of_rows h (row := 352) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_5
  simp only [← c23127, ← c23128, k2, ← c23129] at e_352_5
  have hr := e_352_5
  simp only [e_352_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 86)) := by
  have c23130 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23131 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23132 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23133 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23134 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23135 := (publicBatchWrapper89_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_6 := arithEq_of_rows h (row := 352) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_6
  simp only [← c23130, ← c23131, ← c23132] at e_352_6
  have e_352_7 := arithEq_of_rows h (row := 352) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_7
  simp only [← c23133, ← c23134, k2, ← c23135] at e_352_7
  have hr := e_352_7
  simp only [e_352_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 87)) := by
  have c23136 := (publicBatchWrapper89_copies723 a h).1
  have c23137 := (publicBatchWrapper89_copies723 a h).2.1
  have c23138 := (publicBatchWrapper89_copies723 a h).2.2.1
  have c23139 := (publicBatchWrapper89_copies723 a h).2.2.2.1
  have c23140 := (publicBatchWrapper89_copies723 a h).2.2.2.2.1
  have c23141 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_8 := arithEq_of_rows h (row := 352) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_8
  simp only [← c23136, ← c23137, ← c23138] at e_352_8
  have e_352_9 := arithEq_of_rows h (row := 352) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_9
  simp only [← c23139, ← c23140, k2, ← c23141] at e_352_9
  have hr := e_352_9
  simp only [e_352_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 88)) := by
  have c23142 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.1
  have c23143 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.1
  have c23144 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.1
  have c23145 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.1
  have c23146 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23147 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_10 := arithEq_of_rows h (row := 352) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_10
  simp only [← c23142, ← c23143, ← c23144] at e_352_10
  have e_352_11 := arithEq_of_rows h (row := 352) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_11
  simp only [← c23145, ← c23146, k2, ← c23147] at e_352_11
  have hr := e_352_11
  simp only [e_352_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 89)) := by
  have c23148 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23149 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23150 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23151 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23152 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23153 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_12 := arithEq_of_rows h (row := 352) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_12
  simp only [← c23148, ← c23149, ← c23150] at e_352_12
  have e_352_13 := arithEq_of_rows h (row := 352) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_13
  simp only [← c23151, ← c23152, k2, ← c23153] at e_352_13
  have hr := e_352_13
  simp only [e_352_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 90)) := by
  have c23154 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23155 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23156 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23157 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23158 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23159 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_14 := arithEq_of_rows h (row := 352) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_14
  simp only [← c23154, ← c23155, ← c23156] at e_352_14
  have e_352_15 := arithEq_of_rows h (row := 352) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_15
  simp only [← c23157, ← c23158, k2, ← c23159] at e_352_15
  have hr := e_352_15
  simp only [e_352_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 91)) := by
  have c23160 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23161 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23162 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23163 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23164 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23165 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_16 := arithEq_of_rows h (row := 352) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_16
  simp only [← c23160, ← c23161, ← c23162] at e_352_16
  have e_352_17 := arithEq_of_rows h (row := 352) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_17
  simp only [← c23163, ← c23164, k2, ← c23165] at e_352_17
  have hr := e_352_17
  simp only [e_352_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 352 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 92)) := by
  have c23166 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23167 := (publicBatchWrapper89_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23168 := (publicBatchWrapper89_copies724 a h).1
  have c23169 := (publicBatchWrapper89_copies724 a h).2.1
  have c23170 := (publicBatchWrapper89_copies724 a h).2.2.1
  have c23171 := (publicBatchWrapper89_copies724 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_352_18 := arithEq_of_rows h (row := 352) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_18
  simp only [← c23166, ← c23167, ← c23168] at e_352_18
  have e_352_19 := arithEq_of_rows h (row := 352) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_19
  simp only [← c23169, ← c23170, k2, ← c23171] at e_352_19
  have hr := e_352_19
  simp only [e_352_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 93)) := by
  have c23172 := (publicBatchWrapper89_copies724 a h).2.2.2.2.1
  have c23173 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.1
  have c23174 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.1
  have c23175 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.1
  have c23176 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.1
  have c23177 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_0 := arithEq_of_rows h (row := 353) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_0
  simp only [← c23172, ← c23173, ← c23174] at e_353_0
  have e_353_1 := arithEq_of_rows h (row := 353) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_1
  simp only [← c23175, ← c23176, k2, ← c23177] at e_353_1
  have hr := e_353_1
  simp only [e_353_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 94)) := by
  have c23178 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23179 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23180 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23181 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23182 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23183 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_2 := arithEq_of_rows h (row := 353) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_2
  simp only [← c23178, ← c23179, ← c23180] at e_353_2
  have e_353_3 := arithEq_of_rows h (row := 353) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_3
  simp only [← c23181, ← c23182, k2, ← c23183] at e_353_3
  have hr := e_353_3
  simp only [e_353_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 95)) := by
  have c23184 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23185 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23186 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23187 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23188 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23189 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_4 := arithEq_of_rows h (row := 353) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_4
  simp only [← c23184, ← c23185, ← c23186] at e_353_4
  have e_353_5 := arithEq_of_rows h (row := 353) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_5
  simp only [← c23187, ← c23188, k2, ← c23189] at e_353_5
  have hr := e_353_5
  simp only [e_353_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 96)) := by
  have c23190 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23191 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23192 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23193 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23194 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23195 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_6 := arithEq_of_rows h (row := 353) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_6
  simp only [← c23190, ← c23191, ← c23192] at e_353_6
  have e_353_7 := arithEq_of_rows h (row := 353) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_7
  simp only [← c23193, ← c23194, k2, ← c23195] at e_353_7
  have hr := e_353_7
  simp only [e_353_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 97)) := by
  have c23196 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23197 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23198 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23199 := (publicBatchWrapper89_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23200 := (publicBatchWrapper89_copies725 a h).1
  have c23201 := (publicBatchWrapper89_copies725 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_8 := arithEq_of_rows h (row := 353) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_8
  simp only [← c23196, ← c23197, ← c23198] at e_353_8
  have e_353_9 := arithEq_of_rows h (row := 353) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_9
  simp only [← c23199, ← c23200, k2, ← c23201] at e_353_9
  have hr := e_353_9
  simp only [e_353_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 98)) := by
  have c23202 := (publicBatchWrapper89_copies725 a h).2.2.1
  have c23203 := (publicBatchWrapper89_copies725 a h).2.2.2.1
  have c23204 := (publicBatchWrapper89_copies725 a h).2.2.2.2.1
  have c23205 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.1
  have c23206 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.1
  have c23207 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_10 := arithEq_of_rows h (row := 353) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_10
  simp only [← c23202, ← c23203, ← c23204] at e_353_10
  have e_353_11 := arithEq_of_rows h (row := 353) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_11
  simp only [← c23205, ← c23206, k2, ← c23207] at e_353_11
  have hr := e_353_11
  simp only [e_353_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 99)) := by
  have c23208 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.1
  have c23209 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.1
  have c23210 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23211 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23212 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23213 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_12 := arithEq_of_rows h (row := 353) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_12
  simp only [← c23208, ← c23209, ← c23210] at e_353_12
  have e_353_13 := arithEq_of_rows h (row := 353) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_13
  simp only [← c23211, ← c23212, k2, ← c23213] at e_353_13
  have hr := e_353_13
  simp only [e_353_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 100)) := by
  have c23214 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23215 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23216 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23217 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23218 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23219 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_14 := arithEq_of_rows h (row := 353) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_14
  simp only [← c23214, ← c23215, ← c23216] at e_353_14
  have e_353_15 := arithEq_of_rows h (row := 353) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_15
  simp only [← c23217, ← c23218, k2, ← c23219] at e_353_15
  have hr := e_353_15
  simp only [e_353_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 101)) := by
  have c23220 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23221 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23222 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23223 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23224 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23225 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_16 := arithEq_of_rows h (row := 353) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_16
  simp only [← c23220, ← c23221, ← c23222] at e_353_16
  have e_353_17 := arithEq_of_rows h (row := 353) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_17
  simp only [← c23223, ← c23224, k2, ← c23225] at e_353_17
  have hr := e_353_17
  simp only [e_353_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 353 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 102)) := by
  have c23226 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23227 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23228 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23229 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23230 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23231 := (publicBatchWrapper89_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_353_18 := arithEq_of_rows h (row := 353) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_18
  simp only [← c23226, ← c23227, ← c23228] at e_353_18
  have e_353_19 := arithEq_of_rows h (row := 353) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_19
  simp only [← c23229, ← c23230, k2, ← c23231] at e_353_19
  have hr := e_353_19
  simp only [e_353_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 103)) := by
  have c23232 := (publicBatchWrapper89_copies726 a h).1
  have c23233 := (publicBatchWrapper89_copies726 a h).2.1
  have c23234 := (publicBatchWrapper89_copies726 a h).2.2.1
  have c23235 := (publicBatchWrapper89_copies726 a h).2.2.2.1
  have c23236 := (publicBatchWrapper89_copies726 a h).2.2.2.2.1
  have c23237 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_0 := arithEq_of_rows h (row := 354) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_0
  simp only [← c23232, ← c23233, ← c23234] at e_354_0
  have e_354_1 := arithEq_of_rows h (row := 354) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_1
  simp only [← c23235, ← c23236, k2, ← c23237] at e_354_1
  have hr := e_354_1
  simp only [e_354_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 104)) := by
  have c23238 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.1
  have c23239 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.1
  have c23240 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.1
  have c23241 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.1
  have c23242 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23243 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_2 := arithEq_of_rows h (row := 354) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_2
  simp only [← c23238, ← c23239, ← c23240] at e_354_2
  have e_354_3 := arithEq_of_rows h (row := 354) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_3
  simp only [← c23241, ← c23242, k2, ← c23243] at e_354_3
  have hr := e_354_3
  simp only [e_354_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 105)) := by
  have c23244 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23245 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23246 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23247 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23248 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23249 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_4 := arithEq_of_rows h (row := 354) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_4
  simp only [← c23244, ← c23245, ← c23246] at e_354_4
  have e_354_5 := arithEq_of_rows h (row := 354) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_5
  simp only [← c23247, ← c23248, k2, ← c23249] at e_354_5
  have hr := e_354_5
  simp only [e_354_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 106)) := by
  have c23250 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23251 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23252 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23253 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23254 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23255 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_6 := arithEq_of_rows h (row := 354) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_6
  simp only [← c23250, ← c23251, ← c23252] at e_354_6
  have e_354_7 := arithEq_of_rows h (row := 354) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_7
  simp only [← c23253, ← c23254, k2, ← c23255] at e_354_7
  have hr := e_354_7
  simp only [e_354_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 107)) := by
  have c23256 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23257 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23258 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23259 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23260 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23261 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_8 := arithEq_of_rows h (row := 354) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_8
  simp only [← c23256, ← c23257, ← c23258] at e_354_8
  have e_354_9 := arithEq_of_rows h (row := 354) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_9
  simp only [← c23259, ← c23260, k2, ← c23261] at e_354_9
  have hr := e_354_9
  simp only [e_354_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 108)) := by
  have c23262 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23263 := (publicBatchWrapper89_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23264 := (publicBatchWrapper89_copies727 a h).1
  have c23265 := (publicBatchWrapper89_copies727 a h).2.1
  have c23266 := (publicBatchWrapper89_copies727 a h).2.2.1
  have c23267 := (publicBatchWrapper89_copies727 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_10 := arithEq_of_rows h (row := 354) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_10
  simp only [← c23262, ← c23263, ← c23264] at e_354_10
  have e_354_11 := arithEq_of_rows h (row := 354) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_11
  simp only [← c23265, ← c23266, k2, ← c23267] at e_354_11
  have hr := e_354_11
  simp only [e_354_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 109)) := by
  have c23268 := (publicBatchWrapper89_copies727 a h).2.2.2.2.1
  have c23269 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.1
  have c23270 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.1
  have c23271 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.1
  have c23272 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.1
  have c23273 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_12 := arithEq_of_rows h (row := 354) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_12
  simp only [← c23268, ← c23269, ← c23270] at e_354_12
  have e_354_13 := arithEq_of_rows h (row := 354) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_13
  simp only [← c23271, ← c23272, k2, ← c23273] at e_354_13
  have hr := e_354_13
  simp only [e_354_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 110)) := by
  have c23274 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23275 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23276 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23277 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23278 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23279 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_14 := arithEq_of_rows h (row := 354) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_14
  simp only [← c23274, ← c23275, ← c23276] at e_354_14
  have e_354_15 := arithEq_of_rows h (row := 354) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_15
  simp only [← c23277, ← c23278, k2, ← c23279] at e_354_15
  have hr := e_354_15
  simp only [e_354_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 111)) := by
  have c23280 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23281 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23282 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23283 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23284 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23285 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_16 := arithEq_of_rows h (row := 354) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_16
  simp only [← c23280, ← c23281, ← c23282] at e_354_16
  have e_354_17 := arithEq_of_rows h (row := 354) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_17
  simp only [← c23283, ← c23284, k2, ← c23285] at e_354_17
  have hr := e_354_17
  simp only [e_354_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 354 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 112)) := by
  have c23286 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23287 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23288 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23289 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23290 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23291 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_354_18 := arithEq_of_rows h (row := 354) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_18
  simp only [← c23286, ← c23287, ← c23288] at e_354_18
  have e_354_19 := arithEq_of_rows h (row := 354) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_19
  simp only [← c23289, ← c23290, k2, ← c23291] at e_354_19
  have hr := e_354_19
  simp only [e_354_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 113)) := by
  have c23292 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23293 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23294 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23295 := (publicBatchWrapper89_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23296 := (publicBatchWrapper89_copies728 a h).1
  have c23297 := (publicBatchWrapper89_copies728 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_0 := arithEq_of_rows h (row := 355) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_0
  simp only [← c23292, ← c23293, ← c23294] at e_355_0
  have e_355_1 := arithEq_of_rows h (row := 355) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_1
  simp only [← c23295, ← c23296, k2, ← c23297] at e_355_1
  have hr := e_355_1
  simp only [e_355_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 114)) := by
  have c23298 := (publicBatchWrapper89_copies728 a h).2.2.1
  have c23299 := (publicBatchWrapper89_copies728 a h).2.2.2.1
  have c23300 := (publicBatchWrapper89_copies728 a h).2.2.2.2.1
  have c23301 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.1
  have c23302 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.1
  have c23303 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_2 := arithEq_of_rows h (row := 355) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_2
  simp only [← c23298, ← c23299, ← c23300] at e_355_2
  have e_355_3 := arithEq_of_rows h (row := 355) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_3
  simp only [← c23301, ← c23302, k2, ← c23303] at e_355_3
  have hr := e_355_3
  simp only [e_355_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 115)) := by
  have c23304 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.1
  have c23305 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.1
  have c23306 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23307 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23308 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23309 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_4 := arithEq_of_rows h (row := 355) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_4
  simp only [← c23304, ← c23305, ← c23306] at e_355_4
  have e_355_5 := arithEq_of_rows h (row := 355) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_5
  simp only [← c23307, ← c23308, k2, ← c23309] at e_355_5
  have hr := e_355_5
  simp only [e_355_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 116)) := by
  have c23310 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23311 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23312 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23313 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23314 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23315 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_6 := arithEq_of_rows h (row := 355) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_6
  simp only [← c23310, ← c23311, ← c23312] at e_355_6
  have e_355_7 := arithEq_of_rows h (row := 355) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_7
  simp only [← c23313, ← c23314, k2, ← c23315] at e_355_7
  have hr := e_355_7
  simp only [e_355_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 117)) := by
  have c23316 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23317 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23318 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23319 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23320 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23321 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_8 := arithEq_of_rows h (row := 355) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_8
  simp only [← c23316, ← c23317, ← c23318] at e_355_8
  have e_355_9 := arithEq_of_rows h (row := 355) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_9
  simp only [← c23319, ← c23320, k2, ← c23321] at e_355_9
  have hr := e_355_9
  simp only [e_355_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 118)) := by
  have c23322 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23323 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23324 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23325 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23326 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23327 := (publicBatchWrapper89_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_10 := arithEq_of_rows h (row := 355) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_10
  simp only [← c23322, ← c23323, ← c23324] at e_355_10
  have e_355_11 := arithEq_of_rows h (row := 355) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_11
  simp only [← c23325, ← c23326, k2, ← c23327] at e_355_11
  have hr := e_355_11
  simp only [e_355_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 119)) := by
  have c23328 := (publicBatchWrapper89_copies729 a h).1
  have c23329 := (publicBatchWrapper89_copies729 a h).2.1
  have c23330 := (publicBatchWrapper89_copies729 a h).2.2.1
  have c23331 := (publicBatchWrapper89_copies729 a h).2.2.2.1
  have c23332 := (publicBatchWrapper89_copies729 a h).2.2.2.2.1
  have c23333 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_12 := arithEq_of_rows h (row := 355) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_12
  simp only [← c23328, ← c23329, ← c23330] at e_355_12
  have e_355_13 := arithEq_of_rows h (row := 355) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_13
  simp only [← c23331, ← c23332, k2, ← c23333] at e_355_13
  have hr := e_355_13
  simp only [e_355_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 120)) := by
  have c23334 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.1
  have c23335 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.1
  have c23336 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.1
  have c23337 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.1
  have c23338 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23339 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_14 := arithEq_of_rows h (row := 355) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_14
  simp only [← c23334, ← c23335, ← c23336] at e_355_14
  have e_355_15 := arithEq_of_rows h (row := 355) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_15
  simp only [← c23337, ← c23338, k2, ← c23339] at e_355_15
  have hr := e_355_15
  simp only [e_355_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 121)) := by
  have c23340 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23341 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23342 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23343 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23344 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23345 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_16 := arithEq_of_rows h (row := 355) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_16
  simp only [← c23340, ← c23341, ← c23342] at e_355_16
  have e_355_17 := arithEq_of_rows h (row := 355) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_17
  simp only [← c23343, ← c23344, k2, ← c23345] at e_355_17
  have hr := e_355_17
  simp only [e_355_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 355 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 122)) := by
  have c23346 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23347 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23348 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23349 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23350 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23351 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_355_18 := arithEq_of_rows h (row := 355) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_18
  simp only [← c23346, ← c23347, ← c23348] at e_355_18
  have e_355_19 := arithEq_of_rows h (row := 355) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_19
  simp only [← c23349, ← c23350, k2, ← c23351] at e_355_19
  have hr := e_355_19
  simp only [e_355_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 123)) := by
  have c23352 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23353 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23354 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23355 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23356 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23357 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_0 := arithEq_of_rows h (row := 356) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_0
  simp only [← c23352, ← c23353, ← c23354] at e_356_0
  have e_356_1 := arithEq_of_rows h (row := 356) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_1
  simp only [← c23355, ← c23356, k2, ← c23357] at e_356_1
  have hr := e_356_1
  simp only [e_356_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 124)) := by
  have c23358 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23359 := (publicBatchWrapper89_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23360 := (publicBatchWrapper89_copies730 a h).1
  have c23361 := (publicBatchWrapper89_copies730 a h).2.1
  have c23362 := (publicBatchWrapper89_copies730 a h).2.2.1
  have c23363 := (publicBatchWrapper89_copies730 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_2 := arithEq_of_rows h (row := 356) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_2
  simp only [← c23358, ← c23359, ← c23360] at e_356_2
  have e_356_3 := arithEq_of_rows h (row := 356) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_3
  simp only [← c23361, ← c23362, k2, ← c23363] at e_356_3
  have hr := e_356_3
  simp only [e_356_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 125)) := by
  have c23364 := (publicBatchWrapper89_copies730 a h).2.2.2.2.1
  have c23365 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.1
  have c23366 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.1
  have c23367 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.1
  have c23368 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.1
  have c23369 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_4 := arithEq_of_rows h (row := 356) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_4
  simp only [← c23364, ← c23365, ← c23366] at e_356_4
  have e_356_5 := arithEq_of_rows h (row := 356) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_5
  simp only [← c23367, ← c23368, k2, ← c23369] at e_356_5
  have hr := e_356_5
  simp only [e_356_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 126)) := by
  have c23370 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23371 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23372 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23373 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23374 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23375 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_6 := arithEq_of_rows h (row := 356) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_6
  simp only [← c23370, ← c23371, ← c23372] at e_356_6
  have e_356_7 := arithEq_of_rows h (row := 356) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_7
  simp only [← c23373, ← c23374, k2, ← c23375] at e_356_7
  have hr := e_356_7
  simp only [e_356_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 127)) := by
  have c23376 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23377 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23378 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23379 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23380 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23381 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_8 := arithEq_of_rows h (row := 356) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_8
  simp only [← c23376, ← c23377, ← c23378] at e_356_8
  have e_356_9 := arithEq_of_rows h (row := 356) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_9
  simp only [← c23379, ← c23380, k2, ← c23381] at e_356_9
  have hr := e_356_9
  simp only [e_356_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 128)) := by
  have c23382 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23383 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23384 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23385 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23386 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23387 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_10 := arithEq_of_rows h (row := 356) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_10
  simp only [← c23382, ← c23383, ← c23384] at e_356_10
  have e_356_11 := arithEq_of_rows h (row := 356) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_11
  simp only [← c23385, ← c23386, k2, ← c23387] at e_356_11
  have hr := e_356_11
  simp only [e_356_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 129)) := by
  have c23388 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23389 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23390 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23391 := (publicBatchWrapper89_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23392 := (publicBatchWrapper89_copies731 a h).1
  have c23393 := (publicBatchWrapper89_copies731 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_12 := arithEq_of_rows h (row := 356) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_12
  simp only [← c23388, ← c23389, ← c23390] at e_356_12
  have e_356_13 := arithEq_of_rows h (row := 356) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_13
  simp only [← c23391, ← c23392, k2, ← c23393] at e_356_13
  have hr := e_356_13
  simp only [e_356_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 130)) := by
  have c23394 := (publicBatchWrapper89_copies731 a h).2.2.1
  have c23395 := (publicBatchWrapper89_copies731 a h).2.2.2.1
  have c23396 := (publicBatchWrapper89_copies731 a h).2.2.2.2.1
  have c23397 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.1
  have c23398 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.1
  have c23399 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_14 := arithEq_of_rows h (row := 356) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_14
  simp only [← c23394, ← c23395, ← c23396] at e_356_14
  have e_356_15 := arithEq_of_rows h (row := 356) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_15
  simp only [← c23397, ← c23398, k2, ← c23399] at e_356_15
  have hr := e_356_15
  simp only [e_356_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 131)) := by
  have c23400 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.1
  have c23401 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.1
  have c23402 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23403 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23404 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23405 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_16 := arithEq_of_rows h (row := 356) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_16
  simp only [← c23400, ← c23401, ← c23402] at e_356_16
  have e_356_17 := arithEq_of_rows h (row := 356) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_17
  simp only [← c23403, ← c23404, k2, ← c23405] at e_356_17
  have hr := e_356_17
  simp only [e_356_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 356 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 132)) := by
  have c23406 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23407 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23408 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23409 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23410 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23411 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_356_18 := arithEq_of_rows h (row := 356) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_18
  simp only [← c23406, ← c23407, ← c23408] at e_356_18
  have e_356_19 := arithEq_of_rows h (row := 356) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_19
  simp only [← c23409, ← c23410, k2, ← c23411] at e_356_19
  have hr := e_356_19
  simp only [e_356_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 133)) := by
  have c23412 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23413 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23414 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23415 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23416 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23417 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_0 := arithEq_of_rows h (row := 357) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_0
  simp only [← c23412, ← c23413, ← c23414] at e_357_0
  have e_357_1 := arithEq_of_rows h (row := 357) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_1
  simp only [← c23415, ← c23416, k2, ← c23417] at e_357_1
  have hr := e_357_1
  simp only [e_357_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 134)) := by
  have c23418 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23419 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23420 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23421 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23422 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23423 := (publicBatchWrapper89_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_2 := arithEq_of_rows h (row := 357) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_2
  simp only [← c23418, ← c23419, ← c23420] at e_357_2
  have e_357_3 := arithEq_of_rows h (row := 357) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_3
  simp only [← c23421, ← c23422, k2, ← c23423] at e_357_3
  have hr := e_357_3
  simp only [e_357_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 135)) := by
  have c23424 := (publicBatchWrapper89_copies732 a h).1
  have c23425 := (publicBatchWrapper89_copies732 a h).2.1
  have c23426 := (publicBatchWrapper89_copies732 a h).2.2.1
  have c23427 := (publicBatchWrapper89_copies732 a h).2.2.2.1
  have c23428 := (publicBatchWrapper89_copies732 a h).2.2.2.2.1
  have c23429 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_4 := arithEq_of_rows h (row := 357) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_4
  simp only [← c23424, ← c23425, ← c23426] at e_357_4
  have e_357_5 := arithEq_of_rows h (row := 357) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_5
  simp only [← c23427, ← c23428, k2, ← c23429] at e_357_5
  have hr := e_357_5
  simp only [e_357_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 136)) := by
  have c23430 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.1
  have c23431 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.1
  have c23432 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.1
  have c23433 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.1
  have c23434 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23435 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_6 := arithEq_of_rows h (row := 357) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_6
  simp only [← c23430, ← c23431, ← c23432] at e_357_6
  have e_357_7 := arithEq_of_rows h (row := 357) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_7
  simp only [← c23433, ← c23434, k2, ← c23435] at e_357_7
  have hr := e_357_7
  simp only [e_357_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 137)) := by
  have c23436 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23437 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23438 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23439 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23440 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23441 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_8 := arithEq_of_rows h (row := 357) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_8
  simp only [← c23436, ← c23437, ← c23438] at e_357_8
  have e_357_9 := arithEq_of_rows h (row := 357) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_9
  simp only [← c23439, ← c23440, k2, ← c23441] at e_357_9
  have hr := e_357_9
  simp only [e_357_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 138)) := by
  have c23442 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23443 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23444 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23445 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23446 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23447 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_10 := arithEq_of_rows h (row := 357) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_10
  simp only [← c23442, ← c23443, ← c23444] at e_357_10
  have e_357_11 := arithEq_of_rows h (row := 357) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_11
  simp only [← c23445, ← c23446, k2, ← c23447] at e_357_11
  have hr := e_357_11
  simp only [e_357_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 139)) := by
  have c23448 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23449 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23450 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23451 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23452 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23453 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_12 := arithEq_of_rows h (row := 357) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_12
  simp only [← c23448, ← c23449, ← c23450] at e_357_12
  have e_357_13 := arithEq_of_rows h (row := 357) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_13
  simp only [← c23451, ← c23452, k2, ← c23453] at e_357_13
  have hr := e_357_13
  simp only [e_357_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 140)) := by
  have c23454 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23455 := (publicBatchWrapper89_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23456 := (publicBatchWrapper89_copies733 a h).1
  have c23457 := (publicBatchWrapper89_copies733 a h).2.1
  have c23458 := (publicBatchWrapper89_copies733 a h).2.2.1
  have c23459 := (publicBatchWrapper89_copies733 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_14 := arithEq_of_rows h (row := 357) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_14
  simp only [← c23454, ← c23455, ← c23456] at e_357_14
  have e_357_15 := arithEq_of_rows h (row := 357) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_15
  simp only [← c23457, ← c23458, k2, ← c23459] at e_357_15
  have hr := e_357_15
  simp only [e_357_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 141)) := by
  have c23460 := (publicBatchWrapper89_copies733 a h).2.2.2.2.1
  have c23461 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.1
  have c23462 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.1
  have c23463 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.1
  have c23464 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.1
  have c23465 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_16 := arithEq_of_rows h (row := 357) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_16
  simp only [← c23460, ← c23461, ← c23462] at e_357_16
  have e_357_17 := arithEq_of_rows h (row := 357) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_17
  simp only [← c23463, ← c23464, k2, ← c23465] at e_357_17
  have hr := e_357_17
  simp only [e_357_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 357 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 142)) := by
  have c23466 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23467 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23468 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23469 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23470 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23471 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_357_18 := arithEq_of_rows h (row := 357) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_18
  simp only [← c23466, ← c23467, ← c23468] at e_357_18
  have e_357_19 := arithEq_of_rows h (row := 357) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_19
  simp only [← c23469, ← c23470, k2, ← c23471] at e_357_19
  have hr := e_357_19
  simp only [e_357_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 143)) := by
  have c23472 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23473 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23474 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23475 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23476 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23477 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_0 := arithEq_of_rows h (row := 358) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_0
  simp only [← c23472, ← c23473, ← c23474] at e_358_0
  have e_358_1 := arithEq_of_rows h (row := 358) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_1
  simp only [← c23475, ← c23476, k2, ← c23477] at e_358_1
  have hr := e_358_1
  simp only [e_358_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 144)) := by
  have c23478 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23479 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23480 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23481 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23482 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23483 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_2 := arithEq_of_rows h (row := 358) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_2
  simp only [← c23478, ← c23479, ← c23480] at e_358_2
  have e_358_3 := arithEq_of_rows h (row := 358) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_3
  simp only [← c23481, ← c23482, k2, ← c23483] at e_358_3
  have hr := e_358_3
  simp only [e_358_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 145)) := by
  have c23484 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23485 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23486 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23487 := (publicBatchWrapper89_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23488 := (publicBatchWrapper89_copies734 a h).1
  have c23489 := (publicBatchWrapper89_copies734 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_4 := arithEq_of_rows h (row := 358) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_4
  simp only [← c23484, ← c23485, ← c23486] at e_358_4
  have e_358_5 := arithEq_of_rows h (row := 358) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_5
  simp only [← c23487, ← c23488, k2, ← c23489] at e_358_5
  have hr := e_358_5
  simp only [e_358_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 146)) := by
  have c23490 := (publicBatchWrapper89_copies734 a h).2.2.1
  have c23491 := (publicBatchWrapper89_copies734 a h).2.2.2.1
  have c23492 := (publicBatchWrapper89_copies734 a h).2.2.2.2.1
  have c23493 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.1
  have c23494 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.1
  have c23495 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_6 := arithEq_of_rows h (row := 358) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_6
  simp only [← c23490, ← c23491, ← c23492] at e_358_6
  have e_358_7 := arithEq_of_rows h (row := 358) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_7
  simp only [← c23493, ← c23494, k2, ← c23495] at e_358_7
  have hr := e_358_7
  simp only [e_358_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 147)) := by
  have c23496 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.1
  have c23497 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.1
  have c23498 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23499 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23500 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23501 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_8 := arithEq_of_rows h (row := 358) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_8
  simp only [← c23496, ← c23497, ← c23498] at e_358_8
  have e_358_9 := arithEq_of_rows h (row := 358) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_9
  simp only [← c23499, ← c23500, k2, ← c23501] at e_358_9
  have hr := e_358_9
  simp only [e_358_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 148)) := by
  have c23502 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23503 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23504 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23505 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23506 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23507 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_10 := arithEq_of_rows h (row := 358) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_10
  simp only [← c23502, ← c23503, ← c23504] at e_358_10
  have e_358_11 := arithEq_of_rows h (row := 358) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_11
  simp only [← c23505, ← c23506, k2, ← c23507] at e_358_11
  have hr := e_358_11
  simp only [e_358_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 149)) := by
  have c23508 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23509 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23510 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23511 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23512 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23513 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_12 := arithEq_of_rows h (row := 358) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_12
  simp only [← c23508, ← c23509, ← c23510] at e_358_12
  have e_358_13 := arithEq_of_rows h (row := 358) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_13
  simp only [← c23511, ← c23512, k2, ← c23513] at e_358_13
  have hr := e_358_13
  simp only [e_358_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 150)) := by
  have c23514 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23515 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23516 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23517 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23518 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23519 := (publicBatchWrapper89_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_14 := arithEq_of_rows h (row := 358) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_14
  simp only [← c23514, ← c23515, ← c23516] at e_358_14
  have e_358_15 := arithEq_of_rows h (row := 358) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_15
  simp only [← c23517, ← c23518, k2, ← c23519] at e_358_15
  have hr := e_358_15
  simp only [e_358_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 151)) := by
  have c23520 := (publicBatchWrapper89_copies735 a h).1
  have c23521 := (publicBatchWrapper89_copies735 a h).2.1
  have c23522 := (publicBatchWrapper89_copies735 a h).2.2.1
  have c23523 := (publicBatchWrapper89_copies735 a h).2.2.2.1
  have c23524 := (publicBatchWrapper89_copies735 a h).2.2.2.2.1
  have c23525 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_16 := arithEq_of_rows h (row := 358) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_16
  simp only [← c23520, ← c23521, ← c23522] at e_358_16
  have e_358_17 := arithEq_of_rows h (row := 358) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_17
  simp only [← c23523, ← c23524, k2, ← c23525] at e_358_17
  have hr := e_358_17
  simp only [e_358_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 358 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 152)) := by
  have c23526 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.1
  have c23527 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.1
  have c23528 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.1
  have c23529 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.1
  have c23530 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23531 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_358_18 := arithEq_of_rows h (row := 358) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_18
  simp only [← c23526, ← c23527, ← c23528] at e_358_18
  have e_358_19 := arithEq_of_rows h (row := 358) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_19
  simp only [← c23529, ← c23530, k2, ← c23531] at e_358_19
  have hr := e_358_19
  simp only [e_358_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 153)) := by
  have c23532 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23533 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23534 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23535 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23536 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23537 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_0 := arithEq_of_rows h (row := 359) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_0
  simp only [← c23532, ← c23533, ← c23534] at e_359_0
  have e_359_1 := arithEq_of_rows h (row := 359) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_1
  simp only [← c23535, ← c23536, k2, ← c23537] at e_359_1
  have hr := e_359_1
  simp only [e_359_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 154)) := by
  have c23538 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23539 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23540 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23541 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23542 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23543 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_2 := arithEq_of_rows h (row := 359) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_2
  simp only [← c23538, ← c23539, ← c23540] at e_359_2
  have e_359_3 := arithEq_of_rows h (row := 359) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_3
  simp only [← c23541, ← c23542, k2, ← c23543] at e_359_3
  have hr := e_359_3
  simp only [e_359_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 155)) := by
  have c23544 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23545 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23546 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23547 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23548 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23549 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_4 := arithEq_of_rows h (row := 359) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_4
  simp only [← c23544, ← c23545, ← c23546] at e_359_4
  have e_359_5 := arithEq_of_rows h (row := 359) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_5
  simp only [← c23547, ← c23548, k2, ← c23549] at e_359_5
  have hr := e_359_5
  simp only [e_359_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 156)) := by
  have c23550 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23551 := (publicBatchWrapper89_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23552 := (publicBatchWrapper89_copies736 a h).1
  have c23553 := (publicBatchWrapper89_copies736 a h).2.1
  have c23554 := (publicBatchWrapper89_copies736 a h).2.2.1
  have c23555 := (publicBatchWrapper89_copies736 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_6 := arithEq_of_rows h (row := 359) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_6
  simp only [← c23550, ← c23551, ← c23552] at e_359_6
  have e_359_7 := arithEq_of_rows h (row := 359) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_7
  simp only [← c23553, ← c23554, k2, ← c23555] at e_359_7
  have hr := e_359_7
  simp only [e_359_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 157)) := by
  have c23556 := (publicBatchWrapper89_copies736 a h).2.2.2.2.1
  have c23557 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.1
  have c23558 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.1
  have c23559 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.1
  have c23560 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.1
  have c23561 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_8 := arithEq_of_rows h (row := 359) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_8
  simp only [← c23556, ← c23557, ← c23558] at e_359_8
  have e_359_9 := arithEq_of_rows h (row := 359) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_9
  simp only [← c23559, ← c23560, k2, ← c23561] at e_359_9
  have hr := e_359_9
  simp only [e_359_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 158)) := by
  have c23562 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23563 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23564 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23565 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23566 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23567 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_10 := arithEq_of_rows h (row := 359) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_10
  simp only [← c23562, ← c23563, ← c23564] at e_359_10
  have e_359_11 := arithEq_of_rows h (row := 359) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_11
  simp only [← c23565, ← c23566, k2, ← c23567] at e_359_11
  have hr := e_359_11
  simp only [e_359_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 159)) := by
  have c23568 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23569 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23570 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23571 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23572 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23573 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_12 := arithEq_of_rows h (row := 359) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_12
  simp only [← c23568, ← c23569, ← c23570] at e_359_12
  have e_359_13 := arithEq_of_rows h (row := 359) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_13
  simp only [← c23571, ← c23572, k2, ← c23573] at e_359_13
  have hr := e_359_13
  simp only [e_359_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 160)) := by
  have c23574 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23575 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23576 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23577 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23578 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23579 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_14 := arithEq_of_rows h (row := 359) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_14
  simp only [← c23574, ← c23575, ← c23576] at e_359_14
  have e_359_15 := arithEq_of_rows h (row := 359) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_15
  simp only [← c23577, ← c23578, k2, ← c23579] at e_359_15
  have hr := e_359_15
  simp only [e_359_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 161)) := by
  have c23580 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23581 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23582 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23583 := (publicBatchWrapper89_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23584 := (publicBatchWrapper89_copies737 a h).1
  have c23585 := (publicBatchWrapper89_copies737 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_16 := arithEq_of_rows h (row := 359) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_16
  simp only [← c23580, ← c23581, ← c23582] at e_359_16
  have e_359_17 := arithEq_of_rows h (row := 359) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_17
  simp only [← c23583, ← c23584, k2, ← c23585] at e_359_17
  have hr := e_359_17
  simp only [e_359_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 359 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 162)) := by
  have c23586 := (publicBatchWrapper89_copies737 a h).2.2.1
  have c23587 := (publicBatchWrapper89_copies737 a h).2.2.2.1
  have c23588 := (publicBatchWrapper89_copies737 a h).2.2.2.2.1
  have c23589 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.1
  have c23590 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.1
  have c23591 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_359_18 := arithEq_of_rows h (row := 359) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_18
  simp only [← c23586, ← c23587, ← c23588] at e_359_18
  have e_359_19 := arithEq_of_rows h (row := 359) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_19
  simp only [← c23589, ← c23590, k2, ← c23591] at e_359_19
  have hr := e_359_19
  simp only [e_359_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 163)) := by
  have c23592 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.1
  have c23593 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.1
  have c23594 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23595 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23596 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23597 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_0 := arithEq_of_rows h (row := 360) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_0
  simp only [← c23592, ← c23593, ← c23594] at e_360_0
  have e_360_1 := arithEq_of_rows h (row := 360) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_1
  simp only [← c23595, ← c23596, k2, ← c23597] at e_360_1
  have hr := e_360_1
  simp only [e_360_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 164)) := by
  have c23598 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23599 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23600 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23601 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23602 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23603 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_2 := arithEq_of_rows h (row := 360) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_2
  simp only [← c23598, ← c23599, ← c23600] at e_360_2
  have e_360_3 := arithEq_of_rows h (row := 360) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_3
  simp only [← c23601, ← c23602, k2, ← c23603] at e_360_3
  have hr := e_360_3
  simp only [e_360_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 165)) := by
  have c23604 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23605 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23606 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23607 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23608 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23609 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_4 := arithEq_of_rows h (row := 360) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_4
  simp only [← c23604, ← c23605, ← c23606] at e_360_4
  have e_360_5 := arithEq_of_rows h (row := 360) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_5
  simp only [← c23607, ← c23608, k2, ← c23609] at e_360_5
  have hr := e_360_5
  simp only [e_360_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 166)) := by
  have c23610 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23611 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23612 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23613 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23614 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23615 := (publicBatchWrapper89_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_6 := arithEq_of_rows h (row := 360) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_6
  simp only [← c23610, ← c23611, ← c23612] at e_360_6
  have e_360_7 := arithEq_of_rows h (row := 360) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_7
  simp only [← c23613, ← c23614, k2, ← c23615] at e_360_7
  have hr := e_360_7
  simp only [e_360_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 167)) := by
  have c23616 := (publicBatchWrapper89_copies738 a h).1
  have c23617 := (publicBatchWrapper89_copies738 a h).2.1
  have c23618 := (publicBatchWrapper89_copies738 a h).2.2.1
  have c23619 := (publicBatchWrapper89_copies738 a h).2.2.2.1
  have c23620 := (publicBatchWrapper89_copies738 a h).2.2.2.2.1
  have c23621 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_8 := arithEq_of_rows h (row := 360) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_8
  simp only [← c23616, ← c23617, ← c23618] at e_360_8
  have e_360_9 := arithEq_of_rows h (row := 360) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_9
  simp only [← c23619, ← c23620, k2, ← c23621] at e_360_9
  have hr := e_360_9
  simp only [e_360_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 210)) := by
  have c23622 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.1
  have c23623 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.1
  have c23624 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.1
  have c23625 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.1
  have c23626 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23627 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_10 := arithEq_of_rows h (row := 360) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_10
  simp only [← c23622, ← c23623, ← c23624] at e_360_10
  have e_360_11 := arithEq_of_rows h (row := 360) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_11
  simp only [← c23625, ← c23626, k2, ← c23627] at e_360_11
  have hr := e_360_11
  simp only [e_360_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 211)) := by
  have c23628 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23629 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23630 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23631 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23632 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23633 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_12 := arithEq_of_rows h (row := 360) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_12
  simp only [← c23628, ← c23629, ← c23630] at e_360_12
  have e_360_13 := arithEq_of_rows h (row := 360) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_13
  simp only [← c23631, ← c23632, k2, ← c23633] at e_360_13
  have hr := e_360_13
  simp only [e_360_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 212)) := by
  have c23634 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23635 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23636 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23637 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23638 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23639 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_14 := arithEq_of_rows h (row := 360) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_14
  simp only [← c23634, ← c23635, ← c23636] at e_360_14
  have e_360_15 := arithEq_of_rows h (row := 360) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_15
  simp only [← c23637, ← c23638, k2, ← c23639] at e_360_15
  have hr := e_360_15
  simp only [e_360_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 213)) := by
  have c23640 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23641 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23642 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23643 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23644 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23645 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_16 := arithEq_of_rows h (row := 360) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_16
  simp only [← c23640, ← c23641, ← c23642] at e_360_16
  have e_360_17 := arithEq_of_rows h (row := 360) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_17
  simp only [← c23643, ← c23644, k2, ← c23645] at e_360_17
  have hr := e_360_17
  simp only [e_360_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 360 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 214)) := by
  have c23646 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23647 := (publicBatchWrapper89_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23648 := (publicBatchWrapper89_copies739 a h).1
  have c23649 := (publicBatchWrapper89_copies739 a h).2.1
  have c23650 := (publicBatchWrapper89_copies739 a h).2.2.1
  have c23651 := (publicBatchWrapper89_copies739 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_360_18 := arithEq_of_rows h (row := 360) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_18
  simp only [← c23646, ← c23647, ← c23648] at e_360_18
  have e_360_19 := arithEq_of_rows h (row := 360) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_19
  simp only [← c23649, ← c23650, k2, ← c23651] at e_360_19
  have hr := e_360_19
  simp only [e_360_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 215)) := by
  have c23652 := (publicBatchWrapper89_copies739 a h).2.2.2.2.1
  have c23653 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.1
  have c23654 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.1
  have c23655 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.1
  have c23656 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.1
  have c23657 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_0 := arithEq_of_rows h (row := 361) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_0
  simp only [← c23652, ← c23653, ← c23654] at e_361_0
  have e_361_1 := arithEq_of_rows h (row := 361) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_1
  simp only [← c23655, ← c23656, k2, ← c23657] at e_361_1
  have hr := e_361_1
  simp only [e_361_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 216)) := by
  have c23658 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23659 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23660 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23661 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23662 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23663 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_2 := arithEq_of_rows h (row := 361) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_2
  simp only [← c23658, ← c23659, ← c23660] at e_361_2
  have e_361_3 := arithEq_of_rows h (row := 361) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_3
  simp only [← c23661, ← c23662, k2, ← c23663] at e_361_3
  have hr := e_361_3
  simp only [e_361_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 217)) := by
  have c23664 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23665 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23666 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23667 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23668 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23669 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_4 := arithEq_of_rows h (row := 361) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_4
  simp only [← c23664, ← c23665, ← c23666] at e_361_4
  have e_361_5 := arithEq_of_rows h (row := 361) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_5
  simp only [← c23667, ← c23668, k2, ← c23669] at e_361_5
  have hr := e_361_5
  simp only [e_361_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 218)) := by
  have c23670 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23671 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23672 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23673 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23674 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23675 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_6 := arithEq_of_rows h (row := 361) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_6
  simp only [← c23670, ← c23671, ← c23672] at e_361_6
  have e_361_7 := arithEq_of_rows h (row := 361) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_7
  simp only [← c23673, ← c23674, k2, ← c23675] at e_361_7
  have hr := e_361_7
  simp only [e_361_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 219)) := by
  have c23676 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23677 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23678 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23679 := (publicBatchWrapper89_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23680 := (publicBatchWrapper89_copies740 a h).1
  have c23681 := (publicBatchWrapper89_copies740 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_8 := arithEq_of_rows h (row := 361) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_8
  simp only [← c23676, ← c23677, ← c23678] at e_361_8
  have e_361_9 := arithEq_of_rows h (row := 361) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_9
  simp only [← c23679, ← c23680, k2, ← c23681] at e_361_9
  have hr := e_361_9
  simp only [e_361_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 220)) := by
  have c23682 := (publicBatchWrapper89_copies740 a h).2.2.1
  have c23683 := (publicBatchWrapper89_copies740 a h).2.2.2.1
  have c23684 := (publicBatchWrapper89_copies740 a h).2.2.2.2.1
  have c23685 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.1
  have c23686 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.1
  have c23687 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_10 := arithEq_of_rows h (row := 361) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_10
  simp only [← c23682, ← c23683, ← c23684] at e_361_10
  have e_361_11 := arithEq_of_rows h (row := 361) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_11
  simp only [← c23685, ← c23686, k2, ← c23687] at e_361_11
  have hr := e_361_11
  simp only [e_361_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 221)) := by
  have c23688 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.1
  have c23689 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.1
  have c23690 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23691 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23692 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23693 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_12 := arithEq_of_rows h (row := 361) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_12
  simp only [← c23688, ← c23689, ← c23690] at e_361_12
  have e_361_13 := arithEq_of_rows h (row := 361) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_13
  simp only [← c23691, ← c23692, k2, ← c23693] at e_361_13
  have hr := e_361_13
  simp only [e_361_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 222)) := by
  have c23694 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23695 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23696 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23697 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23698 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23699 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_14 := arithEq_of_rows h (row := 361) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_14
  simp only [← c23694, ← c23695, ← c23696] at e_361_14
  have e_361_15 := arithEq_of_rows h (row := 361) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_15
  simp only [← c23697, ← c23698, k2, ← c23699] at e_361_15
  have hr := e_361_15
  simp only [e_361_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 223)) := by
  have c23700 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23701 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23702 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23703 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23704 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23705 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_16 := arithEq_of_rows h (row := 361) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_16
  simp only [← c23700, ← c23701, ← c23702] at e_361_16
  have e_361_17 := arithEq_of_rows h (row := 361) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_17
  simp only [← c23703, ← c23704, k2, ← c23705] at e_361_17
  have hr := e_361_17
  simp only [e_361_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 361 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 224)) := by
  have c23706 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23707 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23708 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23709 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23710 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23711 := (publicBatchWrapper89_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_361_18 := arithEq_of_rows h (row := 361) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_18
  simp only [← c23706, ← c23707, ← c23708] at e_361_18
  have e_361_19 := arithEq_of_rows h (row := 361) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_19
  simp only [← c23709, ← c23710, k2, ← c23711] at e_361_19
  have hr := e_361_19
  simp only [e_361_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 225)) := by
  have c23712 := (publicBatchWrapper89_copies741 a h).1
  have c23713 := (publicBatchWrapper89_copies741 a h).2.1
  have c23714 := (publicBatchWrapper89_copies741 a h).2.2.1
  have c23715 := (publicBatchWrapper89_copies741 a h).2.2.2.1
  have c23716 := (publicBatchWrapper89_copies741 a h).2.2.2.2.1
  have c23717 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_0 := arithEq_of_rows h (row := 362) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_0
  simp only [← c23712, ← c23713, ← c23714] at e_362_0
  have e_362_1 := arithEq_of_rows h (row := 362) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_1
  simp only [← c23715, ← c23716, k2, ← c23717] at e_362_1
  have hr := e_362_1
  simp only [e_362_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 226)) := by
  have c23718 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.1
  have c23719 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.1
  have c23720 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.1
  have c23721 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.1
  have c23722 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23723 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_2 := arithEq_of_rows h (row := 362) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_2
  simp only [← c23718, ← c23719, ← c23720] at e_362_2
  have e_362_3 := arithEq_of_rows h (row := 362) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_3
  simp only [← c23721, ← c23722, k2, ← c23723] at e_362_3
  have hr := e_362_3
  simp only [e_362_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 227)) := by
  have c23724 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23725 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23726 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23727 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23728 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23729 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_4 := arithEq_of_rows h (row := 362) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_4
  simp only [← c23724, ← c23725, ← c23726] at e_362_4
  have e_362_5 := arithEq_of_rows h (row := 362) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_5
  simp only [← c23727, ← c23728, k2, ← c23729] at e_362_5
  have hr := e_362_5
  simp only [e_362_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 228)) := by
  have c23730 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23731 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23732 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23733 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23734 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23735 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_6 := arithEq_of_rows h (row := 362) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_6
  simp only [← c23730, ← c23731, ← c23732] at e_362_6
  have e_362_7 := arithEq_of_rows h (row := 362) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_7
  simp only [← c23733, ← c23734, k2, ← c23735] at e_362_7
  have hr := e_362_7
  simp only [e_362_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 229)) := by
  have c23736 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23737 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23738 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23739 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23740 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23741 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_8 := arithEq_of_rows h (row := 362) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_8
  simp only [← c23736, ← c23737, ← c23738] at e_362_8
  have e_362_9 := arithEq_of_rows h (row := 362) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_9
  simp only [← c23739, ← c23740, k2, ← c23741] at e_362_9
  have hr := e_362_9
  simp only [e_362_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 230)) := by
  have c23742 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23743 := (publicBatchWrapper89_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23744 := (publicBatchWrapper89_copies742 a h).1
  have c23745 := (publicBatchWrapper89_copies742 a h).2.1
  have c23746 := (publicBatchWrapper89_copies742 a h).2.2.1
  have c23747 := (publicBatchWrapper89_copies742 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_10 := arithEq_of_rows h (row := 362) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_10
  simp only [← c23742, ← c23743, ← c23744] at e_362_10
  have e_362_11 := arithEq_of_rows h (row := 362) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_11
  simp only [← c23745, ← c23746, k2, ← c23747] at e_362_11
  have hr := e_362_11
  simp only [e_362_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 231)) := by
  have c23748 := (publicBatchWrapper89_copies742 a h).2.2.2.2.1
  have c23749 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.1
  have c23750 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.1
  have c23751 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.1
  have c23752 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.1
  have c23753 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_12 := arithEq_of_rows h (row := 362) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_12
  simp only [← c23748, ← c23749, ← c23750] at e_362_12
  have e_362_13 := arithEq_of_rows h (row := 362) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_13
  simp only [← c23751, ← c23752, k2, ← c23753] at e_362_13
  have hr := e_362_13
  simp only [e_362_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 232)) := by
  have c23754 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23755 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23756 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23757 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23758 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23759 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_14 := arithEq_of_rows h (row := 362) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_14
  simp only [← c23754, ← c23755, ← c23756] at e_362_14
  have e_362_15 := arithEq_of_rows h (row := 362) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_15
  simp only [← c23757, ← c23758, k2, ← c23759] at e_362_15
  have hr := e_362_15
  simp only [e_362_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 233)) := by
  have c23760 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23761 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23762 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23763 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23764 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23765 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_16 := arithEq_of_rows h (row := 362) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_16
  simp only [← c23760, ← c23761, ← c23762] at e_362_16
  have e_362_17 := arithEq_of_rows h (row := 362) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_17
  simp only [← c23763, ← c23764, k2, ← c23765] at e_362_17
  have hr := e_362_17
  simp only [e_362_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 362 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 234)) := by
  have c23766 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23767 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23768 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23769 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23770 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23771 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_362_18 := arithEq_of_rows h (row := 362) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_18
  simp only [← c23766, ← c23767, ← c23768] at e_362_18
  have e_362_19 := arithEq_of_rows h (row := 362) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_19
  simp only [← c23769, ← c23770, k2, ← c23771] at e_362_19
  have hr := e_362_19
  simp only [e_362_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 235)) := by
  have c23772 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23773 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23774 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23775 := (publicBatchWrapper89_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23776 := (publicBatchWrapper89_copies743 a h).1
  have c23777 := (publicBatchWrapper89_copies743 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_0 := arithEq_of_rows h (row := 363) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_0
  simp only [← c23772, ← c23773, ← c23774] at e_363_0
  have e_363_1 := arithEq_of_rows h (row := 363) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_1
  simp only [← c23775, ← c23776, k2, ← c23777] at e_363_1
  have hr := e_363_1
  simp only [e_363_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 236)) := by
  have c23778 := (publicBatchWrapper89_copies743 a h).2.2.1
  have c23779 := (publicBatchWrapper89_copies743 a h).2.2.2.1
  have c23780 := (publicBatchWrapper89_copies743 a h).2.2.2.2.1
  have c23781 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.1
  have c23782 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.1
  have c23783 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_2 := arithEq_of_rows h (row := 363) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_2
  simp only [← c23778, ← c23779, ← c23780] at e_363_2
  have e_363_3 := arithEq_of_rows h (row := 363) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_3
  simp only [← c23781, ← c23782, k2, ← c23783] at e_363_3
  have hr := e_363_3
  simp only [e_363_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 237)) := by
  have c23784 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.1
  have c23785 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.1
  have c23786 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23787 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23788 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23789 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_4 := arithEq_of_rows h (row := 363) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_4
  simp only [← c23784, ← c23785, ← c23786] at e_363_4
  have e_363_5 := arithEq_of_rows h (row := 363) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_5
  simp only [← c23787, ← c23788, k2, ← c23789] at e_363_5
  have hr := e_363_5
  simp only [e_363_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 238)) := by
  have c23790 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23791 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23792 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23793 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23794 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23795 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_6 := arithEq_of_rows h (row := 363) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_6
  simp only [← c23790, ← c23791, ← c23792] at e_363_6
  have e_363_7 := arithEq_of_rows h (row := 363) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_7
  simp only [← c23793, ← c23794, k2, ← c23795] at e_363_7
  have hr := e_363_7
  simp only [e_363_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 239)) := by
  have c23796 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23797 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23798 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23799 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23800 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23801 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_8 := arithEq_of_rows h (row := 363) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_8
  simp only [← c23796, ← c23797, ← c23798] at e_363_8
  have e_363_9 := arithEq_of_rows h (row := 363) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_9
  simp only [← c23799, ← c23800, k2, ← c23801] at e_363_9
  have hr := e_363_9
  simp only [e_363_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 240)) := by
  have c23802 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23803 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23804 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23805 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23806 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23807 := (publicBatchWrapper89_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_10 := arithEq_of_rows h (row := 363) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_10
  simp only [← c23802, ← c23803, ← c23804] at e_363_10
  have e_363_11 := arithEq_of_rows h (row := 363) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_11
  simp only [← c23805, ← c23806, k2, ← c23807] at e_363_11
  have hr := e_363_11
  simp only [e_363_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 241)) := by
  have c23808 := (publicBatchWrapper89_copies744 a h).1
  have c23809 := (publicBatchWrapper89_copies744 a h).2.1
  have c23810 := (publicBatchWrapper89_copies744 a h).2.2.1
  have c23811 := (publicBatchWrapper89_copies744 a h).2.2.2.1
  have c23812 := (publicBatchWrapper89_copies744 a h).2.2.2.2.1
  have c23813 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_12 := arithEq_of_rows h (row := 363) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_12
  simp only [← c23808, ← c23809, ← c23810] at e_363_12
  have e_363_13 := arithEq_of_rows h (row := 363) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_13
  simp only [← c23811, ← c23812, k2, ← c23813] at e_363_13
  have hr := e_363_13
  simp only [e_363_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 242)) := by
  have c23814 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.1
  have c23815 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.1
  have c23816 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.1
  have c23817 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.1
  have c23818 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23819 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_14 := arithEq_of_rows h (row := 363) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_14
  simp only [← c23814, ← c23815, ← c23816] at e_363_14
  have e_363_15 := arithEq_of_rows h (row := 363) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_15
  simp only [← c23817, ← c23818, k2, ← c23819] at e_363_15
  have hr := e_363_15
  simp only [e_363_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 243)) := by
  have c23820 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23821 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23822 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23823 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23824 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23825 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_16 := arithEq_of_rows h (row := 363) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_16
  simp only [← c23820, ← c23821, ← c23822] at e_363_16
  have e_363_17 := arithEq_of_rows h (row := 363) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_17
  simp only [← c23823, ← c23824, k2, ← c23825] at e_363_17
  have hr := e_363_17
  simp only [e_363_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 363 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 244)) := by
  have c23826 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23827 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23828 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23829 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23830 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23831 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_363_18 := arithEq_of_rows h (row := 363) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_18
  simp only [← c23826, ← c23827, ← c23828] at e_363_18
  have e_363_19 := arithEq_of_rows h (row := 363) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_19
  simp only [← c23829, ← c23830, k2, ← c23831] at e_363_19
  have hr := e_363_19
  simp only [e_363_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 245)) := by
  have c23832 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23833 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23834 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23835 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23836 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23837 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_0 := arithEq_of_rows h (row := 364) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_0
  simp only [← c23832, ← c23833, ← c23834] at e_364_0
  have e_364_1 := arithEq_of_rows h (row := 364) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_1
  simp only [← c23835, ← c23836, k2, ← c23837] at e_364_1
  have hr := e_364_1
  simp only [e_364_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 246)) := by
  have c23838 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23839 := (publicBatchWrapper89_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23840 := (publicBatchWrapper89_copies745 a h).1
  have c23841 := (publicBatchWrapper89_copies745 a h).2.1
  have c23842 := (publicBatchWrapper89_copies745 a h).2.2.1
  have c23843 := (publicBatchWrapper89_copies745 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_2 := arithEq_of_rows h (row := 364) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_2
  simp only [← c23838, ← c23839, ← c23840] at e_364_2
  have e_364_3 := arithEq_of_rows h (row := 364) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_3
  simp only [← c23841, ← c23842, k2, ← c23843] at e_364_3
  have hr := e_364_3
  simp only [e_364_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 247)) := by
  have c23844 := (publicBatchWrapper89_copies745 a h).2.2.2.2.1
  have c23845 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.1
  have c23846 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.1
  have c23847 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.1
  have c23848 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.1
  have c23849 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_4 := arithEq_of_rows h (row := 364) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_4
  simp only [← c23844, ← c23845, ← c23846] at e_364_4
  have e_364_5 := arithEq_of_rows h (row := 364) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_5
  simp only [← c23847, ← c23848, k2, ← c23849] at e_364_5
  have hr := e_364_5
  simp only [e_364_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 248)) := by
  have c23850 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23851 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23852 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23853 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23854 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23855 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_6 := arithEq_of_rows h (row := 364) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_6
  simp only [← c23850, ← c23851, ← c23852] at e_364_6
  have e_364_7 := arithEq_of_rows h (row := 364) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_7
  simp only [← c23853, ← c23854, k2, ← c23855] at e_364_7
  have hr := e_364_7
  simp only [e_364_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 249)) := by
  have c23856 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23857 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23858 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23859 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23860 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23861 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_8 := arithEq_of_rows h (row := 364) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_8
  simp only [← c23856, ← c23857, ← c23858] at e_364_8
  have e_364_9 := arithEq_of_rows h (row := 364) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_9
  simp only [← c23859, ← c23860, k2, ← c23861] at e_364_9
  have hr := e_364_9
  simp only [e_364_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 250)) := by
  have c23862 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23863 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23864 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23865 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23866 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23867 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_10 := arithEq_of_rows h (row := 364) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_10
  simp only [← c23862, ← c23863, ← c23864] at e_364_10
  have e_364_11 := arithEq_of_rows h (row := 364) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_11
  simp only [← c23865, ← c23866, k2, ← c23867] at e_364_11
  have hr := e_364_11
  simp only [e_364_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 251)) := by
  have c23868 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23869 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23870 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23871 := (publicBatchWrapper89_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23872 := (publicBatchWrapper89_copies746 a h).1
  have c23873 := (publicBatchWrapper89_copies746 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_12 := arithEq_of_rows h (row := 364) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_12
  simp only [← c23868, ← c23869, ← c23870] at e_364_12
  have e_364_13 := arithEq_of_rows h (row := 364) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_13
  simp only [← c23871, ← c23872, k2, ← c23873] at e_364_13
  have hr := e_364_13
  simp only [e_364_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 252)) := by
  have c23874 := (publicBatchWrapper89_copies746 a h).2.2.1
  have c23875 := (publicBatchWrapper89_copies746 a h).2.2.2.1
  have c23876 := (publicBatchWrapper89_copies746 a h).2.2.2.2.1
  have c23877 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.1
  have c23878 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.1
  have c23879 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_14 := arithEq_of_rows h (row := 364) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_14
  simp only [← c23874, ← c23875, ← c23876] at e_364_14
  have e_364_15 := arithEq_of_rows h (row := 364) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_15
  simp only [← c23877, ← c23878, k2, ← c23879] at e_364_15
  have hr := e_364_15
  simp only [e_364_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 253)) := by
  have c23880 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.1
  have c23881 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.1
  have c23882 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23883 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23884 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23885 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_16 := arithEq_of_rows h (row := 364) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_16
  simp only [← c23880, ← c23881, ← c23882] at e_364_16
  have e_364_17 := arithEq_of_rows h (row := 364) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_17
  simp only [← c23883, ← c23884, k2, ← c23885] at e_364_17
  have hr := e_364_17
  simp only [e_364_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 364 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 254)) := by
  have c23886 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23887 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23888 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23889 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23890 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23891 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_364_18 := arithEq_of_rows h (row := 364) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_18
  simp only [← c23886, ← c23887, ← c23888] at e_364_18
  have e_364_19 := arithEq_of_rows h (row := 364) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_19
  simp only [← c23889, ← c23890, k2, ← c23891] at e_364_19
  have hr := e_364_19
  simp only [e_364_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 255)) := by
  have c23892 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23893 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23894 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23895 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23896 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23897 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_0 := arithEq_of_rows h (row := 365) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_0
  simp only [← c23892, ← c23893, ← c23894] at e_365_0
  have e_365_1 := arithEq_of_rows h (row := 365) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_1
  simp only [← c23895, ← c23896, k2, ← c23897] at e_365_1
  have hr := e_365_1
  simp only [e_365_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 256)) := by
  have c23898 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23899 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23900 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23901 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23902 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23903 := (publicBatchWrapper89_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_2 := arithEq_of_rows h (row := 365) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_2
  simp only [← c23898, ← c23899, ← c23900] at e_365_2
  have e_365_3 := arithEq_of_rows h (row := 365) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_3
  simp only [← c23901, ← c23902, k2, ← c23903] at e_365_3
  have hr := e_365_3
  simp only [e_365_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 257)) := by
  have c23904 := (publicBatchWrapper89_copies747 a h).1
  have c23905 := (publicBatchWrapper89_copies747 a h).2.1
  have c23906 := (publicBatchWrapper89_copies747 a h).2.2.1
  have c23907 := (publicBatchWrapper89_copies747 a h).2.2.2.1
  have c23908 := (publicBatchWrapper89_copies747 a h).2.2.2.2.1
  have c23909 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_4 := arithEq_of_rows h (row := 365) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_4
  simp only [← c23904, ← c23905, ← c23906] at e_365_4
  have e_365_5 := arithEq_of_rows h (row := 365) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_5
  simp only [← c23907, ← c23908, k2, ← c23909] at e_365_5
  have hr := e_365_5
  simp only [e_365_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 258)) := by
  have c23910 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.1
  have c23911 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.1
  have c23912 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.1
  have c23913 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.1
  have c23914 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23915 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_6 := arithEq_of_rows h (row := 365) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_6
  simp only [← c23910, ← c23911, ← c23912] at e_365_6
  have e_365_7 := arithEq_of_rows h (row := 365) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_7
  simp only [← c23913, ← c23914, k2, ← c23915] at e_365_7
  have hr := e_365_7
  simp only [e_365_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 259)) := by
  have c23916 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23917 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23918 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23919 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23920 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23921 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_8 := arithEq_of_rows h (row := 365) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_8
  simp only [← c23916, ← c23917, ← c23918] at e_365_8
  have e_365_9 := arithEq_of_rows h (row := 365) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_9
  simp only [← c23919, ← c23920, k2, ← c23921] at e_365_9
  have hr := e_365_9
  simp only [e_365_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 260)) := by
  have c23922 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23923 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23924 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23925 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23926 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23927 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_10 := arithEq_of_rows h (row := 365) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_10
  simp only [← c23922, ← c23923, ← c23924] at e_365_10
  have e_365_11 := arithEq_of_rows h (row := 365) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_11
  simp only [← c23925, ← c23926, k2, ← c23927] at e_365_11
  have hr := e_365_11
  simp only [e_365_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 261)) := by
  have c23928 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23929 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23930 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23931 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23932 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23933 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_12 := arithEq_of_rows h (row := 365) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_12
  simp only [← c23928, ← c23929, ← c23930] at e_365_12
  have e_365_13 := arithEq_of_rows h (row := 365) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_13
  simp only [← c23931, ← c23932, k2, ← c23933] at e_365_13
  have hr := e_365_13
  simp only [e_365_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 262)) := by
  have c23934 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23935 := (publicBatchWrapper89_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23936 := (publicBatchWrapper89_copies748 a h).1
  have c23937 := (publicBatchWrapper89_copies748 a h).2.1
  have c23938 := (publicBatchWrapper89_copies748 a h).2.2.1
  have c23939 := (publicBatchWrapper89_copies748 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_14 := arithEq_of_rows h (row := 365) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_14
  simp only [← c23934, ← c23935, ← c23936] at e_365_14
  have e_365_15 := arithEq_of_rows h (row := 365) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_15
  simp only [← c23937, ← c23938, k2, ← c23939] at e_365_15
  have hr := e_365_15
  simp only [e_365_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 263)) := by
  have c23940 := (publicBatchWrapper89_copies748 a h).2.2.2.2.1
  have c23941 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.1
  have c23942 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.1
  have c23943 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.1
  have c23944 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.1
  have c23945 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_16 := arithEq_of_rows h (row := 365) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_16
  simp only [← c23940, ← c23941, ← c23942] at e_365_16
  have e_365_17 := arithEq_of_rows h (row := 365) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_17
  simp only [← c23943, ← c23944, k2, ← c23945] at e_365_17
  have hr := e_365_17
  simp only [e_365_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 365 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 264)) := by
  have c23946 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23947 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23948 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23949 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23950 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23951 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_365_18 := arithEq_of_rows h (row := 365) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_18
  simp only [← c23946, ← c23947, ← c23948] at e_365_18
  have e_365_19 := arithEq_of_rows h (row := 365) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_19
  simp only [← c23949, ← c23950, k2, ← c23951] at e_365_19
  have hr := e_365_19
  simp only [e_365_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 265)) := by
  have c23952 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23953 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23954 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23955 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23956 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23957 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_0 := arithEq_of_rows h (row := 366) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_0
  simp only [← c23952, ← c23953, ← c23954] at e_366_0
  have e_366_1 := arithEq_of_rows h (row := 366) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_1
  simp only [← c23955, ← c23956, k2, ← c23957] at e_366_1
  have hr := e_366_1
  simp only [e_366_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 266)) := by
  have c23958 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23959 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23960 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23961 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23962 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23963 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_2 := arithEq_of_rows h (row := 366) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_2
  simp only [← c23958, ← c23959, ← c23960] at e_366_2
  have e_366_3 := arithEq_of_rows h (row := 366) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_3
  simp only [← c23961, ← c23962, k2, ← c23963] at e_366_3
  have hr := e_366_3
  simp only [e_366_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 267)) := by
  have c23964 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23965 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23966 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23967 := (publicBatchWrapper89_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23968 := (publicBatchWrapper89_copies749 a h).1
  have c23969 := (publicBatchWrapper89_copies749 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_4 := arithEq_of_rows h (row := 366) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_4
  simp only [← c23964, ← c23965, ← c23966] at e_366_4
  have e_366_5 := arithEq_of_rows h (row := 366) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_5
  simp only [← c23967, ← c23968, k2, ← c23969] at e_366_5
  have hr := e_366_5
  simp only [e_366_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 268)) := by
  have c23970 := (publicBatchWrapper89_copies749 a h).2.2.1
  have c23971 := (publicBatchWrapper89_copies749 a h).2.2.2.1
  have c23972 := (publicBatchWrapper89_copies749 a h).2.2.2.2.1
  have c23973 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.1
  have c23974 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.1
  have c23975 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_6 := arithEq_of_rows h (row := 366) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_6
  simp only [← c23970, ← c23971, ← c23972] at e_366_6
  have e_366_7 := arithEq_of_rows h (row := 366) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_7
  simp only [← c23973, ← c23974, k2, ← c23975] at e_366_7
  have hr := e_366_7
  simp only [e_366_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 269)) := by
  have c23976 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.1
  have c23977 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.1
  have c23978 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23979 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23980 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23981 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_8 := arithEq_of_rows h (row := 366) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_8
  simp only [← c23976, ← c23977, ← c23978] at e_366_8
  have e_366_9 := arithEq_of_rows h (row := 366) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_9
  simp only [← c23979, ← c23980, k2, ← c23981] at e_366_9
  have hr := e_366_9
  simp only [e_366_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 270)) := by
  have c23982 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23983 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23984 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23985 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23986 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23987 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_10 := arithEq_of_rows h (row := 366) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_10
  simp only [← c23982, ← c23983, ← c23984] at e_366_10
  have e_366_11 := arithEq_of_rows h (row := 366) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_11
  simp only [← c23985, ← c23986, k2, ← c23987] at e_366_11
  have hr := e_366_11
  simp only [e_366_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 271)) := by
  have c23988 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23989 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23990 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23991 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23992 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23993 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_12 := arithEq_of_rows h (row := 366) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_12
  simp only [← c23988, ← c23989, ← c23990] at e_366_12
  have e_366_13 := arithEq_of_rows h (row := 366) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_13
  simp only [← c23991, ← c23992, k2, ← c23993] at e_366_13
  have hr := e_366_13
  simp only [e_366_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 272)) := by
  have c23994 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23995 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23996 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23997 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23998 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23999 := (publicBatchWrapper89_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_14 := arithEq_of_rows h (row := 366) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_14
  simp only [← c23994, ← c23995, ← c23996] at e_366_14
  have e_366_15 := arithEq_of_rows h (row := 366) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_15
  simp only [← c23997, ← c23998, k2, ← c23999] at e_366_15
  have hr := e_366_15
  simp only [e_366_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 273)) := by
  have c24000 := (publicBatchWrapper89_copies750 a h).1
  have c24001 := (publicBatchWrapper89_copies750 a h).2.1
  have c24002 := (publicBatchWrapper89_copies750 a h).2.2.1
  have c24003 := (publicBatchWrapper89_copies750 a h).2.2.2.1
  have c24004 := (publicBatchWrapper89_copies750 a h).2.2.2.2.1
  have c24005 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_16 := arithEq_of_rows h (row := 366) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_16
  simp only [← c24000, ← c24001, ← c24002] at e_366_16
  have e_366_17 := arithEq_of_rows h (row := 366) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_17
  simp only [← c24003, ← c24004, k2, ← c24005] at e_366_17
  have hr := e_366_17
  simp only [e_366_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 366 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 274)) := by
  have c24006 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.1
  have c24007 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.1
  have c24008 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.1
  have c24009 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.1
  have c24010 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24011 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_366_18 := arithEq_of_rows h (row := 366) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_18
  simp only [← c24006, ← c24007, ← c24008] at e_366_18
  have e_366_19 := arithEq_of_rows h (row := 366) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_19
  simp only [← c24009, ← c24010, k2, ← c24011] at e_366_19
  have hr := e_366_19
  simp only [e_366_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 275)) := by
  have c24012 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24013 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24014 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24015 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24016 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24017 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_0 := arithEq_of_rows h (row := 367) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_0
  simp only [← c24012, ← c24013, ← c24014] at e_367_0
  have e_367_1 := arithEq_of_rows h (row := 367) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_1
  simp only [← c24015, ← c24016, k2, ← c24017] at e_367_1
  have hr := e_367_1
  simp only [e_367_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 276)) := by
  have c24018 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24019 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24020 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24021 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24022 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24023 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_2 := arithEq_of_rows h (row := 367) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_2
  simp only [← c24018, ← c24019, ← c24020] at e_367_2
  have e_367_3 := arithEq_of_rows h (row := 367) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_3
  simp only [← c24021, ← c24022, k2, ← c24023] at e_367_3
  have hr := e_367_3
  simp only [e_367_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 277)) := by
  have c24024 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24025 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24026 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24027 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24028 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24029 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_4 := arithEq_of_rows h (row := 367) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_4
  simp only [← c24024, ← c24025, ← c24026] at e_367_4
  have e_367_5 := arithEq_of_rows h (row := 367) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_5
  simp only [← c24027, ← c24028, k2, ← c24029] at e_367_5
  have hr := e_367_5
  simp only [e_367_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 278)) := by
  have c24030 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24031 := (publicBatchWrapper89_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24032 := (publicBatchWrapper89_copies751 a h).1
  have c24033 := (publicBatchWrapper89_copies751 a h).2.1
  have c24034 := (publicBatchWrapper89_copies751 a h).2.2.1
  have c24035 := (publicBatchWrapper89_copies751 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_6 := arithEq_of_rows h (row := 367) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_6
  simp only [← c24030, ← c24031, ← c24032] at e_367_6
  have e_367_7 := arithEq_of_rows h (row := 367) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_7
  simp only [← c24033, ← c24034, k2, ← c24035] at e_367_7
  have hr := e_367_7
  simp only [e_367_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 279)) := by
  have c24036 := (publicBatchWrapper89_copies751 a h).2.2.2.2.1
  have c24037 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.1
  have c24038 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.1
  have c24039 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.1
  have c24040 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.1
  have c24041 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_8 := arithEq_of_rows h (row := 367) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_8
  simp only [← c24036, ← c24037, ← c24038] at e_367_8
  have e_367_9 := arithEq_of_rows h (row := 367) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_9
  simp only [← c24039, ← c24040, k2, ← c24041] at e_367_9
  have hr := e_367_9
  simp only [e_367_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 280)) := by
  have c24042 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24043 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24044 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24045 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24046 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24047 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_10 := arithEq_of_rows h (row := 367) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_10
  simp only [← c24042, ← c24043, ← c24044] at e_367_10
  have e_367_11 := arithEq_of_rows h (row := 367) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_11
  simp only [← c24045, ← c24046, k2, ← c24047] at e_367_11
  have hr := e_367_11
  simp only [e_367_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 281)) := by
  have c24048 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24049 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24050 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24051 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24052 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24053 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_12 := arithEq_of_rows h (row := 367) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_12
  simp only [← c24048, ← c24049, ← c24050] at e_367_12
  have e_367_13 := arithEq_of_rows h (row := 367) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_13
  simp only [← c24051, ← c24052, k2, ← c24053] at e_367_13
  have hr := e_367_13
  simp only [e_367_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 282)) := by
  have c24054 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24055 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24056 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24057 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24058 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24059 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_14 := arithEq_of_rows h (row := 367) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_14
  simp only [← c24054, ← c24055, ← c24056] at e_367_14
  have e_367_15 := arithEq_of_rows h (row := 367) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_15
  simp only [← c24057, ← c24058, k2, ← c24059] at e_367_15
  have hr := e_367_15
  simp only [e_367_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 283)) := by
  have c24060 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24061 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24062 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24063 := (publicBatchWrapper89_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24064 := (publicBatchWrapper89_copies752 a h).1
  have c24065 := (publicBatchWrapper89_copies752 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_16 := arithEq_of_rows h (row := 367) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_16
  simp only [← c24060, ← c24061, ← c24062] at e_367_16
  have e_367_17 := arithEq_of_rows h (row := 367) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_17
  simp only [← c24063, ← c24064, k2, ← c24065] at e_367_17
  have hr := e_367_17
  simp only [e_367_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 367 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 284)) := by
  have c24066 := (publicBatchWrapper89_copies752 a h).2.2.1
  have c24067 := (publicBatchWrapper89_copies752 a h).2.2.2.1
  have c24068 := (publicBatchWrapper89_copies752 a h).2.2.2.2.1
  have c24069 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.1
  have c24070 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.1
  have c24071 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_367_18 := arithEq_of_rows h (row := 367) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_18
  simp only [← c24066, ← c24067, ← c24068] at e_367_18
  have e_367_19 := arithEq_of_rows h (row := 367) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_19
  simp only [← c24069, ← c24070, k2, ← c24071] at e_367_19
  have hr := e_367_19
  simp only [e_367_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 285)) := by
  have c24072 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.1
  have c24073 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.1
  have c24074 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24075 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24076 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24077 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_0 := arithEq_of_rows h (row := 368) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_0
  simp only [← c24072, ← c24073, ← c24074] at e_368_0
  have e_368_1 := arithEq_of_rows h (row := 368) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_1
  simp only [← c24075, ← c24076, k2, ← c24077] at e_368_1
  have hr := e_368_1
  simp only [e_368_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 286)) := by
  have c24078 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24079 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24080 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24081 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24082 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24083 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_2 := arithEq_of_rows h (row := 368) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_2
  simp only [← c24078, ← c24079, ← c24080] at e_368_2
  have e_368_3 := arithEq_of_rows h (row := 368) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_3
  simp only [← c24081, ← c24082, k2, ← c24083] at e_368_3
  have hr := e_368_3
  simp only [e_368_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 287)) := by
  have c24084 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24085 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24086 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24087 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24088 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24089 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_4 := arithEq_of_rows h (row := 368) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_4
  simp only [← c24084, ← c24085, ← c24086] at e_368_4
  have e_368_5 := arithEq_of_rows h (row := 368) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_5
  simp only [← c24087, ← c24088, k2, ← c24089] at e_368_5
  have hr := e_368_5
  simp only [e_368_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 288)) := by
  have c24090 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24091 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24092 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24093 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24094 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24095 := (publicBatchWrapper89_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_6 := arithEq_of_rows h (row := 368) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_6
  simp only [← c24090, ← c24091, ← c24092] at e_368_6
  have e_368_7 := arithEq_of_rows h (row := 368) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_7
  simp only [← c24093, ← c24094, k2, ← c24095] at e_368_7
  have hr := e_368_7
  simp only [e_368_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 289)) := by
  have c24096 := (publicBatchWrapper89_copies753 a h).1
  have c24097 := (publicBatchWrapper89_copies753 a h).2.1
  have c24098 := (publicBatchWrapper89_copies753 a h).2.2.1
  have c24099 := (publicBatchWrapper89_copies753 a h).2.2.2.1
  have c24100 := (publicBatchWrapper89_copies753 a h).2.2.2.2.1
  have c24101 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_8 := arithEq_of_rows h (row := 368) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_8
  simp only [← c24096, ← c24097, ← c24098] at e_368_8
  have e_368_9 := arithEq_of_rows h (row := 368) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_9
  simp only [← c24099, ← c24100, k2, ← c24101] at e_368_9
  have hr := e_368_9
  simp only [e_368_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 290)) := by
  have c24102 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.1
  have c24103 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.1
  have c24104 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.1
  have c24105 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.1
  have c24106 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24107 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_10 := arithEq_of_rows h (row := 368) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_10
  simp only [← c24102, ← c24103, ← c24104] at e_368_10
  have e_368_11 := arithEq_of_rows h (row := 368) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_11
  simp only [← c24105, ← c24106, k2, ← c24107] at e_368_11
  have hr := e_368_11
  simp only [e_368_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 291)) := by
  have c24108 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24109 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24110 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24111 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24112 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24113 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_12 := arithEq_of_rows h (row := 368) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_12
  simp only [← c24108, ← c24109, ← c24110] at e_368_12
  have e_368_13 := arithEq_of_rows h (row := 368) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_13
  simp only [← c24111, ← c24112, k2, ← c24113] at e_368_13
  have hr := e_368_13
  simp only [e_368_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 292)) := by
  have c24114 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24115 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24116 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24117 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24118 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24119 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_14 := arithEq_of_rows h (row := 368) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_14
  simp only [← c24114, ← c24115, ← c24116] at e_368_14
  have e_368_15 := arithEq_of_rows h (row := 368) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_15
  simp only [← c24117, ← c24118, k2, ← c24119] at e_368_15
  have hr := e_368_15
  simp only [e_368_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 293)) := by
  have c24120 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24121 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24122 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24123 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24124 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24125 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_16 := arithEq_of_rows h (row := 368) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_16
  simp only [← c24120, ← c24121, ← c24122] at e_368_16
  have e_368_17 := arithEq_of_rows h (row := 368) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_17
  simp only [← c24123, ← c24124, k2, ← c24125] at e_368_17
  have hr := e_368_17
  simp only [e_368_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 368 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 294)) := by
  have c24126 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24127 := (publicBatchWrapper89_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24128 := (publicBatchWrapper89_copies754 a h).1
  have c24129 := (publicBatchWrapper89_copies754 a h).2.1
  have c24130 := (publicBatchWrapper89_copies754 a h).2.2.1
  have c24131 := (publicBatchWrapper89_copies754 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_368_18 := arithEq_of_rows h (row := 368) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_18
  simp only [← c24126, ← c24127, ← c24128] at e_368_18
  have e_368_19 := arithEq_of_rows h (row := 368) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_19
  simp only [← c24129, ← c24130, k2, ← c24131] at e_368_19
  have hr := e_368_19
  simp only [e_368_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 295)) := by
  have c24132 := (publicBatchWrapper89_copies754 a h).2.2.2.2.1
  have c24133 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.1
  have c24134 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.1
  have c24135 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.1
  have c24136 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.1
  have c24137 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_0 := arithEq_of_rows h (row := 369) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_0
  simp only [← c24132, ← c24133, ← c24134] at e_369_0
  have e_369_1 := arithEq_of_rows h (row := 369) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_1
  simp only [← c24135, ← c24136, k2, ← c24137] at e_369_1
  have hr := e_369_1
  simp only [e_369_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 296)) := by
  have c24138 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24139 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24140 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24141 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24142 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24143 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_2 := arithEq_of_rows h (row := 369) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_2
  simp only [← c24138, ← c24139, ← c24140] at e_369_2
  have e_369_3 := arithEq_of_rows h (row := 369) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_3
  simp only [← c24141, ← c24142, k2, ← c24143] at e_369_3
  have hr := e_369_3
  simp only [e_369_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 297)) := by
  have c24144 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24145 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24146 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24147 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24148 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24149 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_4 := arithEq_of_rows h (row := 369) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_4
  simp only [← c24144, ← c24145, ← c24146] at e_369_4
  have e_369_5 := arithEq_of_rows h (row := 369) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_5
  simp only [← c24147, ← c24148, k2, ← c24149] at e_369_5
  have hr := e_369_5
  simp only [e_369_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 298)) := by
  have c24150 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24151 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24152 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24153 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24154 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24155 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_6 := arithEq_of_rows h (row := 369) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_6
  simp only [← c24150, ← c24151, ← c24152] at e_369_6
  have e_369_7 := arithEq_of_rows h (row := 369) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_7
  simp only [← c24153, ← c24154, k2, ← c24155] at e_369_7
  have hr := e_369_7
  simp only [e_369_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 299)) := by
  have c24156 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24157 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24158 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24159 := (publicBatchWrapper89_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24160 := (publicBatchWrapper89_copies755 a h).1
  have c24161 := (publicBatchWrapper89_copies755 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_8 := arithEq_of_rows h (row := 369) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_8
  simp only [← c24156, ← c24157, ← c24158] at e_369_8
  have e_369_9 := arithEq_of_rows h (row := 369) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_9
  simp only [← c24159, ← c24160, k2, ← c24161] at e_369_9
  have hr := e_369_9
  simp only [e_369_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 300)) := by
  have c24162 := (publicBatchWrapper89_copies755 a h).2.2.1
  have c24163 := (publicBatchWrapper89_copies755 a h).2.2.2.1
  have c24164 := (publicBatchWrapper89_copies755 a h).2.2.2.2.1
  have c24165 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.1
  have c24166 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.1
  have c24167 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_10 := arithEq_of_rows h (row := 369) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_10
  simp only [← c24162, ← c24163, ← c24164] at e_369_10
  have e_369_11 := arithEq_of_rows h (row := 369) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_11
  simp only [← c24165, ← c24166, k2, ← c24167] at e_369_11
  have hr := e_369_11
  simp only [e_369_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 301)) := by
  have c24168 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.1
  have c24169 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.1
  have c24170 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24171 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24172 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24173 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_12 := arithEq_of_rows h (row := 369) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_12
  simp only [← c24168, ← c24169, ← c24170] at e_369_12
  have e_369_13 := arithEq_of_rows h (row := 369) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_13
  simp only [← c24171, ← c24172, k2, ← c24173] at e_369_13
  have hr := e_369_13
  simp only [e_369_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 302)) := by
  have c24174 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24175 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24176 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24177 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24178 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24179 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_14 := arithEq_of_rows h (row := 369) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_14
  simp only [← c24174, ← c24175, ← c24176] at e_369_14
  have e_369_15 := arithEq_of_rows h (row := 369) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_15
  simp only [← c24177, ← c24178, k2, ← c24179] at e_369_15
  have hr := e_369_15
  simp only [e_369_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 303)) := by
  have c24180 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24181 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24182 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24183 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24184 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24185 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_16 := arithEq_of_rows h (row := 369) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_16
  simp only [← c24180, ← c24181, ← c24182] at e_369_16
  have e_369_17 := arithEq_of_rows h (row := 369) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_17
  simp only [← c24183, ← c24184, k2, ← c24185] at e_369_17
  have hr := e_369_17
  simp only [e_369_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 369 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 304)) := by
  have c24186 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24187 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24188 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24189 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24190 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24191 := (publicBatchWrapper89_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_369_18 := arithEq_of_rows h (row := 369) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_18
  simp only [← c24186, ← c24187, ← c24188] at e_369_18
  have e_369_19 := arithEq_of_rows h (row := 369) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_19
  simp only [← c24189, ← c24190, k2, ← c24191] at e_369_19
  have hr := e_369_19
  simp only [e_369_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 305)) := by
  have c24192 := (publicBatchWrapper89_copies756 a h).1
  have c24193 := (publicBatchWrapper89_copies756 a h).2.1
  have c24194 := (publicBatchWrapper89_copies756 a h).2.2.1
  have c24195 := (publicBatchWrapper89_copies756 a h).2.2.2.1
  have c24196 := (publicBatchWrapper89_copies756 a h).2.2.2.2.1
  have c24197 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_0 := arithEq_of_rows h (row := 370) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_0
  simp only [← c24192, ← c24193, ← c24194] at e_370_0
  have e_370_1 := arithEq_of_rows h (row := 370) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_1
  simp only [← c24195, ← c24196, k2, ← c24197] at e_370_1
  have hr := e_370_1
  simp only [e_370_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 306)) := by
  have c24198 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.1
  have c24199 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.1
  have c24200 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.1
  have c24201 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.1
  have c24202 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24203 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_2 := arithEq_of_rows h (row := 370) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_2
  simp only [← c24198, ← c24199, ← c24200] at e_370_2
  have e_370_3 := arithEq_of_rows h (row := 370) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_3
  simp only [← c24201, ← c24202, k2, ← c24203] at e_370_3
  have hr := e_370_3
  simp only [e_370_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 307)) := by
  have c24204 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24205 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24206 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24207 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24208 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24209 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_4 := arithEq_of_rows h (row := 370) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_4
  simp only [← c24204, ← c24205, ← c24206] at e_370_4
  have e_370_5 := arithEq_of_rows h (row := 370) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_5
  simp only [← c24207, ← c24208, k2, ← c24209] at e_370_5
  have hr := e_370_5
  simp only [e_370_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 308)) := by
  have c24210 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24211 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24212 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24213 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24214 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24215 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_6 := arithEq_of_rows h (row := 370) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_6
  simp only [← c24210, ← c24211, ← c24212] at e_370_6
  have e_370_7 := arithEq_of_rows h (row := 370) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_7
  simp only [← c24213, ← c24214, k2, ← c24215] at e_370_7
  have hr := e_370_7
  simp only [e_370_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 309)) := by
  have c24216 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24217 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24218 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24219 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24220 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24221 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_8 := arithEq_of_rows h (row := 370) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_8
  simp only [← c24216, ← c24217, ← c24218] at e_370_8
  have e_370_9 := arithEq_of_rows h (row := 370) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_9
  simp only [← c24219, ← c24220, k2, ← c24221] at e_370_9
  have hr := e_370_9
  simp only [e_370_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 310)) := by
  have c24222 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24223 := (publicBatchWrapper89_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24224 := (publicBatchWrapper89_copies757 a h).1
  have c24225 := (publicBatchWrapper89_copies757 a h).2.1
  have c24226 := (publicBatchWrapper89_copies757 a h).2.2.1
  have c24227 := (publicBatchWrapper89_copies757 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_10 := arithEq_of_rows h (row := 370) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_10
  simp only [← c24222, ← c24223, ← c24224] at e_370_10
  have e_370_11 := arithEq_of_rows h (row := 370) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_11
  simp only [← c24225, ← c24226, k2, ← c24227] at e_370_11
  have hr := e_370_11
  simp only [e_370_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 311)) := by
  have c24228 := (publicBatchWrapper89_copies757 a h).2.2.2.2.1
  have c24229 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.1
  have c24230 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.1
  have c24231 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.1
  have c24232 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.1
  have c24233 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_12 := arithEq_of_rows h (row := 370) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_12
  simp only [← c24228, ← c24229, ← c24230] at e_370_12
  have e_370_13 := arithEq_of_rows h (row := 370) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_13
  simp only [← c24231, ← c24232, k2, ← c24233] at e_370_13
  have hr := e_370_13
  simp only [e_370_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 312)) := by
  have c24234 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24235 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24236 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24237 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24238 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24239 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_14 := arithEq_of_rows h (row := 370) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_14
  simp only [← c24234, ← c24235, ← c24236] at e_370_14
  have e_370_15 := arithEq_of_rows h (row := 370) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_15
  simp only [← c24237, ← c24238, k2, ← c24239] at e_370_15
  have hr := e_370_15
  simp only [e_370_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 313)) := by
  have c24240 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24241 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24242 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24243 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24244 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24245 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_16 := arithEq_of_rows h (row := 370) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_16
  simp only [← c24240, ← c24241, ← c24242] at e_370_16
  have e_370_17 := arithEq_of_rows h (row := 370) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_17
  simp only [← c24243, ← c24244, k2, ← c24245] at e_370_17
  have hr := e_370_17
  simp only [e_370_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 370 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 314)) := by
  have c24246 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24247 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24248 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24249 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24250 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24251 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_370_18 := arithEq_of_rows h (row := 370) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_18
  simp only [← c24246, ← c24247, ← c24248] at e_370_18
  have e_370_19 := arithEq_of_rows h (row := 370) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_19
  simp only [← c24249, ← c24250, k2, ← c24251] at e_370_19
  have hr := e_370_19
  simp only [e_370_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 315)) := by
  have c24252 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24253 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24254 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24255 := (publicBatchWrapper89_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24256 := (publicBatchWrapper89_copies758 a h).1
  have c24257 := (publicBatchWrapper89_copies758 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_0 := arithEq_of_rows h (row := 371) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_0
  simp only [← c24252, ← c24253, ← c24254] at e_371_0
  have e_371_1 := arithEq_of_rows h (row := 371) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_1
  simp only [← c24255, ← c24256, k2, ← c24257] at e_371_1
  have hr := e_371_1
  simp only [e_371_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 316)) := by
  have c24258 := (publicBatchWrapper89_copies758 a h).2.2.1
  have c24259 := (publicBatchWrapper89_copies758 a h).2.2.2.1
  have c24260 := (publicBatchWrapper89_copies758 a h).2.2.2.2.1
  have c24261 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.1
  have c24262 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.1
  have c24263 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_2 := arithEq_of_rows h (row := 371) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_2
  simp only [← c24258, ← c24259, ← c24260] at e_371_2
  have e_371_3 := arithEq_of_rows h (row := 371) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_3
  simp only [← c24261, ← c24262, k2, ← c24263] at e_371_3
  have hr := e_371_3
  simp only [e_371_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 317)) := by
  have c24264 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.1
  have c24265 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.1
  have c24266 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24267 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24268 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24269 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_4 := arithEq_of_rows h (row := 371) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_4
  simp only [← c24264, ← c24265, ← c24266] at e_371_4
  have e_371_5 := arithEq_of_rows h (row := 371) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_5
  simp only [← c24267, ← c24268, k2, ← c24269] at e_371_5
  have hr := e_371_5
  simp only [e_371_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 318)) := by
  have c24270 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24271 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24272 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24273 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24274 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24275 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_6 := arithEq_of_rows h (row := 371) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_6
  simp only [← c24270, ← c24271, ← c24272] at e_371_6
  have e_371_7 := arithEq_of_rows h (row := 371) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_7
  simp only [← c24273, ← c24274, k2, ← c24275] at e_371_7
  have hr := e_371_7
  simp only [e_371_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 319)) := by
  have c24276 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24277 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24278 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24279 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24280 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24281 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_8 := arithEq_of_rows h (row := 371) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_8
  simp only [← c24276, ← c24277, ← c24278] at e_371_8
  have e_371_9 := arithEq_of_rows h (row := 371) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_9
  simp only [← c24279, ← c24280, k2, ← c24281] at e_371_9
  have hr := e_371_9
  simp only [e_371_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 320)) := by
  have c24282 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24283 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24284 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24285 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24286 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24287 := (publicBatchWrapper89_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_10 := arithEq_of_rows h (row := 371) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_10
  simp only [← c24282, ← c24283, ← c24284] at e_371_10
  have e_371_11 := arithEq_of_rows h (row := 371) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_11
  simp only [← c24285, ← c24286, k2, ← c24287] at e_371_11
  have hr := e_371_11
  simp only [e_371_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 321)) := by
  have c24288 := (publicBatchWrapper89_copies759 a h).1
  have c24289 := (publicBatchWrapper89_copies759 a h).2.1
  have c24290 := (publicBatchWrapper89_copies759 a h).2.2.1
  have c24291 := (publicBatchWrapper89_copies759 a h).2.2.2.1
  have c24292 := (publicBatchWrapper89_copies759 a h).2.2.2.2.1
  have c24293 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_12 := arithEq_of_rows h (row := 371) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_12
  simp only [← c24288, ← c24289, ← c24290] at e_371_12
  have e_371_13 := arithEq_of_rows h (row := 371) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_13
  simp only [← c24291, ← c24292, k2, ← c24293] at e_371_13
  have hr := e_371_13
  simp only [e_371_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 322)) := by
  have c24294 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.1
  have c24295 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.1
  have c24296 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.1
  have c24297 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.1
  have c24298 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24299 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_14 := arithEq_of_rows h (row := 371) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_14
  simp only [← c24294, ← c24295, ← c24296] at e_371_14
  have e_371_15 := arithEq_of_rows h (row := 371) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_15
  simp only [← c24297, ← c24298, k2, ← c24299] at e_371_15
  have hr := e_371_15
  simp only [e_371_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 323)) := by
  have c24300 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24301 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24302 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24303 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24304 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24305 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_16 := arithEq_of_rows h (row := 371) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_16
  simp only [← c24300, ← c24301, ← c24302] at e_371_16
  have e_371_17 := arithEq_of_rows h (row := 371) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_17
  simp only [← c24303, ← c24304, k2, ← c24305] at e_371_17
  have hr := e_371_17
  simp only [e_371_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 371 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 324)) := by
  have c24306 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24307 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24308 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24309 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24310 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24311 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_371_18 := arithEq_of_rows h (row := 371) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_18
  simp only [← c24306, ← c24307, ← c24308] at e_371_18
  have e_371_19 := arithEq_of_rows h (row := 371) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_19
  simp only [← c24309, ← c24310, k2, ← c24311] at e_371_19
  have hr := e_371_19
  simp only [e_371_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 325)) := by
  have c24312 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24313 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24314 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24315 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24316 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24317 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_0 := arithEq_of_rows h (row := 372) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_0
  simp only [← c24312, ← c24313, ← c24314] at e_372_0
  have e_372_1 := arithEq_of_rows h (row := 372) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_1
  simp only [← c24315, ← c24316, k2, ← c24317] at e_372_1
  have hr := e_372_1
  simp only [e_372_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 326)) := by
  have c24318 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24319 := (publicBatchWrapper89_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24320 := (publicBatchWrapper89_copies760 a h).1
  have c24321 := (publicBatchWrapper89_copies760 a h).2.1
  have c24322 := (publicBatchWrapper89_copies760 a h).2.2.1
  have c24323 := (publicBatchWrapper89_copies760 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_2 := arithEq_of_rows h (row := 372) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_2
  simp only [← c24318, ← c24319, ← c24320] at e_372_2
  have e_372_3 := arithEq_of_rows h (row := 372) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_3
  simp only [← c24321, ← c24322, k2, ← c24323] at e_372_3
  have hr := e_372_3
  simp only [e_372_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 327)) := by
  have c24324 := (publicBatchWrapper89_copies760 a h).2.2.2.2.1
  have c24325 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.1
  have c24326 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.1
  have c24327 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.1
  have c24328 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.1
  have c24329 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_4 := arithEq_of_rows h (row := 372) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_4
  simp only [← c24324, ← c24325, ← c24326] at e_372_4
  have e_372_5 := arithEq_of_rows h (row := 372) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_5
  simp only [← c24327, ← c24328, k2, ← c24329] at e_372_5
  have hr := e_372_5
  simp only [e_372_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 328)) := by
  have c24330 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24331 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24332 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24333 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24334 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24335 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_6 := arithEq_of_rows h (row := 372) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_6
  simp only [← c24330, ← c24331, ← c24332] at e_372_6
  have e_372_7 := arithEq_of_rows h (row := 372) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_7
  simp only [← c24333, ← c24334, k2, ← c24335] at e_372_7
  have hr := e_372_7
  simp only [e_372_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 329)) := by
  have c24336 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24337 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24338 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24339 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24340 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24341 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_8 := arithEq_of_rows h (row := 372) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_8
  simp only [← c24336, ← c24337, ← c24338] at e_372_8
  have e_372_9 := arithEq_of_rows h (row := 372) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_9
  simp only [← c24339, ← c24340, k2, ← c24341] at e_372_9
  have hr := e_372_9
  simp only [e_372_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 330)) := by
  have c24342 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24343 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24344 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24345 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24346 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24347 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_10 := arithEq_of_rows h (row := 372) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_10
  simp only [← c24342, ← c24343, ← c24344] at e_372_10
  have e_372_11 := arithEq_of_rows h (row := 372) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_11
  simp only [← c24345, ← c24346, k2, ← c24347] at e_372_11
  have hr := e_372_11
  simp only [e_372_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 331)) := by
  have c24348 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24349 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24350 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24351 := (publicBatchWrapper89_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24352 := (publicBatchWrapper89_copies761 a h).1
  have c24353 := (publicBatchWrapper89_copies761 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_12 := arithEq_of_rows h (row := 372) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_12
  simp only [← c24348, ← c24349, ← c24350] at e_372_12
  have e_372_13 := arithEq_of_rows h (row := 372) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_13
  simp only [← c24351, ← c24352, k2, ← c24353] at e_372_13
  have hr := e_372_13
  simp only [e_372_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 332)) := by
  have c24354 := (publicBatchWrapper89_copies761 a h).2.2.1
  have c24355 := (publicBatchWrapper89_copies761 a h).2.2.2.1
  have c24356 := (publicBatchWrapper89_copies761 a h).2.2.2.2.1
  have c24357 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.1
  have c24358 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.1
  have c24359 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_14 := arithEq_of_rows h (row := 372) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_14
  simp only [← c24354, ← c24355, ← c24356] at e_372_14
  have e_372_15 := arithEq_of_rows h (row := 372) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_15
  simp only [← c24357, ← c24358, k2, ← c24359] at e_372_15
  have hr := e_372_15
  simp only [e_372_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 333)) := by
  have c24360 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.1
  have c24361 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.1
  have c24362 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24363 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24364 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24365 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_16 := arithEq_of_rows h (row := 372) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_16
  simp only [← c24360, ← c24361, ← c24362] at e_372_16
  have e_372_17 := arithEq_of_rows h (row := 372) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_17
  simp only [← c24363, ← c24364, k2, ← c24365] at e_372_17
  have hr := e_372_17
  simp only [e_372_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 372 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 334)) := by
  have c24366 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24367 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24368 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24369 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24370 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24371 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_372_18 := arithEq_of_rows h (row := 372) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_18
  simp only [← c24366, ← c24367, ← c24368] at e_372_18
  have e_372_19 := arithEq_of_rows h (row := 372) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_19
  simp only [← c24369, ← c24370, k2, ← c24371] at e_372_19
  have hr := e_372_19
  simp only [e_372_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 335)) := by
  have c24372 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24373 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24374 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24375 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24376 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24377 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_0 := arithEq_of_rows h (row := 373) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_0
  simp only [← c24372, ← c24373, ← c24374] at e_373_0
  have e_373_1 := arithEq_of_rows h (row := 373) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_1
  simp only [← c24375, ← c24376, k2, ← c24377] at e_373_1
  have hr := e_373_1
  simp only [e_373_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 336)) := by
  have c24378 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24379 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24380 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24381 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24382 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24383 := (publicBatchWrapper89_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_2 := arithEq_of_rows h (row := 373) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_2
  simp only [← c24378, ← c24379, ← c24380] at e_373_2
  have e_373_3 := arithEq_of_rows h (row := 373) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_3
  simp only [← c24381, ← c24382, k2, ← c24383] at e_373_3
  have hr := e_373_3
  simp only [e_373_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 337)) := by
  have c24384 := (publicBatchWrapper89_copies762 a h).1
  have c24385 := (publicBatchWrapper89_copies762 a h).2.1
  have c24386 := (publicBatchWrapper89_copies762 a h).2.2.1
  have c24387 := (publicBatchWrapper89_copies762 a h).2.2.2.1
  have c24388 := (publicBatchWrapper89_copies762 a h).2.2.2.2.1
  have c24389 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_4 := arithEq_of_rows h (row := 373) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_4
  simp only [← c24384, ← c24385, ← c24386] at e_373_4
  have e_373_5 := arithEq_of_rows h (row := 373) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_5
  simp only [← c24387, ← c24388, k2, ← c24389] at e_373_5
  have hr := e_373_5
  simp only [e_373_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 338)) := by
  have c24390 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.1
  have c24391 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.1
  have c24392 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.1
  have c24393 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.1
  have c24394 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24395 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_6 := arithEq_of_rows h (row := 373) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_6
  simp only [← c24390, ← c24391, ← c24392] at e_373_6
  have e_373_7 := arithEq_of_rows h (row := 373) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_7
  simp only [← c24393, ← c24394, k2, ← c24395] at e_373_7
  have hr := e_373_7
  simp only [e_373_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 339)) := by
  have c24396 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24397 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24398 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24399 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24400 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24401 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_8 := arithEq_of_rows h (row := 373) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_8
  simp only [← c24396, ← c24397, ← c24398] at e_373_8
  have e_373_9 := arithEq_of_rows h (row := 373) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_9
  simp only [← c24399, ← c24400, k2, ← c24401] at e_373_9
  have hr := e_373_9
  simp only [e_373_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 340)) := by
  have c24402 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24403 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24404 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24405 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24406 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24407 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_10 := arithEq_of_rows h (row := 373) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_10
  simp only [← c24402, ← c24403, ← c24404] at e_373_10
  have e_373_11 := arithEq_of_rows h (row := 373) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_11
  simp only [← c24405, ← c24406, k2, ← c24407] at e_373_11
  have hr := e_373_11
  simp only [e_373_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 341)) := by
  have c24408 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24409 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24410 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24411 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24412 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24413 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_12 := arithEq_of_rows h (row := 373) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_12
  simp only [← c24408, ← c24409, ← c24410] at e_373_12
  have e_373_13 := arithEq_of_rows h (row := 373) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_13
  simp only [← c24411, ← c24412, k2, ← c24413] at e_373_13
  have hr := e_373_13
  simp only [e_373_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 342)) := by
  have c24414 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24415 := (publicBatchWrapper89_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24416 := (publicBatchWrapper89_copies763 a h).1
  have c24417 := (publicBatchWrapper89_copies763 a h).2.1
  have c24418 := (publicBatchWrapper89_copies763 a h).2.2.1
  have c24419 := (publicBatchWrapper89_copies763 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_14 := arithEq_of_rows h (row := 373) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_14
  simp only [← c24414, ← c24415, ← c24416] at e_373_14
  have e_373_15 := arithEq_of_rows h (row := 373) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_15
  simp only [← c24417, ← c24418, k2, ← c24419] at e_373_15
  have hr := e_373_15
  simp only [e_373_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 343)) := by
  have c24420 := (publicBatchWrapper89_copies763 a h).2.2.2.2.1
  have c24421 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.1
  have c24422 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.1
  have c24423 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.1
  have c24424 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.1
  have c24425 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_16 := arithEq_of_rows h (row := 373) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_16
  simp only [← c24420, ← c24421, ← c24422] at e_373_16
  have e_373_17 := arithEq_of_rows h (row := 373) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_17
  simp only [← c24423, ← c24424, k2, ← c24425] at e_373_17
  have hr := e_373_17
  simp only [e_373_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 373 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 344)) := by
  have c24426 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24427 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24428 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24429 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24430 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24431 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_373_18 := arithEq_of_rows h (row := 373) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_18
  simp only [← c24426, ← c24427, ← c24428] at e_373_18
  have e_373_19 := arithEq_of_rows h (row := 373) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_19
  simp only [← c24429, ← c24430, k2, ← c24431] at e_373_19
  have hr := e_373_19
  simp only [e_373_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 345)) := by
  have c24432 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24433 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24434 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24435 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24436 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24437 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_0 := arithEq_of_rows h (row := 374) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_0
  simp only [← c24432, ← c24433, ← c24434] at e_374_0
  have e_374_1 := arithEq_of_rows h (row := 374) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_1
  simp only [← c24435, ← c24436, k2, ← c24437] at e_374_1
  have hr := e_374_1
  simp only [e_374_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 346)) := by
  have c24438 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24439 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24440 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24441 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24442 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24443 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_2 := arithEq_of_rows h (row := 374) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_2
  simp only [← c24438, ← c24439, ← c24440] at e_374_2
  have e_374_3 := arithEq_of_rows h (row := 374) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_3
  simp only [← c24441, ← c24442, k2, ← c24443] at e_374_3
  have hr := e_374_3
  simp only [e_374_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 347)) := by
  have c24444 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24445 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24446 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24447 := (publicBatchWrapper89_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24448 := (publicBatchWrapper89_copies764 a h).1
  have c24449 := (publicBatchWrapper89_copies764 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_4 := arithEq_of_rows h (row := 374) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_4
  simp only [← c24444, ← c24445, ← c24446] at e_374_4
  have e_374_5 := arithEq_of_rows h (row := 374) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_5
  simp only [← c24447, ← c24448, k2, ← c24449] at e_374_5
  have hr := e_374_5
  simp only [e_374_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 348)) := by
  have c24450 := (publicBatchWrapper89_copies764 a h).2.2.1
  have c24451 := (publicBatchWrapper89_copies764 a h).2.2.2.1
  have c24452 := (publicBatchWrapper89_copies764 a h).2.2.2.2.1
  have c24453 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.1
  have c24454 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.1
  have c24455 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_6 := arithEq_of_rows h (row := 374) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_6
  simp only [← c24450, ← c24451, ← c24452] at e_374_6
  have e_374_7 := arithEq_of_rows h (row := 374) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_7
  simp only [← c24453, ← c24454, k2, ← c24455] at e_374_7
  have hr := e_374_7
  simp only [e_374_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 349)) := by
  have c24456 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.1
  have c24457 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.1
  have c24458 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24459 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24460 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24461 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_8 := arithEq_of_rows h (row := 374) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_8
  simp only [← c24456, ← c24457, ← c24458] at e_374_8
  have e_374_9 := arithEq_of_rows h (row := 374) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_9
  simp only [← c24459, ← c24460, k2, ← c24461] at e_374_9
  have hr := e_374_9
  simp only [e_374_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 350)) := by
  have c24462 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24463 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24464 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24465 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24466 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24467 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_10 := arithEq_of_rows h (row := 374) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_10
  simp only [← c24462, ← c24463, ← c24464] at e_374_10
  have e_374_11 := arithEq_of_rows h (row := 374) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_11
  simp only [← c24465, ← c24466, k2, ← c24467] at e_374_11
  have hr := e_374_11
  simp only [e_374_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 351)) := by
  have c24468 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24469 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24470 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24471 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24472 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24473 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_12 := arithEq_of_rows h (row := 374) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_12
  simp only [← c24468, ← c24469, ← c24470] at e_374_12
  have e_374_13 := arithEq_of_rows h (row := 374) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_13
  simp only [← c24471, ← c24472, k2, ← c24473] at e_374_13
  have hr := e_374_13
  simp only [e_374_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 352)) := by
  have c24474 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24475 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24476 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24477 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24478 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24479 := (publicBatchWrapper89_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_14 := arithEq_of_rows h (row := 374) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_14
  simp only [← c24474, ← c24475, ← c24476] at e_374_14
  have e_374_15 := arithEq_of_rows h (row := 374) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_15
  simp only [← c24477, ← c24478, k2, ← c24479] at e_374_15
  have hr := e_374_15
  simp only [e_374_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 353)) := by
  have c24480 := (publicBatchWrapper89_copies765 a h).1
  have c24481 := (publicBatchWrapper89_copies765 a h).2.1
  have c24482 := (publicBatchWrapper89_copies765 a h).2.2.1
  have c24483 := (publicBatchWrapper89_copies765 a h).2.2.2.1
  have c24484 := (publicBatchWrapper89_copies765 a h).2.2.2.2.1
  have c24485 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_16 := arithEq_of_rows h (row := 374) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_16
  simp only [← c24480, ← c24481, ← c24482] at e_374_16
  have e_374_17 := arithEq_of_rows h (row := 374) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_17
  simp only [← c24483, ← c24484, k2, ← c24485] at e_374_17
  have hr := e_374_17
  simp only [e_374_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 374 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 354)) := by
  have c24486 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.1
  have c24487 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.1
  have c24488 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.1
  have c24489 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.1
  have c24490 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24491 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_374_18 := arithEq_of_rows h (row := 374) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_18
  simp only [← c24486, ← c24487, ← c24488] at e_374_18
  have e_374_19 := arithEq_of_rows h (row := 374) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_19
  simp only [← c24489, ← c24490, k2, ← c24491] at e_374_19
  have hr := e_374_19
  simp only [e_374_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 355)) := by
  have c24492 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24493 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24494 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24495 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24496 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24497 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_0 := arithEq_of_rows h (row := 375) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_0
  simp only [← c24492, ← c24493, ← c24494] at e_375_0
  have e_375_1 := arithEq_of_rows h (row := 375) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_1
  simp only [← c24495, ← c24496, k2, ← c24497] at e_375_1
  have hr := e_375_1
  simp only [e_375_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 356)) := by
  have c24498 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24499 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24500 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24501 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24502 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24503 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_2 := arithEq_of_rows h (row := 375) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_2
  simp only [← c24498, ← c24499, ← c24500] at e_375_2
  have e_375_3 := arithEq_of_rows h (row := 375) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_3
  simp only [← c24501, ← c24502, k2, ← c24503] at e_375_3
  have hr := e_375_3
  simp only [e_375_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 357)) := by
  have c24504 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24505 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24506 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24507 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24508 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24509 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_4 := arithEq_of_rows h (row := 375) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_4
  simp only [← c24504, ← c24505, ← c24506] at e_375_4
  have e_375_5 := arithEq_of_rows h (row := 375) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_5
  simp only [← c24507, ← c24508, k2, ← c24509] at e_375_5
  have hr := e_375_5
  simp only [e_375_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 358)) := by
  have c24510 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24511 := (publicBatchWrapper89_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24512 := (publicBatchWrapper89_copies766 a h).1
  have c24513 := (publicBatchWrapper89_copies766 a h).2.1
  have c24514 := (publicBatchWrapper89_copies766 a h).2.2.1
  have c24515 := (publicBatchWrapper89_copies766 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_6 := arithEq_of_rows h (row := 375) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_6
  simp only [← c24510, ← c24511, ← c24512] at e_375_6
  have e_375_7 := arithEq_of_rows h (row := 375) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_7
  simp only [← c24513, ← c24514, k2, ← c24515] at e_375_7
  have hr := e_375_7
  simp only [e_375_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 359)) := by
  have c24516 := (publicBatchWrapper89_copies766 a h).2.2.2.2.1
  have c24517 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.1
  have c24518 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.1
  have c24519 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.1
  have c24520 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.1
  have c24521 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_8 := arithEq_of_rows h (row := 375) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_8
  simp only [← c24516, ← c24517, ← c24518] at e_375_8
  have e_375_9 := arithEq_of_rows h (row := 375) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_9
  simp only [← c24519, ← c24520, k2, ← c24521] at e_375_9
  have hr := e_375_9
  simp only [e_375_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 360)) := by
  have c24522 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24523 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24524 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24525 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24526 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24527 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_10 := arithEq_of_rows h (row := 375) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_10
  simp only [← c24522, ← c24523, ← c24524] at e_375_10
  have e_375_11 := arithEq_of_rows h (row := 375) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_11
  simp only [← c24525, ← c24526, k2, ← c24527] at e_375_11
  have hr := e_375_11
  simp only [e_375_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 361)) := by
  have c24528 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24529 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24530 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24531 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24532 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24533 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_12 := arithEq_of_rows h (row := 375) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_12
  simp only [← c24528, ← c24529, ← c24530] at e_375_12
  have e_375_13 := arithEq_of_rows h (row := 375) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_13
  simp only [← c24531, ← c24532, k2, ← c24533] at e_375_13
  have hr := e_375_13
  simp only [e_375_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 362)) := by
  have c24534 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24535 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24536 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24537 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24538 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24539 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_14 := arithEq_of_rows h (row := 375) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_14
  simp only [← c24534, ← c24535, ← c24536] at e_375_14
  have e_375_15 := arithEq_of_rows h (row := 375) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_15
  simp only [← c24537, ← c24538, k2, ← c24539] at e_375_15
  have hr := e_375_15
  simp only [e_375_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 363)) := by
  have c24540 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24541 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24542 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24543 := (publicBatchWrapper89_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24544 := (publicBatchWrapper89_copies767 a h).1
  have c24545 := (publicBatchWrapper89_copies767 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_16 := arithEq_of_rows h (row := 375) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_16
  simp only [← c24540, ← c24541, ← c24542] at e_375_16
  have e_375_17 := arithEq_of_rows h (row := 375) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_17
  simp only [← c24543, ← c24544, k2, ← c24545] at e_375_17
  have hr := e_375_17
  simp only [e_375_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 375 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 364)) := by
  have c24546 := (publicBatchWrapper89_copies767 a h).2.2.1
  have c24547 := (publicBatchWrapper89_copies767 a h).2.2.2.1
  have c24548 := (publicBatchWrapper89_copies767 a h).2.2.2.2.1
  have c24549 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.1
  have c24550 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.1
  have c24551 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_375_18 := arithEq_of_rows h (row := 375) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_18
  simp only [← c24546, ← c24547, ← c24548] at e_375_18
  have e_375_19 := arithEq_of_rows h (row := 375) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_19
  simp only [← c24549, ← c24550, k2, ← c24551] at e_375_19
  have hr := e_375_19
  simp only [e_375_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 365)) := by
  have c24552 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.1
  have c24553 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.1
  have c24554 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24555 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24556 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24557 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_0 := arithEq_of_rows h (row := 376) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_0
  simp only [← c24552, ← c24553, ← c24554] at e_376_0
  have e_376_1 := arithEq_of_rows h (row := 376) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_1
  simp only [← c24555, ← c24556, k2, ← c24557] at e_376_1
  have hr := e_376_1
  simp only [e_376_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 366)) := by
  have c24558 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24559 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24560 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24561 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24562 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24563 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_2 := arithEq_of_rows h (row := 376) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_2
  simp only [← c24558, ← c24559, ← c24560] at e_376_2
  have e_376_3 := arithEq_of_rows h (row := 376) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_3
  simp only [← c24561, ← c24562, k2, ← c24563] at e_376_3
  have hr := e_376_3
  simp only [e_376_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 367)) := by
  have c24564 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24565 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24566 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24567 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24568 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24569 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_4 := arithEq_of_rows h (row := 376) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_4
  simp only [← c24564, ← c24565, ← c24566] at e_376_4
  have e_376_5 := arithEq_of_rows h (row := 376) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_5
  simp only [← c24567, ← c24568, k2, ← c24569] at e_376_5
  have hr := e_376_5
  simp only [e_376_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 368)) := by
  have c24570 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24571 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24572 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24573 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24574 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24575 := (publicBatchWrapper89_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_6 := arithEq_of_rows h (row := 376) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_6
  simp only [← c24570, ← c24571, ← c24572] at e_376_6
  have e_376_7 := arithEq_of_rows h (row := 376) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_7
  simp only [← c24573, ← c24574, k2, ← c24575] at e_376_7
  have hr := e_376_7
  simp only [e_376_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 369)) := by
  have c24576 := (publicBatchWrapper89_copies768 a h).1
  have c24577 := (publicBatchWrapper89_copies768 a h).2.1
  have c24578 := (publicBatchWrapper89_copies768 a h).2.2.1
  have c24579 := (publicBatchWrapper89_copies768 a h).2.2.2.1
  have c24580 := (publicBatchWrapper89_copies768 a h).2.2.2.2.1
  have c24581 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_8 := arithEq_of_rows h (row := 376) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_8
  simp only [← c24576, ← c24577, ← c24578] at e_376_8
  have e_376_9 := arithEq_of_rows h (row := 376) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_9
  simp only [← c24579, ← c24580, k2, ← c24581] at e_376_9
  have hr := e_376_9
  simp only [e_376_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 410)) := by
  have c24582 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.1
  have c24583 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.1
  have c24584 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.1
  have c24585 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.1
  have c24586 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24587 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_10 := arithEq_of_rows h (row := 376) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_10
  simp only [← c24582, ← c24583, ← c24584] at e_376_10
  have e_376_11 := arithEq_of_rows h (row := 376) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_11
  simp only [← c24585, ← c24586, k2, ← c24587] at e_376_11
  have hr := e_376_11
  simp only [e_376_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 411)) := by
  have c24588 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24589 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24590 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24591 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24592 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24593 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_12 := arithEq_of_rows h (row := 376) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_12
  simp only [← c24588, ← c24589, ← c24590] at e_376_12
  have e_376_13 := arithEq_of_rows h (row := 376) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_13
  simp only [← c24591, ← c24592, k2, ← c24593] at e_376_13
  have hr := e_376_13
  simp only [e_376_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 412)) := by
  have c24594 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24595 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24596 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24597 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24598 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24599 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_14 := arithEq_of_rows h (row := 376) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_14
  simp only [← c24594, ← c24595, ← c24596] at e_376_14
  have e_376_15 := arithEq_of_rows h (row := 376) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_15
  simp only [← c24597, ← c24598, k2, ← c24599] at e_376_15
  have hr := e_376_15
  simp only [e_376_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 413)) := by
  have c24600 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24601 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24602 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24603 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24604 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24605 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_16 := arithEq_of_rows h (row := 376) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_16
  simp only [← c24600, ← c24601, ← c24602] at e_376_16
  have e_376_17 := arithEq_of_rows h (row := 376) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_17
  simp only [← c24603, ← c24604, k2, ← c24605] at e_376_17
  have hr := e_376_17
  simp only [e_376_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 376 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 414)) := by
  have c24606 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24607 := (publicBatchWrapper89_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24608 := (publicBatchWrapper89_copies769 a h).1
  have c24609 := (publicBatchWrapper89_copies769 a h).2.1
  have c24610 := (publicBatchWrapper89_copies769 a h).2.2.1
  have c24611 := (publicBatchWrapper89_copies769 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_376_18 := arithEq_of_rows h (row := 376) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_18
  simp only [← c24606, ← c24607, ← c24608] at e_376_18
  have e_376_19 := arithEq_of_rows h (row := 376) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_19
  simp only [← c24609, ← c24610, k2, ← c24611] at e_376_19
  have hr := e_376_19
  simp only [e_376_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 415)) := by
  have c24612 := (publicBatchWrapper89_copies769 a h).2.2.2.2.1
  have c24613 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.1
  have c24614 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.1
  have c24615 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.1
  have c24616 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.1
  have c24617 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_0 := arithEq_of_rows h (row := 377) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_0
  simp only [← c24612, ← c24613, ← c24614] at e_377_0
  have e_377_1 := arithEq_of_rows h (row := 377) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_1
  simp only [← c24615, ← c24616, k2, ← c24617] at e_377_1
  have hr := e_377_1
  simp only [e_377_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 416)) := by
  have c24618 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24619 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24620 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24621 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24622 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24623 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_2 := arithEq_of_rows h (row := 377) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_2
  simp only [← c24618, ← c24619, ← c24620] at e_377_2
  have e_377_3 := arithEq_of_rows h (row := 377) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_3
  simp only [← c24621, ← c24622, k2, ← c24623] at e_377_3
  have hr := e_377_3
  simp only [e_377_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 417)) := by
  have c24624 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24625 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24626 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24627 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24628 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24629 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_4 := arithEq_of_rows h (row := 377) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_4
  simp only [← c24624, ← c24625, ← c24626] at e_377_4
  have e_377_5 := arithEq_of_rows h (row := 377) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_5
  simp only [← c24627, ← c24628, k2, ← c24629] at e_377_5
  have hr := e_377_5
  simp only [e_377_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 418)) := by
  have c24630 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24631 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24632 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24633 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24634 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24635 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_6 := arithEq_of_rows h (row := 377) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_6
  simp only [← c24630, ← c24631, ← c24632] at e_377_6
  have e_377_7 := arithEq_of_rows h (row := 377) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_7
  simp only [← c24633, ← c24634, k2, ← c24635] at e_377_7
  have hr := e_377_7
  simp only [e_377_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 419)) := by
  have c24636 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24637 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24638 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24639 := (publicBatchWrapper89_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24640 := (publicBatchWrapper89_copies770 a h).1
  have c24641 := (publicBatchWrapper89_copies770 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_8 := arithEq_of_rows h (row := 377) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_8
  simp only [← c24636, ← c24637, ← c24638] at e_377_8
  have e_377_9 := arithEq_of_rows h (row := 377) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_9
  simp only [← c24639, ← c24640, k2, ← c24641] at e_377_9
  have hr := e_377_9
  simp only [e_377_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 420)) := by
  have c24642 := (publicBatchWrapper89_copies770 a h).2.2.1
  have c24643 := (publicBatchWrapper89_copies770 a h).2.2.2.1
  have c24644 := (publicBatchWrapper89_copies770 a h).2.2.2.2.1
  have c24645 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.1
  have c24646 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.1
  have c24647 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_10 := arithEq_of_rows h (row := 377) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_10
  simp only [← c24642, ← c24643, ← c24644] at e_377_10
  have e_377_11 := arithEq_of_rows h (row := 377) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_11
  simp only [← c24645, ← c24646, k2, ← c24647] at e_377_11
  have hr := e_377_11
  simp only [e_377_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 421)) := by
  have c24648 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.1
  have c24649 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.1
  have c24650 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24651 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24652 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24653 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_12 := arithEq_of_rows h (row := 377) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_12
  simp only [← c24648, ← c24649, ← c24650] at e_377_12
  have e_377_13 := arithEq_of_rows h (row := 377) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_13
  simp only [← c24651, ← c24652, k2, ← c24653] at e_377_13
  have hr := e_377_13
  simp only [e_377_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 422)) := by
  have c24654 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24655 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24656 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24657 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24658 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24659 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_14 := arithEq_of_rows h (row := 377) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_14
  simp only [← c24654, ← c24655, ← c24656] at e_377_14
  have e_377_15 := arithEq_of_rows h (row := 377) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_15
  simp only [← c24657, ← c24658, k2, ← c24659] at e_377_15
  have hr := e_377_15
  simp only [e_377_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 423)) := by
  have c24660 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24661 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24662 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24663 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24664 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24665 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_16 := arithEq_of_rows h (row := 377) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_16
  simp only [← c24660, ← c24661, ← c24662] at e_377_16
  have e_377_17 := arithEq_of_rows h (row := 377) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_17
  simp only [← c24663, ← c24664, k2, ← c24665] at e_377_17
  have hr := e_377_17
  simp only [e_377_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 377 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 424)) := by
  have c24666 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24667 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24668 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24669 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24670 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24671 := (publicBatchWrapper89_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_377_18 := arithEq_of_rows h (row := 377) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_18
  simp only [← c24666, ← c24667, ← c24668] at e_377_18
  have e_377_19 := arithEq_of_rows h (row := 377) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_19
  simp only [← c24669, ← c24670, k2, ← c24671] at e_377_19
  have hr := e_377_19
  simp only [e_377_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 425)) := by
  have c24672 := (publicBatchWrapper89_copies771 a h).1
  have c24673 := (publicBatchWrapper89_copies771 a h).2.1
  have c24674 := (publicBatchWrapper89_copies771 a h).2.2.1
  have c24675 := (publicBatchWrapper89_copies771 a h).2.2.2.1
  have c24676 := (publicBatchWrapper89_copies771 a h).2.2.2.2.1
  have c24677 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_0 := arithEq_of_rows h (row := 378) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_0
  simp only [← c24672, ← c24673, ← c24674] at e_378_0
  have e_378_1 := arithEq_of_rows h (row := 378) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_1
  simp only [← c24675, ← c24676, k2, ← c24677] at e_378_1
  have hr := e_378_1
  simp only [e_378_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 426)) := by
  have c24678 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.1
  have c24679 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.1
  have c24680 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.1
  have c24681 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.1
  have c24682 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24683 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_2 := arithEq_of_rows h (row := 378) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_2
  simp only [← c24678, ← c24679, ← c24680] at e_378_2
  have e_378_3 := arithEq_of_rows h (row := 378) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_3
  simp only [← c24681, ← c24682, k2, ← c24683] at e_378_3
  have hr := e_378_3
  simp only [e_378_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 427)) := by
  have c24684 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24685 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24686 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24687 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24688 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24689 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_4 := arithEq_of_rows h (row := 378) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_4
  simp only [← c24684, ← c24685, ← c24686] at e_378_4
  have e_378_5 := arithEq_of_rows h (row := 378) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_5
  simp only [← c24687, ← c24688, k2, ← c24689] at e_378_5
  have hr := e_378_5
  simp only [e_378_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 428)) := by
  have c24690 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24691 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24692 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24693 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24694 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24695 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_6 := arithEq_of_rows h (row := 378) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_6
  simp only [← c24690, ← c24691, ← c24692] at e_378_6
  have e_378_7 := arithEq_of_rows h (row := 378) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_7
  simp only [← c24693, ← c24694, k2, ← c24695] at e_378_7
  have hr := e_378_7
  simp only [e_378_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 429)) := by
  have c24696 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24697 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24698 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24699 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24700 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24701 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_8 := arithEq_of_rows h (row := 378) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_8
  simp only [← c24696, ← c24697, ← c24698] at e_378_8
  have e_378_9 := arithEq_of_rows h (row := 378) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_9
  simp only [← c24699, ← c24700, k2, ← c24701] at e_378_9
  have hr := e_378_9
  simp only [e_378_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 430)) := by
  have c24702 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24703 := (publicBatchWrapper89_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24704 := (publicBatchWrapper89_copies772 a h).1
  have c24705 := (publicBatchWrapper89_copies772 a h).2.1
  have c24706 := (publicBatchWrapper89_copies772 a h).2.2.1
  have c24707 := (publicBatchWrapper89_copies772 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_10 := arithEq_of_rows h (row := 378) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_10
  simp only [← c24702, ← c24703, ← c24704] at e_378_10
  have e_378_11 := arithEq_of_rows h (row := 378) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_11
  simp only [← c24705, ← c24706, k2, ← c24707] at e_378_11
  have hr := e_378_11
  simp only [e_378_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 431)) := by
  have c24708 := (publicBatchWrapper89_copies772 a h).2.2.2.2.1
  have c24709 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.1
  have c24710 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.1
  have c24711 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.1
  have c24712 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.1
  have c24713 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_12 := arithEq_of_rows h (row := 378) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_12
  simp only [← c24708, ← c24709, ← c24710] at e_378_12
  have e_378_13 := arithEq_of_rows h (row := 378) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_13
  simp only [← c24711, ← c24712, k2, ← c24713] at e_378_13
  have hr := e_378_13
  simp only [e_378_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 432)) := by
  have c24714 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24715 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24716 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24717 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24718 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24719 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_14 := arithEq_of_rows h (row := 378) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_14
  simp only [← c24714, ← c24715, ← c24716] at e_378_14
  have e_378_15 := arithEq_of_rows h (row := 378) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_15
  simp only [← c24717, ← c24718, k2, ← c24719] at e_378_15
  have hr := e_378_15
  simp only [e_378_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 433)) := by
  have c24720 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24721 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24722 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24723 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24724 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24725 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_16 := arithEq_of_rows h (row := 378) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_16
  simp only [← c24720, ← c24721, ← c24722] at e_378_16
  have e_378_17 := arithEq_of_rows h (row := 378) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_17
  simp only [← c24723, ← c24724, k2, ← c24725] at e_378_17
  have hr := e_378_17
  simp only [e_378_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 378 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 434)) := by
  have c24726 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24727 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24728 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24729 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24730 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24731 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_378_18 := arithEq_of_rows h (row := 378) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_18
  simp only [← c24726, ← c24727, ← c24728] at e_378_18
  have e_378_19 := arithEq_of_rows h (row := 378) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_19
  simp only [← c24729, ← c24730, k2, ← c24731] at e_378_19
  have hr := e_378_19
  simp only [e_378_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 435)) := by
  have c24732 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24733 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24734 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24735 := (publicBatchWrapper89_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24736 := (publicBatchWrapper89_copies773 a h).1
  have c24737 := (publicBatchWrapper89_copies773 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_0 := arithEq_of_rows h (row := 379) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_0
  simp only [← c24732, ← c24733, ← c24734] at e_379_0
  have e_379_1 := arithEq_of_rows h (row := 379) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_1
  simp only [← c24735, ← c24736, k2, ← c24737] at e_379_1
  have hr := e_379_1
  simp only [e_379_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 436)) := by
  have c24738 := (publicBatchWrapper89_copies773 a h).2.2.1
  have c24739 := (publicBatchWrapper89_copies773 a h).2.2.2.1
  have c24740 := (publicBatchWrapper89_copies773 a h).2.2.2.2.1
  have c24741 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.1
  have c24742 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.1
  have c24743 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_2 := arithEq_of_rows h (row := 379) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_2
  simp only [← c24738, ← c24739, ← c24740] at e_379_2
  have e_379_3 := arithEq_of_rows h (row := 379) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_3
  simp only [← c24741, ← c24742, k2, ← c24743] at e_379_3
  have hr := e_379_3
  simp only [e_379_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 437)) := by
  have c24744 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.1
  have c24745 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.1
  have c24746 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24747 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24748 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24749 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_4 := arithEq_of_rows h (row := 379) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_4
  simp only [← c24744, ← c24745, ← c24746] at e_379_4
  have e_379_5 := arithEq_of_rows h (row := 379) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_5
  simp only [← c24747, ← c24748, k2, ← c24749] at e_379_5
  have hr := e_379_5
  simp only [e_379_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 438)) := by
  have c24750 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24751 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24752 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24753 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24754 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24755 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_6 := arithEq_of_rows h (row := 379) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_6
  simp only [← c24750, ← c24751, ← c24752] at e_379_6
  have e_379_7 := arithEq_of_rows h (row := 379) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_7
  simp only [← c24753, ← c24754, k2, ← c24755] at e_379_7
  have hr := e_379_7
  simp only [e_379_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 439)) := by
  have c24756 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24757 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24758 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24759 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24760 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24761 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_8 := arithEq_of_rows h (row := 379) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_8
  simp only [← c24756, ← c24757, ← c24758] at e_379_8
  have e_379_9 := arithEq_of_rows h (row := 379) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_9
  simp only [← c24759, ← c24760, k2, ← c24761] at e_379_9
  have hr := e_379_9
  simp only [e_379_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 440)) := by
  have c24762 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24763 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24764 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24765 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24766 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24767 := (publicBatchWrapper89_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_10 := arithEq_of_rows h (row := 379) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_10
  simp only [← c24762, ← c24763, ← c24764] at e_379_10
  have e_379_11 := arithEq_of_rows h (row := 379) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_11
  simp only [← c24765, ← c24766, k2, ← c24767] at e_379_11
  have hr := e_379_11
  simp only [e_379_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 441)) := by
  have c24768 := (publicBatchWrapper89_copies774 a h).1
  have c24769 := (publicBatchWrapper89_copies774 a h).2.1
  have c24770 := (publicBatchWrapper89_copies774 a h).2.2.1
  have c24771 := (publicBatchWrapper89_copies774 a h).2.2.2.1
  have c24772 := (publicBatchWrapper89_copies774 a h).2.2.2.2.1
  have c24773 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_12 := arithEq_of_rows h (row := 379) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_12
  simp only [← c24768, ← c24769, ← c24770] at e_379_12
  have e_379_13 := arithEq_of_rows h (row := 379) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_13
  simp only [← c24771, ← c24772, k2, ← c24773] at e_379_13
  have hr := e_379_13
  simp only [e_379_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 442)) := by
  have c24774 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.1
  have c24775 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.1
  have c24776 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.1
  have c24777 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.1
  have c24778 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24779 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_14 := arithEq_of_rows h (row := 379) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_14
  simp only [← c24774, ← c24775, ← c24776] at e_379_14
  have e_379_15 := arithEq_of_rows h (row := 379) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_15
  simp only [← c24777, ← c24778, k2, ← c24779] at e_379_15
  have hr := e_379_15
  simp only [e_379_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 443)) := by
  have c24780 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24781 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24782 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24783 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24784 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24785 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_16 := arithEq_of_rows h (row := 379) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_16
  simp only [← c24780, ← c24781, ← c24782] at e_379_16
  have e_379_17 := arithEq_of_rows h (row := 379) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_17
  simp only [← c24783, ← c24784, k2, ← c24785] at e_379_17
  have hr := e_379_17
  simp only [e_379_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 379 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 444)) := by
  have c24786 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24787 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24788 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24789 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24790 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24791 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_379_18 := arithEq_of_rows h (row := 379) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_18
  simp only [← c24786, ← c24787, ← c24788] at e_379_18
  have e_379_19 := arithEq_of_rows h (row := 379) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_19
  simp only [← c24789, ← c24790, k2, ← c24791] at e_379_19
  have hr := e_379_19
  simp only [e_379_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 445)) := by
  have c24792 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24793 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24794 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24795 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24796 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24797 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_0 := arithEq_of_rows h (row := 380) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_0
  simp only [← c24792, ← c24793, ← c24794] at e_380_0
  have e_380_1 := arithEq_of_rows h (row := 380) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_1
  simp only [← c24795, ← c24796, k2, ← c24797] at e_380_1
  have hr := e_380_1
  simp only [e_380_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 446)) := by
  have c24798 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24799 := (publicBatchWrapper89_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24800 := (publicBatchWrapper89_copies775 a h).1
  have c24801 := (publicBatchWrapper89_copies775 a h).2.1
  have c24802 := (publicBatchWrapper89_copies775 a h).2.2.1
  have c24803 := (publicBatchWrapper89_copies775 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_2 := arithEq_of_rows h (row := 380) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_2
  simp only [← c24798, ← c24799, ← c24800] at e_380_2
  have e_380_3 := arithEq_of_rows h (row := 380) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_3
  simp only [← c24801, ← c24802, k2, ← c24803] at e_380_3
  have hr := e_380_3
  simp only [e_380_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 447)) := by
  have c24804 := (publicBatchWrapper89_copies775 a h).2.2.2.2.1
  have c24805 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.1
  have c24806 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.1
  have c24807 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.1
  have c24808 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.1
  have c24809 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_4 := arithEq_of_rows h (row := 380) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_4
  simp only [← c24804, ← c24805, ← c24806] at e_380_4
  have e_380_5 := arithEq_of_rows h (row := 380) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_5
  simp only [← c24807, ← c24808, k2, ← c24809] at e_380_5
  have hr := e_380_5
  simp only [e_380_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 448)) := by
  have c24810 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24811 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24812 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24813 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24814 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24815 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_6 := arithEq_of_rows h (row := 380) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_6
  simp only [← c24810, ← c24811, ← c24812] at e_380_6
  have e_380_7 := arithEq_of_rows h (row := 380) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_7
  simp only [← c24813, ← c24814, k2, ← c24815] at e_380_7
  have hr := e_380_7
  simp only [e_380_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 449)) := by
  have c24816 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24817 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24818 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24819 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24820 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24821 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_8 := arithEq_of_rows h (row := 380) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_8
  simp only [← c24816, ← c24817, ← c24818] at e_380_8
  have e_380_9 := arithEq_of_rows h (row := 380) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_9
  simp only [← c24819, ← c24820, k2, ← c24821] at e_380_9
  have hr := e_380_9
  simp only [e_380_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 450)) := by
  have c24822 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24823 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24824 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24825 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24826 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24827 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_10 := arithEq_of_rows h (row := 380) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_10
  simp only [← c24822, ← c24823, ← c24824] at e_380_10
  have e_380_11 := arithEq_of_rows h (row := 380) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_11
  simp only [← c24825, ← c24826, k2, ← c24827] at e_380_11
  have hr := e_380_11
  simp only [e_380_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 451)) := by
  have c24828 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24829 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24830 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24831 := (publicBatchWrapper89_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24832 := (publicBatchWrapper89_copies776 a h).1
  have c24833 := (publicBatchWrapper89_copies776 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_12 := arithEq_of_rows h (row := 380) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_12
  simp only [← c24828, ← c24829, ← c24830] at e_380_12
  have e_380_13 := arithEq_of_rows h (row := 380) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_13
  simp only [← c24831, ← c24832, k2, ← c24833] at e_380_13
  have hr := e_380_13
  simp only [e_380_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 452)) := by
  have c24834 := (publicBatchWrapper89_copies776 a h).2.2.1
  have c24835 := (publicBatchWrapper89_copies776 a h).2.2.2.1
  have c24836 := (publicBatchWrapper89_copies776 a h).2.2.2.2.1
  have c24837 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.1
  have c24838 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.1
  have c24839 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_14 := arithEq_of_rows h (row := 380) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_14
  simp only [← c24834, ← c24835, ← c24836] at e_380_14
  have e_380_15 := arithEq_of_rows h (row := 380) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_15
  simp only [← c24837, ← c24838, k2, ← c24839] at e_380_15
  have hr := e_380_15
  simp only [e_380_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 453)) := by
  have c24840 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.1
  have c24841 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.1
  have c24842 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24843 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24844 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24845 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_16 := arithEq_of_rows h (row := 380) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_16
  simp only [← c24840, ← c24841, ← c24842] at e_380_16
  have e_380_17 := arithEq_of_rows h (row := 380) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_17
  simp only [← c24843, ← c24844, k2, ← c24845] at e_380_17
  have hr := e_380_17
  simp only [e_380_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 380 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 454)) := by
  have c24846 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24847 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24848 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24849 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24850 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24851 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_380_18 := arithEq_of_rows h (row := 380) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_18
  simp only [← c24846, ← c24847, ← c24848] at e_380_18
  have e_380_19 := arithEq_of_rows h (row := 380) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_19
  simp only [← c24849, ← c24850, k2, ← c24851] at e_380_19
  have hr := e_380_19
  simp only [e_380_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 455)) := by
  have c24852 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24853 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24854 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24855 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24856 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24857 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_0 := arithEq_of_rows h (row := 381) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_0
  simp only [← c24852, ← c24853, ← c24854] at e_381_0
  have e_381_1 := arithEq_of_rows h (row := 381) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_1
  simp only [← c24855, ← c24856, k2, ← c24857] at e_381_1
  have hr := e_381_1
  simp only [e_381_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 456)) := by
  have c24858 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24859 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24860 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24861 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24862 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24863 := (publicBatchWrapper89_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_2 := arithEq_of_rows h (row := 381) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_2
  simp only [← c24858, ← c24859, ← c24860] at e_381_2
  have e_381_3 := arithEq_of_rows h (row := 381) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_3
  simp only [← c24861, ← c24862, k2, ← c24863] at e_381_3
  have hr := e_381_3
  simp only [e_381_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 457)) := by
  have c24864 := (publicBatchWrapper89_copies777 a h).1
  have c24865 := (publicBatchWrapper89_copies777 a h).2.1
  have c24866 := (publicBatchWrapper89_copies777 a h).2.2.1
  have c24867 := (publicBatchWrapper89_copies777 a h).2.2.2.1
  have c24868 := (publicBatchWrapper89_copies777 a h).2.2.2.2.1
  have c24869 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_4 := arithEq_of_rows h (row := 381) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_4
  simp only [← c24864, ← c24865, ← c24866] at e_381_4
  have e_381_5 := arithEq_of_rows h (row := 381) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_5
  simp only [← c24867, ← c24868, k2, ← c24869] at e_381_5
  have hr := e_381_5
  simp only [e_381_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 458)) := by
  have c24870 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.1
  have c24871 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.1
  have c24872 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.1
  have c24873 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.1
  have c24874 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24875 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_6 := arithEq_of_rows h (row := 381) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_6
  simp only [← c24870, ← c24871, ← c24872] at e_381_6
  have e_381_7 := arithEq_of_rows h (row := 381) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_7
  simp only [← c24873, ← c24874, k2, ← c24875] at e_381_7
  have hr := e_381_7
  simp only [e_381_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 459)) := by
  have c24876 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24877 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24878 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24879 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24880 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24881 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_8 := arithEq_of_rows h (row := 381) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_8
  simp only [← c24876, ← c24877, ← c24878] at e_381_8
  have e_381_9 := arithEq_of_rows h (row := 381) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_9
  simp only [← c24879, ← c24880, k2, ← c24881] at e_381_9
  have hr := e_381_9
  simp only [e_381_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 460)) := by
  have c24882 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24883 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24884 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24885 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24886 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24887 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_10 := arithEq_of_rows h (row := 381) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_10
  simp only [← c24882, ← c24883, ← c24884] at e_381_10
  have e_381_11 := arithEq_of_rows h (row := 381) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_11
  simp only [← c24885, ← c24886, k2, ← c24887] at e_381_11
  have hr := e_381_11
  simp only [e_381_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 461)) := by
  have c24888 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24889 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24890 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24891 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24892 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24893 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_12 := arithEq_of_rows h (row := 381) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_12
  simp only [← c24888, ← c24889, ← c24890] at e_381_12
  have e_381_13 := arithEq_of_rows h (row := 381) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_13
  simp only [← c24891, ← c24892, k2, ← c24893] at e_381_13
  have hr := e_381_13
  simp only [e_381_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 462)) := by
  have c24894 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24895 := (publicBatchWrapper89_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24896 := (publicBatchWrapper89_copies778 a h).1
  have c24897 := (publicBatchWrapper89_copies778 a h).2.1
  have c24898 := (publicBatchWrapper89_copies778 a h).2.2.1
  have c24899 := (publicBatchWrapper89_copies778 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_14 := arithEq_of_rows h (row := 381) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_14
  simp only [← c24894, ← c24895, ← c24896] at e_381_14
  have e_381_15 := arithEq_of_rows h (row := 381) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_15
  simp only [← c24897, ← c24898, k2, ← c24899] at e_381_15
  have hr := e_381_15
  simp only [e_381_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 463)) := by
  have c24900 := (publicBatchWrapper89_copies778 a h).2.2.2.2.1
  have c24901 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.1
  have c24902 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.1
  have c24903 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.1
  have c24904 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.1
  have c24905 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_16 := arithEq_of_rows h (row := 381) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_16
  simp only [← c24900, ← c24901, ← c24902] at e_381_16
  have e_381_17 := arithEq_of_rows h (row := 381) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_17
  simp only [← c24903, ← c24904, k2, ← c24905] at e_381_17
  have hr := e_381_17
  simp only [e_381_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 381 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 464)) := by
  have c24906 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24907 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24908 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24909 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24910 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24911 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_381_18 := arithEq_of_rows h (row := 381) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_18
  simp only [← c24906, ← c24907, ← c24908] at e_381_18
  have e_381_19 := arithEq_of_rows h (row := 381) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_19
  simp only [← c24909, ← c24910, k2, ← c24911] at e_381_19
  have hr := e_381_19
  simp only [e_381_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 465)) := by
  have c24912 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24913 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24914 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24915 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24916 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24917 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_0 := arithEq_of_rows h (row := 382) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_0
  simp only [← c24912, ← c24913, ← c24914] at e_382_0
  have e_382_1 := arithEq_of_rows h (row := 382) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_1
  simp only [← c24915, ← c24916, k2, ← c24917] at e_382_1
  have hr := e_382_1
  simp only [e_382_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 466)) := by
  have c24918 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24919 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24920 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24921 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24922 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24923 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_2 := arithEq_of_rows h (row := 382) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_2
  simp only [← c24918, ← c24919, ← c24920] at e_382_2
  have e_382_3 := arithEq_of_rows h (row := 382) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_3
  simp only [← c24921, ← c24922, k2, ← c24923] at e_382_3
  have hr := e_382_3
  simp only [e_382_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 467)) := by
  have c24924 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24925 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24926 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24927 := (publicBatchWrapper89_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24928 := (publicBatchWrapper89_copies779 a h).1
  have c24929 := (publicBatchWrapper89_copies779 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_4 := arithEq_of_rows h (row := 382) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_4
  simp only [← c24924, ← c24925, ← c24926] at e_382_4
  have e_382_5 := arithEq_of_rows h (row := 382) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_5
  simp only [← c24927, ← c24928, k2, ← c24929] at e_382_5
  have hr := e_382_5
  simp only [e_382_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 468)) := by
  have c24930 := (publicBatchWrapper89_copies779 a h).2.2.1
  have c24931 := (publicBatchWrapper89_copies779 a h).2.2.2.1
  have c24932 := (publicBatchWrapper89_copies779 a h).2.2.2.2.1
  have c24933 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.1
  have c24934 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.1
  have c24935 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_6 := arithEq_of_rows h (row := 382) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_6
  simp only [← c24930, ← c24931, ← c24932] at e_382_6
  have e_382_7 := arithEq_of_rows h (row := 382) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_7
  simp only [← c24933, ← c24934, k2, ← c24935] at e_382_7
  have hr := e_382_7
  simp only [e_382_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 469)) := by
  have c24936 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.1
  have c24937 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.1
  have c24938 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24939 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24940 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24941 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_8 := arithEq_of_rows h (row := 382) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_8
  simp only [← c24936, ← c24937, ← c24938] at e_382_8
  have e_382_9 := arithEq_of_rows h (row := 382) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_9
  simp only [← c24939, ← c24940, k2, ← c24941] at e_382_9
  have hr := e_382_9
  simp only [e_382_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 470)) := by
  have c24942 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24943 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24944 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24945 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24946 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24947 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_10 := arithEq_of_rows h (row := 382) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_10
  simp only [← c24942, ← c24943, ← c24944] at e_382_10
  have e_382_11 := arithEq_of_rows h (row := 382) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_11
  simp only [← c24945, ← c24946, k2, ← c24947] at e_382_11
  have hr := e_382_11
  simp only [e_382_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 471)) := by
  have c24948 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24949 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24950 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24951 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24952 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24953 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_12 := arithEq_of_rows h (row := 382) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_12
  simp only [← c24948, ← c24949, ← c24950] at e_382_12
  have e_382_13 := arithEq_of_rows h (row := 382) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_13
  simp only [← c24951, ← c24952, k2, ← c24953] at e_382_13
  have hr := e_382_13
  simp only [e_382_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 472)) := by
  have c24954 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24955 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24956 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24957 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24958 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24959 := (publicBatchWrapper89_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_14 := arithEq_of_rows h (row := 382) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_14
  simp only [← c24954, ← c24955, ← c24956] at e_382_14
  have e_382_15 := arithEq_of_rows h (row := 382) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_15
  simp only [← c24957, ← c24958, k2, ← c24959] at e_382_15
  have hr := e_382_15
  simp only [e_382_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 473)) := by
  have c24960 := (publicBatchWrapper89_copies780 a h).1
  have c24961 := (publicBatchWrapper89_copies780 a h).2.1
  have c24962 := (publicBatchWrapper89_copies780 a h).2.2.1
  have c24963 := (publicBatchWrapper89_copies780 a h).2.2.2.1
  have c24964 := (publicBatchWrapper89_copies780 a h).2.2.2.2.1
  have c24965 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_16 := arithEq_of_rows h (row := 382) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_16
  simp only [← c24960, ← c24961, ← c24962] at e_382_16
  have e_382_17 := arithEq_of_rows h (row := 382) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_17
  simp only [← c24963, ← c24964, k2, ← c24965] at e_382_17
  have hr := e_382_17
  simp only [e_382_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 382 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 474)) := by
  have c24966 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.1
  have c24967 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.1
  have c24968 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.1
  have c24969 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.1
  have c24970 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24971 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_382_18 := arithEq_of_rows h (row := 382) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_18
  simp only [← c24966, ← c24967, ← c24968] at e_382_18
  have e_382_19 := arithEq_of_rows h (row := 382) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_19
  simp only [← c24969, ← c24970, k2, ← c24971] at e_382_19
  have hr := e_382_19
  simp only [e_382_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 475)) := by
  have c24972 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24973 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24974 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24975 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24976 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24977 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_0 := arithEq_of_rows h (row := 383) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_0
  simp only [← c24972, ← c24973, ← c24974] at e_383_0
  have e_383_1 := arithEq_of_rows h (row := 383) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_1
  simp only [← c24975, ← c24976, k2, ← c24977] at e_383_1
  have hr := e_383_1
  simp only [e_383_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 476)) := by
  have c24978 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24979 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24980 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24981 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24982 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24983 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_2 := arithEq_of_rows h (row := 383) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_2
  simp only [← c24978, ← c24979, ← c24980] at e_383_2
  have e_383_3 := arithEq_of_rows h (row := 383) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_3
  simp only [← c24981, ← c24982, k2, ← c24983] at e_383_3
  have hr := e_383_3
  simp only [e_383_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 477)) := by
  have c24984 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24985 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24986 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24987 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24988 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24989 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_4 := arithEq_of_rows h (row := 383) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_4
  simp only [← c24984, ← c24985, ← c24986] at e_383_4
  have e_383_5 := arithEq_of_rows h (row := 383) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_5
  simp only [← c24987, ← c24988, k2, ← c24989] at e_383_5
  have hr := e_383_5
  simp only [e_383_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 478)) := by
  have c24990 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24991 := (publicBatchWrapper89_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24992 := (publicBatchWrapper89_copies781 a h).1
  have c24993 := (publicBatchWrapper89_copies781 a h).2.1
  have c24994 := (publicBatchWrapper89_copies781 a h).2.2.1
  have c24995 := (publicBatchWrapper89_copies781 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_6 := arithEq_of_rows h (row := 383) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_6
  simp only [← c24990, ← c24991, ← c24992] at e_383_6
  have e_383_7 := arithEq_of_rows h (row := 383) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_7
  simp only [← c24993, ← c24994, k2, ← c24995] at e_383_7
  have hr := e_383_7
  simp only [e_383_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 479)) := by
  have c24996 := (publicBatchWrapper89_copies781 a h).2.2.2.2.1
  have c24997 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.1
  have c24998 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.1
  have c24999 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.1
  have c25000 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.1
  have c25001 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_8 := arithEq_of_rows h (row := 383) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_8
  simp only [← c24996, ← c24997, ← c24998] at e_383_8
  have e_383_9 := arithEq_of_rows h (row := 383) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_9
  simp only [← c24999, ← c25000, k2, ← c25001] at e_383_9
  have hr := e_383_9
  simp only [e_383_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 480)) := by
  have c25002 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25003 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25004 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25005 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25006 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25007 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_10 := arithEq_of_rows h (row := 383) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_10
  simp only [← c25002, ← c25003, ← c25004] at e_383_10
  have e_383_11 := arithEq_of_rows h (row := 383) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_11
  simp only [← c25005, ← c25006, k2, ← c25007] at e_383_11
  have hr := e_383_11
  simp only [e_383_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 481)) := by
  have c25008 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25009 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25010 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25011 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25012 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25013 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_12 := arithEq_of_rows h (row := 383) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_12
  simp only [← c25008, ← c25009, ← c25010] at e_383_12
  have e_383_13 := arithEq_of_rows h (row := 383) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_13
  simp only [← c25011, ← c25012, k2, ← c25013] at e_383_13
  have hr := e_383_13
  simp only [e_383_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 482)) := by
  have c25014 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25015 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25016 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25017 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25018 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25019 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_14 := arithEq_of_rows h (row := 383) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_14
  simp only [← c25014, ← c25015, ← c25016] at e_383_14
  have e_383_15 := arithEq_of_rows h (row := 383) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_15
  simp only [← c25017, ← c25018, k2, ← c25019] at e_383_15
  have hr := e_383_15
  simp only [e_383_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 483)) := by
  have c25020 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25021 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25022 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25023 := (publicBatchWrapper89_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25024 := (publicBatchWrapper89_copies782 a h).1
  have c25025 := (publicBatchWrapper89_copies782 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_16 := arithEq_of_rows h (row := 383) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_16
  simp only [← c25020, ← c25021, ← c25022] at e_383_16
  have e_383_17 := arithEq_of_rows h (row := 383) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_17
  simp only [← c25023, ← c25024, k2, ← c25025] at e_383_17
  have hr := e_383_17
  simp only [e_383_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 383 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 484)) := by
  have c25026 := (publicBatchWrapper89_copies782 a h).2.2.1
  have c25027 := (publicBatchWrapper89_copies782 a h).2.2.2.1
  have c25028 := (publicBatchWrapper89_copies782 a h).2.2.2.2.1
  have c25029 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.1
  have c25030 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.1
  have c25031 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_383_18 := arithEq_of_rows h (row := 383) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_18
  simp only [← c25026, ← c25027, ← c25028] at e_383_18
  have e_383_19 := arithEq_of_rows h (row := 383) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_19
  simp only [← c25029, ← c25030, k2, ← c25031] at e_383_19
  have hr := e_383_19
  simp only [e_383_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 485)) := by
  have c25032 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.1
  have c25033 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.1
  have c25034 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25035 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25036 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25037 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_0 := arithEq_of_rows h (row := 384) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_0
  simp only [← c25032, ← c25033, ← c25034] at e_384_0
  have e_384_1 := arithEq_of_rows h (row := 384) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_1
  simp only [← c25035, ← c25036, k2, ← c25037] at e_384_1
  have hr := e_384_1
  simp only [e_384_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 486)) := by
  have c25038 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25039 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25040 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25041 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25042 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25043 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_2 := arithEq_of_rows h (row := 384) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_2
  simp only [← c25038, ← c25039, ← c25040] at e_384_2
  have e_384_3 := arithEq_of_rows h (row := 384) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_3
  simp only [← c25041, ← c25042, k2, ← c25043] at e_384_3
  have hr := e_384_3
  simp only [e_384_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 487)) := by
  have c25044 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25045 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25046 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25047 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25048 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25049 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_4 := arithEq_of_rows h (row := 384) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_4
  simp only [← c25044, ← c25045, ← c25046] at e_384_4
  have e_384_5 := arithEq_of_rows h (row := 384) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_5
  simp only [← c25047, ← c25048, k2, ← c25049] at e_384_5
  have hr := e_384_5
  simp only [e_384_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 488)) := by
  have c25050 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25051 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25052 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25053 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25054 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25055 := (publicBatchWrapper89_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_6 := arithEq_of_rows h (row := 384) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_6
  simp only [← c25050, ← c25051, ← c25052] at e_384_6
  have e_384_7 := arithEq_of_rows h (row := 384) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_7
  simp only [← c25053, ← c25054, k2, ← c25055] at e_384_7
  have hr := e_384_7
  simp only [e_384_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 489)) := by
  have c25056 := (publicBatchWrapper89_copies783 a h).1
  have c25057 := (publicBatchWrapper89_copies783 a h).2.1
  have c25058 := (publicBatchWrapper89_copies783 a h).2.2.1
  have c25059 := (publicBatchWrapper89_copies783 a h).2.2.2.1
  have c25060 := (publicBatchWrapper89_copies783 a h).2.2.2.2.1
  have c25061 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_8 := arithEq_of_rows h (row := 384) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_8
  simp only [← c25056, ← c25057, ← c25058] at e_384_8
  have e_384_9 := arithEq_of_rows h (row := 384) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_9
  simp only [← c25059, ← c25060, k2, ← c25061] at e_384_9
  have hr := e_384_9
  simp only [e_384_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 490)) := by
  have c25062 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.1
  have c25063 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.1
  have c25064 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.1
  have c25065 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.1
  have c25066 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25067 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_10 := arithEq_of_rows h (row := 384) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_10
  simp only [← c25062, ← c25063, ← c25064] at e_384_10
  have e_384_11 := arithEq_of_rows h (row := 384) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_11
  simp only [← c25065, ← c25066, k2, ← c25067] at e_384_11
  have hr := e_384_11
  simp only [e_384_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 491)) := by
  have c25068 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25069 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25070 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25071 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25072 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25073 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_12 := arithEq_of_rows h (row := 384) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_12
  simp only [← c25068, ← c25069, ← c25070] at e_384_12
  have e_384_13 := arithEq_of_rows h (row := 384) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_13
  simp only [← c25071, ← c25072, k2, ← c25073] at e_384_13
  have hr := e_384_13
  simp only [e_384_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 492)) := by
  have c25074 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25075 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25076 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25077 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25078 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25079 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_14 := arithEq_of_rows h (row := 384) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_14
  simp only [← c25074, ← c25075, ← c25076] at e_384_14
  have e_384_15 := arithEq_of_rows h (row := 384) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_15
  simp only [← c25077, ← c25078, k2, ← c25079] at e_384_15
  have hr := e_384_15
  simp only [e_384_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 493)) := by
  have c25080 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25081 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25082 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25083 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25084 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25085 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_16 := arithEq_of_rows h (row := 384) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_16
  simp only [← c25080, ← c25081, ← c25082] at e_384_16
  have e_384_17 := arithEq_of_rows h (row := 384) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_17
  simp only [← c25083, ← c25084, k2, ← c25085] at e_384_17
  have hr := e_384_17
  simp only [e_384_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 384 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 494)) := by
  have c25086 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25087 := (publicBatchWrapper89_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25088 := (publicBatchWrapper89_copies784 a h).1
  have c25089 := (publicBatchWrapper89_copies784 a h).2.1
  have c25090 := (publicBatchWrapper89_copies784 a h).2.2.1
  have c25091 := (publicBatchWrapper89_copies784 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_384_18 := arithEq_of_rows h (row := 384) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_18
  simp only [← c25086, ← c25087, ← c25088] at e_384_18
  have e_384_19 := arithEq_of_rows h (row := 384) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_19
  simp only [← c25089, ← c25090, k2, ← c25091] at e_384_19
  have hr := e_384_19
  simp only [e_384_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 495)) := by
  have c25092 := (publicBatchWrapper89_copies784 a h).2.2.2.2.1
  have c25093 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.1
  have c25094 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.1
  have c25095 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.1
  have c25096 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.1
  have c25097 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_0 := arithEq_of_rows h (row := 385) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_0
  simp only [← c25092, ← c25093, ← c25094] at e_385_0
  have e_385_1 := arithEq_of_rows h (row := 385) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_1
  simp only [← c25095, ← c25096, k2, ← c25097] at e_385_1
  have hr := e_385_1
  simp only [e_385_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 496)) := by
  have c25098 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25099 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25100 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25101 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25102 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25103 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_2 := arithEq_of_rows h (row := 385) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_2
  simp only [← c25098, ← c25099, ← c25100] at e_385_2
  have e_385_3 := arithEq_of_rows h (row := 385) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_3
  simp only [← c25101, ← c25102, k2, ← c25103] at e_385_3
  have hr := e_385_3
  simp only [e_385_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 497)) := by
  have c25104 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25105 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25106 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25107 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25108 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25109 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_4 := arithEq_of_rows h (row := 385) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_4
  simp only [← c25104, ← c25105, ← c25106] at e_385_4
  have e_385_5 := arithEq_of_rows h (row := 385) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_5
  simp only [← c25107, ← c25108, k2, ← c25109] at e_385_5
  have hr := e_385_5
  simp only [e_385_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 498)) := by
  have c25110 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25111 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25112 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25113 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25114 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25115 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_6 := arithEq_of_rows h (row := 385) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_6
  simp only [← c25110, ← c25111, ← c25112] at e_385_6
  have e_385_7 := arithEq_of_rows h (row := 385) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_7
  simp only [← c25113, ← c25114, k2, ← c25115] at e_385_7
  have hr := e_385_7
  simp only [e_385_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 499)) := by
  have c25116 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25117 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25118 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25119 := (publicBatchWrapper89_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25120 := (publicBatchWrapper89_copies785 a h).1
  have c25121 := (publicBatchWrapper89_copies785 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_8 := arithEq_of_rows h (row := 385) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_8
  simp only [← c25116, ← c25117, ← c25118] at e_385_8
  have e_385_9 := arithEq_of_rows h (row := 385) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_9
  simp only [← c25119, ← c25120, k2, ← c25121] at e_385_9
  have hr := e_385_9
  simp only [e_385_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 500)) := by
  have c25122 := (publicBatchWrapper89_copies785 a h).2.2.1
  have c25123 := (publicBatchWrapper89_copies785 a h).2.2.2.1
  have c25124 := (publicBatchWrapper89_copies785 a h).2.2.2.2.1
  have c25125 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.1
  have c25126 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.1
  have c25127 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_10 := arithEq_of_rows h (row := 385) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_10
  simp only [← c25122, ← c25123, ← c25124] at e_385_10
  have e_385_11 := arithEq_of_rows h (row := 385) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_11
  simp only [← c25125, ← c25126, k2, ← c25127] at e_385_11
  have hr := e_385_11
  simp only [e_385_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 501)) := by
  have c25128 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.1
  have c25129 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.1
  have c25130 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25131 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25132 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25133 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_12 := arithEq_of_rows h (row := 385) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_12
  simp only [← c25128, ← c25129, ← c25130] at e_385_12
  have e_385_13 := arithEq_of_rows h (row := 385) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_13
  simp only [← c25131, ← c25132, k2, ← c25133] at e_385_13
  have hr := e_385_13
  simp only [e_385_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 502)) := by
  have c25134 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25135 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25136 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25137 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25138 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25139 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_14 := arithEq_of_rows h (row := 385) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_14
  simp only [← c25134, ← c25135, ← c25136] at e_385_14
  have e_385_15 := arithEq_of_rows h (row := 385) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_15
  simp only [← c25137, ← c25138, k2, ← c25139] at e_385_15
  have hr := e_385_15
  simp only [e_385_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 503)) := by
  have c25140 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25141 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25142 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25143 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25144 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25145 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_16 := arithEq_of_rows h (row := 385) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_16
  simp only [← c25140, ← c25141, ← c25142] at e_385_16
  have e_385_17 := arithEq_of_rows h (row := 385) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_17
  simp only [← c25143, ← c25144, k2, ← c25145] at e_385_17
  have hr := e_385_17
  simp only [e_385_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 385 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 504)) := by
  have c25146 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25147 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25148 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25149 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25150 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25151 := (publicBatchWrapper89_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_385_18 := arithEq_of_rows h (row := 385) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_18
  simp only [← c25146, ← c25147, ← c25148] at e_385_18
  have e_385_19 := arithEq_of_rows h (row := 385) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_19
  simp only [← c25149, ← c25150, k2, ← c25151] at e_385_19
  have hr := e_385_19
  simp only [e_385_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 505)) := by
  have c25152 := (publicBatchWrapper89_copies786 a h).1
  have c25153 := (publicBatchWrapper89_copies786 a h).2.1
  have c25154 := (publicBatchWrapper89_copies786 a h).2.2.1
  have c25155 := (publicBatchWrapper89_copies786 a h).2.2.2.1
  have c25156 := (publicBatchWrapper89_copies786 a h).2.2.2.2.1
  have c25157 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_0 := arithEq_of_rows h (row := 386) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_0
  simp only [← c25152, ← c25153, ← c25154] at e_386_0
  have e_386_1 := arithEq_of_rows h (row := 386) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_1
  simp only [← c25155, ← c25156, k2, ← c25157] at e_386_1
  have hr := e_386_1
  simp only [e_386_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 506)) := by
  have c25158 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.1
  have c25159 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.1
  have c25160 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.1
  have c25161 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.1
  have c25162 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25163 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_2 := arithEq_of_rows h (row := 386) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_2
  simp only [← c25158, ← c25159, ← c25160] at e_386_2
  have e_386_3 := arithEq_of_rows h (row := 386) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_3
  simp only [← c25161, ← c25162, k2, ← c25163] at e_386_3
  have hr := e_386_3
  simp only [e_386_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 507)) := by
  have c25164 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25165 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25166 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25167 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25168 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25169 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_4 := arithEq_of_rows h (row := 386) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_4
  simp only [← c25164, ← c25165, ← c25166] at e_386_4
  have e_386_5 := arithEq_of_rows h (row := 386) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_5
  simp only [← c25167, ← c25168, k2, ← c25169] at e_386_5
  have hr := e_386_5
  simp only [e_386_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 508)) := by
  have c25170 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25171 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25172 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25173 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25174 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25175 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_6 := arithEq_of_rows h (row := 386) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_6
  simp only [← c25170, ← c25171, ← c25172] at e_386_6
  have e_386_7 := arithEq_of_rows h (row := 386) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_7
  simp only [← c25173, ← c25174, k2, ← c25175] at e_386_7
  have hr := e_386_7
  simp only [e_386_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 509)) := by
  have c25176 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25177 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25178 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25179 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25180 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25181 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_8 := arithEq_of_rows h (row := 386) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_8
  simp only [← c25176, ← c25177, ← c25178] at e_386_8
  have e_386_9 := arithEq_of_rows h (row := 386) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_9
  simp only [← c25179, ← c25180, k2, ← c25181] at e_386_9
  have hr := e_386_9
  simp only [e_386_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 510)) := by
  have c25182 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25183 := (publicBatchWrapper89_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25184 := (publicBatchWrapper89_copies787 a h).1
  have c25185 := (publicBatchWrapper89_copies787 a h).2.1
  have c25186 := (publicBatchWrapper89_copies787 a h).2.2.1
  have c25187 := (publicBatchWrapper89_copies787 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_10 := arithEq_of_rows h (row := 386) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_10
  simp only [← c25182, ← c25183, ← c25184] at e_386_10
  have e_386_11 := arithEq_of_rows h (row := 386) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_11
  simp only [← c25185, ← c25186, k2, ← c25187] at e_386_11
  have hr := e_386_11
  simp only [e_386_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 511)) := by
  have c25188 := (publicBatchWrapper89_copies787 a h).2.2.2.2.1
  have c25189 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.1
  have c25190 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.1
  have c25191 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.1
  have c25192 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.1
  have c25193 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_12 := arithEq_of_rows h (row := 386) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_12
  simp only [← c25188, ← c25189, ← c25190] at e_386_12
  have e_386_13 := arithEq_of_rows h (row := 386) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_13
  simp only [← c25191, ← c25192, k2, ← c25193] at e_386_13
  have hr := e_386_13
  simp only [e_386_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 512)) := by
  have c25194 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25195 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25196 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25197 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25198 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25199 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_14 := arithEq_of_rows h (row := 386) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_14
  simp only [← c25194, ← c25195, ← c25196] at e_386_14
  have e_386_15 := arithEq_of_rows h (row := 386) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_15
  simp only [← c25197, ← c25198, k2, ← c25199] at e_386_15
  have hr := e_386_15
  simp only [e_386_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 513)) := by
  have c25200 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25201 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25202 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25203 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25204 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25205 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_16 := arithEq_of_rows h (row := 386) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_16
  simp only [← c25200, ← c25201, ← c25202] at e_386_16
  have e_386_17 := arithEq_of_rows h (row := 386) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_17
  simp only [← c25203, ← c25204, k2, ← c25205] at e_386_17
  have hr := e_386_17
  simp only [e_386_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 386 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 514)) := by
  have c25206 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25207 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25208 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25209 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25210 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25211 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_386_18 := arithEq_of_rows h (row := 386) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_18
  simp only [← c25206, ← c25207, ← c25208] at e_386_18
  have e_386_19 := arithEq_of_rows h (row := 386) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_19
  simp only [← c25209, ← c25210, k2, ← c25211] at e_386_19
  have hr := e_386_19
  simp only [e_386_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 515)) := by
  have c25212 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25213 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25214 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25215 := (publicBatchWrapper89_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25216 := (publicBatchWrapper89_copies788 a h).1
  have c25217 := (publicBatchWrapper89_copies788 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_0 := arithEq_of_rows h (row := 387) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_0
  simp only [← c25212, ← c25213, ← c25214] at e_387_0
  have e_387_1 := arithEq_of_rows h (row := 387) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_1
  simp only [← c25215, ← c25216, k2, ← c25217] at e_387_1
  have hr := e_387_1
  simp only [e_387_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 516)) := by
  have c25218 := (publicBatchWrapper89_copies788 a h).2.2.1
  have c25219 := (publicBatchWrapper89_copies788 a h).2.2.2.1
  have c25220 := (publicBatchWrapper89_copies788 a h).2.2.2.2.1
  have c25221 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.1
  have c25222 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.1
  have c25223 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_2 := arithEq_of_rows h (row := 387) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_2
  simp only [← c25218, ← c25219, ← c25220] at e_387_2
  have e_387_3 := arithEq_of_rows h (row := 387) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_3
  simp only [← c25221, ← c25222, k2, ← c25223] at e_387_3
  have hr := e_387_3
  simp only [e_387_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 517)) := by
  have c25224 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.1
  have c25225 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.1
  have c25226 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25227 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25228 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25229 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_4 := arithEq_of_rows h (row := 387) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_4
  simp only [← c25224, ← c25225, ← c25226] at e_387_4
  have e_387_5 := arithEq_of_rows h (row := 387) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_5
  simp only [← c25227, ← c25228, k2, ← c25229] at e_387_5
  have hr := e_387_5
  simp only [e_387_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 518)) := by
  have c25230 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25231 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25232 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25233 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25234 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25235 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_6 := arithEq_of_rows h (row := 387) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_6
  simp only [← c25230, ← c25231, ← c25232] at e_387_6
  have e_387_7 := arithEq_of_rows h (row := 387) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_7
  simp only [← c25233, ← c25234, k2, ← c25235] at e_387_7
  have hr := e_387_7
  simp only [e_387_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 519)) := by
  have c25236 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25237 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25238 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25239 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25240 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25241 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_8 := arithEq_of_rows h (row := 387) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_8
  simp only [← c25236, ← c25237, ← c25238] at e_387_8
  have e_387_9 := arithEq_of_rows h (row := 387) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_9
  simp only [← c25239, ← c25240, k2, ← c25241] at e_387_9
  have hr := e_387_9
  simp only [e_387_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 520)) := by
  have c25242 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25243 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25244 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25245 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25246 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25247 := (publicBatchWrapper89_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_10 := arithEq_of_rows h (row := 387) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_10
  simp only [← c25242, ← c25243, ← c25244] at e_387_10
  have e_387_11 := arithEq_of_rows h (row := 387) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_11
  simp only [← c25245, ← c25246, k2, ← c25247] at e_387_11
  have hr := e_387_11
  simp only [e_387_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 521)) := by
  have c25248 := (publicBatchWrapper89_copies789 a h).1
  have c25249 := (publicBatchWrapper89_copies789 a h).2.1
  have c25250 := (publicBatchWrapper89_copies789 a h).2.2.1
  have c25251 := (publicBatchWrapper89_copies789 a h).2.2.2.1
  have c25252 := (publicBatchWrapper89_copies789 a h).2.2.2.2.1
  have c25253 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_12 := arithEq_of_rows h (row := 387) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_12
  simp only [← c25248, ← c25249, ← c25250] at e_387_12
  have e_387_13 := arithEq_of_rows h (row := 387) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_13
  simp only [← c25251, ← c25252, k2, ← c25253] at e_387_13
  have hr := e_387_13
  simp only [e_387_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 522)) := by
  have c25254 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.1
  have c25255 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.1
  have c25256 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.1
  have c25257 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.1
  have c25258 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25259 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_14 := arithEq_of_rows h (row := 387) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_14
  simp only [← c25254, ← c25255, ← c25256] at e_387_14
  have e_387_15 := arithEq_of_rows h (row := 387) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_15
  simp only [← c25257, ← c25258, k2, ← c25259] at e_387_15
  have hr := e_387_15
  simp only [e_387_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 523)) := by
  have c25260 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25261 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25262 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25263 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25264 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25265 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_16 := arithEq_of_rows h (row := 387) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_16
  simp only [← c25260, ← c25261, ← c25262] at e_387_16
  have e_387_17 := arithEq_of_rows h (row := 387) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_17
  simp only [← c25263, ← c25264, k2, ← c25265] at e_387_17
  have hr := e_387_17
  simp only [e_387_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 387 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 524)) := by
  have c25266 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25267 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25268 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25269 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25270 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25271 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_387_18 := arithEq_of_rows h (row := 387) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_18
  simp only [← c25266, ← c25267, ← c25268] at e_387_18
  have e_387_19 := arithEq_of_rows h (row := 387) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_19
  simp only [← c25269, ← c25270, k2, ← c25271] at e_387_19
  have hr := e_387_19
  simp only [e_387_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 525)) := by
  have c25272 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25273 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25274 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25275 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25276 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25277 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_0 := arithEq_of_rows h (row := 388) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_0
  simp only [← c25272, ← c25273, ← c25274] at e_388_0
  have e_388_1 := arithEq_of_rows h (row := 388) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_1
  simp only [← c25275, ← c25276, k2, ← c25277] at e_388_1
  have hr := e_388_1
  simp only [e_388_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 526)) := by
  have c25278 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25279 := (publicBatchWrapper89_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25280 := (publicBatchWrapper89_copies790 a h).1
  have c25281 := (publicBatchWrapper89_copies790 a h).2.1
  have c25282 := (publicBatchWrapper89_copies790 a h).2.2.1
  have c25283 := (publicBatchWrapper89_copies790 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_2 := arithEq_of_rows h (row := 388) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_2
  simp only [← c25278, ← c25279, ← c25280] at e_388_2
  have e_388_3 := arithEq_of_rows h (row := 388) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_3
  simp only [← c25281, ← c25282, k2, ← c25283] at e_388_3
  have hr := e_388_3
  simp only [e_388_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 527)) := by
  have c25284 := (publicBatchWrapper89_copies790 a h).2.2.2.2.1
  have c25285 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.1
  have c25286 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.1
  have c25287 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.1
  have c25288 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.1
  have c25289 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_4 := arithEq_of_rows h (row := 388) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_4
  simp only [← c25284, ← c25285, ← c25286] at e_388_4
  have e_388_5 := arithEq_of_rows h (row := 388) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_5
  simp only [← c25287, ← c25288, k2, ← c25289] at e_388_5
  have hr := e_388_5
  simp only [e_388_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 528)) := by
  have c25290 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25291 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25292 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25293 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25294 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25295 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_6 := arithEq_of_rows h (row := 388) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_6
  simp only [← c25290, ← c25291, ← c25292] at e_388_6
  have e_388_7 := arithEq_of_rows h (row := 388) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_7
  simp only [← c25293, ← c25294, k2, ← c25295] at e_388_7
  have hr := e_388_7
  simp only [e_388_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 529)) := by
  have c25296 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25297 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25298 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25299 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25300 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25301 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_8 := arithEq_of_rows h (row := 388) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_8
  simp only [← c25296, ← c25297, ← c25298] at e_388_8
  have e_388_9 := arithEq_of_rows h (row := 388) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_9
  simp only [← c25299, ← c25300, k2, ← c25301] at e_388_9
  have hr := e_388_9
  simp only [e_388_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 530)) := by
  have c25302 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25303 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25304 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25305 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25306 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25307 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_10 := arithEq_of_rows h (row := 388) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_10
  simp only [← c25302, ← c25303, ← c25304] at e_388_10
  have e_388_11 := arithEq_of_rows h (row := 388) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_11
  simp only [← c25305, ← c25306, k2, ← c25307] at e_388_11
  have hr := e_388_11
  simp only [e_388_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 531)) := by
  have c25308 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25309 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25310 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25311 := (publicBatchWrapper89_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25312 := (publicBatchWrapper89_copies791 a h).1
  have c25313 := (publicBatchWrapper89_copies791 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_12 := arithEq_of_rows h (row := 388) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_12
  simp only [← c25308, ← c25309, ← c25310] at e_388_12
  have e_388_13 := arithEq_of_rows h (row := 388) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_13
  simp only [← c25311, ← c25312, k2, ← c25313] at e_388_13
  have hr := e_388_13
  simp only [e_388_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 532)) := by
  have c25314 := (publicBatchWrapper89_copies791 a h).2.2.1
  have c25315 := (publicBatchWrapper89_copies791 a h).2.2.2.1
  have c25316 := (publicBatchWrapper89_copies791 a h).2.2.2.2.1
  have c25317 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.1
  have c25318 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.1
  have c25319 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_14 := arithEq_of_rows h (row := 388) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_14
  simp only [← c25314, ← c25315, ← c25316] at e_388_14
  have e_388_15 := arithEq_of_rows h (row := 388) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_15
  simp only [← c25317, ← c25318, k2, ← c25319] at e_388_15
  have hr := e_388_15
  simp only [e_388_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 533)) := by
  have c25320 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.1
  have c25321 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.1
  have c25322 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25323 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25324 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25325 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_16 := arithEq_of_rows h (row := 388) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_16
  simp only [← c25320, ← c25321, ← c25322] at e_388_16
  have e_388_17 := arithEq_of_rows h (row := 388) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_17
  simp only [← c25323, ← c25324, k2, ← c25325] at e_388_17
  have hr := e_388_17
  simp only [e_388_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 388 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 534)) := by
  have c25326 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25327 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25328 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25329 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25330 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25331 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_388_18 := arithEq_of_rows h (row := 388) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_18
  simp only [← c25326, ← c25327, ← c25328] at e_388_18
  have e_388_19 := arithEq_of_rows h (row := 388) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_19
  simp only [← c25329, ← c25330, k2, ← c25331] at e_388_19
  have hr := e_388_19
  simp only [e_388_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 535)) := by
  have c25332 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25333 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25334 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25335 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25336 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25337 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_0 := arithEq_of_rows h (row := 389) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_0
  simp only [← c25332, ← c25333, ← c25334] at e_389_0
  have e_389_1 := arithEq_of_rows h (row := 389) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_1
  simp only [← c25335, ← c25336, k2, ← c25337] at e_389_1
  have hr := e_389_1
  simp only [e_389_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 536)) := by
  have c25338 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25339 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25340 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25341 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25342 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25343 := (publicBatchWrapper89_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_2 := arithEq_of_rows h (row := 389) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_2
  simp only [← c25338, ← c25339, ← c25340] at e_389_2
  have e_389_3 := arithEq_of_rows h (row := 389) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_3
  simp only [← c25341, ← c25342, k2, ← c25343] at e_389_3
  have hr := e_389_3
  simp only [e_389_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 537)) := by
  have c25344 := (publicBatchWrapper89_copies792 a h).1
  have c25345 := (publicBatchWrapper89_copies792 a h).2.1
  have c25346 := (publicBatchWrapper89_copies792 a h).2.2.1
  have c25347 := (publicBatchWrapper89_copies792 a h).2.2.2.1
  have c25348 := (publicBatchWrapper89_copies792 a h).2.2.2.2.1
  have c25349 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_4 := arithEq_of_rows h (row := 389) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_4
  simp only [← c25344, ← c25345, ← c25346] at e_389_4
  have e_389_5 := arithEq_of_rows h (row := 389) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_5
  simp only [← c25347, ← c25348, k2, ← c25349] at e_389_5
  have hr := e_389_5
  simp only [e_389_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 538)) := by
  have c25350 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.1
  have c25351 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.1
  have c25352 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.1
  have c25353 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.1
  have c25354 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25355 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_6 := arithEq_of_rows h (row := 389) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_6
  simp only [← c25350, ← c25351, ← c25352] at e_389_6
  have e_389_7 := arithEq_of_rows h (row := 389) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_7
  simp only [← c25353, ← c25354, k2, ← c25355] at e_389_7
  have hr := e_389_7
  simp only [e_389_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 539)) := by
  have c25356 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25357 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25358 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25359 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25360 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25361 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_8 := arithEq_of_rows h (row := 389) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_8
  simp only [← c25356, ← c25357, ← c25358] at e_389_8
  have e_389_9 := arithEq_of_rows h (row := 389) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_9
  simp only [← c25359, ← c25360, k2, ← c25361] at e_389_9
  have hr := e_389_9
  simp only [e_389_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 540)) := by
  have c25362 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25363 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25364 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25365 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25366 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25367 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_10 := arithEq_of_rows h (row := 389) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_10
  simp only [← c25362, ← c25363, ← c25364] at e_389_10
  have e_389_11 := arithEq_of_rows h (row := 389) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_11
  simp only [← c25365, ← c25366, k2, ← c25367] at e_389_11
  have hr := e_389_11
  simp only [e_389_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 541)) := by
  have c25368 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25369 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25370 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25371 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25372 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25373 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_12 := arithEq_of_rows h (row := 389) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_12
  simp only [← c25368, ← c25369, ← c25370] at e_389_12
  have e_389_13 := arithEq_of_rows h (row := 389) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_13
  simp only [← c25371, ← c25372, k2, ← c25373] at e_389_13
  have hr := e_389_13
  simp only [e_389_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 542)) := by
  have c25374 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25375 := (publicBatchWrapper89_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25376 := (publicBatchWrapper89_copies793 a h).1
  have c25377 := (publicBatchWrapper89_copies793 a h).2.1
  have c25378 := (publicBatchWrapper89_copies793 a h).2.2.1
  have c25379 := (publicBatchWrapper89_copies793 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_14 := arithEq_of_rows h (row := 389) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_14
  simp only [← c25374, ← c25375, ← c25376] at e_389_14
  have e_389_15 := arithEq_of_rows h (row := 389) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_15
  simp only [← c25377, ← c25378, k2, ← c25379] at e_389_15
  have hr := e_389_15
  simp only [e_389_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 543)) := by
  have c25380 := (publicBatchWrapper89_copies793 a h).2.2.2.2.1
  have c25381 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.1
  have c25382 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.1
  have c25383 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.1
  have c25384 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.1
  have c25385 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_16 := arithEq_of_rows h (row := 389) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_16
  simp only [← c25380, ← c25381, ← c25382] at e_389_16
  have e_389_17 := arithEq_of_rows h (row := 389) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_17
  simp only [← c25383, ← c25384, k2, ← c25385] at e_389_17
  have hr := e_389_17
  simp only [e_389_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 389 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 544)) := by
  have c25386 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25387 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25388 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25389 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25390 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25391 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_389_18 := arithEq_of_rows h (row := 389) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_18
  simp only [← c25386, ← c25387, ← c25388] at e_389_18
  have e_389_19 := arithEq_of_rows h (row := 389) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_19
  simp only [← c25389, ← c25390, k2, ← c25391] at e_389_19
  have hr := e_389_19
  simp only [e_389_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 545)) := by
  have c25392 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25393 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25394 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25395 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25396 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25397 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_0 := arithEq_of_rows h (row := 390) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_0
  simp only [← c25392, ← c25393, ← c25394] at e_390_0
  have e_390_1 := arithEq_of_rows h (row := 390) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_1
  simp only [← c25395, ← c25396, k2, ← c25397] at e_390_1
  have hr := e_390_1
  simp only [e_390_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 546)) := by
  have c25398 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25399 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25400 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25401 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25402 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25403 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_2 := arithEq_of_rows h (row := 390) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_2
  simp only [← c25398, ← c25399, ← c25400] at e_390_2
  have e_390_3 := arithEq_of_rows h (row := 390) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_3
  simp only [← c25401, ← c25402, k2, ← c25403] at e_390_3
  have hr := e_390_3
  simp only [e_390_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 547)) := by
  have c25404 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25405 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25406 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25407 := (publicBatchWrapper89_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25408 := (publicBatchWrapper89_copies794 a h).1
  have c25409 := (publicBatchWrapper89_copies794 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_4 := arithEq_of_rows h (row := 390) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_4
  simp only [← c25404, ← c25405, ← c25406] at e_390_4
  have e_390_5 := arithEq_of_rows h (row := 390) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_5
  simp only [← c25407, ← c25408, k2, ← c25409] at e_390_5
  have hr := e_390_5
  simp only [e_390_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 548)) := by
  have c25410 := (publicBatchWrapper89_copies794 a h).2.2.1
  have c25411 := (publicBatchWrapper89_copies794 a h).2.2.2.1
  have c25412 := (publicBatchWrapper89_copies794 a h).2.2.2.2.1
  have c25413 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.1
  have c25414 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.1
  have c25415 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_6 := arithEq_of_rows h (row := 390) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_6
  simp only [← c25410, ← c25411, ← c25412] at e_390_6
  have e_390_7 := arithEq_of_rows h (row := 390) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_7
  simp only [← c25413, ← c25414, k2, ← c25415] at e_390_7
  have hr := e_390_7
  simp only [e_390_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 549)) := by
  have c25416 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.1
  have c25417 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.1
  have c25418 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25419 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25420 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25421 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_8 := arithEq_of_rows h (row := 390) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_8
  simp only [← c25416, ← c25417, ← c25418] at e_390_8
  have e_390_9 := arithEq_of_rows h (row := 390) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_9
  simp only [← c25419, ← c25420, k2, ← c25421] at e_390_9
  have hr := e_390_9
  simp only [e_390_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 550)) := by
  have c25422 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25423 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25424 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25425 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25426 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25427 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_10 := arithEq_of_rows h (row := 390) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_10
  simp only [← c25422, ← c25423, ← c25424] at e_390_10
  have e_390_11 := arithEq_of_rows h (row := 390) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_11
  simp only [← c25425, ← c25426, k2, ← c25427] at e_390_11
  have hr := e_390_11
  simp only [e_390_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 551)) := by
  have c25428 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25429 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25430 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25431 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25432 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25433 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_12 := arithEq_of_rows h (row := 390) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_12
  simp only [← c25428, ← c25429, ← c25430] at e_390_12
  have e_390_13 := arithEq_of_rows h (row := 390) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_13
  simp only [← c25431, ← c25432, k2, ← c25433] at e_390_13
  have hr := e_390_13
  simp only [e_390_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 552)) := by
  have c25434 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25435 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25436 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25437 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25438 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25439 := (publicBatchWrapper89_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_14 := arithEq_of_rows h (row := 390) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_14
  simp only [← c25434, ← c25435, ← c25436] at e_390_14
  have e_390_15 := arithEq_of_rows h (row := 390) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_15
  simp only [← c25437, ← c25438, k2, ← c25439] at e_390_15
  have hr := e_390_15
  simp only [e_390_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 553)) := by
  have c25440 := (publicBatchWrapper89_copies795 a h).1
  have c25441 := (publicBatchWrapper89_copies795 a h).2.1
  have c25442 := (publicBatchWrapper89_copies795 a h).2.2.1
  have c25443 := (publicBatchWrapper89_copies795 a h).2.2.2.1
  have c25444 := (publicBatchWrapper89_copies795 a h).2.2.2.2.1
  have c25445 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_16 := arithEq_of_rows h (row := 390) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_16
  simp only [← c25440, ← c25441, ← c25442] at e_390_16
  have e_390_17 := arithEq_of_rows h (row := 390) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_17
  simp only [← c25443, ← c25444, k2, ← c25445] at e_390_17
  have hr := e_390_17
  simp only [e_390_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 390 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 554)) := by
  have c25446 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.1
  have c25447 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.1
  have c25448 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.1
  have c25449 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.1
  have c25450 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25451 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_390_18 := arithEq_of_rows h (row := 390) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_18
  simp only [← c25446, ← c25447, ← c25448] at e_390_18
  have e_390_19 := arithEq_of_rows h (row := 390) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_19
  simp only [← c25449, ← c25450, k2, ← c25451] at e_390_19
  have hr := e_390_19
  simp only [e_390_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 555)) := by
  have c25452 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25453 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25454 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25455 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25456 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25457 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_0 := arithEq_of_rows h (row := 391) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_0
  simp only [← c25452, ← c25453, ← c25454] at e_391_0
  have e_391_1 := arithEq_of_rows h (row := 391) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_1
  simp only [← c25455, ← c25456, k2, ← c25457] at e_391_1
  have hr := e_391_1
  simp only [e_391_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 556)) := by
  have c25458 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25459 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25460 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25461 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25462 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25463 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_2 := arithEq_of_rows h (row := 391) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_2
  simp only [← c25458, ← c25459, ← c25460] at e_391_2
  have e_391_3 := arithEq_of_rows h (row := 391) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_3
  simp only [← c25461, ← c25462, k2, ← c25463] at e_391_3
  have hr := e_391_3
  simp only [e_391_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 557)) := by
  have c25464 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25465 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25466 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25467 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25468 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25469 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_4 := arithEq_of_rows h (row := 391) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_4
  simp only [← c25464, ← c25465, ← c25466] at e_391_4
  have e_391_5 := arithEq_of_rows h (row := 391) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_5
  simp only [← c25467, ← c25468, k2, ← c25469] at e_391_5
  have hr := e_391_5
  simp only [e_391_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 558)) := by
  have c25470 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25471 := (publicBatchWrapper89_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25472 := (publicBatchWrapper89_copies796 a h).1
  have c25473 := (publicBatchWrapper89_copies796 a h).2.1
  have c25474 := (publicBatchWrapper89_copies796 a h).2.2.1
  have c25475 := (publicBatchWrapper89_copies796 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_6 := arithEq_of_rows h (row := 391) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_6
  simp only [← c25470, ← c25471, ← c25472] at e_391_6
  have e_391_7 := arithEq_of_rows h (row := 391) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_7
  simp only [← c25473, ← c25474, k2, ← c25475] at e_391_7
  have hr := e_391_7
  simp only [e_391_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 559)) := by
  have c25476 := (publicBatchWrapper89_copies796 a h).2.2.2.2.1
  have c25477 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.1
  have c25478 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.1
  have c25479 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.1
  have c25480 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.1
  have c25481 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_8 := arithEq_of_rows h (row := 391) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_8
  simp only [← c25476, ← c25477, ← c25478] at e_391_8
  have e_391_9 := arithEq_of_rows h (row := 391) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_9
  simp only [← c25479, ← c25480, k2, ← c25481] at e_391_9
  have hr := e_391_9
  simp only [e_391_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 560)) := by
  have c25482 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25483 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25484 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25485 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25486 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25487 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_10 := arithEq_of_rows h (row := 391) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_10
  simp only [← c25482, ← c25483, ← c25484] at e_391_10
  have e_391_11 := arithEq_of_rows h (row := 391) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_11
  simp only [← c25485, ← c25486, k2, ← c25487] at e_391_11
  have hr := e_391_11
  simp only [e_391_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 561)) := by
  have c25488 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25489 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25490 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25491 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25492 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25493 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_12 := arithEq_of_rows h (row := 391) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_12
  simp only [← c25488, ← c25489, ← c25490] at e_391_12
  have e_391_13 := arithEq_of_rows h (row := 391) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_13
  simp only [← c25491, ← c25492, k2, ← c25493] at e_391_13
  have hr := e_391_13
  simp only [e_391_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 562)) := by
  have c25494 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25495 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25496 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25497 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25498 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25499 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_14 := arithEq_of_rows h (row := 391) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_14
  simp only [← c25494, ← c25495, ← c25496] at e_391_14
  have e_391_15 := arithEq_of_rows h (row := 391) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_15
  simp only [← c25497, ← c25498, k2, ← c25499] at e_391_15
  have hr := e_391_15
  simp only [e_391_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 563)) := by
  have c25500 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25501 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25502 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25503 := (publicBatchWrapper89_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25504 := (publicBatchWrapper89_copies797 a h).1
  have c25505 := (publicBatchWrapper89_copies797 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_16 := arithEq_of_rows h (row := 391) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_16
  simp only [← c25500, ← c25501, ← c25502] at e_391_16
  have e_391_17 := arithEq_of_rows h (row := 391) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_17
  simp only [← c25503, ← c25504, k2, ← c25505] at e_391_17
  have hr := e_391_17
  simp only [e_391_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 391 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 564)) := by
  have c25506 := (publicBatchWrapper89_copies797 a h).2.2.1
  have c25507 := (publicBatchWrapper89_copies797 a h).2.2.2.1
  have c25508 := (publicBatchWrapper89_copies797 a h).2.2.2.2.1
  have c25509 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.1
  have c25510 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.1
  have c25511 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_391_18 := arithEq_of_rows h (row := 391) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_18
  simp only [← c25506, ← c25507, ← c25508] at e_391_18
  have e_391_19 := arithEq_of_rows h (row := 391) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_19
  simp only [← c25509, ← c25510, k2, ← c25511] at e_391_19
  have hr := e_391_19
  simp only [e_391_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 565)) := by
  have c25512 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.1
  have c25513 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.1
  have c25514 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25515 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25516 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25517 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_0 := arithEq_of_rows h (row := 392) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_0
  simp only [← c25512, ← c25513, ← c25514] at e_392_0
  have e_392_1 := arithEq_of_rows h (row := 392) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_1
  simp only [← c25515, ← c25516, k2, ← c25517] at e_392_1
  have hr := e_392_1
  simp only [e_392_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 566)) := by
  have c25518 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25519 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25520 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25521 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25522 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25523 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_2 := arithEq_of_rows h (row := 392) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_2
  simp only [← c25518, ← c25519, ← c25520] at e_392_2
  have e_392_3 := arithEq_of_rows h (row := 392) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_3
  simp only [← c25521, ← c25522, k2, ← c25523] at e_392_3
  have hr := e_392_3
  simp only [e_392_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 567)) := by
  have c25524 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25525 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25526 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25527 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25528 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25529 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_4 := arithEq_of_rows h (row := 392) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_4
  simp only [← c25524, ← c25525, ← c25526] at e_392_4
  have e_392_5 := arithEq_of_rows h (row := 392) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_5
  simp only [← c25527, ← c25528, k2, ← c25529] at e_392_5
  have hr := e_392_5
  simp only [e_392_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 568)) := by
  have c25530 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25531 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25532 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25533 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25534 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25535 := (publicBatchWrapper89_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_6 := arithEq_of_rows h (row := 392) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_6
  simp only [← c25530, ← c25531, ← c25532] at e_392_6
  have e_392_7 := arithEq_of_rows h (row := 392) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_7
  simp only [← c25533, ← c25534, k2, ← c25535] at e_392_7
  have hr := e_392_7
  simp only [e_392_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 569)) := by
  have c25536 := (publicBatchWrapper89_copies798 a h).1
  have c25537 := (publicBatchWrapper89_copies798 a h).2.1
  have c25538 := (publicBatchWrapper89_copies798 a h).2.2.1
  have c25539 := (publicBatchWrapper89_copies798 a h).2.2.2.1
  have c25540 := (publicBatchWrapper89_copies798 a h).2.2.2.2.1
  have c25541 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_8 := arithEq_of_rows h (row := 392) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_8
  simp only [← c25536, ← c25537, ← c25538] at e_392_8
  have e_392_9 := arithEq_of_rows h (row := 392) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_9
  simp only [← c25539, ← c25540, k2, ← c25541] at e_392_9
  have hr := e_392_9
  simp only [e_392_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 610)) := by
  have c25542 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.1
  have c25543 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.1
  have c25544 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.1
  have c25545 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.1
  have c25546 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25547 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_10 := arithEq_of_rows h (row := 392) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_10
  simp only [← c25542, ← c25543, ← c25544] at e_392_10
  have e_392_11 := arithEq_of_rows h (row := 392) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_11
  simp only [← c25545, ← c25546, k2, ← c25547] at e_392_11
  have hr := e_392_11
  simp only [e_392_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 611)) := by
  have c25548 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25549 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25550 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25551 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25552 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25553 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_12 := arithEq_of_rows h (row := 392) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_12
  simp only [← c25548, ← c25549, ← c25550] at e_392_12
  have e_392_13 := arithEq_of_rows h (row := 392) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_13
  simp only [← c25551, ← c25552, k2, ← c25553] at e_392_13
  have hr := e_392_13
  simp only [e_392_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 612)) := by
  have c25554 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25555 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25556 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25557 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25558 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25559 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_14 := arithEq_of_rows h (row := 392) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_14
  simp only [← c25554, ← c25555, ← c25556] at e_392_14
  have e_392_15 := arithEq_of_rows h (row := 392) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_15
  simp only [← c25557, ← c25558, k2, ← c25559] at e_392_15
  have hr := e_392_15
  simp only [e_392_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 613)) := by
  have c25560 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25561 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25562 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25563 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25564 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25565 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_16 := arithEq_of_rows h (row := 392) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_16
  simp only [← c25560, ← c25561, ← c25562] at e_392_16
  have e_392_17 := arithEq_of_rows h (row := 392) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_17
  simp only [← c25563, ← c25564, k2, ← c25565] at e_392_17
  have hr := e_392_17
  simp only [e_392_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 392 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 614)) := by
  have c25566 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25567 := (publicBatchWrapper89_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25568 := (publicBatchWrapper89_copies799 a h).1
  have c25569 := (publicBatchWrapper89_copies799 a h).2.1
  have c25570 := (publicBatchWrapper89_copies799 a h).2.2.1
  have c25571 := (publicBatchWrapper89_copies799 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_392_18 := arithEq_of_rows h (row := 392) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_18
  simp only [← c25566, ← c25567, ← c25568] at e_392_18
  have e_392_19 := arithEq_of_rows h (row := 392) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_19
  simp only [← c25569, ← c25570, k2, ← c25571] at e_392_19
  have hr := e_392_19
  simp only [e_392_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 615)) := by
  have c25572 := (publicBatchWrapper89_copies799 a h).2.2.2.2.1
  have c25573 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.1
  have c25574 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.1
  have c25575 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.1
  have c25576 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.1
  have c25577 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_0 := arithEq_of_rows h (row := 393) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_0
  simp only [← c25572, ← c25573, ← c25574] at e_393_0
  have e_393_1 := arithEq_of_rows h (row := 393) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_1
  simp only [← c25575, ← c25576, k2, ← c25577] at e_393_1
  have hr := e_393_1
  simp only [e_393_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 616)) := by
  have c25578 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25579 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25580 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25581 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25582 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25583 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_2 := arithEq_of_rows h (row := 393) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_2
  simp only [← c25578, ← c25579, ← c25580] at e_393_2
  have e_393_3 := arithEq_of_rows h (row := 393) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_3
  simp only [← c25581, ← c25582, k2, ← c25583] at e_393_3
  have hr := e_393_3
  simp only [e_393_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 617)) := by
  have c25584 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25585 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25586 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25587 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25588 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25589 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_4 := arithEq_of_rows h (row := 393) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_4
  simp only [← c25584, ← c25585, ← c25586] at e_393_4
  have e_393_5 := arithEq_of_rows h (row := 393) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_5
  simp only [← c25587, ← c25588, k2, ← c25589] at e_393_5
  have hr := e_393_5
  simp only [e_393_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 618)) := by
  have c25590 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25591 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25592 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25593 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25594 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25595 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_6 := arithEq_of_rows h (row := 393) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_6
  simp only [← c25590, ← c25591, ← c25592] at e_393_6
  have e_393_7 := arithEq_of_rows h (row := 393) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_7
  simp only [← c25593, ← c25594, k2, ← c25595] at e_393_7
  have hr := e_393_7
  simp only [e_393_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 619)) := by
  have c25596 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25597 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25598 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25599 := (publicBatchWrapper89_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25600 := (publicBatchWrapper89_copies800 a h).1
  have c25601 := (publicBatchWrapper89_copies800 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_8 := arithEq_of_rows h (row := 393) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_8
  simp only [← c25596, ← c25597, ← c25598] at e_393_8
  have e_393_9 := arithEq_of_rows h (row := 393) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_9
  simp only [← c25599, ← c25600, k2, ← c25601] at e_393_9
  have hr := e_393_9
  simp only [e_393_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 620)) := by
  have c25602 := (publicBatchWrapper89_copies800 a h).2.2.1
  have c25603 := (publicBatchWrapper89_copies800 a h).2.2.2.1
  have c25604 := (publicBatchWrapper89_copies800 a h).2.2.2.2.1
  have c25605 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.1
  have c25606 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.1
  have c25607 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_10 := arithEq_of_rows h (row := 393) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_10
  simp only [← c25602, ← c25603, ← c25604] at e_393_10
  have e_393_11 := arithEq_of_rows h (row := 393) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_11
  simp only [← c25605, ← c25606, k2, ← c25607] at e_393_11
  have hr := e_393_11
  simp only [e_393_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 621)) := by
  have c25608 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.1
  have c25609 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.1
  have c25610 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25611 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25612 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25613 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_12 := arithEq_of_rows h (row := 393) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_12
  simp only [← c25608, ← c25609, ← c25610] at e_393_12
  have e_393_13 := arithEq_of_rows h (row := 393) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_13
  simp only [← c25611, ← c25612, k2, ← c25613] at e_393_13
  have hr := e_393_13
  simp only [e_393_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 622)) := by
  have c25614 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25615 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25616 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25617 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25618 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25619 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_14 := arithEq_of_rows h (row := 393) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_14
  simp only [← c25614, ← c25615, ← c25616] at e_393_14
  have e_393_15 := arithEq_of_rows h (row := 393) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_15
  simp only [← c25617, ← c25618, k2, ← c25619] at e_393_15
  have hr := e_393_15
  simp only [e_393_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 623)) := by
  have c25620 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25621 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25622 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25623 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25624 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25625 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_16 := arithEq_of_rows h (row := 393) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_16
  simp only [← c25620, ← c25621, ← c25622] at e_393_16
  have e_393_17 := arithEq_of_rows h (row := 393) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_17
  simp only [← c25623, ← c25624, k2, ← c25625] at e_393_17
  have hr := e_393_17
  simp only [e_393_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 393 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 624)) := by
  have c25626 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25627 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25628 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25629 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25630 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25631 := (publicBatchWrapper89_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_393_18 := arithEq_of_rows h (row := 393) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_18
  simp only [← c25626, ← c25627, ← c25628] at e_393_18
  have e_393_19 := arithEq_of_rows h (row := 393) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_19
  simp only [← c25629, ← c25630, k2, ← c25631] at e_393_19
  have hr := e_393_19
  simp only [e_393_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 625)) := by
  have c25632 := (publicBatchWrapper89_copies801 a h).1
  have c25633 := (publicBatchWrapper89_copies801 a h).2.1
  have c25634 := (publicBatchWrapper89_copies801 a h).2.2.1
  have c25635 := (publicBatchWrapper89_copies801 a h).2.2.2.1
  have c25636 := (publicBatchWrapper89_copies801 a h).2.2.2.2.1
  have c25637 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_0 := arithEq_of_rows h (row := 394) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_0
  simp only [← c25632, ← c25633, ← c25634] at e_394_0
  have e_394_1 := arithEq_of_rows h (row := 394) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_1
  simp only [← c25635, ← c25636, k2, ← c25637] at e_394_1
  have hr := e_394_1
  simp only [e_394_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 626)) := by
  have c25638 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.1
  have c25639 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.1
  have c25640 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.1
  have c25641 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.1
  have c25642 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25643 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_2 := arithEq_of_rows h (row := 394) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_2
  simp only [← c25638, ← c25639, ← c25640] at e_394_2
  have e_394_3 := arithEq_of_rows h (row := 394) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_3
  simp only [← c25641, ← c25642, k2, ← c25643] at e_394_3
  have hr := e_394_3
  simp only [e_394_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 627)) := by
  have c25644 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25645 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25646 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25647 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25648 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25649 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_4 := arithEq_of_rows h (row := 394) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_4
  simp only [← c25644, ← c25645, ← c25646] at e_394_4
  have e_394_5 := arithEq_of_rows h (row := 394) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_5
  simp only [← c25647, ← c25648, k2, ← c25649] at e_394_5
  have hr := e_394_5
  simp only [e_394_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 628)) := by
  have c25650 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25651 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25652 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25653 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25654 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25655 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_6 := arithEq_of_rows h (row := 394) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_6
  simp only [← c25650, ← c25651, ← c25652] at e_394_6
  have e_394_7 := arithEq_of_rows h (row := 394) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_7
  simp only [← c25653, ← c25654, k2, ← c25655] at e_394_7
  have hr := e_394_7
  simp only [e_394_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 629)) := by
  have c25656 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25657 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25658 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25659 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25660 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25661 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_8 := arithEq_of_rows h (row := 394) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_8
  simp only [← c25656, ← c25657, ← c25658] at e_394_8
  have e_394_9 := arithEq_of_rows h (row := 394) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_9
  simp only [← c25659, ← c25660, k2, ← c25661] at e_394_9
  have hr := e_394_9
  simp only [e_394_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 630)) := by
  have c25662 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25663 := (publicBatchWrapper89_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25664 := (publicBatchWrapper89_copies802 a h).1
  have c25665 := (publicBatchWrapper89_copies802 a h).2.1
  have c25666 := (publicBatchWrapper89_copies802 a h).2.2.1
  have c25667 := (publicBatchWrapper89_copies802 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_10 := arithEq_of_rows h (row := 394) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_10
  simp only [← c25662, ← c25663, ← c25664] at e_394_10
  have e_394_11 := arithEq_of_rows h (row := 394) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_11
  simp only [← c25665, ← c25666, k2, ← c25667] at e_394_11
  have hr := e_394_11
  simp only [e_394_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 631)) := by
  have c25668 := (publicBatchWrapper89_copies802 a h).2.2.2.2.1
  have c25669 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.1
  have c25670 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.1
  have c25671 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.1
  have c25672 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.1
  have c25673 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_12 := arithEq_of_rows h (row := 394) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_12
  simp only [← c25668, ← c25669, ← c25670] at e_394_12
  have e_394_13 := arithEq_of_rows h (row := 394) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_13
  simp only [← c25671, ← c25672, k2, ← c25673] at e_394_13
  have hr := e_394_13
  simp only [e_394_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 632)) := by
  have c25674 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25675 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25676 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25677 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25678 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25679 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_14 := arithEq_of_rows h (row := 394) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_14
  simp only [← c25674, ← c25675, ← c25676] at e_394_14
  have e_394_15 := arithEq_of_rows h (row := 394) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_15
  simp only [← c25677, ← c25678, k2, ← c25679] at e_394_15
  have hr := e_394_15
  simp only [e_394_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 633)) := by
  have c25680 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25681 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25682 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25683 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25684 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25685 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_16 := arithEq_of_rows h (row := 394) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_16
  simp only [← c25680, ← c25681, ← c25682] at e_394_16
  have e_394_17 := arithEq_of_rows h (row := 394) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_17
  simp only [← c25683, ← c25684, k2, ← c25685] at e_394_17
  have hr := e_394_17
  simp only [e_394_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 394 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 634)) := by
  have c25686 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25687 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25688 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25689 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25690 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25691 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_394_18 := arithEq_of_rows h (row := 394) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_18
  simp only [← c25686, ← c25687, ← c25688] at e_394_18
  have e_394_19 := arithEq_of_rows h (row := 394) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_19
  simp only [← c25689, ← c25690, k2, ← c25691] at e_394_19
  have hr := e_394_19
  simp only [e_394_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 635)) := by
  have c25692 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25693 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25694 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25695 := (publicBatchWrapper89_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25696 := (publicBatchWrapper89_copies803 a h).1
  have c25697 := (publicBatchWrapper89_copies803 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_0 := arithEq_of_rows h (row := 395) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_0
  simp only [← c25692, ← c25693, ← c25694] at e_395_0
  have e_395_1 := arithEq_of_rows h (row := 395) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_1
  simp only [← c25695, ← c25696, k2, ← c25697] at e_395_1
  have hr := e_395_1
  simp only [e_395_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 636)) := by
  have c25698 := (publicBatchWrapper89_copies803 a h).2.2.1
  have c25699 := (publicBatchWrapper89_copies803 a h).2.2.2.1
  have c25700 := (publicBatchWrapper89_copies803 a h).2.2.2.2.1
  have c25701 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.1
  have c25702 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.1
  have c25703 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_2 := arithEq_of_rows h (row := 395) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_2
  simp only [← c25698, ← c25699, ← c25700] at e_395_2
  have e_395_3 := arithEq_of_rows h (row := 395) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_3
  simp only [← c25701, ← c25702, k2, ← c25703] at e_395_3
  have hr := e_395_3
  simp only [e_395_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 637)) := by
  have c25704 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.1
  have c25705 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.1
  have c25706 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25707 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25708 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25709 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_4 := arithEq_of_rows h (row := 395) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_4
  simp only [← c25704, ← c25705, ← c25706] at e_395_4
  have e_395_5 := arithEq_of_rows h (row := 395) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_5
  simp only [← c25707, ← c25708, k2, ← c25709] at e_395_5
  have hr := e_395_5
  simp only [e_395_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 638)) := by
  have c25710 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25711 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25712 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25713 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25714 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25715 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_6 := arithEq_of_rows h (row := 395) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_6
  simp only [← c25710, ← c25711, ← c25712] at e_395_6
  have e_395_7 := arithEq_of_rows h (row := 395) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_7
  simp only [← c25713, ← c25714, k2, ← c25715] at e_395_7
  have hr := e_395_7
  simp only [e_395_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 639)) := by
  have c25716 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25717 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25718 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25719 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25720 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25721 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_8 := arithEq_of_rows h (row := 395) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_8
  simp only [← c25716, ← c25717, ← c25718] at e_395_8
  have e_395_9 := arithEq_of_rows h (row := 395) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_9
  simp only [← c25719, ← c25720, k2, ← c25721] at e_395_9
  have hr := e_395_9
  simp only [e_395_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 640)) := by
  have c25722 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25723 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25724 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25725 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25726 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25727 := (publicBatchWrapper89_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_10 := arithEq_of_rows h (row := 395) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_10
  simp only [← c25722, ← c25723, ← c25724] at e_395_10
  have e_395_11 := arithEq_of_rows h (row := 395) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_11
  simp only [← c25725, ← c25726, k2, ← c25727] at e_395_11
  have hr := e_395_11
  simp only [e_395_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 641)) := by
  have c25728 := (publicBatchWrapper89_copies804 a h).1
  have c25729 := (publicBatchWrapper89_copies804 a h).2.1
  have c25730 := (publicBatchWrapper89_copies804 a h).2.2.1
  have c25731 := (publicBatchWrapper89_copies804 a h).2.2.2.1
  have c25732 := (publicBatchWrapper89_copies804 a h).2.2.2.2.1
  have c25733 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_12 := arithEq_of_rows h (row := 395) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_12
  simp only [← c25728, ← c25729, ← c25730] at e_395_12
  have e_395_13 := arithEq_of_rows h (row := 395) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_13
  simp only [← c25731, ← c25732, k2, ← c25733] at e_395_13
  have hr := e_395_13
  simp only [e_395_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 642)) := by
  have c25734 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.1
  have c25735 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.1
  have c25736 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.1
  have c25737 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.1
  have c25738 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25739 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_14 := arithEq_of_rows h (row := 395) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_14
  simp only [← c25734, ← c25735, ← c25736] at e_395_14
  have e_395_15 := arithEq_of_rows h (row := 395) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_15
  simp only [← c25737, ← c25738, k2, ← c25739] at e_395_15
  have hr := e_395_15
  simp only [e_395_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 643)) := by
  have c25740 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25741 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25742 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25743 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25744 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25745 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_16 := arithEq_of_rows h (row := 395) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_16
  simp only [← c25740, ← c25741, ← c25742] at e_395_16
  have e_395_17 := arithEq_of_rows h (row := 395) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_17
  simp only [← c25743, ← c25744, k2, ← c25745] at e_395_17
  have hr := e_395_17
  simp only [e_395_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 395 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 644)) := by
  have c25746 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25747 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25748 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25749 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25750 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25751 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_395_18 := arithEq_of_rows h (row := 395) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_18
  simp only [← c25746, ← c25747, ← c25748] at e_395_18
  have e_395_19 := arithEq_of_rows h (row := 395) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_19
  simp only [← c25749, ← c25750, k2, ← c25751] at e_395_19
  have hr := e_395_19
  simp only [e_395_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 645)) := by
  have c25752 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25753 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25754 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25755 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25756 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25757 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_0 := arithEq_of_rows h (row := 396) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_0
  simp only [← c25752, ← c25753, ← c25754] at e_396_0
  have e_396_1 := arithEq_of_rows h (row := 396) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_1
  simp only [← c25755, ← c25756, k2, ← c25757] at e_396_1
  have hr := e_396_1
  simp only [e_396_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 646)) := by
  have c25758 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25759 := (publicBatchWrapper89_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25760 := (publicBatchWrapper89_copies805 a h).1
  have c25761 := (publicBatchWrapper89_copies805 a h).2.1
  have c25762 := (publicBatchWrapper89_copies805 a h).2.2.1
  have c25763 := (publicBatchWrapper89_copies805 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_2 := arithEq_of_rows h (row := 396) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_2
  simp only [← c25758, ← c25759, ← c25760] at e_396_2
  have e_396_3 := arithEq_of_rows h (row := 396) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_3
  simp only [← c25761, ← c25762, k2, ← c25763] at e_396_3
  have hr := e_396_3
  simp only [e_396_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 647)) := by
  have c25764 := (publicBatchWrapper89_copies805 a h).2.2.2.2.1
  have c25765 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.1
  have c25766 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.1
  have c25767 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.1
  have c25768 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.1
  have c25769 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_4 := arithEq_of_rows h (row := 396) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_4
  simp only [← c25764, ← c25765, ← c25766] at e_396_4
  have e_396_5 := arithEq_of_rows h (row := 396) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_5
  simp only [← c25767, ← c25768, k2, ← c25769] at e_396_5
  have hr := e_396_5
  simp only [e_396_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 648)) := by
  have c25770 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25771 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25772 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25773 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25774 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25775 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_6 := arithEq_of_rows h (row := 396) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_6
  simp only [← c25770, ← c25771, ← c25772] at e_396_6
  have e_396_7 := arithEq_of_rows h (row := 396) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_7
  simp only [← c25773, ← c25774, k2, ← c25775] at e_396_7
  have hr := e_396_7
  simp only [e_396_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 649)) := by
  have c25776 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25777 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25778 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25779 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25780 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25781 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_8 := arithEq_of_rows h (row := 396) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_8
  simp only [← c25776, ← c25777, ← c25778] at e_396_8
  have e_396_9 := arithEq_of_rows h (row := 396) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_9
  simp only [← c25779, ← c25780, k2, ← c25781] at e_396_9
  have hr := e_396_9
  simp only [e_396_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 650)) := by
  have c25782 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25783 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25784 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25785 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25786 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25787 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_10 := arithEq_of_rows h (row := 396) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_10
  simp only [← c25782, ← c25783, ← c25784] at e_396_10
  have e_396_11 := arithEq_of_rows h (row := 396) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_11
  simp only [← c25785, ← c25786, k2, ← c25787] at e_396_11
  have hr := e_396_11
  simp only [e_396_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 651)) := by
  have c25788 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25789 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25790 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25791 := (publicBatchWrapper89_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25792 := (publicBatchWrapper89_copies806 a h).1
  have c25793 := (publicBatchWrapper89_copies806 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_12 := arithEq_of_rows h (row := 396) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_12
  simp only [← c25788, ← c25789, ← c25790] at e_396_12
  have e_396_13 := arithEq_of_rows h (row := 396) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_13
  simp only [← c25791, ← c25792, k2, ← c25793] at e_396_13
  have hr := e_396_13
  simp only [e_396_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 652)) := by
  have c25794 := (publicBatchWrapper89_copies806 a h).2.2.1
  have c25795 := (publicBatchWrapper89_copies806 a h).2.2.2.1
  have c25796 := (publicBatchWrapper89_copies806 a h).2.2.2.2.1
  have c25797 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.1
  have c25798 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.1
  have c25799 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_14 := arithEq_of_rows h (row := 396) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_14
  simp only [← c25794, ← c25795, ← c25796] at e_396_14
  have e_396_15 := arithEq_of_rows h (row := 396) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_15
  simp only [← c25797, ← c25798, k2, ← c25799] at e_396_15
  have hr := e_396_15
  simp only [e_396_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 653)) := by
  have c25800 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.1
  have c25801 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.1
  have c25802 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25803 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25804 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25805 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_16 := arithEq_of_rows h (row := 396) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_16
  simp only [← c25800, ← c25801, ← c25802] at e_396_16
  have e_396_17 := arithEq_of_rows h (row := 396) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_17
  simp only [← c25803, ← c25804, k2, ← c25805] at e_396_17
  have hr := e_396_17
  simp only [e_396_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 396 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 654)) := by
  have c25806 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25807 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25808 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25809 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25810 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25811 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_396_18 := arithEq_of_rows h (row := 396) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_18
  simp only [← c25806, ← c25807, ← c25808] at e_396_18
  have e_396_19 := arithEq_of_rows h (row := 396) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_19
  simp only [← c25809, ← c25810, k2, ← c25811] at e_396_19
  have hr := e_396_19
  simp only [e_396_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 655)) := by
  have c25812 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25813 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25814 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25815 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25816 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25817 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_0 := arithEq_of_rows h (row := 397) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_0
  simp only [← c25812, ← c25813, ← c25814] at e_397_0
  have e_397_1 := arithEq_of_rows h (row := 397) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_1
  simp only [← c25815, ← c25816, k2, ← c25817] at e_397_1
  have hr := e_397_1
  simp only [e_397_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 656)) := by
  have c25818 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25819 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25820 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25821 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25822 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25823 := (publicBatchWrapper89_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_2 := arithEq_of_rows h (row := 397) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_2
  simp only [← c25818, ← c25819, ← c25820] at e_397_2
  have e_397_3 := arithEq_of_rows h (row := 397) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_3
  simp only [← c25821, ← c25822, k2, ← c25823] at e_397_3
  have hr := e_397_3
  simp only [e_397_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 657)) := by
  have c25824 := (publicBatchWrapper89_copies807 a h).1
  have c25825 := (publicBatchWrapper89_copies807 a h).2.1
  have c25826 := (publicBatchWrapper89_copies807 a h).2.2.1
  have c25827 := (publicBatchWrapper89_copies807 a h).2.2.2.1
  have c25828 := (publicBatchWrapper89_copies807 a h).2.2.2.2.1
  have c25829 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_4 := arithEq_of_rows h (row := 397) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_4
  simp only [← c25824, ← c25825, ← c25826] at e_397_4
  have e_397_5 := arithEq_of_rows h (row := 397) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_5
  simp only [← c25827, ← c25828, k2, ← c25829] at e_397_5
  have hr := e_397_5
  simp only [e_397_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 658)) := by
  have c25830 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.1
  have c25831 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.1
  have c25832 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.1
  have c25833 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.1
  have c25834 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25835 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_6 := arithEq_of_rows h (row := 397) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_6
  simp only [← c25830, ← c25831, ← c25832] at e_397_6
  have e_397_7 := arithEq_of_rows h (row := 397) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_7
  simp only [← c25833, ← c25834, k2, ← c25835] at e_397_7
  have hr := e_397_7
  simp only [e_397_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 659)) := by
  have c25836 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25837 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25838 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25839 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25840 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25841 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_8 := arithEq_of_rows h (row := 397) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_8
  simp only [← c25836, ← c25837, ← c25838] at e_397_8
  have e_397_9 := arithEq_of_rows h (row := 397) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_9
  simp only [← c25839, ← c25840, k2, ← c25841] at e_397_9
  have hr := e_397_9
  simp only [e_397_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 660)) := by
  have c25842 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25843 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25844 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25845 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25846 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25847 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_10 := arithEq_of_rows h (row := 397) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_10
  simp only [← c25842, ← c25843, ← c25844] at e_397_10
  have e_397_11 := arithEq_of_rows h (row := 397) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_11
  simp only [← c25845, ← c25846, k2, ← c25847] at e_397_11
  have hr := e_397_11
  simp only [e_397_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 661)) := by
  have c25848 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25849 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25850 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25851 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25852 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25853 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_12 := arithEq_of_rows h (row := 397) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_12
  simp only [← c25848, ← c25849, ← c25850] at e_397_12
  have e_397_13 := arithEq_of_rows h (row := 397) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_13
  simp only [← c25851, ← c25852, k2, ← c25853] at e_397_13
  have hr := e_397_13
  simp only [e_397_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 662)) := by
  have c25854 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25855 := (publicBatchWrapper89_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25856 := (publicBatchWrapper89_copies808 a h).1
  have c25857 := (publicBatchWrapper89_copies808 a h).2.1
  have c25858 := (publicBatchWrapper89_copies808 a h).2.2.1
  have c25859 := (publicBatchWrapper89_copies808 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_14 := arithEq_of_rows h (row := 397) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_14
  simp only [← c25854, ← c25855, ← c25856] at e_397_14
  have e_397_15 := arithEq_of_rows h (row := 397) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_15
  simp only [← c25857, ← c25858, k2, ← c25859] at e_397_15
  have hr := e_397_15
  simp only [e_397_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 663)) := by
  have c25860 := (publicBatchWrapper89_copies808 a h).2.2.2.2.1
  have c25861 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.1
  have c25862 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.1
  have c25863 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.1
  have c25864 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.1
  have c25865 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_16 := arithEq_of_rows h (row := 397) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_16
  simp only [← c25860, ← c25861, ← c25862] at e_397_16
  have e_397_17 := arithEq_of_rows h (row := 397) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_17
  simp only [← c25863, ← c25864, k2, ← c25865] at e_397_17
  have hr := e_397_17
  simp only [e_397_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 397 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 664)) := by
  have c25866 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25867 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25868 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25869 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25870 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25871 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_397_18 := arithEq_of_rows h (row := 397) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_18
  simp only [← c25866, ← c25867, ← c25868] at e_397_18
  have e_397_19 := arithEq_of_rows h (row := 397) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_19
  simp only [← c25869, ← c25870, k2, ← c25871] at e_397_19
  have hr := e_397_19
  simp only [e_397_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 665)) := by
  have c25872 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25873 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25874 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25875 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25876 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25877 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_0 := arithEq_of_rows h (row := 398) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_0
  simp only [← c25872, ← c25873, ← c25874] at e_398_0
  have e_398_1 := arithEq_of_rows h (row := 398) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_1
  simp only [← c25875, ← c25876, k2, ← c25877] at e_398_1
  have hr := e_398_1
  simp only [e_398_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 666)) := by
  have c25878 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25879 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25880 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25881 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25882 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25883 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_2 := arithEq_of_rows h (row := 398) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_2
  simp only [← c25878, ← c25879, ← c25880] at e_398_2
  have e_398_3 := arithEq_of_rows h (row := 398) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_3
  simp only [← c25881, ← c25882, k2, ← c25883] at e_398_3
  have hr := e_398_3
  simp only [e_398_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 667)) := by
  have c25884 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25885 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25886 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25887 := (publicBatchWrapper89_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25888 := (publicBatchWrapper89_copies809 a h).1
  have c25889 := (publicBatchWrapper89_copies809 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_4 := arithEq_of_rows h (row := 398) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_4
  simp only [← c25884, ← c25885, ← c25886] at e_398_4
  have e_398_5 := arithEq_of_rows h (row := 398) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_5
  simp only [← c25887, ← c25888, k2, ← c25889] at e_398_5
  have hr := e_398_5
  simp only [e_398_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 668)) := by
  have c25890 := (publicBatchWrapper89_copies809 a h).2.2.1
  have c25891 := (publicBatchWrapper89_copies809 a h).2.2.2.1
  have c25892 := (publicBatchWrapper89_copies809 a h).2.2.2.2.1
  have c25893 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.1
  have c25894 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.1
  have c25895 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_6 := arithEq_of_rows h (row := 398) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_6
  simp only [← c25890, ← c25891, ← c25892] at e_398_6
  have e_398_7 := arithEq_of_rows h (row := 398) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_7
  simp only [← c25893, ← c25894, k2, ← c25895] at e_398_7
  have hr := e_398_7
  simp only [e_398_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 669)) := by
  have c25896 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.1
  have c25897 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.1
  have c25898 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25899 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25900 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25901 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_8 := arithEq_of_rows h (row := 398) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_8
  simp only [← c25896, ← c25897, ← c25898] at e_398_8
  have e_398_9 := arithEq_of_rows h (row := 398) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_9
  simp only [← c25899, ← c25900, k2, ← c25901] at e_398_9
  have hr := e_398_9
  simp only [e_398_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 670)) := by
  have c25902 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25903 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25904 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25905 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25906 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25907 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_10 := arithEq_of_rows h (row := 398) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_10
  simp only [← c25902, ← c25903, ← c25904] at e_398_10
  have e_398_11 := arithEq_of_rows h (row := 398) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_11
  simp only [← c25905, ← c25906, k2, ← c25907] at e_398_11
  have hr := e_398_11
  simp only [e_398_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 671)) := by
  have c25908 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25909 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25910 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25911 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25912 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25913 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_12 := arithEq_of_rows h (row := 398) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_12
  simp only [← c25908, ← c25909, ← c25910] at e_398_12
  have e_398_13 := arithEq_of_rows h (row := 398) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_13
  simp only [← c25911, ← c25912, k2, ← c25913] at e_398_13
  have hr := e_398_13
  simp only [e_398_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 672)) := by
  have c25914 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25915 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25916 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25917 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25918 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25919 := (publicBatchWrapper89_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_14 := arithEq_of_rows h (row := 398) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_14
  simp only [← c25914, ← c25915, ← c25916] at e_398_14
  have e_398_15 := arithEq_of_rows h (row := 398) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_15
  simp only [← c25917, ← c25918, k2, ← c25919] at e_398_15
  have hr := e_398_15
  simp only [e_398_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 673)) := by
  have c25920 := (publicBatchWrapper89_copies810 a h).1
  have c25921 := (publicBatchWrapper89_copies810 a h).2.1
  have c25922 := (publicBatchWrapper89_copies810 a h).2.2.1
  have c25923 := (publicBatchWrapper89_copies810 a h).2.2.2.1
  have c25924 := (publicBatchWrapper89_copies810 a h).2.2.2.2.1
  have c25925 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_16 := arithEq_of_rows h (row := 398) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_16
  simp only [← c25920, ← c25921, ← c25922] at e_398_16
  have e_398_17 := arithEq_of_rows h (row := 398) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_17
  simp only [← c25923, ← c25924, k2, ← c25925] at e_398_17
  have hr := e_398_17
  simp only [e_398_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 674)) := by
  have c25926 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.1
  have c25927 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.1
  have c25928 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.1
  have c25929 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.1
  have c25930 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25931 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_18 := arithEq_of_rows h (row := 398) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_18
  simp only [← c25926, ← c25927, ← c25928] at e_398_18
  have e_398_19 := arithEq_of_rows h (row := 398) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_19
  simp only [← c25929, ← c25930, k2, ← c25931] at e_398_19
  have hr := e_398_19
  simp only [e_398_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 675)) := by
  have c25932 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25933 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25934 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25935 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25936 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25937 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_0 := arithEq_of_rows h (row := 399) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_0
  simp only [← c25932, ← c25933, ← c25934] at e_399_0
  have e_399_1 := arithEq_of_rows h (row := 399) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_1
  simp only [← c25935, ← c25936, k2, ← c25937] at e_399_1
  have hr := e_399_1
  simp only [e_399_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 676)) := by
  have c25938 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25939 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25940 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25941 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25942 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25943 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_2 := arithEq_of_rows h (row := 399) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_2
  simp only [← c25938, ← c25939, ← c25940] at e_399_2
  have e_399_3 := arithEq_of_rows h (row := 399) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_3
  simp only [← c25941, ← c25942, k2, ← c25943] at e_399_3
  have hr := e_399_3
  simp only [e_399_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 677)) := by
  have c25944 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25945 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25946 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25947 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25948 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25949 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_4 := arithEq_of_rows h (row := 399) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_4
  simp only [← c25944, ← c25945, ← c25946] at e_399_4
  have e_399_5 := arithEq_of_rows h (row := 399) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_5
  simp only [← c25947, ← c25948, k2, ← c25949] at e_399_5
  have hr := e_399_5
  simp only [e_399_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 678)) := by
  have c25950 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25951 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25952 := (publicBatchWrapper89_copies811 a h).1
  have c25953 := (publicBatchWrapper89_copies811 a h).2.1
  have c25954 := (publicBatchWrapper89_copies811 a h).2.2.1
  have c25955 := (publicBatchWrapper89_copies811 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_6 := arithEq_of_rows h (row := 399) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_6
  simp only [← c25950, ← c25951, ← c25952] at e_399_6
  have e_399_7 := arithEq_of_rows h (row := 399) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_7
  simp only [← c25953, ← c25954, k2, ← c25955] at e_399_7
  have hr := e_399_7
  simp only [e_399_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 679)) := by
  have c25956 := (publicBatchWrapper89_copies811 a h).2.2.2.2.1
  have c25957 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.1
  have c25958 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.1
  have c25959 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.1
  have c25960 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.1
  have c25961 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_8 := arithEq_of_rows h (row := 399) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_8
  simp only [← c25956, ← c25957, ← c25958] at e_399_8
  have e_399_9 := arithEq_of_rows h (row := 399) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_9
  simp only [← c25959, ← c25960, k2, ← c25961] at e_399_9
  have hr := e_399_9
  simp only [e_399_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 680)) := by
  have c25962 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25963 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25964 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25965 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25966 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25967 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_10 := arithEq_of_rows h (row := 399) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_10
  simp only [← c25962, ← c25963, ← c25964] at e_399_10
  have e_399_11 := arithEq_of_rows h (row := 399) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_11
  simp only [← c25965, ← c25966, k2, ← c25967] at e_399_11
  have hr := e_399_11
  simp only [e_399_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 681)) := by
  have c25968 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25969 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25970 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25971 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25972 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25973 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_12 := arithEq_of_rows h (row := 399) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_12
  simp only [← c25968, ← c25969, ← c25970] at e_399_12
  have e_399_13 := arithEq_of_rows h (row := 399) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_13
  simp only [← c25971, ← c25972, k2, ← c25973] at e_399_13
  have hr := e_399_13
  simp only [e_399_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 682)) := by
  have c25974 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25975 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25976 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25977 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25978 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25979 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_14 := arithEq_of_rows h (row := 399) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_14
  simp only [← c25974, ← c25975, ← c25976] at e_399_14
  have e_399_15 := arithEq_of_rows h (row := 399) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_15
  simp only [← c25977, ← c25978, k2, ← c25979] at e_399_15
  have hr := e_399_15
  simp only [e_399_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 683)) := by
  have c25980 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25981 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25982 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25983 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25984 := (publicBatchWrapper89_copies812 a h).1
  have c25985 := (publicBatchWrapper89_copies812 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_16 := arithEq_of_rows h (row := 399) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_16
  simp only [← c25980, ← c25981, ← c25982] at e_399_16
  have e_399_17 := arithEq_of_rows h (row := 399) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_17
  simp only [← c25983, ← c25984, k2, ← c25985] at e_399_17
  have hr := e_399_17
  simp only [e_399_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 684)) := by
  have c25986 := (publicBatchWrapper89_copies812 a h).2.2.1
  have c25987 := (publicBatchWrapper89_copies812 a h).2.2.2.1
  have c25988 := (publicBatchWrapper89_copies812 a h).2.2.2.2.1
  have c25989 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.1
  have c25990 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.1
  have c25991 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_18 := arithEq_of_rows h (row := 399) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_18
  simp only [← c25986, ← c25987, ← c25988] at e_399_18
  have e_399_19 := arithEq_of_rows h (row := 399) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_19
  simp only [← c25989, ← c25990, k2, ← c25991] at e_399_19
  have hr := e_399_19
  simp only [e_399_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 685)) := by
  have c25992 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.1
  have c25993 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.1
  have c25994 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25995 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25996 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25997 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_0 := arithEq_of_rows h (row := 400) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_0
  simp only [← c25992, ← c25993, ← c25994] at e_400_0
  have e_400_1 := arithEq_of_rows h (row := 400) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_1
  simp only [← c25995, ← c25996, k2, ← c25997] at e_400_1
  have hr := e_400_1
  simp only [e_400_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 686)) := by
  have c25998 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25999 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26000 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26001 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26002 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26003 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_2 := arithEq_of_rows h (row := 400) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_2
  simp only [← c25998, ← c25999, ← c26000] at e_400_2
  have e_400_3 := arithEq_of_rows h (row := 400) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_3
  simp only [← c26001, ← c26002, k2, ← c26003] at e_400_3
  have hr := e_400_3
  simp only [e_400_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 687)) := by
  have c26004 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26005 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26006 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26007 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26008 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26009 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_4 := arithEq_of_rows h (row := 400) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_4
  simp only [← c26004, ← c26005, ← c26006] at e_400_4
  have e_400_5 := arithEq_of_rows h (row := 400) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_5
  simp only [← c26007, ← c26008, k2, ← c26009] at e_400_5
  have hr := e_400_5
  simp only [e_400_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 688)) := by
  have c26010 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26011 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26012 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26013 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26014 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26015 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_6 := arithEq_of_rows h (row := 400) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_6
  simp only [← c26010, ← c26011, ← c26012] at e_400_6
  have e_400_7 := arithEq_of_rows h (row := 400) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_7
  simp only [← c26013, ← c26014, k2, ← c26015] at e_400_7
  have hr := e_400_7
  simp only [e_400_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 689)) := by
  have c26016 := (publicBatchWrapper89_copies813 a h).1
  have c26017 := (publicBatchWrapper89_copies813 a h).2.1
  have c26018 := (publicBatchWrapper89_copies813 a h).2.2.1
  have c26019 := (publicBatchWrapper89_copies813 a h).2.2.2.1
  have c26020 := (publicBatchWrapper89_copies813 a h).2.2.2.2.1
  have c26021 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_8 := arithEq_of_rows h (row := 400) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_8
  simp only [← c26016, ← c26017, ← c26018] at e_400_8
  have e_400_9 := arithEq_of_rows h (row := 400) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_9
  simp only [← c26019, ← c26020, k2, ← c26021] at e_400_9
  have hr := e_400_9
  simp only [e_400_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 690)) := by
  have c26022 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.1
  have c26023 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.1
  have c26024 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.1
  have c26025 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.1
  have c26026 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26027 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_10 := arithEq_of_rows h (row := 400) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_10
  simp only [← c26022, ← c26023, ← c26024] at e_400_10
  have e_400_11 := arithEq_of_rows h (row := 400) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_11
  simp only [← c26025, ← c26026, k2, ← c26027] at e_400_11
  have hr := e_400_11
  simp only [e_400_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 691)) := by
  have c26028 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26029 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26030 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26031 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26032 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26033 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_12 := arithEq_of_rows h (row := 400) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_12
  simp only [← c26028, ← c26029, ← c26030] at e_400_12
  have e_400_13 := arithEq_of_rows h (row := 400) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_13
  simp only [← c26031, ← c26032, k2, ← c26033] at e_400_13
  have hr := e_400_13
  simp only [e_400_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 692)) := by
  have c26034 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26035 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26036 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26037 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26038 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26039 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_14 := arithEq_of_rows h (row := 400) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_14
  simp only [← c26034, ← c26035, ← c26036] at e_400_14
  have e_400_15 := arithEq_of_rows h (row := 400) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_15
  simp only [← c26037, ← c26038, k2, ← c26039] at e_400_15
  have hr := e_400_15
  simp only [e_400_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 693)) := by
  have c26040 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26041 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26042 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26043 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26044 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26045 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_16 := arithEq_of_rows h (row := 400) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_16
  simp only [← c26040, ← c26041, ← c26042] at e_400_16
  have e_400_17 := arithEq_of_rows h (row := 400) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_17
  simp only [← c26043, ← c26044, k2, ← c26045] at e_400_17
  have hr := e_400_17
  simp only [e_400_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 694)) := by
  have c26046 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26047 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26048 := (publicBatchWrapper89_copies814 a h).1
  have c26049 := (publicBatchWrapper89_copies814 a h).2.1
  have c26050 := (publicBatchWrapper89_copies814 a h).2.2.1
  have c26051 := (publicBatchWrapper89_copies814 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_18 := arithEq_of_rows h (row := 400) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_18
  simp only [← c26046, ← c26047, ← c26048] at e_400_18
  have e_400_19 := arithEq_of_rows h (row := 400) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_19
  simp only [← c26049, ← c26050, k2, ← c26051] at e_400_19
  have hr := e_400_19
  simp only [e_400_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 695)) := by
  have c26052 := (publicBatchWrapper89_copies814 a h).2.2.2.2.1
  have c26053 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.1
  have c26054 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.1
  have c26055 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.1
  have c26056 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.1
  have c26057 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_0 := arithEq_of_rows h (row := 401) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_0
  simp only [← c26052, ← c26053, ← c26054] at e_401_0
  have e_401_1 := arithEq_of_rows h (row := 401) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_1
  simp only [← c26055, ← c26056, k2, ← c26057] at e_401_1
  have hr := e_401_1
  simp only [e_401_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 696)) := by
  have c26058 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26059 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26060 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26061 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26062 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26063 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_2 := arithEq_of_rows h (row := 401) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_2
  simp only [← c26058, ← c26059, ← c26060] at e_401_2
  have e_401_3 := arithEq_of_rows h (row := 401) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_3
  simp only [← c26061, ← c26062, k2, ← c26063] at e_401_3
  have hr := e_401_3
  simp only [e_401_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 697)) := by
  have c26064 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26065 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26066 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26067 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26068 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26069 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_4 := arithEq_of_rows h (row := 401) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_4
  simp only [← c26064, ← c26065, ← c26066] at e_401_4
  have e_401_5 := arithEq_of_rows h (row := 401) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_5
  simp only [← c26067, ← c26068, k2, ← c26069] at e_401_5
  have hr := e_401_5
  simp only [e_401_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 698)) := by
  have c26070 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26071 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26072 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26073 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26074 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26075 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_6 := arithEq_of_rows h (row := 401) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_6
  simp only [← c26070, ← c26071, ← c26072] at e_401_6
  have e_401_7 := arithEq_of_rows h (row := 401) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_7
  simp only [← c26073, ← c26074, k2, ← c26075] at e_401_7
  have hr := e_401_7
  simp only [e_401_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 699)) := by
  have c26076 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26077 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26078 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26079 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26080 := (publicBatchWrapper89_copies815 a h).1
  have c26081 := (publicBatchWrapper89_copies815 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_8 := arithEq_of_rows h (row := 401) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_8
  simp only [← c26076, ← c26077, ← c26078] at e_401_8
  have e_401_9 := arithEq_of_rows h (row := 401) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_9
  simp only [← c26079, ← c26080, k2, ← c26081] at e_401_9
  have hr := e_401_9
  simp only [e_401_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 700)) := by
  have c26082 := (publicBatchWrapper89_copies815 a h).2.2.1
  have c26083 := (publicBatchWrapper89_copies815 a h).2.2.2.1
  have c26084 := (publicBatchWrapper89_copies815 a h).2.2.2.2.1
  have c26085 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.1
  have c26086 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.1
  have c26087 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_10 := arithEq_of_rows h (row := 401) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_10
  simp only [← c26082, ← c26083, ← c26084] at e_401_10
  have e_401_11 := arithEq_of_rows h (row := 401) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_11
  simp only [← c26085, ← c26086, k2, ← c26087] at e_401_11
  have hr := e_401_11
  simp only [e_401_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 701)) := by
  have c26088 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.1
  have c26089 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.1
  have c26090 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26091 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26092 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26093 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_12 := arithEq_of_rows h (row := 401) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_12
  simp only [← c26088, ← c26089, ← c26090] at e_401_12
  have e_401_13 := arithEq_of_rows h (row := 401) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_13
  simp only [← c26091, ← c26092, k2, ← c26093] at e_401_13
  have hr := e_401_13
  simp only [e_401_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 702)) := by
  have c26094 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26095 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26096 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26097 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26098 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26099 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_14 := arithEq_of_rows h (row := 401) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_14
  simp only [← c26094, ← c26095, ← c26096] at e_401_14
  have e_401_15 := arithEq_of_rows h (row := 401) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_15
  simp only [← c26097, ← c26098, k2, ← c26099] at e_401_15
  have hr := e_401_15
  simp only [e_401_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 703)) := by
  have c26100 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26101 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26102 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26103 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26104 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26105 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_16 := arithEq_of_rows h (row := 401) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_16
  simp only [← c26100, ← c26101, ← c26102] at e_401_16
  have e_401_17 := arithEq_of_rows h (row := 401) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_17
  simp only [← c26103, ← c26104, k2, ← c26105] at e_401_17
  have hr := e_401_17
  simp only [e_401_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 704)) := by
  have c26106 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26107 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26108 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26109 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26110 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26111 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_18 := arithEq_of_rows h (row := 401) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_18
  simp only [← c26106, ← c26107, ← c26108] at e_401_18
  have e_401_19 := arithEq_of_rows h (row := 401) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_19
  simp only [← c26109, ← c26110, k2, ← c26111] at e_401_19
  have hr := e_401_19
  simp only [e_401_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 705)) := by
  have c26112 := (publicBatchWrapper89_copies816 a h).1
  have c26113 := (publicBatchWrapper89_copies816 a h).2.1
  have c26114 := (publicBatchWrapper89_copies816 a h).2.2.1
  have c26115 := (publicBatchWrapper89_copies816 a h).2.2.2.1
  have c26116 := (publicBatchWrapper89_copies816 a h).2.2.2.2.1
  have c26117 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_0 := arithEq_of_rows h (row := 402) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_0
  simp only [← c26112, ← c26113, ← c26114] at e_402_0
  have e_402_1 := arithEq_of_rows h (row := 402) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_1
  simp only [← c26115, ← c26116, k2, ← c26117] at e_402_1
  have hr := e_402_1
  simp only [e_402_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 706)) := by
  have c26118 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.1
  have c26119 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.1
  have c26120 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.1
  have c26121 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.1
  have c26122 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26123 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_2 := arithEq_of_rows h (row := 402) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_2
  simp only [← c26118, ← c26119, ← c26120] at e_402_2
  have e_402_3 := arithEq_of_rows h (row := 402) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_3
  simp only [← c26121, ← c26122, k2, ← c26123] at e_402_3
  have hr := e_402_3
  simp only [e_402_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 707)) := by
  have c26124 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26125 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26126 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26127 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26128 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26129 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_4 := arithEq_of_rows h (row := 402) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_4
  simp only [← c26124, ← c26125, ← c26126] at e_402_4
  have e_402_5 := arithEq_of_rows h (row := 402) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_5
  simp only [← c26127, ← c26128, k2, ← c26129] at e_402_5
  have hr := e_402_5
  simp only [e_402_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 708)) := by
  have c26130 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26131 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26132 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26133 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26134 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26135 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_6 := arithEq_of_rows h (row := 402) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_6
  simp only [← c26130, ← c26131, ← c26132] at e_402_6
  have e_402_7 := arithEq_of_rows h (row := 402) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_7
  simp only [← c26133, ← c26134, k2, ← c26135] at e_402_7
  have hr := e_402_7
  simp only [e_402_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 709)) := by
  have c26136 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26137 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26138 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26139 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26140 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26141 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_8 := arithEq_of_rows h (row := 402) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_8
  simp only [← c26136, ← c26137, ← c26138] at e_402_8
  have e_402_9 := arithEq_of_rows h (row := 402) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_9
  simp only [← c26139, ← c26140, k2, ← c26141] at e_402_9
  have hr := e_402_9
  simp only [e_402_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 710)) := by
  have c26142 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26143 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26144 := (publicBatchWrapper89_copies817 a h).1
  have c26145 := (publicBatchWrapper89_copies817 a h).2.1
  have c26146 := (publicBatchWrapper89_copies817 a h).2.2.1
  have c26147 := (publicBatchWrapper89_copies817 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_10 := arithEq_of_rows h (row := 402) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_10
  simp only [← c26142, ← c26143, ← c26144] at e_402_10
  have e_402_11 := arithEq_of_rows h (row := 402) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_11
  simp only [← c26145, ← c26146, k2, ← c26147] at e_402_11
  have hr := e_402_11
  simp only [e_402_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 711)) := by
  have c26148 := (publicBatchWrapper89_copies817 a h).2.2.2.2.1
  have c26149 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.1
  have c26150 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.1
  have c26151 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.1
  have c26152 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.1
  have c26153 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_12 := arithEq_of_rows h (row := 402) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_12
  simp only [← c26148, ← c26149, ← c26150] at e_402_12
  have e_402_13 := arithEq_of_rows h (row := 402) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_13
  simp only [← c26151, ← c26152, k2, ← c26153] at e_402_13
  have hr := e_402_13
  simp only [e_402_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 712)) := by
  have c26154 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26155 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26156 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26157 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26158 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26159 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_14 := arithEq_of_rows h (row := 402) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_14
  simp only [← c26154, ← c26155, ← c26156] at e_402_14
  have e_402_15 := arithEq_of_rows h (row := 402) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_15
  simp only [← c26157, ← c26158, k2, ← c26159] at e_402_15
  have hr := e_402_15
  simp only [e_402_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 713)) := by
  have c26160 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26161 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26162 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26163 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26164 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26165 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_16 := arithEq_of_rows h (row := 402) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_16
  simp only [← c26160, ← c26161, ← c26162] at e_402_16
  have e_402_17 := arithEq_of_rows h (row := 402) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_17
  simp only [← c26163, ← c26164, k2, ← c26165] at e_402_17
  have hr := e_402_17
  simp only [e_402_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 714)) := by
  have c26166 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26167 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26168 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26169 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26170 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26171 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_18 := arithEq_of_rows h (row := 402) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_18
  simp only [← c26166, ← c26167, ← c26168] at e_402_18
  have e_402_19 := arithEq_of_rows h (row := 402) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_19
  simp only [← c26169, ← c26170, k2, ← c26171] at e_402_19
  have hr := e_402_19
  simp only [e_402_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 715)) := by
  have c26172 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26173 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26174 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26175 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26176 := (publicBatchWrapper89_copies818 a h).1
  have c26177 := (publicBatchWrapper89_copies818 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_0 := arithEq_of_rows h (row := 403) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_0
  simp only [← c26172, ← c26173, ← c26174] at e_403_0
  have e_403_1 := arithEq_of_rows h (row := 403) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_1
  simp only [← c26175, ← c26176, k2, ← c26177] at e_403_1
  have hr := e_403_1
  simp only [e_403_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 716)) := by
  have c26178 := (publicBatchWrapper89_copies818 a h).2.2.1
  have c26179 := (publicBatchWrapper89_copies818 a h).2.2.2.1
  have c26180 := (publicBatchWrapper89_copies818 a h).2.2.2.2.1
  have c26181 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.1
  have c26182 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.1
  have c26183 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_2 := arithEq_of_rows h (row := 403) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_2
  simp only [← c26178, ← c26179, ← c26180] at e_403_2
  have e_403_3 := arithEq_of_rows h (row := 403) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_3
  simp only [← c26181, ← c26182, k2, ← c26183] at e_403_3
  have hr := e_403_3
  simp only [e_403_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 717)) := by
  have c26184 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.1
  have c26185 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.1
  have c26186 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26187 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26188 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26189 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_4 := arithEq_of_rows h (row := 403) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_4
  simp only [← c26184, ← c26185, ← c26186] at e_403_4
  have e_403_5 := arithEq_of_rows h (row := 403) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_5
  simp only [← c26187, ← c26188, k2, ← c26189] at e_403_5
  have hr := e_403_5
  simp only [e_403_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 718)) := by
  have c26190 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26191 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26192 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26193 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26194 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26195 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_6 := arithEq_of_rows h (row := 403) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_6
  simp only [← c26190, ← c26191, ← c26192] at e_403_6
  have e_403_7 := arithEq_of_rows h (row := 403) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_7
  simp only [← c26193, ← c26194, k2, ← c26195] at e_403_7
  have hr := e_403_7
  simp only [e_403_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 719)) := by
  have c26196 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26197 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26198 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26199 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26200 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26201 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_8 := arithEq_of_rows h (row := 403) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_8
  simp only [← c26196, ← c26197, ← c26198] at e_403_8
  have e_403_9 := arithEq_of_rows h (row := 403) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_9
  simp only [← c26199, ← c26200, k2, ← c26201] at e_403_9
  have hr := e_403_9
  simp only [e_403_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 720)) := by
  have c26202 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26203 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26204 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26205 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26206 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26207 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_10 := arithEq_of_rows h (row := 403) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_10
  simp only [← c26202, ← c26203, ← c26204] at e_403_10
  have e_403_11 := arithEq_of_rows h (row := 403) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_11
  simp only [← c26205, ← c26206, k2, ← c26207] at e_403_11
  have hr := e_403_11
  simp only [e_403_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 721)) := by
  have c26208 := (publicBatchWrapper89_copies819 a h).1
  have c26209 := (publicBatchWrapper89_copies819 a h).2.1
  have c26210 := (publicBatchWrapper89_copies819 a h).2.2.1
  have c26211 := (publicBatchWrapper89_copies819 a h).2.2.2.1
  have c26212 := (publicBatchWrapper89_copies819 a h).2.2.2.2.1
  have c26213 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_12 := arithEq_of_rows h (row := 403) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_12
  simp only [← c26208, ← c26209, ← c26210] at e_403_12
  have e_403_13 := arithEq_of_rows h (row := 403) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_13
  simp only [← c26211, ← c26212, k2, ← c26213] at e_403_13
  have hr := e_403_13
  simp only [e_403_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 722)) := by
  have c26214 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.1
  have c26215 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.1
  have c26216 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.1
  have c26217 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.1
  have c26218 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26219 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_14 := arithEq_of_rows h (row := 403) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_14
  simp only [← c26214, ← c26215, ← c26216] at e_403_14
  have e_403_15 := arithEq_of_rows h (row := 403) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_15
  simp only [← c26217, ← c26218, k2, ← c26219] at e_403_15
  have hr := e_403_15
  simp only [e_403_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 723)) := by
  have c26220 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26221 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26222 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26223 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26224 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26225 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_16 := arithEq_of_rows h (row := 403) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_16
  simp only [← c26220, ← c26221, ← c26222] at e_403_16
  have e_403_17 := arithEq_of_rows h (row := 403) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_17
  simp only [← c26223, ← c26224, k2, ← c26225] at e_403_17
  have hr := e_403_17
  simp only [e_403_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 724)) := by
  have c26226 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26227 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26228 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26229 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26230 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26231 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_18 := arithEq_of_rows h (row := 403) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_18
  simp only [← c26226, ← c26227, ← c26228] at e_403_18
  have e_403_19 := arithEq_of_rows h (row := 403) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_19
  simp only [← c26229, ← c26230, k2, ← c26231] at e_403_19
  have hr := e_403_19
  simp only [e_403_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 725)) := by
  have c26232 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26233 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26234 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26235 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26236 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26237 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_0 := arithEq_of_rows h (row := 404) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_0
  simp only [← c26232, ← c26233, ← c26234] at e_404_0
  have e_404_1 := arithEq_of_rows h (row := 404) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_1
  simp only [← c26235, ← c26236, k2, ← c26237] at e_404_1
  have hr := e_404_1
  simp only [e_404_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 726)) := by
  have c26238 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26239 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26240 := (publicBatchWrapper89_copies820 a h).1
  have c26241 := (publicBatchWrapper89_copies820 a h).2.1
  have c26242 := (publicBatchWrapper89_copies820 a h).2.2.1
  have c26243 := (publicBatchWrapper89_copies820 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_2 := arithEq_of_rows h (row := 404) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_2
  simp only [← c26238, ← c26239, ← c26240] at e_404_2
  have e_404_3 := arithEq_of_rows h (row := 404) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_3
  simp only [← c26241, ← c26242, k2, ← c26243] at e_404_3
  have hr := e_404_3
  simp only [e_404_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 727)) := by
  have c26244 := (publicBatchWrapper89_copies820 a h).2.2.2.2.1
  have c26245 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.1
  have c26246 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.1
  have c26247 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.1
  have c26248 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.1
  have c26249 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_4 := arithEq_of_rows h (row := 404) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_4
  simp only [← c26244, ← c26245, ← c26246] at e_404_4
  have e_404_5 := arithEq_of_rows h (row := 404) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_5
  simp only [← c26247, ← c26248, k2, ← c26249] at e_404_5
  have hr := e_404_5
  simp only [e_404_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 728)) := by
  have c26250 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26251 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26252 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26253 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26254 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26255 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_6 := arithEq_of_rows h (row := 404) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_6
  simp only [← c26250, ← c26251, ← c26252] at e_404_6
  have e_404_7 := arithEq_of_rows h (row := 404) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_7
  simp only [← c26253, ← c26254, k2, ← c26255] at e_404_7
  have hr := e_404_7
  simp only [e_404_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 729)) := by
  have c26256 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26257 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26258 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26259 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26260 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26261 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_8 := arithEq_of_rows h (row := 404) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_8
  simp only [← c26256, ← c26257, ← c26258] at e_404_8
  have e_404_9 := arithEq_of_rows h (row := 404) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_9
  simp only [← c26259, ← c26260, k2, ← c26261] at e_404_9
  have hr := e_404_9
  simp only [e_404_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 730)) := by
  have c26262 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26263 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26264 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26265 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26266 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26267 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_10 := arithEq_of_rows h (row := 404) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_10
  simp only [← c26262, ← c26263, ← c26264] at e_404_10
  have e_404_11 := arithEq_of_rows h (row := 404) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_11
  simp only [← c26265, ← c26266, k2, ← c26267] at e_404_11
  have hr := e_404_11
  simp only [e_404_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 731)) := by
  have c26268 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26269 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26270 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26271 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26272 := (publicBatchWrapper89_copies821 a h).1
  have c26273 := (publicBatchWrapper89_copies821 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_12 := arithEq_of_rows h (row := 404) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_12
  simp only [← c26268, ← c26269, ← c26270] at e_404_12
  have e_404_13 := arithEq_of_rows h (row := 404) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_13
  simp only [← c26271, ← c26272, k2, ← c26273] at e_404_13
  have hr := e_404_13
  simp only [e_404_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 732)) := by
  have c26274 := (publicBatchWrapper89_copies821 a h).2.2.1
  have c26275 := (publicBatchWrapper89_copies821 a h).2.2.2.1
  have c26276 := (publicBatchWrapper89_copies821 a h).2.2.2.2.1
  have c26277 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.1
  have c26278 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.1
  have c26279 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_14 := arithEq_of_rows h (row := 404) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_14
  simp only [← c26274, ← c26275, ← c26276] at e_404_14
  have e_404_15 := arithEq_of_rows h (row := 404) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_15
  simp only [← c26277, ← c26278, k2, ← c26279] at e_404_15
  have hr := e_404_15
  simp only [e_404_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 733)) := by
  have c26280 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.1
  have c26281 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.1
  have c26282 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26283 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26284 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26285 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_16 := arithEq_of_rows h (row := 404) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_16
  simp only [← c26280, ← c26281, ← c26282] at e_404_16
  have e_404_17 := arithEq_of_rows h (row := 404) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_17
  simp only [← c26283, ← c26284, k2, ← c26285] at e_404_17
  have hr := e_404_17
  simp only [e_404_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 734)) := by
  have c26286 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26287 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26288 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26289 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26290 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26291 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_18 := arithEq_of_rows h (row := 404) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_18
  simp only [← c26286, ← c26287, ← c26288] at e_404_18
  have e_404_19 := arithEq_of_rows h (row := 404) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_19
  simp only [← c26289, ← c26290, k2, ← c26291] at e_404_19
  have hr := e_404_19
  simp only [e_404_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 735)) := by
  have c26292 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26293 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26294 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26295 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26296 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26297 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_0 := arithEq_of_rows h (row := 405) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_0
  simp only [← c26292, ← c26293, ← c26294] at e_405_0
  have e_405_1 := arithEq_of_rows h (row := 405) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_1
  simp only [← c26295, ← c26296, k2, ← c26297] at e_405_1
  have hr := e_405_1
  simp only [e_405_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 736)) := by
  have c26298 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26299 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26300 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26301 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26302 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26303 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_2 := arithEq_of_rows h (row := 405) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_2
  simp only [← c26298, ← c26299, ← c26300] at e_405_2
  have e_405_3 := arithEq_of_rows h (row := 405) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_3
  simp only [← c26301, ← c26302, k2, ← c26303] at e_405_3
  have hr := e_405_3
  simp only [e_405_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 737)) := by
  have c26304 := (publicBatchWrapper89_copies822 a h).1
  have c26305 := (publicBatchWrapper89_copies822 a h).2.1
  have c26306 := (publicBatchWrapper89_copies822 a h).2.2.1
  have c26307 := (publicBatchWrapper89_copies822 a h).2.2.2.1
  have c26308 := (publicBatchWrapper89_copies822 a h).2.2.2.2.1
  have c26309 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_4 := arithEq_of_rows h (row := 405) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_4
  simp only [← c26304, ← c26305, ← c26306] at e_405_4
  have e_405_5 := arithEq_of_rows h (row := 405) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_5
  simp only [← c26307, ← c26308, k2, ← c26309] at e_405_5
  have hr := e_405_5
  simp only [e_405_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 738)) := by
  have c26310 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.1
  have c26311 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.1
  have c26312 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.1
  have c26313 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.1
  have c26314 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26315 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_6 := arithEq_of_rows h (row := 405) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_6
  simp only [← c26310, ← c26311, ← c26312] at e_405_6
  have e_405_7 := arithEq_of_rows h (row := 405) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_7
  simp only [← c26313, ← c26314, k2, ← c26315] at e_405_7
  have hr := e_405_7
  simp only [e_405_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 739)) := by
  have c26316 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26317 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26318 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26319 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26320 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26321 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_8 := arithEq_of_rows h (row := 405) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_8
  simp only [← c26316, ← c26317, ← c26318] at e_405_8
  have e_405_9 := arithEq_of_rows h (row := 405) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_9
  simp only [← c26319, ← c26320, k2, ← c26321] at e_405_9
  have hr := e_405_9
  simp only [e_405_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
