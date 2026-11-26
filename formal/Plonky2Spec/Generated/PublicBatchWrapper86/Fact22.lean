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

theorem publicBatchWrapper86_f17004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2170)) := by
  have c106377 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.1
  have c106378 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106379 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106380 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106381 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106382 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_4 := arithEq_of_rows h (row := 1741) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_4
  simp only [← c106377, ← c106378, ← c106379] at e_1741_4
  have e_1741_5 := arithEq_of_rows h (row := 1741) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_5
  simp only [← c106380, ← c106381, k2, ← c106382] at e_1741_5
  have hr := e_1741_5
  simp only [e_1741_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2171)) := by
  have c106383 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106384 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106385 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106386 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106387 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106388 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_6 := arithEq_of_rows h (row := 1741) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_6
  simp only [← c106383, ← c106384, ← c106385] at e_1741_6
  have e_1741_7 := arithEq_of_rows h (row := 1741) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_7
  simp only [← c106386, ← c106387, k2, ← c106388] at e_1741_7
  have hr := e_1741_7
  simp only [e_1741_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2172)) := by
  have c106389 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106390 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106391 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106392 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106393 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106394 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_8 := arithEq_of_rows h (row := 1741) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_8
  simp only [← c106389, ← c106390, ← c106391] at e_1741_8
  have e_1741_9 := arithEq_of_rows h (row := 1741) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_9
  simp only [← c106392, ← c106393, k2, ← c106394] at e_1741_9
  have hr := e_1741_9
  simp only [e_1741_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2173)) := by
  have c106395 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106396 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106397 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106398 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106399 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106400 := (publicBatchWrapper86_copies3325 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_10 := arithEq_of_rows h (row := 1741) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_10
  simp only [← c106395, ← c106396, ← c106397] at e_1741_10
  have e_1741_11 := arithEq_of_rows h (row := 1741) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_11
  simp only [← c106398, ← c106399, k2, ← c106400] at e_1741_11
  have hr := e_1741_11
  simp only [e_1741_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2174)) := by
  have c106401 := (publicBatchWrapper86_copies3325 a h).2.1
  have c106402 := (publicBatchWrapper86_copies3325 a h).2.2.1
  have c106403 := (publicBatchWrapper86_copies3325 a h).2.2.2.1
  have c106404 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.1
  have c106405 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.1
  have c106406 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_12 := arithEq_of_rows h (row := 1741) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_12
  simp only [← c106401, ← c106402, ← c106403] at e_1741_12
  have e_1741_13 := arithEq_of_rows h (row := 1741) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_13
  simp only [← c106404, ← c106405, k2, ← c106406] at e_1741_13
  have hr := e_1741_13
  simp only [e_1741_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2175)) := by
  have c106407 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.1
  have c106408 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.1
  have c106409 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.1
  have c106410 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106411 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106412 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_14 := arithEq_of_rows h (row := 1741) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_14
  simp only [← c106407, ← c106408, ← c106409] at e_1741_14
  have e_1741_15 := arithEq_of_rows h (row := 1741) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_15
  simp only [← c106410, ← c106411, k2, ← c106412] at e_1741_15
  have hr := e_1741_15
  simp only [e_1741_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2176)) := by
  have c106413 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106414 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106415 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106416 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106417 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106418 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_16 := arithEq_of_rows h (row := 1741) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_16
  simp only [← c106413, ← c106414, ← c106415] at e_1741_16
  have e_1741_17 := arithEq_of_rows h (row := 1741) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_17
  simp only [← c106416, ← c106417, k2, ← c106418] at e_1741_17
  have hr := e_1741_17
  simp only [e_1741_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2177)) := by
  have c106419 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106420 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106421 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106422 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106423 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106424 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_18 := arithEq_of_rows h (row := 1741) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_18
  simp only [← c106419, ← c106420, ← c106421] at e_1741_18
  have e_1741_19 := arithEq_of_rows h (row := 1741) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_19
  simp only [← c106422, ← c106423, k2, ← c106424] at e_1741_19
  have hr := e_1741_19
  simp only [e_1741_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2178)) := by
  have c106425 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106426 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106427 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106428 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106429 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106430 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_0 := arithEq_of_rows h (row := 1742) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_0
  simp only [← c106425, ← c106426, ← c106427] at e_1742_0
  have e_1742_1 := arithEq_of_rows h (row := 1742) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_1
  simp only [← c106428, ← c106429, k2, ← c106430] at e_1742_1
  have hr := e_1742_1
  simp only [e_1742_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2179)) := by
  have c106431 := (publicBatchWrapper86_copies3325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106432 := (publicBatchWrapper86_copies3326 a h).1
  have c106433 := (publicBatchWrapper86_copies3326 a h).2.1
  have c106434 := (publicBatchWrapper86_copies3326 a h).2.2.1
  have c106435 := (publicBatchWrapper86_copies3326 a h).2.2.2.1
  have c106436 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_2 := arithEq_of_rows h (row := 1742) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_2
  simp only [← c106431, ← c106432, ← c106433] at e_1742_2
  have e_1742_3 := arithEq_of_rows h (row := 1742) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_3
  simp only [← c106434, ← c106435, k2, ← c106436] at e_1742_3
  have hr := e_1742_3
  simp only [e_1742_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2180)) := by
  have c106437 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.1
  have c106438 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.1
  have c106439 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.1
  have c106440 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.1
  have c106441 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.1
  have c106442 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_4 := arithEq_of_rows h (row := 1742) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_4
  simp only [← c106437, ← c106438, ← c106439] at e_1742_4
  have e_1742_5 := arithEq_of_rows h (row := 1742) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_5
  simp only [← c106440, ← c106441, k2, ← c106442] at e_1742_5
  have hr := e_1742_5
  simp only [e_1742_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2181)) := by
  have c106443 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106444 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106445 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106446 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106447 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106448 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_6 := arithEq_of_rows h (row := 1742) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_6
  simp only [← c106443, ← c106444, ← c106445] at e_1742_6
  have e_1742_7 := arithEq_of_rows h (row := 1742) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_7
  simp only [← c106446, ← c106447, k2, ← c106448] at e_1742_7
  have hr := e_1742_7
  simp only [e_1742_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2182)) := by
  have c106449 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106450 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106451 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106452 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106453 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106454 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_8 := arithEq_of_rows h (row := 1742) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_8
  simp only [← c106449, ← c106450, ← c106451] at e_1742_8
  have e_1742_9 := arithEq_of_rows h (row := 1742) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_9
  simp only [← c106452, ← c106453, k2, ← c106454] at e_1742_9
  have hr := e_1742_9
  simp only [e_1742_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2183)) := by
  have c106455 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106456 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106457 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106458 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106459 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106460 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_10 := arithEq_of_rows h (row := 1742) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_10
  simp only [← c106455, ← c106456, ← c106457] at e_1742_10
  have e_1742_11 := arithEq_of_rows h (row := 1742) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_11
  simp only [← c106458, ← c106459, k2, ← c106460] at e_1742_11
  have hr := e_1742_11
  simp only [e_1742_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2184)) := by
  have c106461 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106462 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106463 := (publicBatchWrapper86_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106464 := (publicBatchWrapper86_copies3327 a h).1
  have c106465 := (publicBatchWrapper86_copies3327 a h).2.1
  have c106466 := (publicBatchWrapper86_copies3327 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_12 := arithEq_of_rows h (row := 1742) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_12
  simp only [← c106461, ← c106462, ← c106463] at e_1742_12
  have e_1742_13 := arithEq_of_rows h (row := 1742) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_13
  simp only [← c106464, ← c106465, k2, ← c106466] at e_1742_13
  have hr := e_1742_13
  simp only [e_1742_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2185)) := by
  have c106467 := (publicBatchWrapper86_copies3327 a h).2.2.2.1
  have c106468 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.1
  have c106469 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.1
  have c106470 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.1
  have c106471 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.1
  have c106472 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_14 := arithEq_of_rows h (row := 1742) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_14
  simp only [← c106467, ← c106468, ← c106469] at e_1742_14
  have e_1742_15 := arithEq_of_rows h (row := 1742) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_15
  simp only [← c106470, ← c106471, k2, ← c106472] at e_1742_15
  have hr := e_1742_15
  simp only [e_1742_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2186)) := by
  have c106473 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.1
  have c106474 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106475 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106476 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106477 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106478 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_16 := arithEq_of_rows h (row := 1742) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_16
  simp only [← c106473, ← c106474, ← c106475] at e_1742_16
  have e_1742_17 := arithEq_of_rows h (row := 1742) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_17
  simp only [← c106476, ← c106477, k2, ← c106478] at e_1742_17
  have hr := e_1742_17
  simp only [e_1742_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1742 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2187)) := by
  have c106479 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106480 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106481 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106482 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106483 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106484 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1742_18 := arithEq_of_rows h (row := 1742) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_18
  simp only [← c106479, ← c106480, ← c106481] at e_1742_18
  have e_1742_19 := arithEq_of_rows h (row := 1742) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_19
  simp only [← c106482, ← c106483, k2, ← c106484] at e_1742_19
  have hr := e_1742_19
  simp only [e_1742_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2188)) := by
  have c106485 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106486 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106487 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106488 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106489 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106490 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_0 := arithEq_of_rows h (row := 1743) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_0
  simp only [← c106485, ← c106486, ← c106487] at e_1743_0
  have e_1743_1 := arithEq_of_rows h (row := 1743) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_1
  simp only [← c106488, ← c106489, k2, ← c106490] at e_1743_1
  have hr := e_1743_1
  simp only [e_1743_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2189)) := by
  have c106491 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106492 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106493 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106494 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106495 := (publicBatchWrapper86_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106496 := (publicBatchWrapper86_copies3328 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_2 := arithEq_of_rows h (row := 1743) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_2
  simp only [← c106491, ← c106492, ← c106493] at e_1743_2
  have e_1743_3 := arithEq_of_rows h (row := 1743) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_3
  simp only [← c106494, ← c106495, k2, ← c106496] at e_1743_3
  have hr := e_1743_3
  simp only [e_1743_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2190)) := by
  have c106497 := (publicBatchWrapper86_copies3328 a h).2.1
  have c106498 := (publicBatchWrapper86_copies3328 a h).2.2.1
  have c106499 := (publicBatchWrapper86_copies3328 a h).2.2.2.1
  have c106500 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.1
  have c106501 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.1
  have c106502 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_4 := arithEq_of_rows h (row := 1743) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_4
  simp only [← c106497, ← c106498, ← c106499] at e_1743_4
  have e_1743_5 := arithEq_of_rows h (row := 1743) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_5
  simp only [← c106500, ← c106501, k2, ← c106502] at e_1743_5
  have hr := e_1743_5
  simp only [e_1743_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2191)) := by
  have c106503 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.1
  have c106504 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.1
  have c106505 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.1
  have c106506 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106507 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106508 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_6 := arithEq_of_rows h (row := 1743) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_6
  simp only [← c106503, ← c106504, ← c106505] at e_1743_6
  have e_1743_7 := arithEq_of_rows h (row := 1743) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_7
  simp only [← c106506, ← c106507, k2, ← c106508] at e_1743_7
  have hr := e_1743_7
  simp only [e_1743_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2192)) := by
  have c106509 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106510 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106511 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106512 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106513 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106514 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_8 := arithEq_of_rows h (row := 1743) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_8
  simp only [← c106509, ← c106510, ← c106511] at e_1743_8
  have e_1743_9 := arithEq_of_rows h (row := 1743) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_9
  simp only [← c106512, ← c106513, k2, ← c106514] at e_1743_9
  have hr := e_1743_9
  simp only [e_1743_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2193)) := by
  have c106515 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106516 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106517 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106518 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106519 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106520 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_10 := arithEq_of_rows h (row := 1743) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_10
  simp only [← c106515, ← c106516, ← c106517] at e_1743_10
  have e_1743_11 := arithEq_of_rows h (row := 1743) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_11
  simp only [← c106518, ← c106519, k2, ← c106520] at e_1743_11
  have hr := e_1743_11
  simp only [e_1743_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2194)) := by
  have c106521 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106522 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106523 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106524 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106525 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106526 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_12 := arithEq_of_rows h (row := 1743) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_12
  simp only [← c106521, ← c106522, ← c106523] at e_1743_12
  have e_1743_13 := arithEq_of_rows h (row := 1743) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_13
  simp only [← c106524, ← c106525, k2, ← c106526] at e_1743_13
  have hr := e_1743_13
  simp only [e_1743_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2195)) := by
  have c106527 := (publicBatchWrapper86_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106528 := (publicBatchWrapper86_copies3329 a h).1
  have c106529 := (publicBatchWrapper86_copies3329 a h).2.1
  have c106530 := (publicBatchWrapper86_copies3329 a h).2.2.1
  have c106531 := (publicBatchWrapper86_copies3329 a h).2.2.2.1
  have c106532 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_14 := arithEq_of_rows h (row := 1743) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_14
  simp only [← c106527, ← c106528, ← c106529] at e_1743_14
  have e_1743_15 := arithEq_of_rows h (row := 1743) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_15
  simp only [← c106530, ← c106531, k2, ← c106532] at e_1743_15
  have hr := e_1743_15
  simp only [e_1743_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2196)) := by
  have c106533 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.1
  have c106534 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.1
  have c106535 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.1
  have c106536 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.1
  have c106537 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.1
  have c106538 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_16 := arithEq_of_rows h (row := 1743) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_16
  simp only [← c106533, ← c106534, ← c106535] at e_1743_16
  have e_1743_17 := arithEq_of_rows h (row := 1743) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_17
  simp only [← c106536, ← c106537, k2, ← c106538] at e_1743_17
  have hr := e_1743_17
  simp only [e_1743_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1743 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2197)) := by
  have c106539 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106540 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106541 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106542 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106543 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106544 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1743_18 := arithEq_of_rows h (row := 1743) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_18
  simp only [← c106539, ← c106540, ← c106541] at e_1743_18
  have e_1743_19 := arithEq_of_rows h (row := 1743) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_19
  simp only [← c106542, ← c106543, k2, ← c106544] at e_1743_19
  have hr := e_1743_19
  simp only [e_1743_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2198)) := by
  have c106545 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106546 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106547 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106548 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106549 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106550 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_0 := arithEq_of_rows h (row := 1744) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_0
  simp only [← c106545, ← c106546, ← c106547] at e_1744_0
  have e_1744_1 := arithEq_of_rows h (row := 1744) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_1
  simp only [← c106548, ← c106549, k2, ← c106550] at e_1744_1
  have hr := e_1744_1
  simp only [e_1744_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2199)) := by
  have c106551 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106552 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106553 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106554 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106555 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106556 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_2 := arithEq_of_rows h (row := 1744) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_2
  simp only [← c106551, ← c106552, ← c106553] at e_1744_2
  have e_1744_3 := arithEq_of_rows h (row := 1744) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_3
  simp only [← c106554, ← c106555, k2, ← c106556] at e_1744_3
  have hr := e_1744_3
  simp only [e_1744_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2200)) := by
  have c106557 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106558 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106559 := (publicBatchWrapper86_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106560 := (publicBatchWrapper86_copies3330 a h).1
  have c106561 := (publicBatchWrapper86_copies3330 a h).2.1
  have c106562 := (publicBatchWrapper86_copies3330 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_4 := arithEq_of_rows h (row := 1744) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_4
  simp only [← c106557, ← c106558, ← c106559] at e_1744_4
  have e_1744_5 := arithEq_of_rows h (row := 1744) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_5
  simp only [← c106560, ← c106561, k2, ← c106562] at e_1744_5
  have hr := e_1744_5
  simp only [e_1744_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2201)) := by
  have c106563 := (publicBatchWrapper86_copies3330 a h).2.2.2.1
  have c106564 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.1
  have c106565 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.1
  have c106566 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.1
  have c106567 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.1
  have c106568 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_6 := arithEq_of_rows h (row := 1744) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_6
  simp only [← c106563, ← c106564, ← c106565] at e_1744_6
  have e_1744_7 := arithEq_of_rows h (row := 1744) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_7
  simp only [← c106566, ← c106567, k2, ← c106568] at e_1744_7
  have hr := e_1744_7
  simp only [e_1744_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2370)) := by
  have c106569 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.1
  have c106570 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106571 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106572 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106573 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106574 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_8 := arithEq_of_rows h (row := 1744) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_8
  simp only [← c106569, ← c106570, ← c106571] at e_1744_8
  have e_1744_9 := arithEq_of_rows h (row := 1744) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_9
  simp only [← c106572, ← c106573, k2, ← c106574] at e_1744_9
  have hr := e_1744_9
  simp only [e_1744_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2371)) := by
  have c106575 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106576 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106577 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106578 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106579 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106580 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_10 := arithEq_of_rows h (row := 1744) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_10
  simp only [← c106575, ← c106576, ← c106577] at e_1744_10
  have e_1744_11 := arithEq_of_rows h (row := 1744) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_11
  simp only [← c106578, ← c106579, k2, ← c106580] at e_1744_11
  have hr := e_1744_11
  simp only [e_1744_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2372)) := by
  have c106581 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106582 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106583 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106584 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106585 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106586 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_12 := arithEq_of_rows h (row := 1744) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_12
  simp only [← c106581, ← c106582, ← c106583] at e_1744_12
  have e_1744_13 := arithEq_of_rows h (row := 1744) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_13
  simp only [← c106584, ← c106585, k2, ← c106586] at e_1744_13
  have hr := e_1744_13
  simp only [e_1744_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2373)) := by
  have c106587 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106588 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106589 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106590 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106591 := (publicBatchWrapper86_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106592 := (publicBatchWrapper86_copies3331 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_14 := arithEq_of_rows h (row := 1744) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_14
  simp only [← c106587, ← c106588, ← c106589] at e_1744_14
  have e_1744_15 := arithEq_of_rows h (row := 1744) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_15
  simp only [← c106590, ← c106591, k2, ← c106592] at e_1744_15
  have hr := e_1744_15
  simp only [e_1744_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2374)) := by
  have c106593 := (publicBatchWrapper86_copies3331 a h).2.1
  have c106594 := (publicBatchWrapper86_copies3331 a h).2.2.1
  have c106595 := (publicBatchWrapper86_copies3331 a h).2.2.2.1
  have c106596 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.1
  have c106597 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.1
  have c106598 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_16 := arithEq_of_rows h (row := 1744) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_16
  simp only [← c106593, ← c106594, ← c106595] at e_1744_16
  have e_1744_17 := arithEq_of_rows h (row := 1744) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_17
  simp only [← c106596, ← c106597, k2, ← c106598] at e_1744_17
  have hr := e_1744_17
  simp only [e_1744_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1744 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2375)) := by
  have c106599 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.1
  have c106600 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.1
  have c106601 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.1
  have c106602 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106603 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106604 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1744_18 := arithEq_of_rows h (row := 1744) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_18
  simp only [← c106599, ← c106600, ← c106601] at e_1744_18
  have e_1744_19 := arithEq_of_rows h (row := 1744) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_19
  simp only [← c106602, ← c106603, k2, ← c106604] at e_1744_19
  have hr := e_1744_19
  simp only [e_1744_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2376)) := by
  have c106605 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106606 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106607 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106608 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106609 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106610 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_0 := arithEq_of_rows h (row := 1745) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_0
  simp only [← c106605, ← c106606, ← c106607] at e_1745_0
  have e_1745_1 := arithEq_of_rows h (row := 1745) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_1
  simp only [← c106608, ← c106609, k2, ← c106610] at e_1745_1
  have hr := e_1745_1
  simp only [e_1745_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2377)) := by
  have c106611 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106612 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106613 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106614 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106615 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106616 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_2 := arithEq_of_rows h (row := 1745) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_2
  simp only [← c106611, ← c106612, ← c106613] at e_1745_2
  have e_1745_3 := arithEq_of_rows h (row := 1745) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_3
  simp only [← c106614, ← c106615, k2, ← c106616] at e_1745_3
  have hr := e_1745_3
  simp only [e_1745_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2378)) := by
  have c106617 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106618 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106619 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106620 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106621 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106622 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_4 := arithEq_of_rows h (row := 1745) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_4
  simp only [← c106617, ← c106618, ← c106619] at e_1745_4
  have e_1745_5 := arithEq_of_rows h (row := 1745) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_5
  simp only [← c106620, ← c106621, k2, ← c106622] at e_1745_5
  have hr := e_1745_5
  simp only [e_1745_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2379)) := by
  have c106623 := (publicBatchWrapper86_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106624 := (publicBatchWrapper86_copies3332 a h).1
  have c106625 := (publicBatchWrapper86_copies3332 a h).2.1
  have c106626 := (publicBatchWrapper86_copies3332 a h).2.2.1
  have c106627 := (publicBatchWrapper86_copies3332 a h).2.2.2.1
  have c106628 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_6 := arithEq_of_rows h (row := 1745) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_6
  simp only [← c106623, ← c106624, ← c106625] at e_1745_6
  have e_1745_7 := arithEq_of_rows h (row := 1745) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_7
  simp only [← c106626, ← c106627, k2, ← c106628] at e_1745_7
  have hr := e_1745_7
  simp only [e_1745_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2380)) := by
  have c106629 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.1
  have c106630 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.1
  have c106631 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.1
  have c106632 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.1
  have c106633 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.1
  have c106634 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_8 := arithEq_of_rows h (row := 1745) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_8
  simp only [← c106629, ← c106630, ← c106631] at e_1745_8
  have e_1745_9 := arithEq_of_rows h (row := 1745) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_9
  simp only [← c106632, ← c106633, k2, ← c106634] at e_1745_9
  have hr := e_1745_9
  simp only [e_1745_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2381)) := by
  have c106635 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106636 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106637 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106638 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106639 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106640 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_10 := arithEq_of_rows h (row := 1745) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_10
  simp only [← c106635, ← c106636, ← c106637] at e_1745_10
  have e_1745_11 := arithEq_of_rows h (row := 1745) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_11
  simp only [← c106638, ← c106639, k2, ← c106640] at e_1745_11
  have hr := e_1745_11
  simp only [e_1745_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2382)) := by
  have c106641 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106642 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106643 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106644 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106645 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106646 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_12 := arithEq_of_rows h (row := 1745) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_12
  simp only [← c106641, ← c106642, ← c106643] at e_1745_12
  have e_1745_13 := arithEq_of_rows h (row := 1745) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_13
  simp only [← c106644, ← c106645, k2, ← c106646] at e_1745_13
  have hr := e_1745_13
  simp only [e_1745_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2383)) := by
  have c106647 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106648 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106649 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106650 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106651 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106652 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_14 := arithEq_of_rows h (row := 1745) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_14
  simp only [← c106647, ← c106648, ← c106649] at e_1745_14
  have e_1745_15 := arithEq_of_rows h (row := 1745) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_15
  simp only [← c106650, ← c106651, k2, ← c106652] at e_1745_15
  have hr := e_1745_15
  simp only [e_1745_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2384)) := by
  have c106653 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106654 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106655 := (publicBatchWrapper86_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106656 := (publicBatchWrapper86_copies3333 a h).1
  have c106657 := (publicBatchWrapper86_copies3333 a h).2.1
  have c106658 := (publicBatchWrapper86_copies3333 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_16 := arithEq_of_rows h (row := 1745) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_16
  simp only [← c106653, ← c106654, ← c106655] at e_1745_16
  have e_1745_17 := arithEq_of_rows h (row := 1745) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_17
  simp only [← c106656, ← c106657, k2, ← c106658] at e_1745_17
  have hr := e_1745_17
  simp only [e_1745_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1745 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2385)) := by
  have c106659 := (publicBatchWrapper86_copies3333 a h).2.2.2.1
  have c106660 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.1
  have c106661 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.1
  have c106662 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.1
  have c106663 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.1
  have c106664 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1745_18 := arithEq_of_rows h (row := 1745) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_18
  simp only [← c106659, ← c106660, ← c106661] at e_1745_18
  have e_1745_19 := arithEq_of_rows h (row := 1745) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_19
  simp only [← c106662, ← c106663, k2, ← c106664] at e_1745_19
  have hr := e_1745_19
  simp only [e_1745_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2386)) := by
  have c106665 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.1
  have c106666 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106667 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106668 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106669 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106670 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_0 := arithEq_of_rows h (row := 1746) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_0
  simp only [← c106665, ← c106666, ← c106667] at e_1746_0
  have e_1746_1 := arithEq_of_rows h (row := 1746) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_1
  simp only [← c106668, ← c106669, k2, ← c106670] at e_1746_1
  have hr := e_1746_1
  simp only [e_1746_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2387)) := by
  have c106671 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106672 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106673 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106674 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106675 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106676 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_2 := arithEq_of_rows h (row := 1746) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_2
  simp only [← c106671, ← c106672, ← c106673] at e_1746_2
  have e_1746_3 := arithEq_of_rows h (row := 1746) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_3
  simp only [← c106674, ← c106675, k2, ← c106676] at e_1746_3
  have hr := e_1746_3
  simp only [e_1746_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2388)) := by
  have c106677 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106678 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106679 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106680 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106681 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106682 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_4 := arithEq_of_rows h (row := 1746) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_4
  simp only [← c106677, ← c106678, ← c106679] at e_1746_4
  have e_1746_5 := arithEq_of_rows h (row := 1746) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_5
  simp only [← c106680, ← c106681, k2, ← c106682] at e_1746_5
  have hr := e_1746_5
  simp only [e_1746_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2389)) := by
  have c106683 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106684 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106685 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106686 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106687 := (publicBatchWrapper86_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106688 := (publicBatchWrapper86_copies3334 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_6 := arithEq_of_rows h (row := 1746) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_6
  simp only [← c106683, ← c106684, ← c106685] at e_1746_6
  have e_1746_7 := arithEq_of_rows h (row := 1746) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_7
  simp only [← c106686, ← c106687, k2, ← c106688] at e_1746_7
  have hr := e_1746_7
  simp only [e_1746_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2390)) := by
  have c106689 := (publicBatchWrapper86_copies3334 a h).2.1
  have c106690 := (publicBatchWrapper86_copies3334 a h).2.2.1
  have c106691 := (publicBatchWrapper86_copies3334 a h).2.2.2.1
  have c106692 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.1
  have c106693 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.1
  have c106694 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_8 := arithEq_of_rows h (row := 1746) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_8
  simp only [← c106689, ← c106690, ← c106691] at e_1746_8
  have e_1746_9 := arithEq_of_rows h (row := 1746) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_9
  simp only [← c106692, ← c106693, k2, ← c106694] at e_1746_9
  have hr := e_1746_9
  simp only [e_1746_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2391)) := by
  have c106695 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.1
  have c106696 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.1
  have c106697 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.1
  have c106698 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106699 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106700 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_10 := arithEq_of_rows h (row := 1746) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_10
  simp only [← c106695, ← c106696, ← c106697] at e_1746_10
  have e_1746_11 := arithEq_of_rows h (row := 1746) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_11
  simp only [← c106698, ← c106699, k2, ← c106700] at e_1746_11
  have hr := e_1746_11
  simp only [e_1746_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2392)) := by
  have c106701 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106702 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106703 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106704 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106705 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106706 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_12 := arithEq_of_rows h (row := 1746) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_12
  simp only [← c106701, ← c106702, ← c106703] at e_1746_12
  have e_1746_13 := arithEq_of_rows h (row := 1746) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_13
  simp only [← c106704, ← c106705, k2, ← c106706] at e_1746_13
  have hr := e_1746_13
  simp only [e_1746_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2393)) := by
  have c106707 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106708 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106709 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106710 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106711 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106712 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_14 := arithEq_of_rows h (row := 1746) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_14
  simp only [← c106707, ← c106708, ← c106709] at e_1746_14
  have e_1746_15 := arithEq_of_rows h (row := 1746) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_15
  simp only [← c106710, ← c106711, k2, ← c106712] at e_1746_15
  have hr := e_1746_15
  simp only [e_1746_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2394)) := by
  have c106713 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106714 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106715 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106716 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106717 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106718 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_16 := arithEq_of_rows h (row := 1746) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_16
  simp only [← c106713, ← c106714, ← c106715] at e_1746_16
  have e_1746_17 := arithEq_of_rows h (row := 1746) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_17
  simp only [← c106716, ← c106717, k2, ← c106718] at e_1746_17
  have hr := e_1746_17
  simp only [e_1746_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1746 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2395)) := by
  have c106719 := (publicBatchWrapper86_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106720 := (publicBatchWrapper86_copies3335 a h).1
  have c106721 := (publicBatchWrapper86_copies3335 a h).2.1
  have c106722 := (publicBatchWrapper86_copies3335 a h).2.2.1
  have c106723 := (publicBatchWrapper86_copies3335 a h).2.2.2.1
  have c106724 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1746_18 := arithEq_of_rows h (row := 1746) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_18
  simp only [← c106719, ← c106720, ← c106721] at e_1746_18
  have e_1746_19 := arithEq_of_rows h (row := 1746) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_19
  simp only [← c106722, ← c106723, k2, ← c106724] at e_1746_19
  have hr := e_1746_19
  simp only [e_1746_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2396)) := by
  have c106725 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.1
  have c106726 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.1
  have c106727 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.1
  have c106728 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.1
  have c106729 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.1
  have c106730 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_0 := arithEq_of_rows h (row := 1747) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_0
  simp only [← c106725, ← c106726, ← c106727] at e_1747_0
  have e_1747_1 := arithEq_of_rows h (row := 1747) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_1
  simp only [← c106728, ← c106729, k2, ← c106730] at e_1747_1
  have hr := e_1747_1
  simp only [e_1747_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2397)) := by
  have c106731 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106732 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106733 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106734 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106735 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106736 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_2 := arithEq_of_rows h (row := 1747) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_2
  simp only [← c106731, ← c106732, ← c106733] at e_1747_2
  have e_1747_3 := arithEq_of_rows h (row := 1747) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_3
  simp only [← c106734, ← c106735, k2, ← c106736] at e_1747_3
  have hr := e_1747_3
  simp only [e_1747_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2398)) := by
  have c106737 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106738 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106739 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106740 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106741 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106742 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_4 := arithEq_of_rows h (row := 1747) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_4
  simp only [← c106737, ← c106738, ← c106739] at e_1747_4
  have e_1747_5 := arithEq_of_rows h (row := 1747) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_5
  simp only [← c106740, ← c106741, k2, ← c106742] at e_1747_5
  have hr := e_1747_5
  simp only [e_1747_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2399)) := by
  have c106743 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106744 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106745 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106746 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106747 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106748 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_6 := arithEq_of_rows h (row := 1747) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_6
  simp only [← c106743, ← c106744, ← c106745] at e_1747_6
  have e_1747_7 := arithEq_of_rows h (row := 1747) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_7
  simp only [← c106746, ← c106747, k2, ← c106748] at e_1747_7
  have hr := e_1747_7
  simp only [e_1747_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2400)) := by
  have c106749 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106750 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106751 := (publicBatchWrapper86_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106752 := (publicBatchWrapper86_copies3336 a h).1
  have c106753 := (publicBatchWrapper86_copies3336 a h).2.1
  have c106754 := (publicBatchWrapper86_copies3336 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_8 := arithEq_of_rows h (row := 1747) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_8
  simp only [← c106749, ← c106750, ← c106751] at e_1747_8
  have e_1747_9 := arithEq_of_rows h (row := 1747) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_9
  simp only [← c106752, ← c106753, k2, ← c106754] at e_1747_9
  have hr := e_1747_9
  simp only [e_1747_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2401)) := by
  have c106755 := (publicBatchWrapper86_copies3336 a h).2.2.2.1
  have c106756 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.1
  have c106757 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.1
  have c106758 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.1
  have c106759 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.1
  have c106760 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_10 := arithEq_of_rows h (row := 1747) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_10
  simp only [← c106755, ← c106756, ← c106757] at e_1747_10
  have e_1747_11 := arithEq_of_rows h (row := 1747) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_11
  simp only [← c106758, ← c106759, k2, ← c106760] at e_1747_11
  have hr := e_1747_11
  simp only [e_1747_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2570)) := by
  have c106761 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.1
  have c106762 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106763 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106764 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106765 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106766 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_12 := arithEq_of_rows h (row := 1747) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_12
  simp only [← c106761, ← c106762, ← c106763] at e_1747_12
  have e_1747_13 := arithEq_of_rows h (row := 1747) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_13
  simp only [← c106764, ← c106765, k2, ← c106766] at e_1747_13
  have hr := e_1747_13
  simp only [e_1747_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2571)) := by
  have c106767 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106768 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106769 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106770 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106771 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106772 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_14 := arithEq_of_rows h (row := 1747) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_14
  simp only [← c106767, ← c106768, ← c106769] at e_1747_14
  have e_1747_15 := arithEq_of_rows h (row := 1747) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_15
  simp only [← c106770, ← c106771, k2, ← c106772] at e_1747_15
  have hr := e_1747_15
  simp only [e_1747_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2572)) := by
  have c106773 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106774 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106775 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106776 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106777 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106778 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_16 := arithEq_of_rows h (row := 1747) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_16
  simp only [← c106773, ← c106774, ← c106775] at e_1747_16
  have e_1747_17 := arithEq_of_rows h (row := 1747) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_17
  simp only [← c106776, ← c106777, k2, ← c106778] at e_1747_17
  have hr := e_1747_17
  simp only [e_1747_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1747 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2573)) := by
  have c106779 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106780 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106781 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106782 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106783 := (publicBatchWrapper86_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106784 := (publicBatchWrapper86_copies3337 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1747_18 := arithEq_of_rows h (row := 1747) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_18
  simp only [← c106779, ← c106780, ← c106781] at e_1747_18
  have e_1747_19 := arithEq_of_rows h (row := 1747) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_19
  simp only [← c106782, ← c106783, k2, ← c106784] at e_1747_19
  have hr := e_1747_19
  simp only [e_1747_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2574)) := by
  have c106785 := (publicBatchWrapper86_copies3337 a h).2.1
  have c106786 := (publicBatchWrapper86_copies3337 a h).2.2.1
  have c106787 := (publicBatchWrapper86_copies3337 a h).2.2.2.1
  have c106788 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.1
  have c106789 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.1
  have c106790 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_0 := arithEq_of_rows h (row := 1748) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_0
  simp only [← c106785, ← c106786, ← c106787] at e_1748_0
  have e_1748_1 := arithEq_of_rows h (row := 1748) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_1
  simp only [← c106788, ← c106789, k2, ← c106790] at e_1748_1
  have hr := e_1748_1
  simp only [e_1748_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2575)) := by
  have c106791 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.1
  have c106792 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.1
  have c106793 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.1
  have c106794 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106795 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106796 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_2 := arithEq_of_rows h (row := 1748) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_2
  simp only [← c106791, ← c106792, ← c106793] at e_1748_2
  have e_1748_3 := arithEq_of_rows h (row := 1748) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_3
  simp only [← c106794, ← c106795, k2, ← c106796] at e_1748_3
  have hr := e_1748_3
  simp only [e_1748_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2576)) := by
  have c106797 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106798 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106799 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106800 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106801 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106802 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_4 := arithEq_of_rows h (row := 1748) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_4
  simp only [← c106797, ← c106798, ← c106799] at e_1748_4
  have e_1748_5 := arithEq_of_rows h (row := 1748) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_5
  simp only [← c106800, ← c106801, k2, ← c106802] at e_1748_5
  have hr := e_1748_5
  simp only [e_1748_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2577)) := by
  have c106803 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106804 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106805 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106806 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106807 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106808 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_6 := arithEq_of_rows h (row := 1748) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_6
  simp only [← c106803, ← c106804, ← c106805] at e_1748_6
  have e_1748_7 := arithEq_of_rows h (row := 1748) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_7
  simp only [← c106806, ← c106807, k2, ← c106808] at e_1748_7
  have hr := e_1748_7
  simp only [e_1748_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2578)) := by
  have c106809 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106810 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106811 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106812 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106813 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106814 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_8 := arithEq_of_rows h (row := 1748) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_8
  simp only [← c106809, ← c106810, ← c106811] at e_1748_8
  have e_1748_9 := arithEq_of_rows h (row := 1748) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_9
  simp only [← c106812, ← c106813, k2, ← c106814] at e_1748_9
  have hr := e_1748_9
  simp only [e_1748_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2579)) := by
  have c106815 := (publicBatchWrapper86_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106816 := (publicBatchWrapper86_copies3338 a h).1
  have c106817 := (publicBatchWrapper86_copies3338 a h).2.1
  have c106818 := (publicBatchWrapper86_copies3338 a h).2.2.1
  have c106819 := (publicBatchWrapper86_copies3338 a h).2.2.2.1
  have c106820 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_10 := arithEq_of_rows h (row := 1748) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_10
  simp only [← c106815, ← c106816, ← c106817] at e_1748_10
  have e_1748_11 := arithEq_of_rows h (row := 1748) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_11
  simp only [← c106818, ← c106819, k2, ← c106820] at e_1748_11
  have hr := e_1748_11
  simp only [e_1748_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2580)) := by
  have c106821 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.1
  have c106822 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.1
  have c106823 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.1
  have c106824 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.1
  have c106825 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.1
  have c106826 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_12 := arithEq_of_rows h (row := 1748) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_12
  simp only [← c106821, ← c106822, ← c106823] at e_1748_12
  have e_1748_13 := arithEq_of_rows h (row := 1748) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_13
  simp only [← c106824, ← c106825, k2, ← c106826] at e_1748_13
  have hr := e_1748_13
  simp only [e_1748_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2581)) := by
  have c106827 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106828 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106829 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106830 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106831 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106832 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_14 := arithEq_of_rows h (row := 1748) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_14
  simp only [← c106827, ← c106828, ← c106829] at e_1748_14
  have e_1748_15 := arithEq_of_rows h (row := 1748) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_15
  simp only [← c106830, ← c106831, k2, ← c106832] at e_1748_15
  have hr := e_1748_15
  simp only [e_1748_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2582)) := by
  have c106833 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106834 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106835 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106836 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106837 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106838 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_16 := arithEq_of_rows h (row := 1748) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_16
  simp only [← c106833, ← c106834, ← c106835] at e_1748_16
  have e_1748_17 := arithEq_of_rows h (row := 1748) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_17
  simp only [← c106836, ← c106837, k2, ← c106838] at e_1748_17
  have hr := e_1748_17
  simp only [e_1748_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1748 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2583)) := by
  have c106839 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106840 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106841 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106842 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106843 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106844 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1748_18 := arithEq_of_rows h (row := 1748) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_18
  simp only [← c106839, ← c106840, ← c106841] at e_1748_18
  have e_1748_19 := arithEq_of_rows h (row := 1748) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_19
  simp only [← c106842, ← c106843, k2, ← c106844] at e_1748_19
  have hr := e_1748_19
  simp only [e_1748_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2584)) := by
  have c106845 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106846 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106847 := (publicBatchWrapper86_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106848 := (publicBatchWrapper86_copies3339 a h).1
  have c106849 := (publicBatchWrapper86_copies3339 a h).2.1
  have c106850 := (publicBatchWrapper86_copies3339 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_0 := arithEq_of_rows h (row := 1749) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_0
  simp only [← c106845, ← c106846, ← c106847] at e_1749_0
  have e_1749_1 := arithEq_of_rows h (row := 1749) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_1
  simp only [← c106848, ← c106849, k2, ← c106850] at e_1749_1
  have hr := e_1749_1
  simp only [e_1749_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2585)) := by
  have c106851 := (publicBatchWrapper86_copies3339 a h).2.2.2.1
  have c106852 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.1
  have c106853 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.1
  have c106854 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.1
  have c106855 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.1
  have c106856 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_2 := arithEq_of_rows h (row := 1749) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_2
  simp only [← c106851, ← c106852, ← c106853] at e_1749_2
  have e_1749_3 := arithEq_of_rows h (row := 1749) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_3
  simp only [← c106854, ← c106855, k2, ← c106856] at e_1749_3
  have hr := e_1749_3
  simp only [e_1749_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2586)) := by
  have c106857 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.1
  have c106858 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106859 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106860 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106861 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106862 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_4 := arithEq_of_rows h (row := 1749) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_4
  simp only [← c106857, ← c106858, ← c106859] at e_1749_4
  have e_1749_5 := arithEq_of_rows h (row := 1749) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_5
  simp only [← c106860, ← c106861, k2, ← c106862] at e_1749_5
  have hr := e_1749_5
  simp only [e_1749_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2587)) := by
  have c106863 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106864 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106865 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106866 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106867 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106868 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_6 := arithEq_of_rows h (row := 1749) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_6
  simp only [← c106863, ← c106864, ← c106865] at e_1749_6
  have e_1749_7 := arithEq_of_rows h (row := 1749) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_7
  simp only [← c106866, ← c106867, k2, ← c106868] at e_1749_7
  have hr := e_1749_7
  simp only [e_1749_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2588)) := by
  have c106869 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106870 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106871 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106872 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106873 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106874 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_8 := arithEq_of_rows h (row := 1749) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_8
  simp only [← c106869, ← c106870, ← c106871] at e_1749_8
  have e_1749_9 := arithEq_of_rows h (row := 1749) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_9
  simp only [← c106872, ← c106873, k2, ← c106874] at e_1749_9
  have hr := e_1749_9
  simp only [e_1749_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2589)) := by
  have c106875 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106876 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106877 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106878 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106879 := (publicBatchWrapper86_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106880 := (publicBatchWrapper86_copies3340 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_10 := arithEq_of_rows h (row := 1749) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_10
  simp only [← c106875, ← c106876, ← c106877] at e_1749_10
  have e_1749_11 := arithEq_of_rows h (row := 1749) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_11
  simp only [← c106878, ← c106879, k2, ← c106880] at e_1749_11
  have hr := e_1749_11
  simp only [e_1749_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2590)) := by
  have c106881 := (publicBatchWrapper86_copies3340 a h).2.1
  have c106882 := (publicBatchWrapper86_copies3340 a h).2.2.1
  have c106883 := (publicBatchWrapper86_copies3340 a h).2.2.2.1
  have c106884 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.1
  have c106885 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.1
  have c106886 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_12 := arithEq_of_rows h (row := 1749) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_12
  simp only [← c106881, ← c106882, ← c106883] at e_1749_12
  have e_1749_13 := arithEq_of_rows h (row := 1749) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_13
  simp only [← c106884, ← c106885, k2, ← c106886] at e_1749_13
  have hr := e_1749_13
  simp only [e_1749_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2591)) := by
  have c106887 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.1
  have c106888 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.1
  have c106889 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.1
  have c106890 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106891 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106892 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_14 := arithEq_of_rows h (row := 1749) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_14
  simp only [← c106887, ← c106888, ← c106889] at e_1749_14
  have e_1749_15 := arithEq_of_rows h (row := 1749) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_15
  simp only [← c106890, ← c106891, k2, ← c106892] at e_1749_15
  have hr := e_1749_15
  simp only [e_1749_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2592)) := by
  have c106893 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106894 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106895 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106896 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106897 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106898 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_16 := arithEq_of_rows h (row := 1749) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_16
  simp only [← c106893, ← c106894, ← c106895] at e_1749_16
  have e_1749_17 := arithEq_of_rows h (row := 1749) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_17
  simp only [← c106896, ← c106897, k2, ← c106898] at e_1749_17
  have hr := e_1749_17
  simp only [e_1749_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1749 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2593)) := by
  have c106899 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106900 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106901 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106902 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106903 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106904 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1749_18 := arithEq_of_rows h (row := 1749) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_18
  simp only [← c106899, ← c106900, ← c106901] at e_1749_18
  have e_1749_19 := arithEq_of_rows h (row := 1749) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_19
  simp only [← c106902, ← c106903, k2, ← c106904] at e_1749_19
  have hr := e_1749_19
  simp only [e_1749_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2594)) := by
  have c106905 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106906 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106907 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106908 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106909 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106910 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_0 := arithEq_of_rows h (row := 1750) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_0
  simp only [← c106905, ← c106906, ← c106907] at e_1750_0
  have e_1750_1 := arithEq_of_rows h (row := 1750) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_1
  simp only [← c106908, ← c106909, k2, ← c106910] at e_1750_1
  have hr := e_1750_1
  simp only [e_1750_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2595)) := by
  have c106911 := (publicBatchWrapper86_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106912 := (publicBatchWrapper86_copies3341 a h).1
  have c106913 := (publicBatchWrapper86_copies3341 a h).2.1
  have c106914 := (publicBatchWrapper86_copies3341 a h).2.2.1
  have c106915 := (publicBatchWrapper86_copies3341 a h).2.2.2.1
  have c106916 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_2 := arithEq_of_rows h (row := 1750) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_2
  simp only [← c106911, ← c106912, ← c106913] at e_1750_2
  have e_1750_3 := arithEq_of_rows h (row := 1750) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_3
  simp only [← c106914, ← c106915, k2, ← c106916] at e_1750_3
  have hr := e_1750_3
  simp only [e_1750_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2596)) := by
  have c106917 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.1
  have c106918 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.1
  have c106919 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.1
  have c106920 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.1
  have c106921 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.1
  have c106922 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_4 := arithEq_of_rows h (row := 1750) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_4
  simp only [← c106917, ← c106918, ← c106919] at e_1750_4
  have e_1750_5 := arithEq_of_rows h (row := 1750) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_5
  simp only [← c106920, ← c106921, k2, ← c106922] at e_1750_5
  have hr := e_1750_5
  simp only [e_1750_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2597)) := by
  have c106923 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106924 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106925 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106926 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106927 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106928 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_6 := arithEq_of_rows h (row := 1750) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_6
  simp only [← c106923, ← c106924, ← c106925] at e_1750_6
  have e_1750_7 := arithEq_of_rows h (row := 1750) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_7
  simp only [← c106926, ← c106927, k2, ← c106928] at e_1750_7
  have hr := e_1750_7
  simp only [e_1750_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2598)) := by
  have c106929 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106930 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106931 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106932 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106933 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106934 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_8 := arithEq_of_rows h (row := 1750) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_8
  simp only [← c106929, ← c106930, ← c106931] at e_1750_8
  have e_1750_9 := arithEq_of_rows h (row := 1750) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_9
  simp only [← c106932, ← c106933, k2, ← c106934] at e_1750_9
  have hr := e_1750_9
  simp only [e_1750_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2599)) := by
  have c106935 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106936 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106937 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106938 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106939 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106940 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_10 := arithEq_of_rows h (row := 1750) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_10
  simp only [← c106935, ← c106936, ← c106937] at e_1750_10
  have e_1750_11 := arithEq_of_rows h (row := 1750) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_11
  simp only [← c106938, ← c106939, k2, ← c106940] at e_1750_11
  have hr := e_1750_11
  simp only [e_1750_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2600)) := by
  have c106941 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106942 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106943 := (publicBatchWrapper86_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106944 := (publicBatchWrapper86_copies3342 a h).1
  have c106945 := (publicBatchWrapper86_copies3342 a h).2.1
  have c106946 := (publicBatchWrapper86_copies3342 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_12 := arithEq_of_rows h (row := 1750) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_12
  simp only [← c106941, ← c106942, ← c106943] at e_1750_12
  have e_1750_13 := arithEq_of_rows h (row := 1750) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_13
  simp only [← c106944, ← c106945, k2, ← c106946] at e_1750_13
  have hr := e_1750_13
  simp only [e_1750_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2601)) := by
  have c106947 := (publicBatchWrapper86_copies3342 a h).2.2.2.1
  have c106948 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.1
  have c106949 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.1
  have c106950 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.1
  have c106951 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.1
  have c106952 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_14 := arithEq_of_rows h (row := 1750) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_14
  simp only [← c106947, ← c106948, ← c106949] at e_1750_14
  have e_1750_15 := arithEq_of_rows h (row := 1750) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_15
  simp only [← c106950, ← c106951, k2, ← c106952] at e_1750_15
  have hr := e_1750_15
  simp only [e_1750_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2770)) := by
  have c106953 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.1
  have c106954 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106955 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106956 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106957 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106958 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_16 := arithEq_of_rows h (row := 1750) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_16
  simp only [← c106953, ← c106954, ← c106955] at e_1750_16
  have e_1750_17 := arithEq_of_rows h (row := 1750) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_17
  simp only [← c106956, ← c106957, k2, ← c106958] at e_1750_17
  have hr := e_1750_17
  simp only [e_1750_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1750 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2771)) := by
  have c106959 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106960 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106961 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106962 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106963 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106964 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1750_18 := arithEq_of_rows h (row := 1750) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_18
  simp only [← c106959, ← c106960, ← c106961] at e_1750_18
  have e_1750_19 := arithEq_of_rows h (row := 1750) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_19
  simp only [← c106962, ← c106963, k2, ← c106964] at e_1750_19
  have hr := e_1750_19
  simp only [e_1750_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2772)) := by
  have c106965 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106966 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106967 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106968 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106969 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106970 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_0 := arithEq_of_rows h (row := 1751) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_0
  simp only [← c106965, ← c106966, ← c106967] at e_1751_0
  have e_1751_1 := arithEq_of_rows h (row := 1751) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_1
  simp only [← c106968, ← c106969, k2, ← c106970] at e_1751_1
  have hr := e_1751_1
  simp only [e_1751_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2773)) := by
  have c106971 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106972 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106973 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106974 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106975 := (publicBatchWrapper86_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106976 := (publicBatchWrapper86_copies3343 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_2 := arithEq_of_rows h (row := 1751) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_2
  simp only [← c106971, ← c106972, ← c106973] at e_1751_2
  have e_1751_3 := arithEq_of_rows h (row := 1751) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_3
  simp only [← c106974, ← c106975, k2, ← c106976] at e_1751_3
  have hr := e_1751_3
  simp only [e_1751_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2774)) := by
  have c106977 := (publicBatchWrapper86_copies3343 a h).2.1
  have c106978 := (publicBatchWrapper86_copies3343 a h).2.2.1
  have c106979 := (publicBatchWrapper86_copies3343 a h).2.2.2.1
  have c106980 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.1
  have c106981 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.1
  have c106982 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_4 := arithEq_of_rows h (row := 1751) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_4
  simp only [← c106977, ← c106978, ← c106979] at e_1751_4
  have e_1751_5 := arithEq_of_rows h (row := 1751) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_5
  simp only [← c106980, ← c106981, k2, ← c106982] at e_1751_5
  have hr := e_1751_5
  simp only [e_1751_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2775)) := by
  have c106983 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.1
  have c106984 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.1
  have c106985 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.1
  have c106986 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106987 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106988 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_6 := arithEq_of_rows h (row := 1751) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_6
  simp only [← c106983, ← c106984, ← c106985] at e_1751_6
  have e_1751_7 := arithEq_of_rows h (row := 1751) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_7
  simp only [← c106986, ← c106987, k2, ← c106988] at e_1751_7
  have hr := e_1751_7
  simp only [e_1751_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2776)) := by
  have c106989 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106990 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106991 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106992 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106993 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106994 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_8 := arithEq_of_rows h (row := 1751) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_8
  simp only [← c106989, ← c106990, ← c106991] at e_1751_8
  have e_1751_9 := arithEq_of_rows h (row := 1751) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_9
  simp only [← c106992, ← c106993, k2, ← c106994] at e_1751_9
  have hr := e_1751_9
  simp only [e_1751_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2777)) := by
  have c106995 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106996 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106997 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106998 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106999 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107000 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_10 := arithEq_of_rows h (row := 1751) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_10
  simp only [← c106995, ← c106996, ← c106997] at e_1751_10
  have e_1751_11 := arithEq_of_rows h (row := 1751) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_11
  simp only [← c106998, ← c106999, k2, ← c107000] at e_1751_11
  have hr := e_1751_11
  simp only [e_1751_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2778)) := by
  have c107001 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107002 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107003 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107004 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107005 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107006 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_12 := arithEq_of_rows h (row := 1751) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_12
  simp only [← c107001, ← c107002, ← c107003] at e_1751_12
  have e_1751_13 := arithEq_of_rows h (row := 1751) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_13
  simp only [← c107004, ← c107005, k2, ← c107006] at e_1751_13
  have hr := e_1751_13
  simp only [e_1751_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2779)) := by
  have c107007 := (publicBatchWrapper86_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107008 := (publicBatchWrapper86_copies3344 a h).1
  have c107009 := (publicBatchWrapper86_copies3344 a h).2.1
  have c107010 := (publicBatchWrapper86_copies3344 a h).2.2.1
  have c107011 := (publicBatchWrapper86_copies3344 a h).2.2.2.1
  have c107012 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_14 := arithEq_of_rows h (row := 1751) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_14
  simp only [← c107007, ← c107008, ← c107009] at e_1751_14
  have e_1751_15 := arithEq_of_rows h (row := 1751) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_15
  simp only [← c107010, ← c107011, k2, ← c107012] at e_1751_15
  have hr := e_1751_15
  simp only [e_1751_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2780)) := by
  have c107013 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.1
  have c107014 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.1
  have c107015 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.1
  have c107016 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.1
  have c107017 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.1
  have c107018 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_16 := arithEq_of_rows h (row := 1751) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_16
  simp only [← c107013, ← c107014, ← c107015] at e_1751_16
  have e_1751_17 := arithEq_of_rows h (row := 1751) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_17
  simp only [← c107016, ← c107017, k2, ← c107018] at e_1751_17
  have hr := e_1751_17
  simp only [e_1751_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1751 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2781)) := by
  have c107019 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107020 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107021 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107022 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107023 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107024 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1751_18 := arithEq_of_rows h (row := 1751) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_18
  simp only [← c107019, ← c107020, ← c107021] at e_1751_18
  have e_1751_19 := arithEq_of_rows h (row := 1751) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_19
  simp only [← c107022, ← c107023, k2, ← c107024] at e_1751_19
  have hr := e_1751_19
  simp only [e_1751_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2782)) := by
  have c107025 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107026 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107027 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107028 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107029 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107030 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_0 := arithEq_of_rows h (row := 1752) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_0
  simp only [← c107025, ← c107026, ← c107027] at e_1752_0
  have e_1752_1 := arithEq_of_rows h (row := 1752) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_1
  simp only [← c107028, ← c107029, k2, ← c107030] at e_1752_1
  have hr := e_1752_1
  simp only [e_1752_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2783)) := by
  have c107031 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107032 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107033 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107034 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107035 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107036 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_2 := arithEq_of_rows h (row := 1752) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_2
  simp only [← c107031, ← c107032, ← c107033] at e_1752_2
  have e_1752_3 := arithEq_of_rows h (row := 1752) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_3
  simp only [← c107034, ← c107035, k2, ← c107036] at e_1752_3
  have hr := e_1752_3
  simp only [e_1752_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2784)) := by
  have c107037 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107038 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107039 := (publicBatchWrapper86_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107040 := (publicBatchWrapper86_copies3345 a h).1
  have c107041 := (publicBatchWrapper86_copies3345 a h).2.1
  have c107042 := (publicBatchWrapper86_copies3345 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_4 := arithEq_of_rows h (row := 1752) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_4
  simp only [← c107037, ← c107038, ← c107039] at e_1752_4
  have e_1752_5 := arithEq_of_rows h (row := 1752) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_5
  simp only [← c107040, ← c107041, k2, ← c107042] at e_1752_5
  have hr := e_1752_5
  simp only [e_1752_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2785)) := by
  have c107043 := (publicBatchWrapper86_copies3345 a h).2.2.2.1
  have c107044 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.1
  have c107045 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.1
  have c107046 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.1
  have c107047 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.1
  have c107048 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_6 := arithEq_of_rows h (row := 1752) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_6
  simp only [← c107043, ← c107044, ← c107045] at e_1752_6
  have e_1752_7 := arithEq_of_rows h (row := 1752) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_7
  simp only [← c107046, ← c107047, k2, ← c107048] at e_1752_7
  have hr := e_1752_7
  simp only [e_1752_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2786)) := by
  have c107049 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.1
  have c107050 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107051 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107052 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107053 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107054 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_8 := arithEq_of_rows h (row := 1752) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_8
  simp only [← c107049, ← c107050, ← c107051] at e_1752_8
  have e_1752_9 := arithEq_of_rows h (row := 1752) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_9
  simp only [← c107052, ← c107053, k2, ← c107054] at e_1752_9
  have hr := e_1752_9
  simp only [e_1752_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2787)) := by
  have c107055 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107056 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107057 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107058 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107059 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107060 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_10 := arithEq_of_rows h (row := 1752) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_10
  simp only [← c107055, ← c107056, ← c107057] at e_1752_10
  have e_1752_11 := arithEq_of_rows h (row := 1752) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_11
  simp only [← c107058, ← c107059, k2, ← c107060] at e_1752_11
  have hr := e_1752_11
  simp only [e_1752_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2788)) := by
  have c107061 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107062 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107063 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107064 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107065 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107066 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_12 := arithEq_of_rows h (row := 1752) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_12
  simp only [← c107061, ← c107062, ← c107063] at e_1752_12
  have e_1752_13 := arithEq_of_rows h (row := 1752) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_13
  simp only [← c107064, ← c107065, k2, ← c107066] at e_1752_13
  have hr := e_1752_13
  simp only [e_1752_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2789)) := by
  have c107067 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107068 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107069 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107070 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107071 := (publicBatchWrapper86_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107072 := (publicBatchWrapper86_copies3346 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_14 := arithEq_of_rows h (row := 1752) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_14
  simp only [← c107067, ← c107068, ← c107069] at e_1752_14
  have e_1752_15 := arithEq_of_rows h (row := 1752) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_15
  simp only [← c107070, ← c107071, k2, ← c107072] at e_1752_15
  have hr := e_1752_15
  simp only [e_1752_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2790)) := by
  have c107073 := (publicBatchWrapper86_copies3346 a h).2.1
  have c107074 := (publicBatchWrapper86_copies3346 a h).2.2.1
  have c107075 := (publicBatchWrapper86_copies3346 a h).2.2.2.1
  have c107076 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.1
  have c107077 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.1
  have c107078 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_16 := arithEq_of_rows h (row := 1752) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_16
  simp only [← c107073, ← c107074, ← c107075] at e_1752_16
  have e_1752_17 := arithEq_of_rows h (row := 1752) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_17
  simp only [← c107076, ← c107077, k2, ← c107078] at e_1752_17
  have hr := e_1752_17
  simp only [e_1752_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1752 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2791)) := by
  have c107079 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.1
  have c107080 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.1
  have c107081 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.1
  have c107082 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107083 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107084 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1752_18 := arithEq_of_rows h (row := 1752) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_18
  simp only [← c107079, ← c107080, ← c107081] at e_1752_18
  have e_1752_19 := arithEq_of_rows h (row := 1752) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_19
  simp only [← c107082, ← c107083, k2, ← c107084] at e_1752_19
  have hr := e_1752_19
  simp only [e_1752_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2792)) := by
  have c107085 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107086 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107087 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107088 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107089 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107090 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_0 := arithEq_of_rows h (row := 1753) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_0
  simp only [← c107085, ← c107086, ← c107087] at e_1753_0
  have e_1753_1 := arithEq_of_rows h (row := 1753) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_1
  simp only [← c107088, ← c107089, k2, ← c107090] at e_1753_1
  have hr := e_1753_1
  simp only [e_1753_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2793)) := by
  have c107091 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107092 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107093 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107094 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107095 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107096 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_2 := arithEq_of_rows h (row := 1753) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_2
  simp only [← c107091, ← c107092, ← c107093] at e_1753_2
  have e_1753_3 := arithEq_of_rows h (row := 1753) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_3
  simp only [← c107094, ← c107095, k2, ← c107096] at e_1753_3
  have hr := e_1753_3
  simp only [e_1753_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2794)) := by
  have c107097 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107098 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107099 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107100 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107101 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107102 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_4 := arithEq_of_rows h (row := 1753) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_4
  simp only [← c107097, ← c107098, ← c107099] at e_1753_4
  have e_1753_5 := arithEq_of_rows h (row := 1753) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_5
  simp only [← c107100, ← c107101, k2, ← c107102] at e_1753_5
  have hr := e_1753_5
  simp only [e_1753_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2795)) := by
  have c107103 := (publicBatchWrapper86_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107104 := (publicBatchWrapper86_copies3347 a h).1
  have c107105 := (publicBatchWrapper86_copies3347 a h).2.1
  have c107106 := (publicBatchWrapper86_copies3347 a h).2.2.1
  have c107107 := (publicBatchWrapper86_copies3347 a h).2.2.2.1
  have c107108 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_6 := arithEq_of_rows h (row := 1753) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_6
  simp only [← c107103, ← c107104, ← c107105] at e_1753_6
  have e_1753_7 := arithEq_of_rows h (row := 1753) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_7
  simp only [← c107106, ← c107107, k2, ← c107108] at e_1753_7
  have hr := e_1753_7
  simp only [e_1753_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2796)) := by
  have c107109 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.1
  have c107110 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.1
  have c107111 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.1
  have c107112 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.1
  have c107113 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.1
  have c107114 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_8 := arithEq_of_rows h (row := 1753) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_8
  simp only [← c107109, ← c107110, ← c107111] at e_1753_8
  have e_1753_9 := arithEq_of_rows h (row := 1753) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_9
  simp only [← c107112, ← c107113, k2, ← c107114] at e_1753_9
  have hr := e_1753_9
  simp only [e_1753_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2797)) := by
  have c107115 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107116 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107117 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107118 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107119 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107120 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_10 := arithEq_of_rows h (row := 1753) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_10
  simp only [← c107115, ← c107116, ← c107117] at e_1753_10
  have e_1753_11 := arithEq_of_rows h (row := 1753) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_11
  simp only [← c107118, ← c107119, k2, ← c107120] at e_1753_11
  have hr := e_1753_11
  simp only [e_1753_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2798)) := by
  have c107121 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107122 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107123 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107124 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107125 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107126 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_12 := arithEq_of_rows h (row := 1753) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_12
  simp only [← c107121, ← c107122, ← c107123] at e_1753_12
  have e_1753_13 := arithEq_of_rows h (row := 1753) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_13
  simp only [← c107124, ← c107125, k2, ← c107126] at e_1753_13
  have hr := e_1753_13
  simp only [e_1753_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2799)) := by
  have c107127 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107128 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107129 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107130 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107131 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107132 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_14 := arithEq_of_rows h (row := 1753) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_14
  simp only [← c107127, ← c107128, ← c107129] at e_1753_14
  have e_1753_15 := arithEq_of_rows h (row := 1753) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_15
  simp only [← c107130, ← c107131, k2, ← c107132] at e_1753_15
  have hr := e_1753_15
  simp only [e_1753_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2800)) := by
  have c107133 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107134 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107135 := (publicBatchWrapper86_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107136 := (publicBatchWrapper86_copies3348 a h).1
  have c107137 := (publicBatchWrapper86_copies3348 a h).2.1
  have c107138 := (publicBatchWrapper86_copies3348 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_16 := arithEq_of_rows h (row := 1753) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_16
  simp only [← c107133, ← c107134, ← c107135] at e_1753_16
  have e_1753_17 := arithEq_of_rows h (row := 1753) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_17
  simp only [← c107136, ← c107137, k2, ← c107138] at e_1753_17
  have hr := e_1753_17
  simp only [e_1753_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1753 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2801)) := by
  have c107139 := (publicBatchWrapper86_copies3348 a h).2.2.2.1
  have c107140 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.1
  have c107141 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.1
  have c107142 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.1
  have c107143 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.1
  have c107144 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1753_18 := arithEq_of_rows h (row := 1753) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_18
  simp only [← c107139, ← c107140, ← c107141] at e_1753_18
  have e_1753_19 := arithEq_of_rows h (row := 1753) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_19
  simp only [← c107142, ← c107143, k2, ← c107144] at e_1753_19
  have hr := e_1753_19
  simp only [e_1753_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2970)) := by
  have c107145 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.1
  have c107146 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107147 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107148 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107149 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107150 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_0 := arithEq_of_rows h (row := 1754) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_0
  simp only [← c107145, ← c107146, ← c107147] at e_1754_0
  have e_1754_1 := arithEq_of_rows h (row := 1754) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_1
  simp only [← c107148, ← c107149, k2, ← c107150] at e_1754_1
  have hr := e_1754_1
  simp only [e_1754_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2971)) := by
  have c107151 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107152 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107153 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107154 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107155 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107156 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_2 := arithEq_of_rows h (row := 1754) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_2
  simp only [← c107151, ← c107152, ← c107153] at e_1754_2
  have e_1754_3 := arithEq_of_rows h (row := 1754) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_3
  simp only [← c107154, ← c107155, k2, ← c107156] at e_1754_3
  have hr := e_1754_3
  simp only [e_1754_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2972)) := by
  have c107157 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107158 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107159 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107160 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107161 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107162 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_4 := arithEq_of_rows h (row := 1754) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_4
  simp only [← c107157, ← c107158, ← c107159] at e_1754_4
  have e_1754_5 := arithEq_of_rows h (row := 1754) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_5
  simp only [← c107160, ← c107161, k2, ← c107162] at e_1754_5
  have hr := e_1754_5
  simp only [e_1754_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2973)) := by
  have c107163 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107164 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107165 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107166 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107167 := (publicBatchWrapper86_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107168 := (publicBatchWrapper86_copies3349 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_6 := arithEq_of_rows h (row := 1754) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_6
  simp only [← c107163, ← c107164, ← c107165] at e_1754_6
  have e_1754_7 := arithEq_of_rows h (row := 1754) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_7
  simp only [← c107166, ← c107167, k2, ← c107168] at e_1754_7
  have hr := e_1754_7
  simp only [e_1754_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2974)) := by
  have c107169 := (publicBatchWrapper86_copies3349 a h).2.1
  have c107170 := (publicBatchWrapper86_copies3349 a h).2.2.1
  have c107171 := (publicBatchWrapper86_copies3349 a h).2.2.2.1
  have c107172 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.1
  have c107173 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.1
  have c107174 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_8 := arithEq_of_rows h (row := 1754) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_8
  simp only [← c107169, ← c107170, ← c107171] at e_1754_8
  have e_1754_9 := arithEq_of_rows h (row := 1754) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_9
  simp only [← c107172, ← c107173, k2, ← c107174] at e_1754_9
  have hr := e_1754_9
  simp only [e_1754_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2975)) := by
  have c107175 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.1
  have c107176 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.1
  have c107177 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.1
  have c107178 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107179 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107180 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_10 := arithEq_of_rows h (row := 1754) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_10
  simp only [← c107175, ← c107176, ← c107177] at e_1754_10
  have e_1754_11 := arithEq_of_rows h (row := 1754) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_11
  simp only [← c107178, ← c107179, k2, ← c107180] at e_1754_11
  have hr := e_1754_11
  simp only [e_1754_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2976)) := by
  have c107181 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107182 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107183 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107184 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107185 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107186 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_12 := arithEq_of_rows h (row := 1754) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_12
  simp only [← c107181, ← c107182, ← c107183] at e_1754_12
  have e_1754_13 := arithEq_of_rows h (row := 1754) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_13
  simp only [← c107184, ← c107185, k2, ← c107186] at e_1754_13
  have hr := e_1754_13
  simp only [e_1754_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2977)) := by
  have c107187 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107188 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107189 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107190 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107191 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107192 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_14 := arithEq_of_rows h (row := 1754) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_14
  simp only [← c107187, ← c107188, ← c107189] at e_1754_14
  have e_1754_15 := arithEq_of_rows h (row := 1754) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_15
  simp only [← c107190, ← c107191, k2, ← c107192] at e_1754_15
  have hr := e_1754_15
  simp only [e_1754_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2978)) := by
  have c107193 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107194 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107195 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107196 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107197 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107198 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_16 := arithEq_of_rows h (row := 1754) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_16
  simp only [← c107193, ← c107194, ← c107195] at e_1754_16
  have e_1754_17 := arithEq_of_rows h (row := 1754) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_17
  simp only [← c107196, ← c107197, k2, ← c107198] at e_1754_17
  have hr := e_1754_17
  simp only [e_1754_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1754 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2979)) := by
  have c107199 := (publicBatchWrapper86_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107200 := (publicBatchWrapper86_copies3350 a h).1
  have c107201 := (publicBatchWrapper86_copies3350 a h).2.1
  have c107202 := (publicBatchWrapper86_copies3350 a h).2.2.1
  have c107203 := (publicBatchWrapper86_copies3350 a h).2.2.2.1
  have c107204 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1754_18 := arithEq_of_rows h (row := 1754) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_18
  simp only [← c107199, ← c107200, ← c107201] at e_1754_18
  have e_1754_19 := arithEq_of_rows h (row := 1754) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_19
  simp only [← c107202, ← c107203, k2, ← c107204] at e_1754_19
  have hr := e_1754_19
  simp only [e_1754_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2980)) := by
  have c107205 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.1
  have c107206 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.1
  have c107207 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.1
  have c107208 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.1
  have c107209 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.1
  have c107210 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_0 := arithEq_of_rows h (row := 1755) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_0
  simp only [← c107205, ← c107206, ← c107207] at e_1755_0
  have e_1755_1 := arithEq_of_rows h (row := 1755) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_1
  simp only [← c107208, ← c107209, k2, ← c107210] at e_1755_1
  have hr := e_1755_1
  simp only [e_1755_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2981)) := by
  have c107211 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107212 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107213 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107214 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107215 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107216 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_2 := arithEq_of_rows h (row := 1755) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_2
  simp only [← c107211, ← c107212, ← c107213] at e_1755_2
  have e_1755_3 := arithEq_of_rows h (row := 1755) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_3
  simp only [← c107214, ← c107215, k2, ← c107216] at e_1755_3
  have hr := e_1755_3
  simp only [e_1755_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2982)) := by
  have c107217 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107218 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107219 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107220 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107221 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107222 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_4 := arithEq_of_rows h (row := 1755) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_4
  simp only [← c107217, ← c107218, ← c107219] at e_1755_4
  have e_1755_5 := arithEq_of_rows h (row := 1755) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_5
  simp only [← c107220, ← c107221, k2, ← c107222] at e_1755_5
  have hr := e_1755_5
  simp only [e_1755_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2983)) := by
  have c107223 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107224 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107225 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107226 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107227 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107228 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_6 := arithEq_of_rows h (row := 1755) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_6
  simp only [← c107223, ← c107224, ← c107225] at e_1755_6
  have e_1755_7 := arithEq_of_rows h (row := 1755) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_7
  simp only [← c107226, ← c107227, k2, ← c107228] at e_1755_7
  have hr := e_1755_7
  simp only [e_1755_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2984)) := by
  have c107229 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107230 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107231 := (publicBatchWrapper86_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107232 := (publicBatchWrapper86_copies3351 a h).1
  have c107233 := (publicBatchWrapper86_copies3351 a h).2.1
  have c107234 := (publicBatchWrapper86_copies3351 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_8 := arithEq_of_rows h (row := 1755) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_8
  simp only [← c107229, ← c107230, ← c107231] at e_1755_8
  have e_1755_9 := arithEq_of_rows h (row := 1755) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_9
  simp only [← c107232, ← c107233, k2, ← c107234] at e_1755_9
  have hr := e_1755_9
  simp only [e_1755_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2985)) := by
  have c107235 := (publicBatchWrapper86_copies3351 a h).2.2.2.1
  have c107236 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.1
  have c107237 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.1
  have c107238 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.1
  have c107239 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.1
  have c107240 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_10 := arithEq_of_rows h (row := 1755) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_10
  simp only [← c107235, ← c107236, ← c107237] at e_1755_10
  have e_1755_11 := arithEq_of_rows h (row := 1755) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_11
  simp only [← c107238, ← c107239, k2, ← c107240] at e_1755_11
  have hr := e_1755_11
  simp only [e_1755_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2986)) := by
  have c107241 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.1
  have c107242 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107243 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107244 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107245 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107246 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_12 := arithEq_of_rows h (row := 1755) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_12
  simp only [← c107241, ← c107242, ← c107243] at e_1755_12
  have e_1755_13 := arithEq_of_rows h (row := 1755) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_13
  simp only [← c107244, ← c107245, k2, ← c107246] at e_1755_13
  have hr := e_1755_13
  simp only [e_1755_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2987)) := by
  have c107247 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107248 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107249 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107250 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107251 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107252 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_14 := arithEq_of_rows h (row := 1755) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_14
  simp only [← c107247, ← c107248, ← c107249] at e_1755_14
  have e_1755_15 := arithEq_of_rows h (row := 1755) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_15
  simp only [← c107250, ← c107251, k2, ← c107252] at e_1755_15
  have hr := e_1755_15
  simp only [e_1755_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2988)) := by
  have c107253 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107254 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107255 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107256 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107257 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107258 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_16 := arithEq_of_rows h (row := 1755) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_16
  simp only [← c107253, ← c107254, ← c107255] at e_1755_16
  have e_1755_17 := arithEq_of_rows h (row := 1755) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_17
  simp only [← c107256, ← c107257, k2, ← c107258] at e_1755_17
  have hr := e_1755_17
  simp only [e_1755_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1755 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2989)) := by
  have c107259 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107260 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107261 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107262 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107263 := (publicBatchWrapper86_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107264 := (publicBatchWrapper86_copies3352 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1755_18 := arithEq_of_rows h (row := 1755) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_18
  simp only [← c107259, ← c107260, ← c107261] at e_1755_18
  have e_1755_19 := arithEq_of_rows h (row := 1755) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_19
  simp only [← c107262, ← c107263, k2, ← c107264] at e_1755_19
  have hr := e_1755_19
  simp only [e_1755_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2990)) := by
  have c107265 := (publicBatchWrapper86_copies3352 a h).2.1
  have c107266 := (publicBatchWrapper86_copies3352 a h).2.2.1
  have c107267 := (publicBatchWrapper86_copies3352 a h).2.2.2.1
  have c107268 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.1
  have c107269 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.1
  have c107270 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_0 := arithEq_of_rows h (row := 1756) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_0
  simp only [← c107265, ← c107266, ← c107267] at e_1756_0
  have e_1756_1 := arithEq_of_rows h (row := 1756) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_1
  simp only [← c107268, ← c107269, k2, ← c107270] at e_1756_1
  have hr := e_1756_1
  simp only [e_1756_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2991)) := by
  have c107271 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.1
  have c107272 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.1
  have c107273 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.1
  have c107274 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107275 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107276 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_2 := arithEq_of_rows h (row := 1756) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_2
  simp only [← c107271, ← c107272, ← c107273] at e_1756_2
  have e_1756_3 := arithEq_of_rows h (row := 1756) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_3
  simp only [← c107274, ← c107275, k2, ← c107276] at e_1756_3
  have hr := e_1756_3
  simp only [e_1756_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2992)) := by
  have c107277 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107278 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107279 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107280 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107281 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107282 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_4 := arithEq_of_rows h (row := 1756) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_4
  simp only [← c107277, ← c107278, ← c107279] at e_1756_4
  have e_1756_5 := arithEq_of_rows h (row := 1756) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_5
  simp only [← c107280, ← c107281, k2, ← c107282] at e_1756_5
  have hr := e_1756_5
  simp only [e_1756_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2993)) := by
  have c107283 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107284 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107285 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107286 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107287 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107288 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_6 := arithEq_of_rows h (row := 1756) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_6
  simp only [← c107283, ← c107284, ← c107285] at e_1756_6
  have e_1756_7 := arithEq_of_rows h (row := 1756) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_7
  simp only [← c107286, ← c107287, k2, ← c107288] at e_1756_7
  have hr := e_1756_7
  simp only [e_1756_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2994)) := by
  have c107289 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107290 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107291 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107292 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107293 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107294 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_8 := arithEq_of_rows h (row := 1756) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_8
  simp only [← c107289, ← c107290, ← c107291] at e_1756_8
  have e_1756_9 := arithEq_of_rows h (row := 1756) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_9
  simp only [← c107292, ← c107293, k2, ← c107294] at e_1756_9
  have hr := e_1756_9
  simp only [e_1756_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2995)) := by
  have c107295 := (publicBatchWrapper86_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107296 := (publicBatchWrapper86_copies3353 a h).1
  have c107297 := (publicBatchWrapper86_copies3353 a h).2.1
  have c107298 := (publicBatchWrapper86_copies3353 a h).2.2.1
  have c107299 := (publicBatchWrapper86_copies3353 a h).2.2.2.1
  have c107300 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_10 := arithEq_of_rows h (row := 1756) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_10
  simp only [← c107295, ← c107296, ← c107297] at e_1756_10
  have e_1756_11 := arithEq_of_rows h (row := 1756) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_11
  simp only [← c107298, ← c107299, k2, ← c107300] at e_1756_11
  have hr := e_1756_11
  simp only [e_1756_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2996)) := by
  have c107301 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.1
  have c107302 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.1
  have c107303 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.1
  have c107304 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.1
  have c107305 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.1
  have c107306 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_12 := arithEq_of_rows h (row := 1756) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_12
  simp only [← c107301, ← c107302, ← c107303] at e_1756_12
  have e_1756_13 := arithEq_of_rows h (row := 1756) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_13
  simp only [← c107304, ← c107305, k2, ← c107306] at e_1756_13
  have hr := e_1756_13
  simp only [e_1756_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2997)) := by
  have c107307 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107308 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107309 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107310 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107311 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107312 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_14 := arithEq_of_rows h (row := 1756) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_14
  simp only [← c107307, ← c107308, ← c107309] at e_1756_14
  have e_1756_15 := arithEq_of_rows h (row := 1756) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_15
  simp only [← c107310, ← c107311, k2, ← c107312] at e_1756_15
  have hr := e_1756_15
  simp only [e_1756_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2998)) := by
  have c107313 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107314 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107315 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107316 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107317 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107318 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_16 := arithEq_of_rows h (row := 1756) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_16
  simp only [← c107313, ← c107314, ← c107315] at e_1756_16
  have e_1756_17 := arithEq_of_rows h (row := 1756) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_17
  simp only [← c107316, ← c107317, k2, ← c107318] at e_1756_17
  have hr := e_1756_17
  simp only [e_1756_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1756 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2999)) := by
  have c107319 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107320 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107321 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107322 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107323 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107324 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1756_18 := arithEq_of_rows h (row := 1756) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_18
  simp only [← c107319, ← c107320, ← c107321] at e_1756_18
  have e_1756_19 := arithEq_of_rows h (row := 1756) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_19
  simp only [← c107322, ← c107323, k2, ← c107324] at e_1756_19
  have hr := e_1756_19
  simp only [e_1756_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 3000)) := by
  have c107325 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107326 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107327 := (publicBatchWrapper86_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107328 := (publicBatchWrapper86_copies3354 a h).1
  have c107329 := (publicBatchWrapper86_copies3354 a h).2.1
  have c107330 := (publicBatchWrapper86_copies3354 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_0 := arithEq_of_rows h (row := 1757) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_0
  simp only [← c107325, ← c107326, ← c107327] at e_1757_0
  have e_1757_1 := arithEq_of_rows h (row := 1757) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_1
  simp only [← c107328, ← c107329, k2, ← c107330] at e_1757_1
  have hr := e_1757_1
  simp only [e_1757_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 3001)) := by
  have c107331 := (publicBatchWrapper86_copies3354 a h).2.2.2.1
  have c107332 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.1
  have c107333 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.1
  have c107334 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.1
  have c107335 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.1
  have c107336 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_2 := arithEq_of_rows h (row := 1757) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_2
  simp only [← c107331, ← c107332, ← c107333] at e_1757_2
  have e_1757_3 := arithEq_of_rows h (row := 1757) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_3
  simp only [← c107334, ← c107335, k2, ← c107336] at e_1757_3
  have hr := e_1757_3
  simp only [e_1757_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3170)) := by
  have c107337 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.1
  have c107338 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107339 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107340 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107341 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107342 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_4 := arithEq_of_rows h (row := 1757) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_4
  simp only [← c107337, ← c107338, ← c107339] at e_1757_4
  have e_1757_5 := arithEq_of_rows h (row := 1757) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_5
  simp only [← c107340, ← c107341, k2, ← c107342] at e_1757_5
  have hr := e_1757_5
  simp only [e_1757_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3171)) := by
  have c107343 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107344 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107345 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107346 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107347 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107348 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_6 := arithEq_of_rows h (row := 1757) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_6
  simp only [← c107343, ← c107344, ← c107345] at e_1757_6
  have e_1757_7 := arithEq_of_rows h (row := 1757) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_7
  simp only [← c107346, ← c107347, k2, ← c107348] at e_1757_7
  have hr := e_1757_7
  simp only [e_1757_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3172)) := by
  have c107349 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107350 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107351 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107352 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107353 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107354 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_8 := arithEq_of_rows h (row := 1757) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_8
  simp only [← c107349, ← c107350, ← c107351] at e_1757_8
  have e_1757_9 := arithEq_of_rows h (row := 1757) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_9
  simp only [← c107352, ← c107353, k2, ← c107354] at e_1757_9
  have hr := e_1757_9
  simp only [e_1757_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3173)) := by
  have c107355 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107356 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107357 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107358 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107359 := (publicBatchWrapper86_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107360 := (publicBatchWrapper86_copies3355 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_10 := arithEq_of_rows h (row := 1757) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_10
  simp only [← c107355, ← c107356, ← c107357] at e_1757_10
  have e_1757_11 := arithEq_of_rows h (row := 1757) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_11
  simp only [← c107358, ← c107359, k2, ← c107360] at e_1757_11
  have hr := e_1757_11
  simp only [e_1757_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3174)) := by
  have c107361 := (publicBatchWrapper86_copies3355 a h).2.1
  have c107362 := (publicBatchWrapper86_copies3355 a h).2.2.1
  have c107363 := (publicBatchWrapper86_copies3355 a h).2.2.2.1
  have c107364 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.1
  have c107365 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.1
  have c107366 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_12 := arithEq_of_rows h (row := 1757) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_12
  simp only [← c107361, ← c107362, ← c107363] at e_1757_12
  have e_1757_13 := arithEq_of_rows h (row := 1757) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_13
  simp only [← c107364, ← c107365, k2, ← c107366] at e_1757_13
  have hr := e_1757_13
  simp only [e_1757_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3175)) := by
  have c107367 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.1
  have c107368 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.1
  have c107369 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.1
  have c107370 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107371 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107372 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_14 := arithEq_of_rows h (row := 1757) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_14
  simp only [← c107367, ← c107368, ← c107369] at e_1757_14
  have e_1757_15 := arithEq_of_rows h (row := 1757) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_15
  simp only [← c107370, ← c107371, k2, ← c107372] at e_1757_15
  have hr := e_1757_15
  simp only [e_1757_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3176)) := by
  have c107373 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107374 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107375 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107376 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107377 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107378 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_16 := arithEq_of_rows h (row := 1757) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_16
  simp only [← c107373, ← c107374, ← c107375] at e_1757_16
  have e_1757_17 := arithEq_of_rows h (row := 1757) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_17
  simp only [← c107376, ← c107377, k2, ← c107378] at e_1757_17
  have hr := e_1757_17
  simp only [e_1757_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1757 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3177)) := by
  have c107379 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107380 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107381 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107382 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107383 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107384 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1757_18 := arithEq_of_rows h (row := 1757) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_18
  simp only [← c107379, ← c107380, ← c107381] at e_1757_18
  have e_1757_19 := arithEq_of_rows h (row := 1757) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_19
  simp only [← c107382, ← c107383, k2, ← c107384] at e_1757_19
  have hr := e_1757_19
  simp only [e_1757_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3178)) := by
  have c107385 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107386 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107387 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107388 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107389 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107390 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_0 := arithEq_of_rows h (row := 1758) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_0
  simp only [← c107385, ← c107386, ← c107387] at e_1758_0
  have e_1758_1 := arithEq_of_rows h (row := 1758) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_1
  simp only [← c107388, ← c107389, k2, ← c107390] at e_1758_1
  have hr := e_1758_1
  simp only [e_1758_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3179)) := by
  have c107391 := (publicBatchWrapper86_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107392 := (publicBatchWrapper86_copies3356 a h).1
  have c107393 := (publicBatchWrapper86_copies3356 a h).2.1
  have c107394 := (publicBatchWrapper86_copies3356 a h).2.2.1
  have c107395 := (publicBatchWrapper86_copies3356 a h).2.2.2.1
  have c107396 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_2 := arithEq_of_rows h (row := 1758) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_2
  simp only [← c107391, ← c107392, ← c107393] at e_1758_2
  have e_1758_3 := arithEq_of_rows h (row := 1758) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_3
  simp only [← c107394, ← c107395, k2, ← c107396] at e_1758_3
  have hr := e_1758_3
  simp only [e_1758_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3180)) := by
  have c107397 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.1
  have c107398 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.1
  have c107399 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.1
  have c107400 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.1
  have c107401 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.1
  have c107402 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_4 := arithEq_of_rows h (row := 1758) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_4
  simp only [← c107397, ← c107398, ← c107399] at e_1758_4
  have e_1758_5 := arithEq_of_rows h (row := 1758) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_5
  simp only [← c107400, ← c107401, k2, ← c107402] at e_1758_5
  have hr := e_1758_5
  simp only [e_1758_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3181)) := by
  have c107403 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107404 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107405 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107406 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107407 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107408 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_6 := arithEq_of_rows h (row := 1758) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_6
  simp only [← c107403, ← c107404, ← c107405] at e_1758_6
  have e_1758_7 := arithEq_of_rows h (row := 1758) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_7
  simp only [← c107406, ← c107407, k2, ← c107408] at e_1758_7
  have hr := e_1758_7
  simp only [e_1758_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3182)) := by
  have c107409 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107410 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107411 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107412 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107413 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107414 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_8 := arithEq_of_rows h (row := 1758) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_8
  simp only [← c107409, ← c107410, ← c107411] at e_1758_8
  have e_1758_9 := arithEq_of_rows h (row := 1758) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_9
  simp only [← c107412, ← c107413, k2, ← c107414] at e_1758_9
  have hr := e_1758_9
  simp only [e_1758_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3183)) := by
  have c107415 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107416 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107417 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107418 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107419 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107420 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_10 := arithEq_of_rows h (row := 1758) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_10
  simp only [← c107415, ← c107416, ← c107417] at e_1758_10
  have e_1758_11 := arithEq_of_rows h (row := 1758) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_11
  simp only [← c107418, ← c107419, k2, ← c107420] at e_1758_11
  have hr := e_1758_11
  simp only [e_1758_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3184)) := by
  have c107421 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107422 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107423 := (publicBatchWrapper86_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107424 := (publicBatchWrapper86_copies3357 a h).1
  have c107425 := (publicBatchWrapper86_copies3357 a h).2.1
  have c107426 := (publicBatchWrapper86_copies3357 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_12 := arithEq_of_rows h (row := 1758) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_12
  simp only [← c107421, ← c107422, ← c107423] at e_1758_12
  have e_1758_13 := arithEq_of_rows h (row := 1758) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_13
  simp only [← c107424, ← c107425, k2, ← c107426] at e_1758_13
  have hr := e_1758_13
  simp only [e_1758_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3185)) := by
  have c107427 := (publicBatchWrapper86_copies3357 a h).2.2.2.1
  have c107428 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.1
  have c107429 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.1
  have c107430 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.1
  have c107431 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.1
  have c107432 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_14 := arithEq_of_rows h (row := 1758) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_14
  simp only [← c107427, ← c107428, ← c107429] at e_1758_14
  have e_1758_15 := arithEq_of_rows h (row := 1758) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_15
  simp only [← c107430, ← c107431, k2, ← c107432] at e_1758_15
  have hr := e_1758_15
  simp only [e_1758_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3186)) := by
  have c107433 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.1
  have c107434 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107435 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107436 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107437 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107438 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_16 := arithEq_of_rows h (row := 1758) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_16
  simp only [← c107433, ← c107434, ← c107435] at e_1758_16
  have e_1758_17 := arithEq_of_rows h (row := 1758) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_17
  simp only [← c107436, ← c107437, k2, ← c107438] at e_1758_17
  have hr := e_1758_17
  simp only [e_1758_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1758 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3187)) := by
  have c107439 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107440 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107441 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107442 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107443 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107444 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1758_18 := arithEq_of_rows h (row := 1758) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_18
  simp only [← c107439, ← c107440, ← c107441] at e_1758_18
  have e_1758_19 := arithEq_of_rows h (row := 1758) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_19
  simp only [← c107442, ← c107443, k2, ← c107444] at e_1758_19
  have hr := e_1758_19
  simp only [e_1758_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3188)) := by
  have c107445 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107446 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107447 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107448 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107449 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107450 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_0 := arithEq_of_rows h (row := 1759) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_0
  simp only [← c107445, ← c107446, ← c107447] at e_1759_0
  have e_1759_1 := arithEq_of_rows h (row := 1759) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_1
  simp only [← c107448, ← c107449, k2, ← c107450] at e_1759_1
  have hr := e_1759_1
  simp only [e_1759_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3189)) := by
  have c107451 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107452 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107453 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107454 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107455 := (publicBatchWrapper86_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107456 := (publicBatchWrapper86_copies3358 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_2 := arithEq_of_rows h (row := 1759) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_2
  simp only [← c107451, ← c107452, ← c107453] at e_1759_2
  have e_1759_3 := arithEq_of_rows h (row := 1759) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_3
  simp only [← c107454, ← c107455, k2, ← c107456] at e_1759_3
  have hr := e_1759_3
  simp only [e_1759_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3190)) := by
  have c107457 := (publicBatchWrapper86_copies3358 a h).2.1
  have c107458 := (publicBatchWrapper86_copies3358 a h).2.2.1
  have c107459 := (publicBatchWrapper86_copies3358 a h).2.2.2.1
  have c107460 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.1
  have c107461 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.1
  have c107462 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_4 := arithEq_of_rows h (row := 1759) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_4
  simp only [← c107457, ← c107458, ← c107459] at e_1759_4
  have e_1759_5 := arithEq_of_rows h (row := 1759) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_5
  simp only [← c107460, ← c107461, k2, ← c107462] at e_1759_5
  have hr := e_1759_5
  simp only [e_1759_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3191)) := by
  have c107463 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.1
  have c107464 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.1
  have c107465 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.1
  have c107466 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107467 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107468 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_6 := arithEq_of_rows h (row := 1759) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_6
  simp only [← c107463, ← c107464, ← c107465] at e_1759_6
  have e_1759_7 := arithEq_of_rows h (row := 1759) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_7
  simp only [← c107466, ← c107467, k2, ← c107468] at e_1759_7
  have hr := e_1759_7
  simp only [e_1759_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3192)) := by
  have c107469 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107470 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107471 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107472 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107473 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107474 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_8 := arithEq_of_rows h (row := 1759) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_8
  simp only [← c107469, ← c107470, ← c107471] at e_1759_8
  have e_1759_9 := arithEq_of_rows h (row := 1759) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_9
  simp only [← c107472, ← c107473, k2, ← c107474] at e_1759_9
  have hr := e_1759_9
  simp only [e_1759_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3193)) := by
  have c107475 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107476 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107477 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107478 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107479 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107480 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_10 := arithEq_of_rows h (row := 1759) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_10
  simp only [← c107475, ← c107476, ← c107477] at e_1759_10
  have e_1759_11 := arithEq_of_rows h (row := 1759) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_11
  simp only [← c107478, ← c107479, k2, ← c107480] at e_1759_11
  have hr := e_1759_11
  simp only [e_1759_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3194)) := by
  have c107481 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107482 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107483 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107484 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107485 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107486 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_12 := arithEq_of_rows h (row := 1759) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_12
  simp only [← c107481, ← c107482, ← c107483] at e_1759_12
  have e_1759_13 := arithEq_of_rows h (row := 1759) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_13
  simp only [← c107484, ← c107485, k2, ← c107486] at e_1759_13
  have hr := e_1759_13
  simp only [e_1759_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3195)) := by
  have c107487 := (publicBatchWrapper86_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107488 := (publicBatchWrapper86_copies3359 a h).1
  have c107489 := (publicBatchWrapper86_copies3359 a h).2.1
  have c107490 := (publicBatchWrapper86_copies3359 a h).2.2.1
  have c107491 := (publicBatchWrapper86_copies3359 a h).2.2.2.1
  have c107492 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_14 := arithEq_of_rows h (row := 1759) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_14
  simp only [← c107487, ← c107488, ← c107489] at e_1759_14
  have e_1759_15 := arithEq_of_rows h (row := 1759) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_15
  simp only [← c107490, ← c107491, k2, ← c107492] at e_1759_15
  have hr := e_1759_15
  simp only [e_1759_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3196)) := by
  have c107493 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.1
  have c107494 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.1
  have c107495 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.1
  have c107496 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.1
  have c107497 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.1
  have c107498 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_16 := arithEq_of_rows h (row := 1759) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_16
  simp only [← c107493, ← c107494, ← c107495] at e_1759_16
  have e_1759_17 := arithEq_of_rows h (row := 1759) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_17
  simp only [← c107496, ← c107497, k2, ← c107498] at e_1759_17
  have hr := e_1759_17
  simp only [e_1759_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1759 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3197)) := by
  have c107499 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107500 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107501 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107502 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107503 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107504 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1759_18 := arithEq_of_rows h (row := 1759) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_18
  simp only [← c107499, ← c107500, ← c107501] at e_1759_18
  have e_1759_19 := arithEq_of_rows h (row := 1759) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_19
  simp only [← c107502, ← c107503, k2, ← c107504] at e_1759_19
  have hr := e_1759_19
  simp only [e_1759_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3198)) := by
  have c107505 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107506 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107507 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107508 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107509 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107510 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_0 := arithEq_of_rows h (row := 1760) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_0
  simp only [← c107505, ← c107506, ← c107507] at e_1760_0
  have e_1760_1 := arithEq_of_rows h (row := 1760) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_1
  simp only [← c107508, ← c107509, k2, ← c107510] at e_1760_1
  have hr := e_1760_1
  simp only [e_1760_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3199)) := by
  have c107511 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107512 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107513 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107514 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107515 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107516 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_2 := arithEq_of_rows h (row := 1760) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_2
  simp only [← c107511, ← c107512, ← c107513] at e_1760_2
  have e_1760_3 := arithEq_of_rows h (row := 1760) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_3
  simp only [← c107514, ← c107515, k2, ← c107516] at e_1760_3
  have hr := e_1760_3
  simp only [e_1760_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3200)) := by
  have c107517 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107518 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107519 := (publicBatchWrapper86_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107520 := (publicBatchWrapper86_copies3360 a h).1
  have c107521 := (publicBatchWrapper86_copies3360 a h).2.1
  have c107522 := (publicBatchWrapper86_copies3360 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_4 := arithEq_of_rows h (row := 1760) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_4
  simp only [← c107517, ← c107518, ← c107519] at e_1760_4
  have e_1760_5 := arithEq_of_rows h (row := 1760) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_5
  simp only [← c107520, ← c107521, k2, ← c107522] at e_1760_5
  have hr := e_1760_5
  simp only [e_1760_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3201)) := by
  have c107523 := (publicBatchWrapper86_copies3360 a h).2.2.2.1
  have c107524 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.1
  have c107525 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.1
  have c107526 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.1
  have c107527 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.1
  have c107528 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_6 := arithEq_of_rows h (row := 1760) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_6
  simp only [← c107523, ← c107524, ← c107525] at e_1760_6
  have e_1760_7 := arithEq_of_rows h (row := 1760) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_7
  simp only [← c107526, ← c107527, k2, ← c107528] at e_1760_7
  have hr := e_1760_7
  simp only [e_1760_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3370)) := by
  have c107529 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.1
  have c107530 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107531 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107532 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107533 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107534 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_8 := arithEq_of_rows h (row := 1760) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_8
  simp only [← c107529, ← c107530, ← c107531] at e_1760_8
  have e_1760_9 := arithEq_of_rows h (row := 1760) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_9
  simp only [← c107532, ← c107533, k2, ← c107534] at e_1760_9
  have hr := e_1760_9
  simp only [e_1760_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3371)) := by
  have c107535 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107536 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107537 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107538 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107539 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107540 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_10 := arithEq_of_rows h (row := 1760) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_10
  simp only [← c107535, ← c107536, ← c107537] at e_1760_10
  have e_1760_11 := arithEq_of_rows h (row := 1760) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_11
  simp only [← c107538, ← c107539, k2, ← c107540] at e_1760_11
  have hr := e_1760_11
  simp only [e_1760_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3372)) := by
  have c107541 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107542 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107543 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107544 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107545 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107546 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_12 := arithEq_of_rows h (row := 1760) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_12
  simp only [← c107541, ← c107542, ← c107543] at e_1760_12
  have e_1760_13 := arithEq_of_rows h (row := 1760) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_13
  simp only [← c107544, ← c107545, k2, ← c107546] at e_1760_13
  have hr := e_1760_13
  simp only [e_1760_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3373)) := by
  have c107547 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107548 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107549 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107550 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107551 := (publicBatchWrapper86_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107552 := (publicBatchWrapper86_copies3361 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_14 := arithEq_of_rows h (row := 1760) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_14
  simp only [← c107547, ← c107548, ← c107549] at e_1760_14
  have e_1760_15 := arithEq_of_rows h (row := 1760) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_15
  simp only [← c107550, ← c107551, k2, ← c107552] at e_1760_15
  have hr := e_1760_15
  simp only [e_1760_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3374)) := by
  have c107553 := (publicBatchWrapper86_copies3361 a h).2.1
  have c107554 := (publicBatchWrapper86_copies3361 a h).2.2.1
  have c107555 := (publicBatchWrapper86_copies3361 a h).2.2.2.1
  have c107556 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.1
  have c107557 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.1
  have c107558 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_16 := arithEq_of_rows h (row := 1760) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_16
  simp only [← c107553, ← c107554, ← c107555] at e_1760_16
  have e_1760_17 := arithEq_of_rows h (row := 1760) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_17
  simp only [← c107556, ← c107557, k2, ← c107558] at e_1760_17
  have hr := e_1760_17
  simp only [e_1760_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1760 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3375)) := by
  have c107559 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.1
  have c107560 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.1
  have c107561 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.1
  have c107562 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107563 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107564 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1760_18 := arithEq_of_rows h (row := 1760) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_18
  simp only [← c107559, ← c107560, ← c107561] at e_1760_18
  have e_1760_19 := arithEq_of_rows h (row := 1760) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_19
  simp only [← c107562, ← c107563, k2, ← c107564] at e_1760_19
  have hr := e_1760_19
  simp only [e_1760_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3376)) := by
  have c107565 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107566 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107567 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107568 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107569 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107570 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_0 := arithEq_of_rows h (row := 1761) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_0
  simp only [← c107565, ← c107566, ← c107567] at e_1761_0
  have e_1761_1 := arithEq_of_rows h (row := 1761) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_1
  simp only [← c107568, ← c107569, k2, ← c107570] at e_1761_1
  have hr := e_1761_1
  simp only [e_1761_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3377)) := by
  have c107571 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107572 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107573 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107574 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107575 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107576 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_2 := arithEq_of_rows h (row := 1761) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_2
  simp only [← c107571, ← c107572, ← c107573] at e_1761_2
  have e_1761_3 := arithEq_of_rows h (row := 1761) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_3
  simp only [← c107574, ← c107575, k2, ← c107576] at e_1761_3
  have hr := e_1761_3
  simp only [e_1761_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3378)) := by
  have c107577 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107578 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107579 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107580 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107581 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107582 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_4 := arithEq_of_rows h (row := 1761) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_4
  simp only [← c107577, ← c107578, ← c107579] at e_1761_4
  have e_1761_5 := arithEq_of_rows h (row := 1761) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_5
  simp only [← c107580, ← c107581, k2, ← c107582] at e_1761_5
  have hr := e_1761_5
  simp only [e_1761_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3379)) := by
  have c107583 := (publicBatchWrapper86_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107584 := (publicBatchWrapper86_copies3362 a h).1
  have c107585 := (publicBatchWrapper86_copies3362 a h).2.1
  have c107586 := (publicBatchWrapper86_copies3362 a h).2.2.1
  have c107587 := (publicBatchWrapper86_copies3362 a h).2.2.2.1
  have c107588 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_6 := arithEq_of_rows h (row := 1761) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_6
  simp only [← c107583, ← c107584, ← c107585] at e_1761_6
  have e_1761_7 := arithEq_of_rows h (row := 1761) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_7
  simp only [← c107586, ← c107587, k2, ← c107588] at e_1761_7
  have hr := e_1761_7
  simp only [e_1761_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3380)) := by
  have c107589 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.1
  have c107590 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.1
  have c107591 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.1
  have c107592 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.1
  have c107593 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.1
  have c107594 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_8 := arithEq_of_rows h (row := 1761) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_8
  simp only [← c107589, ← c107590, ← c107591] at e_1761_8
  have e_1761_9 := arithEq_of_rows h (row := 1761) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_9
  simp only [← c107592, ← c107593, k2, ← c107594] at e_1761_9
  have hr := e_1761_9
  simp only [e_1761_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3381)) := by
  have c107595 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107596 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107597 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107598 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107599 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107600 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_10 := arithEq_of_rows h (row := 1761) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_10
  simp only [← c107595, ← c107596, ← c107597] at e_1761_10
  have e_1761_11 := arithEq_of_rows h (row := 1761) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_11
  simp only [← c107598, ← c107599, k2, ← c107600] at e_1761_11
  have hr := e_1761_11
  simp only [e_1761_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3382)) := by
  have c107601 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107602 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107603 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107604 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107605 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107606 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_12 := arithEq_of_rows h (row := 1761) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_12
  simp only [← c107601, ← c107602, ← c107603] at e_1761_12
  have e_1761_13 := arithEq_of_rows h (row := 1761) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_13
  simp only [← c107604, ← c107605, k2, ← c107606] at e_1761_13
  have hr := e_1761_13
  simp only [e_1761_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3383)) := by
  have c107607 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107608 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107609 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107610 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107611 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107612 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_14 := arithEq_of_rows h (row := 1761) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_14
  simp only [← c107607, ← c107608, ← c107609] at e_1761_14
  have e_1761_15 := arithEq_of_rows h (row := 1761) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_15
  simp only [← c107610, ← c107611, k2, ← c107612] at e_1761_15
  have hr := e_1761_15
  simp only [e_1761_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3384)) := by
  have c107613 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107614 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107615 := (publicBatchWrapper86_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107616 := (publicBatchWrapper86_copies3363 a h).1
  have c107617 := (publicBatchWrapper86_copies3363 a h).2.1
  have c107618 := (publicBatchWrapper86_copies3363 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_16 := arithEq_of_rows h (row := 1761) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_16
  simp only [← c107613, ← c107614, ← c107615] at e_1761_16
  have e_1761_17 := arithEq_of_rows h (row := 1761) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_17
  simp only [← c107616, ← c107617, k2, ← c107618] at e_1761_17
  have hr := e_1761_17
  simp only [e_1761_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1761 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3385)) := by
  have c107619 := (publicBatchWrapper86_copies3363 a h).2.2.2.1
  have c107620 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.1
  have c107621 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.1
  have c107622 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.1
  have c107623 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.1
  have c107624 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1761_18 := arithEq_of_rows h (row := 1761) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_18
  simp only [← c107619, ← c107620, ← c107621] at e_1761_18
  have e_1761_19 := arithEq_of_rows h (row := 1761) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_19
  simp only [← c107622, ← c107623, k2, ← c107624] at e_1761_19
  have hr := e_1761_19
  simp only [e_1761_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3386)) := by
  have c107625 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.1
  have c107626 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107627 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107628 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107629 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107630 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_0 := arithEq_of_rows h (row := 1762) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_0
  simp only [← c107625, ← c107626, ← c107627] at e_1762_0
  have e_1762_1 := arithEq_of_rows h (row := 1762) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_1
  simp only [← c107628, ← c107629, k2, ← c107630] at e_1762_1
  have hr := e_1762_1
  simp only [e_1762_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3387)) := by
  have c107631 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107632 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107633 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107634 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107635 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107636 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_2 := arithEq_of_rows h (row := 1762) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_2
  simp only [← c107631, ← c107632, ← c107633] at e_1762_2
  have e_1762_3 := arithEq_of_rows h (row := 1762) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_3
  simp only [← c107634, ← c107635, k2, ← c107636] at e_1762_3
  have hr := e_1762_3
  simp only [e_1762_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3388)) := by
  have c107637 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107638 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107639 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107640 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107641 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107642 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_4 := arithEq_of_rows h (row := 1762) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_4
  simp only [← c107637, ← c107638, ← c107639] at e_1762_4
  have e_1762_5 := arithEq_of_rows h (row := 1762) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_5
  simp only [← c107640, ← c107641, k2, ← c107642] at e_1762_5
  have hr := e_1762_5
  simp only [e_1762_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3389)) := by
  have c107643 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107644 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107645 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107646 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107647 := (publicBatchWrapper86_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107648 := (publicBatchWrapper86_copies3364 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_6 := arithEq_of_rows h (row := 1762) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_6
  simp only [← c107643, ← c107644, ← c107645] at e_1762_6
  have e_1762_7 := arithEq_of_rows h (row := 1762) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_7
  simp only [← c107646, ← c107647, k2, ← c107648] at e_1762_7
  have hr := e_1762_7
  simp only [e_1762_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3390)) := by
  have c107649 := (publicBatchWrapper86_copies3364 a h).2.1
  have c107650 := (publicBatchWrapper86_copies3364 a h).2.2.1
  have c107651 := (publicBatchWrapper86_copies3364 a h).2.2.2.1
  have c107652 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.1
  have c107653 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.1
  have c107654 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_8 := arithEq_of_rows h (row := 1762) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_8
  simp only [← c107649, ← c107650, ← c107651] at e_1762_8
  have e_1762_9 := arithEq_of_rows h (row := 1762) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_9
  simp only [← c107652, ← c107653, k2, ← c107654] at e_1762_9
  have hr := e_1762_9
  simp only [e_1762_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3391)) := by
  have c107655 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.1
  have c107656 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.1
  have c107657 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.1
  have c107658 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107659 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107660 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_10 := arithEq_of_rows h (row := 1762) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_10
  simp only [← c107655, ← c107656, ← c107657] at e_1762_10
  have e_1762_11 := arithEq_of_rows h (row := 1762) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_11
  simp only [← c107658, ← c107659, k2, ← c107660] at e_1762_11
  have hr := e_1762_11
  simp only [e_1762_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3392)) := by
  have c107661 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107662 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107663 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107664 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107665 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107666 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_12 := arithEq_of_rows h (row := 1762) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_12
  simp only [← c107661, ← c107662, ← c107663] at e_1762_12
  have e_1762_13 := arithEq_of_rows h (row := 1762) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_13
  simp only [← c107664, ← c107665, k2, ← c107666] at e_1762_13
  have hr := e_1762_13
  simp only [e_1762_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3393)) := by
  have c107667 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107668 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107669 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107670 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107671 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107672 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_14 := arithEq_of_rows h (row := 1762) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_14
  simp only [← c107667, ← c107668, ← c107669] at e_1762_14
  have e_1762_15 := arithEq_of_rows h (row := 1762) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_15
  simp only [← c107670, ← c107671, k2, ← c107672] at e_1762_15
  have hr := e_1762_15
  simp only [e_1762_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3394)) := by
  have c107673 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107674 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107675 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107676 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107677 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107678 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_16 := arithEq_of_rows h (row := 1762) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_16
  simp only [← c107673, ← c107674, ← c107675] at e_1762_16
  have e_1762_17 := arithEq_of_rows h (row := 1762) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_17
  simp only [← c107676, ← c107677, k2, ← c107678] at e_1762_17
  have hr := e_1762_17
  simp only [e_1762_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1762 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3395)) := by
  have c107679 := (publicBatchWrapper86_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107680 := (publicBatchWrapper86_copies3365 a h).1
  have c107681 := (publicBatchWrapper86_copies3365 a h).2.1
  have c107682 := (publicBatchWrapper86_copies3365 a h).2.2.1
  have c107683 := (publicBatchWrapper86_copies3365 a h).2.2.2.1
  have c107684 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1762_18 := arithEq_of_rows h (row := 1762) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_18
  simp only [← c107679, ← c107680, ← c107681] at e_1762_18
  have e_1762_19 := arithEq_of_rows h (row := 1762) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_19
  simp only [← c107682, ← c107683, k2, ← c107684] at e_1762_19
  have hr := e_1762_19
  simp only [e_1762_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3396)) := by
  have c107685 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.1
  have c107686 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.1
  have c107687 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.1
  have c107688 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.1
  have c107689 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.1
  have c107690 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_0 := arithEq_of_rows h (row := 1763) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_0
  simp only [← c107685, ← c107686, ← c107687] at e_1763_0
  have e_1763_1 := arithEq_of_rows h (row := 1763) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_1
  simp only [← c107688, ← c107689, k2, ← c107690] at e_1763_1
  have hr := e_1763_1
  simp only [e_1763_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3397)) := by
  have c107691 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107692 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107693 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107694 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107695 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107696 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_2 := arithEq_of_rows h (row := 1763) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_2
  simp only [← c107691, ← c107692, ← c107693] at e_1763_2
  have e_1763_3 := arithEq_of_rows h (row := 1763) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_3
  simp only [← c107694, ← c107695, k2, ← c107696] at e_1763_3
  have hr := e_1763_3
  simp only [e_1763_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3398)) := by
  have c107697 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107698 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107699 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107700 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107701 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107702 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_4 := arithEq_of_rows h (row := 1763) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_4
  simp only [← c107697, ← c107698, ← c107699] at e_1763_4
  have e_1763_5 := arithEq_of_rows h (row := 1763) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_5
  simp only [← c107700, ← c107701, k2, ← c107702] at e_1763_5
  have hr := e_1763_5
  simp only [e_1763_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3399)) := by
  have c107703 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107704 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107705 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107706 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107707 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107708 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_6 := arithEq_of_rows h (row := 1763) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_6
  simp only [← c107703, ← c107704, ← c107705] at e_1763_6
  have e_1763_7 := arithEq_of_rows h (row := 1763) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_7
  simp only [← c107706, ← c107707, k2, ← c107708] at e_1763_7
  have hr := e_1763_7
  simp only [e_1763_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3400)) := by
  have c107709 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107710 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107711 := (publicBatchWrapper86_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107712 := (publicBatchWrapper86_copies3366 a h).1
  have c107713 := (publicBatchWrapper86_copies3366 a h).2.1
  have c107714 := (publicBatchWrapper86_copies3366 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_8 := arithEq_of_rows h (row := 1763) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_8
  simp only [← c107709, ← c107710, ← c107711] at e_1763_8
  have e_1763_9 := arithEq_of_rows h (row := 1763) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_9
  simp only [← c107712, ← c107713, k2, ← c107714] at e_1763_9
  have hr := e_1763_9
  simp only [e_1763_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3401)) := by
  have c107715 := (publicBatchWrapper86_copies3366 a h).2.2.2.1
  have c107716 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.1
  have c107717 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.1
  have c107718 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.1
  have c107719 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.1
  have c107720 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_10 := arithEq_of_rows h (row := 1763) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_10
  simp only [← c107715, ← c107716, ← c107717] at e_1763_10
  have e_1763_11 := arithEq_of_rows h (row := 1763) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_11
  simp only [← c107718, ← c107719, k2, ← c107720] at e_1763_11
  have hr := e_1763_11
  simp only [e_1763_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3570)) := by
  have c107721 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.1
  have c107722 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107723 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107724 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107725 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107726 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_12 := arithEq_of_rows h (row := 1763) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_12
  simp only [← c107721, ← c107722, ← c107723] at e_1763_12
  have e_1763_13 := arithEq_of_rows h (row := 1763) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_13
  simp only [← c107724, ← c107725, k2, ← c107726] at e_1763_13
  have hr := e_1763_13
  simp only [e_1763_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3571)) := by
  have c107727 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107728 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107729 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107730 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107731 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107732 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_14 := arithEq_of_rows h (row := 1763) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_14
  simp only [← c107727, ← c107728, ← c107729] at e_1763_14
  have e_1763_15 := arithEq_of_rows h (row := 1763) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_15
  simp only [← c107730, ← c107731, k2, ← c107732] at e_1763_15
  have hr := e_1763_15
  simp only [e_1763_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3572)) := by
  have c107733 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107734 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107735 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107736 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107737 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107738 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_16 := arithEq_of_rows h (row := 1763) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_16
  simp only [← c107733, ← c107734, ← c107735] at e_1763_16
  have e_1763_17 := arithEq_of_rows h (row := 1763) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_17
  simp only [← c107736, ← c107737, k2, ← c107738] at e_1763_17
  have hr := e_1763_17
  simp only [e_1763_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1763 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3573)) := by
  have c107739 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107740 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107741 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107742 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107743 := (publicBatchWrapper86_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107744 := (publicBatchWrapper86_copies3367 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1763_18 := arithEq_of_rows h (row := 1763) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_18
  simp only [← c107739, ← c107740, ← c107741] at e_1763_18
  have e_1763_19 := arithEq_of_rows h (row := 1763) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_19
  simp only [← c107742, ← c107743, k2, ← c107744] at e_1763_19
  have hr := e_1763_19
  simp only [e_1763_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3574)) := by
  have c107745 := (publicBatchWrapper86_copies3367 a h).2.1
  have c107746 := (publicBatchWrapper86_copies3367 a h).2.2.1
  have c107747 := (publicBatchWrapper86_copies3367 a h).2.2.2.1
  have c107748 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.1
  have c107749 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.1
  have c107750 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_0 := arithEq_of_rows h (row := 1764) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_0
  simp only [← c107745, ← c107746, ← c107747] at e_1764_0
  have e_1764_1 := arithEq_of_rows h (row := 1764) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_1
  simp only [← c107748, ← c107749, k2, ← c107750] at e_1764_1
  have hr := e_1764_1
  simp only [e_1764_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3575)) := by
  have c107751 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.1
  have c107752 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.1
  have c107753 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.1
  have c107754 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107755 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107756 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_2 := arithEq_of_rows h (row := 1764) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_2
  simp only [← c107751, ← c107752, ← c107753] at e_1764_2
  have e_1764_3 := arithEq_of_rows h (row := 1764) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_3
  simp only [← c107754, ← c107755, k2, ← c107756] at e_1764_3
  have hr := e_1764_3
  simp only [e_1764_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3576)) := by
  have c107757 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107758 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107759 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107760 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107761 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107762 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_4 := arithEq_of_rows h (row := 1764) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_4
  simp only [← c107757, ← c107758, ← c107759] at e_1764_4
  have e_1764_5 := arithEq_of_rows h (row := 1764) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_5
  simp only [← c107760, ← c107761, k2, ← c107762] at e_1764_5
  have hr := e_1764_5
  simp only [e_1764_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3577)) := by
  have c107763 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107764 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107765 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107766 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107767 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107768 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_6 := arithEq_of_rows h (row := 1764) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_6
  simp only [← c107763, ← c107764, ← c107765] at e_1764_6
  have e_1764_7 := arithEq_of_rows h (row := 1764) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_7
  simp only [← c107766, ← c107767, k2, ← c107768] at e_1764_7
  have hr := e_1764_7
  simp only [e_1764_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3578)) := by
  have c107769 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107770 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107771 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107772 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107773 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107774 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_8 := arithEq_of_rows h (row := 1764) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_8
  simp only [← c107769, ← c107770, ← c107771] at e_1764_8
  have e_1764_9 := arithEq_of_rows h (row := 1764) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_9
  simp only [← c107772, ← c107773, k2, ← c107774] at e_1764_9
  have hr := e_1764_9
  simp only [e_1764_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3579)) := by
  have c107775 := (publicBatchWrapper86_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107776 := (publicBatchWrapper86_copies3368 a h).1
  have c107777 := (publicBatchWrapper86_copies3368 a h).2.1
  have c107778 := (publicBatchWrapper86_copies3368 a h).2.2.1
  have c107779 := (publicBatchWrapper86_copies3368 a h).2.2.2.1
  have c107780 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_10 := arithEq_of_rows h (row := 1764) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_10
  simp only [← c107775, ← c107776, ← c107777] at e_1764_10
  have e_1764_11 := arithEq_of_rows h (row := 1764) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_11
  simp only [← c107778, ← c107779, k2, ← c107780] at e_1764_11
  have hr := e_1764_11
  simp only [e_1764_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3580)) := by
  have c107781 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.1
  have c107782 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.1
  have c107783 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.1
  have c107784 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.1
  have c107785 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.1
  have c107786 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_12 := arithEq_of_rows h (row := 1764) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_12
  simp only [← c107781, ← c107782, ← c107783] at e_1764_12
  have e_1764_13 := arithEq_of_rows h (row := 1764) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_13
  simp only [← c107784, ← c107785, k2, ← c107786] at e_1764_13
  have hr := e_1764_13
  simp only [e_1764_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3581)) := by
  have c107787 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107788 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107789 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107790 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107791 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107792 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_14 := arithEq_of_rows h (row := 1764) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_14
  simp only [← c107787, ← c107788, ← c107789] at e_1764_14
  have e_1764_15 := arithEq_of_rows h (row := 1764) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_15
  simp only [← c107790, ← c107791, k2, ← c107792] at e_1764_15
  have hr := e_1764_15
  simp only [e_1764_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3582)) := by
  have c107793 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107794 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107795 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107796 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107797 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107798 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_16 := arithEq_of_rows h (row := 1764) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_16
  simp only [← c107793, ← c107794, ← c107795] at e_1764_16
  have e_1764_17 := arithEq_of_rows h (row := 1764) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_17
  simp only [← c107796, ← c107797, k2, ← c107798] at e_1764_17
  have hr := e_1764_17
  simp only [e_1764_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1764 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3583)) := by
  have c107799 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107800 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107801 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107802 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107803 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107804 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1764_18 := arithEq_of_rows h (row := 1764) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_18
  simp only [← c107799, ← c107800, ← c107801] at e_1764_18
  have e_1764_19 := arithEq_of_rows h (row := 1764) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_19
  simp only [← c107802, ← c107803, k2, ← c107804] at e_1764_19
  have hr := e_1764_19
  simp only [e_1764_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3584)) := by
  have c107805 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107806 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107807 := (publicBatchWrapper86_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107808 := (publicBatchWrapper86_copies3369 a h).1
  have c107809 := (publicBatchWrapper86_copies3369 a h).2.1
  have c107810 := (publicBatchWrapper86_copies3369 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_0 := arithEq_of_rows h (row := 1765) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_0
  simp only [← c107805, ← c107806, ← c107807] at e_1765_0
  have e_1765_1 := arithEq_of_rows h (row := 1765) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_1
  simp only [← c107808, ← c107809, k2, ← c107810] at e_1765_1
  have hr := e_1765_1
  simp only [e_1765_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3585)) := by
  have c107811 := (publicBatchWrapper86_copies3369 a h).2.2.2.1
  have c107812 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.1
  have c107813 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.1
  have c107814 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.1
  have c107815 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.1
  have c107816 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_2 := arithEq_of_rows h (row := 1765) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_2
  simp only [← c107811, ← c107812, ← c107813] at e_1765_2
  have e_1765_3 := arithEq_of_rows h (row := 1765) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_3
  simp only [← c107814, ← c107815, k2, ← c107816] at e_1765_3
  have hr := e_1765_3
  simp only [e_1765_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3586)) := by
  have c107817 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.1
  have c107818 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107819 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107820 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107821 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107822 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_4 := arithEq_of_rows h (row := 1765) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_4
  simp only [← c107817, ← c107818, ← c107819] at e_1765_4
  have e_1765_5 := arithEq_of_rows h (row := 1765) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_5
  simp only [← c107820, ← c107821, k2, ← c107822] at e_1765_5
  have hr := e_1765_5
  simp only [e_1765_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3587)) := by
  have c107823 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107824 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107825 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107826 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107827 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107828 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_6 := arithEq_of_rows h (row := 1765) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_6
  simp only [← c107823, ← c107824, ← c107825] at e_1765_6
  have e_1765_7 := arithEq_of_rows h (row := 1765) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_7
  simp only [← c107826, ← c107827, k2, ← c107828] at e_1765_7
  have hr := e_1765_7
  simp only [e_1765_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3588)) := by
  have c107829 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107830 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107831 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107832 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107833 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107834 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_8 := arithEq_of_rows h (row := 1765) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_8
  simp only [← c107829, ← c107830, ← c107831] at e_1765_8
  have e_1765_9 := arithEq_of_rows h (row := 1765) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_9
  simp only [← c107832, ← c107833, k2, ← c107834] at e_1765_9
  have hr := e_1765_9
  simp only [e_1765_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3589)) := by
  have c107835 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107836 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107837 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107838 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107839 := (publicBatchWrapper86_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107840 := (publicBatchWrapper86_copies3370 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_10 := arithEq_of_rows h (row := 1765) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_10
  simp only [← c107835, ← c107836, ← c107837] at e_1765_10
  have e_1765_11 := arithEq_of_rows h (row := 1765) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_11
  simp only [← c107838, ← c107839, k2, ← c107840] at e_1765_11
  have hr := e_1765_11
  simp only [e_1765_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3590)) := by
  have c107841 := (publicBatchWrapper86_copies3370 a h).2.1
  have c107842 := (publicBatchWrapper86_copies3370 a h).2.2.1
  have c107843 := (publicBatchWrapper86_copies3370 a h).2.2.2.1
  have c107844 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.1
  have c107845 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.1
  have c107846 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_12 := arithEq_of_rows h (row := 1765) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_12
  simp only [← c107841, ← c107842, ← c107843] at e_1765_12
  have e_1765_13 := arithEq_of_rows h (row := 1765) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_13
  simp only [← c107844, ← c107845, k2, ← c107846] at e_1765_13
  have hr := e_1765_13
  simp only [e_1765_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3591)) := by
  have c107847 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.1
  have c107848 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.1
  have c107849 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.1
  have c107850 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107851 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107852 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_14 := arithEq_of_rows h (row := 1765) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_14
  simp only [← c107847, ← c107848, ← c107849] at e_1765_14
  have e_1765_15 := arithEq_of_rows h (row := 1765) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_15
  simp only [← c107850, ← c107851, k2, ← c107852] at e_1765_15
  have hr := e_1765_15
  simp only [e_1765_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3592)) := by
  have c107853 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107854 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107855 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107856 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107857 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107858 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_16 := arithEq_of_rows h (row := 1765) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_16
  simp only [← c107853, ← c107854, ← c107855] at e_1765_16
  have e_1765_17 := arithEq_of_rows h (row := 1765) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_17
  simp only [← c107856, ← c107857, k2, ← c107858] at e_1765_17
  have hr := e_1765_17
  simp only [e_1765_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1765 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3593)) := by
  have c107859 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107860 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107861 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107862 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107863 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107864 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1765_18 := arithEq_of_rows h (row := 1765) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_18
  simp only [← c107859, ← c107860, ← c107861] at e_1765_18
  have e_1765_19 := arithEq_of_rows h (row := 1765) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_19
  simp only [← c107862, ← c107863, k2, ← c107864] at e_1765_19
  have hr := e_1765_19
  simp only [e_1765_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3594)) := by
  have c107865 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107866 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107867 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107868 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107869 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107870 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_0 := arithEq_of_rows h (row := 1766) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_0
  simp only [← c107865, ← c107866, ← c107867] at e_1766_0
  have e_1766_1 := arithEq_of_rows h (row := 1766) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_1
  simp only [← c107868, ← c107869, k2, ← c107870] at e_1766_1
  have hr := e_1766_1
  simp only [e_1766_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3595)) := by
  have c107871 := (publicBatchWrapper86_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107872 := (publicBatchWrapper86_copies3371 a h).1
  have c107873 := (publicBatchWrapper86_copies3371 a h).2.1
  have c107874 := (publicBatchWrapper86_copies3371 a h).2.2.1
  have c107875 := (publicBatchWrapper86_copies3371 a h).2.2.2.1
  have c107876 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_2 := arithEq_of_rows h (row := 1766) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_2
  simp only [← c107871, ← c107872, ← c107873] at e_1766_2
  have e_1766_3 := arithEq_of_rows h (row := 1766) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_3
  simp only [← c107874, ← c107875, k2, ← c107876] at e_1766_3
  have hr := e_1766_3
  simp only [e_1766_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3596)) := by
  have c107877 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.1
  have c107878 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.1
  have c107879 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.1
  have c107880 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.1
  have c107881 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.1
  have c107882 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_4 := arithEq_of_rows h (row := 1766) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_4
  simp only [← c107877, ← c107878, ← c107879] at e_1766_4
  have e_1766_5 := arithEq_of_rows h (row := 1766) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_5
  simp only [← c107880, ← c107881, k2, ← c107882] at e_1766_5
  have hr := e_1766_5
  simp only [e_1766_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3597)) := by
  have c107883 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107884 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107885 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107886 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107887 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107888 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_6 := arithEq_of_rows h (row := 1766) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_6
  simp only [← c107883, ← c107884, ← c107885] at e_1766_6
  have e_1766_7 := arithEq_of_rows h (row := 1766) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_7
  simp only [← c107886, ← c107887, k2, ← c107888] at e_1766_7
  have hr := e_1766_7
  simp only [e_1766_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3598)) := by
  have c107889 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107890 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107891 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107892 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107893 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107894 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_8 := arithEq_of_rows h (row := 1766) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_8
  simp only [← c107889, ← c107890, ← c107891] at e_1766_8
  have e_1766_9 := arithEq_of_rows h (row := 1766) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_9
  simp only [← c107892, ← c107893, k2, ← c107894] at e_1766_9
  have hr := e_1766_9
  simp only [e_1766_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3599)) := by
  have c107895 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107896 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107897 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107898 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107899 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107900 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_10 := arithEq_of_rows h (row := 1766) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_10
  simp only [← c107895, ← c107896, ← c107897] at e_1766_10
  have e_1766_11 := arithEq_of_rows h (row := 1766) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_11
  simp only [← c107898, ← c107899, k2, ← c107900] at e_1766_11
  have hr := e_1766_11
  simp only [e_1766_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3600)) := by
  have c107901 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107902 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107903 := (publicBatchWrapper86_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107904 := (publicBatchWrapper86_copies3372 a h).1
  have c107905 := (publicBatchWrapper86_copies3372 a h).2.1
  have c107906 := (publicBatchWrapper86_copies3372 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_12 := arithEq_of_rows h (row := 1766) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_12
  simp only [← c107901, ← c107902, ← c107903] at e_1766_12
  have e_1766_13 := arithEq_of_rows h (row := 1766) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_13
  simp only [← c107904, ← c107905, k2, ← c107906] at e_1766_13
  have hr := e_1766_13
  simp only [e_1766_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3601)) := by
  have c107907 := (publicBatchWrapper86_copies3372 a h).2.2.2.1
  have c107908 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.1
  have c107909 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.1
  have c107910 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.1
  have c107911 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.1
  have c107912 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_14 := arithEq_of_rows h (row := 1766) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_14
  simp only [← c107907, ← c107908, ← c107909] at e_1766_14
  have e_1766_15 := arithEq_of_rows h (row := 1766) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_15
  simp only [← c107910, ← c107911, k2, ← c107912] at e_1766_15
  have hr := e_1766_15
  simp only [e_1766_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3770)) := by
  have c107913 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.1
  have c107914 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107915 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107916 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107917 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107918 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_16 := arithEq_of_rows h (row := 1766) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_16
  simp only [← c107913, ← c107914, ← c107915] at e_1766_16
  have e_1766_17 := arithEq_of_rows h (row := 1766) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_17
  simp only [← c107916, ← c107917, k2, ← c107918] at e_1766_17
  have hr := e_1766_17
  simp only [e_1766_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1766 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3771)) := by
  have c107919 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107920 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107921 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107922 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107923 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107924 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1766_18 := arithEq_of_rows h (row := 1766) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_18
  simp only [← c107919, ← c107920, ← c107921] at e_1766_18
  have e_1766_19 := arithEq_of_rows h (row := 1766) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_19
  simp only [← c107922, ← c107923, k2, ← c107924] at e_1766_19
  have hr := e_1766_19
  simp only [e_1766_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3772)) := by
  have c107925 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107926 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107927 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107928 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107929 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107930 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_0 := arithEq_of_rows h (row := 1767) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_0
  simp only [← c107925, ← c107926, ← c107927] at e_1767_0
  have e_1767_1 := arithEq_of_rows h (row := 1767) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_1
  simp only [← c107928, ← c107929, k2, ← c107930] at e_1767_1
  have hr := e_1767_1
  simp only [e_1767_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3773)) := by
  have c107931 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107932 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107933 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107934 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107935 := (publicBatchWrapper86_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107936 := (publicBatchWrapper86_copies3373 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_2 := arithEq_of_rows h (row := 1767) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_2
  simp only [← c107931, ← c107932, ← c107933] at e_1767_2
  have e_1767_3 := arithEq_of_rows h (row := 1767) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_3
  simp only [← c107934, ← c107935, k2, ← c107936] at e_1767_3
  have hr := e_1767_3
  simp only [e_1767_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3774)) := by
  have c107937 := (publicBatchWrapper86_copies3373 a h).2.1
  have c107938 := (publicBatchWrapper86_copies3373 a h).2.2.1
  have c107939 := (publicBatchWrapper86_copies3373 a h).2.2.2.1
  have c107940 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.1
  have c107941 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.1
  have c107942 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_4 := arithEq_of_rows h (row := 1767) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_4
  simp only [← c107937, ← c107938, ← c107939] at e_1767_4
  have e_1767_5 := arithEq_of_rows h (row := 1767) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_5
  simp only [← c107940, ← c107941, k2, ← c107942] at e_1767_5
  have hr := e_1767_5
  simp only [e_1767_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3775)) := by
  have c107943 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.1
  have c107944 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.1
  have c107945 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.1
  have c107946 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107947 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107948 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_6 := arithEq_of_rows h (row := 1767) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_6
  simp only [← c107943, ← c107944, ← c107945] at e_1767_6
  have e_1767_7 := arithEq_of_rows h (row := 1767) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_7
  simp only [← c107946, ← c107947, k2, ← c107948] at e_1767_7
  have hr := e_1767_7
  simp only [e_1767_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3776)) := by
  have c107949 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107950 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107951 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107952 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107953 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107954 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_8 := arithEq_of_rows h (row := 1767) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_8
  simp only [← c107949, ← c107950, ← c107951] at e_1767_8
  have e_1767_9 := arithEq_of_rows h (row := 1767) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_9
  simp only [← c107952, ← c107953, k2, ← c107954] at e_1767_9
  have hr := e_1767_9
  simp only [e_1767_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3777)) := by
  have c107955 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107956 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107957 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107958 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107959 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107960 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_10 := arithEq_of_rows h (row := 1767) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_10
  simp only [← c107955, ← c107956, ← c107957] at e_1767_10
  have e_1767_11 := arithEq_of_rows h (row := 1767) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_11
  simp only [← c107958, ← c107959, k2, ← c107960] at e_1767_11
  have hr := e_1767_11
  simp only [e_1767_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3778)) := by
  have c107961 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107962 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107963 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107964 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107965 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107966 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_12 := arithEq_of_rows h (row := 1767) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_12
  simp only [← c107961, ← c107962, ← c107963] at e_1767_12
  have e_1767_13 := arithEq_of_rows h (row := 1767) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_13
  simp only [← c107964, ← c107965, k2, ← c107966] at e_1767_13
  have hr := e_1767_13
  simp only [e_1767_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3779)) := by
  have c107967 := (publicBatchWrapper86_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107968 := (publicBatchWrapper86_copies3374 a h).1
  have c107969 := (publicBatchWrapper86_copies3374 a h).2.1
  have c107970 := (publicBatchWrapper86_copies3374 a h).2.2.1
  have c107971 := (publicBatchWrapper86_copies3374 a h).2.2.2.1
  have c107972 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_14 := arithEq_of_rows h (row := 1767) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_14
  simp only [← c107967, ← c107968, ← c107969] at e_1767_14
  have e_1767_15 := arithEq_of_rows h (row := 1767) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_15
  simp only [← c107970, ← c107971, k2, ← c107972] at e_1767_15
  have hr := e_1767_15
  simp only [e_1767_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3780)) := by
  have c107973 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.1
  have c107974 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.1
  have c107975 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.1
  have c107976 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.1
  have c107977 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.1
  have c107978 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_16 := arithEq_of_rows h (row := 1767) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_16
  simp only [← c107973, ← c107974, ← c107975] at e_1767_16
  have e_1767_17 := arithEq_of_rows h (row := 1767) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_17
  simp only [← c107976, ← c107977, k2, ← c107978] at e_1767_17
  have hr := e_1767_17
  simp only [e_1767_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1767 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3781)) := by
  have c107979 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107980 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107981 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107982 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107983 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107984 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1767_18 := arithEq_of_rows h (row := 1767) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_18
  simp only [← c107979, ← c107980, ← c107981] at e_1767_18
  have e_1767_19 := arithEq_of_rows h (row := 1767) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_19
  simp only [← c107982, ← c107983, k2, ← c107984] at e_1767_19
  have hr := e_1767_19
  simp only [e_1767_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3782)) := by
  have c107985 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107986 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107987 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107988 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107989 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107990 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_0 := arithEq_of_rows h (row := 1768) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_0
  simp only [← c107985, ← c107986, ← c107987] at e_1768_0
  have e_1768_1 := arithEq_of_rows h (row := 1768) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_1
  simp only [← c107988, ← c107989, k2, ← c107990] at e_1768_1
  have hr := e_1768_1
  simp only [e_1768_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3783)) := by
  have c107991 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107992 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107993 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107994 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107995 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107996 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_2 := arithEq_of_rows h (row := 1768) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_2
  simp only [← c107991, ← c107992, ← c107993] at e_1768_2
  have e_1768_3 := arithEq_of_rows h (row := 1768) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_3
  simp only [← c107994, ← c107995, k2, ← c107996] at e_1768_3
  have hr := e_1768_3
  simp only [e_1768_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3784)) := by
  have c107997 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107998 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107999 := (publicBatchWrapper86_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108000 := (publicBatchWrapper86_copies3375 a h).1
  have c108001 := (publicBatchWrapper86_copies3375 a h).2.1
  have c108002 := (publicBatchWrapper86_copies3375 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_4 := arithEq_of_rows h (row := 1768) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_4
  simp only [← c107997, ← c107998, ← c107999] at e_1768_4
  have e_1768_5 := arithEq_of_rows h (row := 1768) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_5
  simp only [← c108000, ← c108001, k2, ← c108002] at e_1768_5
  have hr := e_1768_5
  simp only [e_1768_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3785)) := by
  have c108003 := (publicBatchWrapper86_copies3375 a h).2.2.2.1
  have c108004 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.1
  have c108005 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.1
  have c108006 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.1
  have c108007 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.1
  have c108008 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_6 := arithEq_of_rows h (row := 1768) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_6
  simp only [← c108003, ← c108004, ← c108005] at e_1768_6
  have e_1768_7 := arithEq_of_rows h (row := 1768) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_7
  simp only [← c108006, ← c108007, k2, ← c108008] at e_1768_7
  have hr := e_1768_7
  simp only [e_1768_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3786)) := by
  have c108009 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.1
  have c108010 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108011 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108012 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108013 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108014 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_8 := arithEq_of_rows h (row := 1768) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_8
  simp only [← c108009, ← c108010, ← c108011] at e_1768_8
  have e_1768_9 := arithEq_of_rows h (row := 1768) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_9
  simp only [← c108012, ← c108013, k2, ← c108014] at e_1768_9
  have hr := e_1768_9
  simp only [e_1768_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3787)) := by
  have c108015 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108016 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108017 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108018 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108019 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108020 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_10 := arithEq_of_rows h (row := 1768) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_10
  simp only [← c108015, ← c108016, ← c108017] at e_1768_10
  have e_1768_11 := arithEq_of_rows h (row := 1768) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_11
  simp only [← c108018, ← c108019, k2, ← c108020] at e_1768_11
  have hr := e_1768_11
  simp only [e_1768_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3788)) := by
  have c108021 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108022 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108023 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108024 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108025 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108026 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_12 := arithEq_of_rows h (row := 1768) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_12
  simp only [← c108021, ← c108022, ← c108023] at e_1768_12
  have e_1768_13 := arithEq_of_rows h (row := 1768) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_13
  simp only [← c108024, ← c108025, k2, ← c108026] at e_1768_13
  have hr := e_1768_13
  simp only [e_1768_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3789)) := by
  have c108027 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108028 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108029 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108030 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108031 := (publicBatchWrapper86_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108032 := (publicBatchWrapper86_copies3376 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_14 := arithEq_of_rows h (row := 1768) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_14
  simp only [← c108027, ← c108028, ← c108029] at e_1768_14
  have e_1768_15 := arithEq_of_rows h (row := 1768) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_15
  simp only [← c108030, ← c108031, k2, ← c108032] at e_1768_15
  have hr := e_1768_15
  simp only [e_1768_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3790)) := by
  have c108033 := (publicBatchWrapper86_copies3376 a h).2.1
  have c108034 := (publicBatchWrapper86_copies3376 a h).2.2.1
  have c108035 := (publicBatchWrapper86_copies3376 a h).2.2.2.1
  have c108036 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.1
  have c108037 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.1
  have c108038 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_16 := arithEq_of_rows h (row := 1768) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_16
  simp only [← c108033, ← c108034, ← c108035] at e_1768_16
  have e_1768_17 := arithEq_of_rows h (row := 1768) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_17
  simp only [← c108036, ← c108037, k2, ← c108038] at e_1768_17
  have hr := e_1768_17
  simp only [e_1768_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1768 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3791)) := by
  have c108039 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.1
  have c108040 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.1
  have c108041 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.1
  have c108042 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108043 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108044 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1768_18 := arithEq_of_rows h (row := 1768) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_18
  simp only [← c108039, ← c108040, ← c108041] at e_1768_18
  have e_1768_19 := arithEq_of_rows h (row := 1768) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_19
  simp only [← c108042, ← c108043, k2, ← c108044] at e_1768_19
  have hr := e_1768_19
  simp only [e_1768_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3792)) := by
  have c108045 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108046 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108047 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108048 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108049 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108050 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_0 := arithEq_of_rows h (row := 1769) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_0
  simp only [← c108045, ← c108046, ← c108047] at e_1769_0
  have e_1769_1 := arithEq_of_rows h (row := 1769) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_1
  simp only [← c108048, ← c108049, k2, ← c108050] at e_1769_1
  have hr := e_1769_1
  simp only [e_1769_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3793)) := by
  have c108051 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108052 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108053 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108054 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108055 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108056 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_2 := arithEq_of_rows h (row := 1769) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_2
  simp only [← c108051, ← c108052, ← c108053] at e_1769_2
  have e_1769_3 := arithEq_of_rows h (row := 1769) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_3
  simp only [← c108054, ← c108055, k2, ← c108056] at e_1769_3
  have hr := e_1769_3
  simp only [e_1769_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3794)) := by
  have c108057 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108058 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108059 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108060 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108061 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108062 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_4 := arithEq_of_rows h (row := 1769) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_4
  simp only [← c108057, ← c108058, ← c108059] at e_1769_4
  have e_1769_5 := arithEq_of_rows h (row := 1769) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_5
  simp only [← c108060, ← c108061, k2, ← c108062] at e_1769_5
  have hr := e_1769_5
  simp only [e_1769_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3795)) := by
  have c108063 := (publicBatchWrapper86_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108064 := (publicBatchWrapper86_copies3377 a h).1
  have c108065 := (publicBatchWrapper86_copies3377 a h).2.1
  have c108066 := (publicBatchWrapper86_copies3377 a h).2.2.1
  have c108067 := (publicBatchWrapper86_copies3377 a h).2.2.2.1
  have c108068 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_6 := arithEq_of_rows h (row := 1769) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_6
  simp only [← c108063, ← c108064, ← c108065] at e_1769_6
  have e_1769_7 := arithEq_of_rows h (row := 1769) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_7
  simp only [← c108066, ← c108067, k2, ← c108068] at e_1769_7
  have hr := e_1769_7
  simp only [e_1769_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3796)) := by
  have c108069 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.1
  have c108070 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.1
  have c108071 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.1
  have c108072 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.1
  have c108073 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.1
  have c108074 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_8 := arithEq_of_rows h (row := 1769) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_8
  simp only [← c108069, ← c108070, ← c108071] at e_1769_8
  have e_1769_9 := arithEq_of_rows h (row := 1769) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_9
  simp only [← c108072, ← c108073, k2, ← c108074] at e_1769_9
  have hr := e_1769_9
  simp only [e_1769_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3797)) := by
  have c108075 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108076 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108077 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108078 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108079 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108080 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_10 := arithEq_of_rows h (row := 1769) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_10
  simp only [← c108075, ← c108076, ← c108077] at e_1769_10
  have e_1769_11 := arithEq_of_rows h (row := 1769) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_11
  simp only [← c108078, ← c108079, k2, ← c108080] at e_1769_11
  have hr := e_1769_11
  simp only [e_1769_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3798)) := by
  have c108081 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108082 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108083 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108084 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108085 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108086 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_12 := arithEq_of_rows h (row := 1769) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_12
  simp only [← c108081, ← c108082, ← c108083] at e_1769_12
  have e_1769_13 := arithEq_of_rows h (row := 1769) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_13
  simp only [← c108084, ← c108085, k2, ← c108086] at e_1769_13
  have hr := e_1769_13
  simp only [e_1769_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3799)) := by
  have c108087 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108088 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108089 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108090 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108091 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108092 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_14 := arithEq_of_rows h (row := 1769) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_14
  simp only [← c108087, ← c108088, ← c108089] at e_1769_14
  have e_1769_15 := arithEq_of_rows h (row := 1769) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_15
  simp only [← c108090, ← c108091, k2, ← c108092] at e_1769_15
  have hr := e_1769_15
  simp only [e_1769_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3800)) := by
  have c108093 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108094 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108095 := (publicBatchWrapper86_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108096 := (publicBatchWrapper86_copies3378 a h).1
  have c108097 := (publicBatchWrapper86_copies3378 a h).2.1
  have c108098 := (publicBatchWrapper86_copies3378 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_16 := arithEq_of_rows h (row := 1769) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_16
  simp only [← c108093, ← c108094, ← c108095] at e_1769_16
  have e_1769_17 := arithEq_of_rows h (row := 1769) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_17
  simp only [← c108096, ← c108097, k2, ← c108098] at e_1769_17
  have hr := e_1769_17
  simp only [e_1769_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1769 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3801)) := by
  have c108099 := (publicBatchWrapper86_copies3378 a h).2.2.2.1
  have c108100 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.1
  have c108101 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.1
  have c108102 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.1
  have c108103 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.1
  have c108104 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1769_18 := arithEq_of_rows h (row := 1769) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_18
  simp only [← c108099, ← c108100, ← c108101] at e_1769_18
  have e_1769_19 := arithEq_of_rows h (row := 1769) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_19
  simp only [← c108102, ← c108103, k2, ← c108104] at e_1769_19
  have hr := e_1769_19
  simp only [e_1769_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3970)) := by
  have c108105 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.1
  have c108106 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108107 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108108 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108109 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108110 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_0 := arithEq_of_rows h (row := 1770) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_0
  simp only [← c108105, ← c108106, ← c108107] at e_1770_0
  have e_1770_1 := arithEq_of_rows h (row := 1770) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_1
  simp only [← c108108, ← c108109, k2, ← c108110] at e_1770_1
  have hr := e_1770_1
  simp only [e_1770_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3971)) := by
  have c108111 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108112 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108113 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108114 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108115 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108116 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_2 := arithEq_of_rows h (row := 1770) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_2
  simp only [← c108111, ← c108112, ← c108113] at e_1770_2
  have e_1770_3 := arithEq_of_rows h (row := 1770) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_3
  simp only [← c108114, ← c108115, k2, ← c108116] at e_1770_3
  have hr := e_1770_3
  simp only [e_1770_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3972)) := by
  have c108117 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108118 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108119 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108120 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108121 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108122 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_4 := arithEq_of_rows h (row := 1770) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_4
  simp only [← c108117, ← c108118, ← c108119] at e_1770_4
  have e_1770_5 := arithEq_of_rows h (row := 1770) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_5
  simp only [← c108120, ← c108121, k2, ← c108122] at e_1770_5
  have hr := e_1770_5
  simp only [e_1770_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3973)) := by
  have c108123 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108124 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108125 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108126 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108127 := (publicBatchWrapper86_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108128 := (publicBatchWrapper86_copies3379 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_6 := arithEq_of_rows h (row := 1770) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_6
  simp only [← c108123, ← c108124, ← c108125] at e_1770_6
  have e_1770_7 := arithEq_of_rows h (row := 1770) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_7
  simp only [← c108126, ← c108127, k2, ← c108128] at e_1770_7
  have hr := e_1770_7
  simp only [e_1770_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3974)) := by
  have c108129 := (publicBatchWrapper86_copies3379 a h).2.1
  have c108130 := (publicBatchWrapper86_copies3379 a h).2.2.1
  have c108131 := (publicBatchWrapper86_copies3379 a h).2.2.2.1
  have c108132 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.1
  have c108133 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.1
  have c108134 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_8 := arithEq_of_rows h (row := 1770) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_8
  simp only [← c108129, ← c108130, ← c108131] at e_1770_8
  have e_1770_9 := arithEq_of_rows h (row := 1770) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_9
  simp only [← c108132, ← c108133, k2, ← c108134] at e_1770_9
  have hr := e_1770_9
  simp only [e_1770_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3975)) := by
  have c108135 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.1
  have c108136 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.1
  have c108137 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.1
  have c108138 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108139 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108140 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_10 := arithEq_of_rows h (row := 1770) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_10
  simp only [← c108135, ← c108136, ← c108137] at e_1770_10
  have e_1770_11 := arithEq_of_rows h (row := 1770) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_11
  simp only [← c108138, ← c108139, k2, ← c108140] at e_1770_11
  have hr := e_1770_11
  simp only [e_1770_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3976)) := by
  have c108141 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108142 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108143 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108144 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108145 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108146 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_12 := arithEq_of_rows h (row := 1770) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_12
  simp only [← c108141, ← c108142, ← c108143] at e_1770_12
  have e_1770_13 := arithEq_of_rows h (row := 1770) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_13
  simp only [← c108144, ← c108145, k2, ← c108146] at e_1770_13
  have hr := e_1770_13
  simp only [e_1770_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3977)) := by
  have c108147 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108148 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108149 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108150 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108151 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108152 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_14 := arithEq_of_rows h (row := 1770) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_14
  simp only [← c108147, ← c108148, ← c108149] at e_1770_14
  have e_1770_15 := arithEq_of_rows h (row := 1770) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_15
  simp only [← c108150, ← c108151, k2, ← c108152] at e_1770_15
  have hr := e_1770_15
  simp only [e_1770_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3978)) := by
  have c108153 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108154 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108155 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108156 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108157 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108158 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_16 := arithEq_of_rows h (row := 1770) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_16
  simp only [← c108153, ← c108154, ← c108155] at e_1770_16
  have e_1770_17 := arithEq_of_rows h (row := 1770) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_17
  simp only [← c108156, ← c108157, k2, ← c108158] at e_1770_17
  have hr := e_1770_17
  simp only [e_1770_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1770 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3979)) := by
  have c108159 := (publicBatchWrapper86_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108160 := (publicBatchWrapper86_copies3380 a h).1
  have c108161 := (publicBatchWrapper86_copies3380 a h).2.1
  have c108162 := (publicBatchWrapper86_copies3380 a h).2.2.1
  have c108163 := (publicBatchWrapper86_copies3380 a h).2.2.2.1
  have c108164 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1770_18 := arithEq_of_rows h (row := 1770) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_18
  simp only [← c108159, ← c108160, ← c108161] at e_1770_18
  have e_1770_19 := arithEq_of_rows h (row := 1770) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_19
  simp only [← c108162, ← c108163, k2, ← c108164] at e_1770_19
  have hr := e_1770_19
  simp only [e_1770_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3980)) := by
  have c108165 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.1
  have c108166 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.1
  have c108167 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.1
  have c108168 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.1
  have c108169 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.1
  have c108170 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_0 := arithEq_of_rows h (row := 1771) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_0
  simp only [← c108165, ← c108166, ← c108167] at e_1771_0
  have e_1771_1 := arithEq_of_rows h (row := 1771) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_1
  simp only [← c108168, ← c108169, k2, ← c108170] at e_1771_1
  have hr := e_1771_1
  simp only [e_1771_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3981)) := by
  have c108171 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108172 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108173 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108174 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108175 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108176 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_2 := arithEq_of_rows h (row := 1771) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_2
  simp only [← c108171, ← c108172, ← c108173] at e_1771_2
  have e_1771_3 := arithEq_of_rows h (row := 1771) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_3
  simp only [← c108174, ← c108175, k2, ← c108176] at e_1771_3
  have hr := e_1771_3
  simp only [e_1771_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3982)) := by
  have c108177 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108178 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108179 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108180 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108181 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108182 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_4 := arithEq_of_rows h (row := 1771) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_4
  simp only [← c108177, ← c108178, ← c108179] at e_1771_4
  have e_1771_5 := arithEq_of_rows h (row := 1771) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_5
  simp only [← c108180, ← c108181, k2, ← c108182] at e_1771_5
  have hr := e_1771_5
  simp only [e_1771_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3983)) := by
  have c108183 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108184 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108185 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108186 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108187 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108188 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_6 := arithEq_of_rows h (row := 1771) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_6
  simp only [← c108183, ← c108184, ← c108185] at e_1771_6
  have e_1771_7 := arithEq_of_rows h (row := 1771) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_7
  simp only [← c108186, ← c108187, k2, ← c108188] at e_1771_7
  have hr := e_1771_7
  simp only [e_1771_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3984)) := by
  have c108189 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108190 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108191 := (publicBatchWrapper86_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108192 := (publicBatchWrapper86_copies3381 a h).1
  have c108193 := (publicBatchWrapper86_copies3381 a h).2.1
  have c108194 := (publicBatchWrapper86_copies3381 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_8 := arithEq_of_rows h (row := 1771) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_8
  simp only [← c108189, ← c108190, ← c108191] at e_1771_8
  have e_1771_9 := arithEq_of_rows h (row := 1771) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_9
  simp only [← c108192, ← c108193, k2, ← c108194] at e_1771_9
  have hr := e_1771_9
  simp only [e_1771_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3985)) := by
  have c108195 := (publicBatchWrapper86_copies3381 a h).2.2.2.1
  have c108196 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.1
  have c108197 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.1
  have c108198 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.1
  have c108199 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.1
  have c108200 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_10 := arithEq_of_rows h (row := 1771) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_10
  simp only [← c108195, ← c108196, ← c108197] at e_1771_10
  have e_1771_11 := arithEq_of_rows h (row := 1771) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_11
  simp only [← c108198, ← c108199, k2, ← c108200] at e_1771_11
  have hr := e_1771_11
  simp only [e_1771_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3986)) := by
  have c108201 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.1
  have c108202 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108203 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108204 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108205 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108206 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_12 := arithEq_of_rows h (row := 1771) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_12
  simp only [← c108201, ← c108202, ← c108203] at e_1771_12
  have e_1771_13 := arithEq_of_rows h (row := 1771) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_13
  simp only [← c108204, ← c108205, k2, ← c108206] at e_1771_13
  have hr := e_1771_13
  simp only [e_1771_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3987)) := by
  have c108207 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108208 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108209 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108210 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108211 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108212 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_14 := arithEq_of_rows h (row := 1771) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_14
  simp only [← c108207, ← c108208, ← c108209] at e_1771_14
  have e_1771_15 := arithEq_of_rows h (row := 1771) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_15
  simp only [← c108210, ← c108211, k2, ← c108212] at e_1771_15
  have hr := e_1771_15
  simp only [e_1771_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3988)) := by
  have c108213 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108214 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108215 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108216 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108217 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108218 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_16 := arithEq_of_rows h (row := 1771) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_16
  simp only [← c108213, ← c108214, ← c108215] at e_1771_16
  have e_1771_17 := arithEq_of_rows h (row := 1771) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_17
  simp only [← c108216, ← c108217, k2, ← c108218] at e_1771_17
  have hr := e_1771_17
  simp only [e_1771_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1771 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3989)) := by
  have c108219 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108220 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108221 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108222 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108223 := (publicBatchWrapper86_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108224 := (publicBatchWrapper86_copies3382 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1771_18 := arithEq_of_rows h (row := 1771) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_18
  simp only [← c108219, ← c108220, ← c108221] at e_1771_18
  have e_1771_19 := arithEq_of_rows h (row := 1771) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_19
  simp only [← c108222, ← c108223, k2, ← c108224] at e_1771_19
  have hr := e_1771_19
  simp only [e_1771_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3990)) := by
  have c108225 := (publicBatchWrapper86_copies3382 a h).2.1
  have c108226 := (publicBatchWrapper86_copies3382 a h).2.2.1
  have c108227 := (publicBatchWrapper86_copies3382 a h).2.2.2.1
  have c108228 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.1
  have c108229 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.1
  have c108230 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_0 := arithEq_of_rows h (row := 1772) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_0
  simp only [← c108225, ← c108226, ← c108227] at e_1772_0
  have e_1772_1 := arithEq_of_rows h (row := 1772) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_1
  simp only [← c108228, ← c108229, k2, ← c108230] at e_1772_1
  have hr := e_1772_1
  simp only [e_1772_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3991)) := by
  have c108231 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.1
  have c108232 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.1
  have c108233 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.1
  have c108234 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108235 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108236 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_2 := arithEq_of_rows h (row := 1772) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_2
  simp only [← c108231, ← c108232, ← c108233] at e_1772_2
  have e_1772_3 := arithEq_of_rows h (row := 1772) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_3
  simp only [← c108234, ← c108235, k2, ← c108236] at e_1772_3
  have hr := e_1772_3
  simp only [e_1772_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3992)) := by
  have c108237 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108238 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108239 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108240 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108241 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108242 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_4 := arithEq_of_rows h (row := 1772) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_4
  simp only [← c108237, ← c108238, ← c108239] at e_1772_4
  have e_1772_5 := arithEq_of_rows h (row := 1772) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_5
  simp only [← c108240, ← c108241, k2, ← c108242] at e_1772_5
  have hr := e_1772_5
  simp only [e_1772_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3993)) := by
  have c108243 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108244 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108245 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108246 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108247 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108248 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_6 := arithEq_of_rows h (row := 1772) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_6
  simp only [← c108243, ← c108244, ← c108245] at e_1772_6
  have e_1772_7 := arithEq_of_rows h (row := 1772) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_7
  simp only [← c108246, ← c108247, k2, ← c108248] at e_1772_7
  have hr := e_1772_7
  simp only [e_1772_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3994)) := by
  have c108249 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108250 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108251 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108252 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108253 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108254 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_8 := arithEq_of_rows h (row := 1772) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_8
  simp only [← c108249, ← c108250, ← c108251] at e_1772_8
  have e_1772_9 := arithEq_of_rows h (row := 1772) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_9
  simp only [← c108252, ← c108253, k2, ← c108254] at e_1772_9
  have hr := e_1772_9
  simp only [e_1772_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3995)) := by
  have c108255 := (publicBatchWrapper86_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108256 := (publicBatchWrapper86_copies3383 a h).1
  have c108257 := (publicBatchWrapper86_copies3383 a h).2.1
  have c108258 := (publicBatchWrapper86_copies3383 a h).2.2.1
  have c108259 := (publicBatchWrapper86_copies3383 a h).2.2.2.1
  have c108260 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_10 := arithEq_of_rows h (row := 1772) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_10
  simp only [← c108255, ← c108256, ← c108257] at e_1772_10
  have e_1772_11 := arithEq_of_rows h (row := 1772) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_11
  simp only [← c108258, ← c108259, k2, ← c108260] at e_1772_11
  have hr := e_1772_11
  simp only [e_1772_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3996)) := by
  have c108261 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.1
  have c108262 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.1
  have c108263 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.1
  have c108264 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.1
  have c108265 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.1
  have c108266 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_12 := arithEq_of_rows h (row := 1772) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_12
  simp only [← c108261, ← c108262, ← c108263] at e_1772_12
  have e_1772_13 := arithEq_of_rows h (row := 1772) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_13
  simp only [← c108264, ← c108265, k2, ← c108266] at e_1772_13
  have hr := e_1772_13
  simp only [e_1772_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3997)) := by
  have c108267 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108268 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108269 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108270 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108271 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108272 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_14 := arithEq_of_rows h (row := 1772) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_14
  simp only [← c108267, ← c108268, ← c108269] at e_1772_14
  have e_1772_15 := arithEq_of_rows h (row := 1772) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_15
  simp only [← c108270, ← c108271, k2, ← c108272] at e_1772_15
  have hr := e_1772_15
  simp only [e_1772_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3998)) := by
  have c108273 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108274 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108275 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108276 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108277 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108278 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_16 := arithEq_of_rows h (row := 1772) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_16
  simp only [← c108273, ← c108274, ← c108275] at e_1772_16
  have e_1772_17 := arithEq_of_rows h (row := 1772) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_17
  simp only [← c108276, ← c108277, k2, ← c108278] at e_1772_17
  have hr := e_1772_17
  simp only [e_1772_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1772 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3999)) := by
  have c108279 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108280 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108281 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108282 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108283 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108284 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1772_18 := arithEq_of_rows h (row := 1772) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_18
  simp only [← c108279, ← c108280, ← c108281] at e_1772_18
  have e_1772_19 := arithEq_of_rows h (row := 1772) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_19
  simp only [← c108282, ← c108283, k2, ← c108284] at e_1772_19
  have hr := e_1772_19
  simp only [e_1772_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 4000)) := by
  have c108285 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108286 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108287 := (publicBatchWrapper86_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108288 := (publicBatchWrapper86_copies3384 a h).1
  have c108289 := (publicBatchWrapper86_copies3384 a h).2.1
  have c108290 := (publicBatchWrapper86_copies3384 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_0 := arithEq_of_rows h (row := 1773) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_0
  simp only [← c108285, ← c108286, ← c108287] at e_1773_0
  have e_1773_1 := arithEq_of_rows h (row := 1773) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_1
  simp only [← c108288, ← c108289, k2, ← c108290] at e_1773_1
  have hr := e_1773_1
  simp only [e_1773_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 4001)) := by
  have c108291 := (publicBatchWrapper86_copies3384 a h).2.2.2.1
  have c108292 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.1
  have c108293 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.1
  have c108294 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.1
  have c108295 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.1
  have c108296 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_2 := arithEq_of_rows h (row := 1773) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_2
  simp only [← c108291, ← c108292, ← c108293] at e_1773_2
  have e_1773_3 := arithEq_of_rows h (row := 1773) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_3
  simp only [← c108294, ← c108295, k2, ← c108296] at e_1773_3
  have hr := e_1773_3
  simp only [e_1773_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4170)) := by
  have c108297 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.1
  have c108298 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108299 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108300 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108301 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108302 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_4 := arithEq_of_rows h (row := 1773) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_4
  simp only [← c108297, ← c108298, ← c108299] at e_1773_4
  have e_1773_5 := arithEq_of_rows h (row := 1773) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_5
  simp only [← c108300, ← c108301, k2, ← c108302] at e_1773_5
  have hr := e_1773_5
  simp only [e_1773_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4171)) := by
  have c108303 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108304 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108305 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108306 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108307 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108308 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_6 := arithEq_of_rows h (row := 1773) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_6
  simp only [← c108303, ← c108304, ← c108305] at e_1773_6
  have e_1773_7 := arithEq_of_rows h (row := 1773) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_7
  simp only [← c108306, ← c108307, k2, ← c108308] at e_1773_7
  have hr := e_1773_7
  simp only [e_1773_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4172)) := by
  have c108309 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108310 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108311 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108312 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108313 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108314 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_8 := arithEq_of_rows h (row := 1773) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_8
  simp only [← c108309, ← c108310, ← c108311] at e_1773_8
  have e_1773_9 := arithEq_of_rows h (row := 1773) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_9
  simp only [← c108312, ← c108313, k2, ← c108314] at e_1773_9
  have hr := e_1773_9
  simp only [e_1773_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4173)) := by
  have c108315 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108316 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108317 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108318 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108319 := (publicBatchWrapper86_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108320 := (publicBatchWrapper86_copies3385 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_10 := arithEq_of_rows h (row := 1773) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_10
  simp only [← c108315, ← c108316, ← c108317] at e_1773_10
  have e_1773_11 := arithEq_of_rows h (row := 1773) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_11
  simp only [← c108318, ← c108319, k2, ← c108320] at e_1773_11
  have hr := e_1773_11
  simp only [e_1773_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4174)) := by
  have c108321 := (publicBatchWrapper86_copies3385 a h).2.1
  have c108322 := (publicBatchWrapper86_copies3385 a h).2.2.1
  have c108323 := (publicBatchWrapper86_copies3385 a h).2.2.2.1
  have c108324 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.1
  have c108325 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.1
  have c108326 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_12 := arithEq_of_rows h (row := 1773) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_12
  simp only [← c108321, ← c108322, ← c108323] at e_1773_12
  have e_1773_13 := arithEq_of_rows h (row := 1773) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_13
  simp only [← c108324, ← c108325, k2, ← c108326] at e_1773_13
  have hr := e_1773_13
  simp only [e_1773_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4175)) := by
  have c108327 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.1
  have c108328 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.1
  have c108329 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.1
  have c108330 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108331 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108332 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_14 := arithEq_of_rows h (row := 1773) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_14
  simp only [← c108327, ← c108328, ← c108329] at e_1773_14
  have e_1773_15 := arithEq_of_rows h (row := 1773) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_15
  simp only [← c108330, ← c108331, k2, ← c108332] at e_1773_15
  have hr := e_1773_15
  simp only [e_1773_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4176)) := by
  have c108333 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108334 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108335 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108336 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108337 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108338 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_16 := arithEq_of_rows h (row := 1773) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_16
  simp only [← c108333, ← c108334, ← c108335] at e_1773_16
  have e_1773_17 := arithEq_of_rows h (row := 1773) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_17
  simp only [← c108336, ← c108337, k2, ← c108338] at e_1773_17
  have hr := e_1773_17
  simp only [e_1773_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1773 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4177)) := by
  have c108339 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108340 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108341 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108342 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108343 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108344 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1773_18 := arithEq_of_rows h (row := 1773) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_18
  simp only [← c108339, ← c108340, ← c108341] at e_1773_18
  have e_1773_19 := arithEq_of_rows h (row := 1773) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_19
  simp only [← c108342, ← c108343, k2, ← c108344] at e_1773_19
  have hr := e_1773_19
  simp only [e_1773_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4178)) := by
  have c108345 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108346 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108347 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108348 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108349 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108350 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_0 := arithEq_of_rows h (row := 1774) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_0
  simp only [← c108345, ← c108346, ← c108347] at e_1774_0
  have e_1774_1 := arithEq_of_rows h (row := 1774) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_1
  simp only [← c108348, ← c108349, k2, ← c108350] at e_1774_1
  have hr := e_1774_1
  simp only [e_1774_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4179)) := by
  have c108351 := (publicBatchWrapper86_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108352 := (publicBatchWrapper86_copies3386 a h).1
  have c108353 := (publicBatchWrapper86_copies3386 a h).2.1
  have c108354 := (publicBatchWrapper86_copies3386 a h).2.2.1
  have c108355 := (publicBatchWrapper86_copies3386 a h).2.2.2.1
  have c108356 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_2 := arithEq_of_rows h (row := 1774) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_2
  simp only [← c108351, ← c108352, ← c108353] at e_1774_2
  have e_1774_3 := arithEq_of_rows h (row := 1774) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_3
  simp only [← c108354, ← c108355, k2, ← c108356] at e_1774_3
  have hr := e_1774_3
  simp only [e_1774_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4180)) := by
  have c108357 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.1
  have c108358 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.1
  have c108359 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.1
  have c108360 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.1
  have c108361 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.1
  have c108362 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_4 := arithEq_of_rows h (row := 1774) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_4
  simp only [← c108357, ← c108358, ← c108359] at e_1774_4
  have e_1774_5 := arithEq_of_rows h (row := 1774) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_5
  simp only [← c108360, ← c108361, k2, ← c108362] at e_1774_5
  have hr := e_1774_5
  simp only [e_1774_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4181)) := by
  have c108363 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108364 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108365 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108366 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108367 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108368 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_6 := arithEq_of_rows h (row := 1774) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_6
  simp only [← c108363, ← c108364, ← c108365] at e_1774_6
  have e_1774_7 := arithEq_of_rows h (row := 1774) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_7
  simp only [← c108366, ← c108367, k2, ← c108368] at e_1774_7
  have hr := e_1774_7
  simp only [e_1774_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4182)) := by
  have c108369 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108370 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108371 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108372 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108373 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108374 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_8 := arithEq_of_rows h (row := 1774) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_8
  simp only [← c108369, ← c108370, ← c108371] at e_1774_8
  have e_1774_9 := arithEq_of_rows h (row := 1774) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_9
  simp only [← c108372, ← c108373, k2, ← c108374] at e_1774_9
  have hr := e_1774_9
  simp only [e_1774_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4183)) := by
  have c108375 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108376 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108377 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108378 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108379 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108380 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_10 := arithEq_of_rows h (row := 1774) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_10
  simp only [← c108375, ← c108376, ← c108377] at e_1774_10
  have e_1774_11 := arithEq_of_rows h (row := 1774) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_11
  simp only [← c108378, ← c108379, k2, ← c108380] at e_1774_11
  have hr := e_1774_11
  simp only [e_1774_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4184)) := by
  have c108381 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108382 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108383 := (publicBatchWrapper86_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108384 := (publicBatchWrapper86_copies3387 a h).1
  have c108385 := (publicBatchWrapper86_copies3387 a h).2.1
  have c108386 := (publicBatchWrapper86_copies3387 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_12 := arithEq_of_rows h (row := 1774) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_12
  simp only [← c108381, ← c108382, ← c108383] at e_1774_12
  have e_1774_13 := arithEq_of_rows h (row := 1774) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_13
  simp only [← c108384, ← c108385, k2, ← c108386] at e_1774_13
  have hr := e_1774_13
  simp only [e_1774_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4185)) := by
  have c108387 := (publicBatchWrapper86_copies3387 a h).2.2.2.1
  have c108388 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.1
  have c108389 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.1
  have c108390 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.1
  have c108391 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.1
  have c108392 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_14 := arithEq_of_rows h (row := 1774) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_14
  simp only [← c108387, ← c108388, ← c108389] at e_1774_14
  have e_1774_15 := arithEq_of_rows h (row := 1774) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_15
  simp only [← c108390, ← c108391, k2, ← c108392] at e_1774_15
  have hr := e_1774_15
  simp only [e_1774_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4186)) := by
  have c108393 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.1
  have c108394 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108395 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108396 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108397 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108398 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_16 := arithEq_of_rows h (row := 1774) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_16
  simp only [← c108393, ← c108394, ← c108395] at e_1774_16
  have e_1774_17 := arithEq_of_rows h (row := 1774) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_17
  simp only [← c108396, ← c108397, k2, ← c108398] at e_1774_17
  have hr := e_1774_17
  simp only [e_1774_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1774 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4187)) := by
  have c108399 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108400 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108401 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108402 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108403 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108404 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1774_18 := arithEq_of_rows h (row := 1774) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_18
  simp only [← c108399, ← c108400, ← c108401] at e_1774_18
  have e_1774_19 := arithEq_of_rows h (row := 1774) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_19
  simp only [← c108402, ← c108403, k2, ← c108404] at e_1774_19
  have hr := e_1774_19
  simp only [e_1774_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4188)) := by
  have c108405 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108406 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108407 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108408 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108409 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108410 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_0 := arithEq_of_rows h (row := 1775) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_0
  simp only [← c108405, ← c108406, ← c108407] at e_1775_0
  have e_1775_1 := arithEq_of_rows h (row := 1775) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_1
  simp only [← c108408, ← c108409, k2, ← c108410] at e_1775_1
  have hr := e_1775_1
  simp only [e_1775_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4189)) := by
  have c108411 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108412 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108413 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108414 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108415 := (publicBatchWrapper86_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108416 := (publicBatchWrapper86_copies3388 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_2 := arithEq_of_rows h (row := 1775) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_2
  simp only [← c108411, ← c108412, ← c108413] at e_1775_2
  have e_1775_3 := arithEq_of_rows h (row := 1775) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_3
  simp only [← c108414, ← c108415, k2, ← c108416] at e_1775_3
  have hr := e_1775_3
  simp only [e_1775_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4190)) := by
  have c108417 := (publicBatchWrapper86_copies3388 a h).2.1
  have c108418 := (publicBatchWrapper86_copies3388 a h).2.2.1
  have c108419 := (publicBatchWrapper86_copies3388 a h).2.2.2.1
  have c108420 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.1
  have c108421 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.1
  have c108422 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_4 := arithEq_of_rows h (row := 1775) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_4
  simp only [← c108417, ← c108418, ← c108419] at e_1775_4
  have e_1775_5 := arithEq_of_rows h (row := 1775) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_5
  simp only [← c108420, ← c108421, k2, ← c108422] at e_1775_5
  have hr := e_1775_5
  simp only [e_1775_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4191)) := by
  have c108423 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.1
  have c108424 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.1
  have c108425 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.1
  have c108426 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108427 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108428 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_6 := arithEq_of_rows h (row := 1775) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_6
  simp only [← c108423, ← c108424, ← c108425] at e_1775_6
  have e_1775_7 := arithEq_of_rows h (row := 1775) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_7
  simp only [← c108426, ← c108427, k2, ← c108428] at e_1775_7
  have hr := e_1775_7
  simp only [e_1775_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4192)) := by
  have c108429 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108430 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108431 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108432 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108433 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108434 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_8 := arithEq_of_rows h (row := 1775) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_8
  simp only [← c108429, ← c108430, ← c108431] at e_1775_8
  have e_1775_9 := arithEq_of_rows h (row := 1775) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_9
  simp only [← c108432, ← c108433, k2, ← c108434] at e_1775_9
  have hr := e_1775_9
  simp only [e_1775_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4193)) := by
  have c108435 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108436 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108437 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108438 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108439 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108440 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_10 := arithEq_of_rows h (row := 1775) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_10
  simp only [← c108435, ← c108436, ← c108437] at e_1775_10
  have e_1775_11 := arithEq_of_rows h (row := 1775) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_11
  simp only [← c108438, ← c108439, k2, ← c108440] at e_1775_11
  have hr := e_1775_11
  simp only [e_1775_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4194)) := by
  have c108441 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108442 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108443 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108444 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108445 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108446 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_12 := arithEq_of_rows h (row := 1775) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_12
  simp only [← c108441, ← c108442, ← c108443] at e_1775_12
  have e_1775_13 := arithEq_of_rows h (row := 1775) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_13
  simp only [← c108444, ← c108445, k2, ← c108446] at e_1775_13
  have hr := e_1775_13
  simp only [e_1775_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4195)) := by
  have c108447 := (publicBatchWrapper86_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108448 := (publicBatchWrapper86_copies3389 a h).1
  have c108449 := (publicBatchWrapper86_copies3389 a h).2.1
  have c108450 := (publicBatchWrapper86_copies3389 a h).2.2.1
  have c108451 := (publicBatchWrapper86_copies3389 a h).2.2.2.1
  have c108452 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_14 := arithEq_of_rows h (row := 1775) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_14
  simp only [← c108447, ← c108448, ← c108449] at e_1775_14
  have e_1775_15 := arithEq_of_rows h (row := 1775) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_15
  simp only [← c108450, ← c108451, k2, ← c108452] at e_1775_15
  have hr := e_1775_15
  simp only [e_1775_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4196)) := by
  have c108453 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.1
  have c108454 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.1
  have c108455 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.1
  have c108456 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.1
  have c108457 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.1
  have c108458 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_16 := arithEq_of_rows h (row := 1775) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_16
  simp only [← c108453, ← c108454, ← c108455] at e_1775_16
  have e_1775_17 := arithEq_of_rows h (row := 1775) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_17
  simp only [← c108456, ← c108457, k2, ← c108458] at e_1775_17
  have hr := e_1775_17
  simp only [e_1775_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1775 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4197)) := by
  have c108459 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108460 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108461 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108462 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108463 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108464 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1775_18 := arithEq_of_rows h (row := 1775) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_18
  simp only [← c108459, ← c108460, ← c108461] at e_1775_18
  have e_1775_19 := arithEq_of_rows h (row := 1775) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_19
  simp only [← c108462, ← c108463, k2, ← c108464] at e_1775_19
  have hr := e_1775_19
  simp only [e_1775_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4198)) := by
  have c108465 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108466 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108467 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108468 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108469 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108470 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_0 := arithEq_of_rows h (row := 1776) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_0
  simp only [← c108465, ← c108466, ← c108467] at e_1776_0
  have e_1776_1 := arithEq_of_rows h (row := 1776) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_1
  simp only [← c108468, ← c108469, k2, ← c108470] at e_1776_1
  have hr := e_1776_1
  simp only [e_1776_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4199)) := by
  have c108471 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108472 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108473 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108474 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108475 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108476 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_2 := arithEq_of_rows h (row := 1776) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_2
  simp only [← c108471, ← c108472, ← c108473] at e_1776_2
  have e_1776_3 := arithEq_of_rows h (row := 1776) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_3
  simp only [← c108474, ← c108475, k2, ← c108476] at e_1776_3
  have hr := e_1776_3
  simp only [e_1776_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4200)) := by
  have c108477 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108478 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108479 := (publicBatchWrapper86_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108480 := (publicBatchWrapper86_copies3390 a h).1
  have c108481 := (publicBatchWrapper86_copies3390 a h).2.1
  have c108482 := (publicBatchWrapper86_copies3390 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_4 := arithEq_of_rows h (row := 1776) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_4
  simp only [← c108477, ← c108478, ← c108479] at e_1776_4
  have e_1776_5 := arithEq_of_rows h (row := 1776) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_5
  simp only [← c108480, ← c108481, k2, ← c108482] at e_1776_5
  have hr := e_1776_5
  simp only [e_1776_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4201)) := by
  have c108483 := (publicBatchWrapper86_copies3390 a h).2.2.2.1
  have c108484 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.1
  have c108485 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.1
  have c108486 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.1
  have c108487 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.1
  have c108488 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_6 := arithEq_of_rows h (row := 1776) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_6
  simp only [← c108483, ← c108484, ← c108485] at e_1776_6
  have e_1776_7 := arithEq_of_rows h (row := 1776) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_7
  simp only [← c108486, ← c108487, k2, ← c108488] at e_1776_7
  have hr := e_1776_7
  simp only [e_1776_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4370)) := by
  have c108489 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.1
  have c108490 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108491 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108492 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108493 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108494 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_8 := arithEq_of_rows h (row := 1776) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_8
  simp only [← c108489, ← c108490, ← c108491] at e_1776_8
  have e_1776_9 := arithEq_of_rows h (row := 1776) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_9
  simp only [← c108492, ← c108493, k2, ← c108494] at e_1776_9
  have hr := e_1776_9
  simp only [e_1776_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4371)) := by
  have c108495 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108496 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108497 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108498 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108499 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108500 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_10 := arithEq_of_rows h (row := 1776) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_10
  simp only [← c108495, ← c108496, ← c108497] at e_1776_10
  have e_1776_11 := arithEq_of_rows h (row := 1776) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_11
  simp only [← c108498, ← c108499, k2, ← c108500] at e_1776_11
  have hr := e_1776_11
  simp only [e_1776_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4372)) := by
  have c108501 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108502 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108503 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108504 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108505 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108506 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_12 := arithEq_of_rows h (row := 1776) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_12
  simp only [← c108501, ← c108502, ← c108503] at e_1776_12
  have e_1776_13 := arithEq_of_rows h (row := 1776) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_13
  simp only [← c108504, ← c108505, k2, ← c108506] at e_1776_13
  have hr := e_1776_13
  simp only [e_1776_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4373)) := by
  have c108507 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108508 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108509 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108510 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108511 := (publicBatchWrapper86_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108512 := (publicBatchWrapper86_copies3391 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_14 := arithEq_of_rows h (row := 1776) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_14
  simp only [← c108507, ← c108508, ← c108509] at e_1776_14
  have e_1776_15 := arithEq_of_rows h (row := 1776) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_15
  simp only [← c108510, ← c108511, k2, ← c108512] at e_1776_15
  have hr := e_1776_15
  simp only [e_1776_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4374)) := by
  have c108513 := (publicBatchWrapper86_copies3391 a h).2.1
  have c108514 := (publicBatchWrapper86_copies3391 a h).2.2.1
  have c108515 := (publicBatchWrapper86_copies3391 a h).2.2.2.1
  have c108516 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.1
  have c108517 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.1
  have c108518 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_16 := arithEq_of_rows h (row := 1776) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_16
  simp only [← c108513, ← c108514, ← c108515] at e_1776_16
  have e_1776_17 := arithEq_of_rows h (row := 1776) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_17
  simp only [← c108516, ← c108517, k2, ← c108518] at e_1776_17
  have hr := e_1776_17
  simp only [e_1776_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1776 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4375)) := by
  have c108519 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.1
  have c108520 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.1
  have c108521 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.1
  have c108522 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108523 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108524 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1776_18 := arithEq_of_rows h (row := 1776) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_18
  simp only [← c108519, ← c108520, ← c108521] at e_1776_18
  have e_1776_19 := arithEq_of_rows h (row := 1776) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_19
  simp only [← c108522, ← c108523, k2, ← c108524] at e_1776_19
  have hr := e_1776_19
  simp only [e_1776_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4376)) := by
  have c108525 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108526 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108527 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108528 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108529 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108530 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_0 := arithEq_of_rows h (row := 1777) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_0
  simp only [← c108525, ← c108526, ← c108527] at e_1777_0
  have e_1777_1 := arithEq_of_rows h (row := 1777) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_1
  simp only [← c108528, ← c108529, k2, ← c108530] at e_1777_1
  have hr := e_1777_1
  simp only [e_1777_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4377)) := by
  have c108531 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108532 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108533 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108534 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108535 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108536 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_2 := arithEq_of_rows h (row := 1777) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_2
  simp only [← c108531, ← c108532, ← c108533] at e_1777_2
  have e_1777_3 := arithEq_of_rows h (row := 1777) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_3
  simp only [← c108534, ← c108535, k2, ← c108536] at e_1777_3
  have hr := e_1777_3
  simp only [e_1777_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4378)) := by
  have c108537 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108538 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108539 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108540 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108541 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108542 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_4 := arithEq_of_rows h (row := 1777) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_4
  simp only [← c108537, ← c108538, ← c108539] at e_1777_4
  have e_1777_5 := arithEq_of_rows h (row := 1777) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_5
  simp only [← c108540, ← c108541, k2, ← c108542] at e_1777_5
  have hr := e_1777_5
  simp only [e_1777_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4379)) := by
  have c108543 := (publicBatchWrapper86_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108544 := (publicBatchWrapper86_copies3392 a h).1
  have c108545 := (publicBatchWrapper86_copies3392 a h).2.1
  have c108546 := (publicBatchWrapper86_copies3392 a h).2.2.1
  have c108547 := (publicBatchWrapper86_copies3392 a h).2.2.2.1
  have c108548 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_6 := arithEq_of_rows h (row := 1777) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_6
  simp only [← c108543, ← c108544, ← c108545] at e_1777_6
  have e_1777_7 := arithEq_of_rows h (row := 1777) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_7
  simp only [← c108546, ← c108547, k2, ← c108548] at e_1777_7
  have hr := e_1777_7
  simp only [e_1777_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4380)) := by
  have c108549 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.1
  have c108550 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.1
  have c108551 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.1
  have c108552 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.1
  have c108553 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.1
  have c108554 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_8 := arithEq_of_rows h (row := 1777) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_8
  simp only [← c108549, ← c108550, ← c108551] at e_1777_8
  have e_1777_9 := arithEq_of_rows h (row := 1777) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_9
  simp only [← c108552, ← c108553, k2, ← c108554] at e_1777_9
  have hr := e_1777_9
  simp only [e_1777_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4381)) := by
  have c108555 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108556 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108557 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108558 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108559 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108560 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_10 := arithEq_of_rows h (row := 1777) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_10
  simp only [← c108555, ← c108556, ← c108557] at e_1777_10
  have e_1777_11 := arithEq_of_rows h (row := 1777) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_11
  simp only [← c108558, ← c108559, k2, ← c108560] at e_1777_11
  have hr := e_1777_11
  simp only [e_1777_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4382)) := by
  have c108561 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108562 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108563 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108564 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108565 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108566 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_12 := arithEq_of_rows h (row := 1777) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_12
  simp only [← c108561, ← c108562, ← c108563] at e_1777_12
  have e_1777_13 := arithEq_of_rows h (row := 1777) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_13
  simp only [← c108564, ← c108565, k2, ← c108566] at e_1777_13
  have hr := e_1777_13
  simp only [e_1777_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4383)) := by
  have c108567 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108568 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108569 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108570 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108571 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108572 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_14 := arithEq_of_rows h (row := 1777) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_14
  simp only [← c108567, ← c108568, ← c108569] at e_1777_14
  have e_1777_15 := arithEq_of_rows h (row := 1777) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_15
  simp only [← c108570, ← c108571, k2, ← c108572] at e_1777_15
  have hr := e_1777_15
  simp only [e_1777_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4384)) := by
  have c108573 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108574 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108575 := (publicBatchWrapper86_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108576 := (publicBatchWrapper86_copies3393 a h).1
  have c108577 := (publicBatchWrapper86_copies3393 a h).2.1
  have c108578 := (publicBatchWrapper86_copies3393 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_16 := arithEq_of_rows h (row := 1777) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_16
  simp only [← c108573, ← c108574, ← c108575] at e_1777_16
  have e_1777_17 := arithEq_of_rows h (row := 1777) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_17
  simp only [← c108576, ← c108577, k2, ← c108578] at e_1777_17
  have hr := e_1777_17
  simp only [e_1777_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1777 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4385)) := by
  have c108579 := (publicBatchWrapper86_copies3393 a h).2.2.2.1
  have c108580 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.1
  have c108581 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.1
  have c108582 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.1
  have c108583 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.1
  have c108584 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1777_18 := arithEq_of_rows h (row := 1777) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_18
  simp only [← c108579, ← c108580, ← c108581] at e_1777_18
  have e_1777_19 := arithEq_of_rows h (row := 1777) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_19
  simp only [← c108582, ← c108583, k2, ← c108584] at e_1777_19
  have hr := e_1777_19
  simp only [e_1777_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4386)) := by
  have c108585 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.1
  have c108586 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108587 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108588 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108589 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108590 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_0 := arithEq_of_rows h (row := 1778) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_0
  simp only [← c108585, ← c108586, ← c108587] at e_1778_0
  have e_1778_1 := arithEq_of_rows h (row := 1778) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_1
  simp only [← c108588, ← c108589, k2, ← c108590] at e_1778_1
  have hr := e_1778_1
  simp only [e_1778_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4387)) := by
  have c108591 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108592 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108593 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108594 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108595 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108596 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_2 := arithEq_of_rows h (row := 1778) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_2
  simp only [← c108591, ← c108592, ← c108593] at e_1778_2
  have e_1778_3 := arithEq_of_rows h (row := 1778) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_3
  simp only [← c108594, ← c108595, k2, ← c108596] at e_1778_3
  have hr := e_1778_3
  simp only [e_1778_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4388)) := by
  have c108597 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108598 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108599 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108600 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108601 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108602 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_4 := arithEq_of_rows h (row := 1778) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_4
  simp only [← c108597, ← c108598, ← c108599] at e_1778_4
  have e_1778_5 := arithEq_of_rows h (row := 1778) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_5
  simp only [← c108600, ← c108601, k2, ← c108602] at e_1778_5
  have hr := e_1778_5
  simp only [e_1778_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4389)) := by
  have c108603 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108604 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108605 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108606 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108607 := (publicBatchWrapper86_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108608 := (publicBatchWrapper86_copies3394 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_6 := arithEq_of_rows h (row := 1778) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_6
  simp only [← c108603, ← c108604, ← c108605] at e_1778_6
  have e_1778_7 := arithEq_of_rows h (row := 1778) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_7
  simp only [← c108606, ← c108607, k2, ← c108608] at e_1778_7
  have hr := e_1778_7
  simp only [e_1778_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4390)) := by
  have c108609 := (publicBatchWrapper86_copies3394 a h).2.1
  have c108610 := (publicBatchWrapper86_copies3394 a h).2.2.1
  have c108611 := (publicBatchWrapper86_copies3394 a h).2.2.2.1
  have c108612 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.1
  have c108613 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.1
  have c108614 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_8 := arithEq_of_rows h (row := 1778) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_8
  simp only [← c108609, ← c108610, ← c108611] at e_1778_8
  have e_1778_9 := arithEq_of_rows h (row := 1778) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_9
  simp only [← c108612, ← c108613, k2, ← c108614] at e_1778_9
  have hr := e_1778_9
  simp only [e_1778_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4391)) := by
  have c108615 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.1
  have c108616 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.1
  have c108617 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.1
  have c108618 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108619 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108620 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_10 := arithEq_of_rows h (row := 1778) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_10
  simp only [← c108615, ← c108616, ← c108617] at e_1778_10
  have e_1778_11 := arithEq_of_rows h (row := 1778) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_11
  simp only [← c108618, ← c108619, k2, ← c108620] at e_1778_11
  have hr := e_1778_11
  simp only [e_1778_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4392)) := by
  have c108621 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108622 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108623 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108624 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108625 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108626 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_12 := arithEq_of_rows h (row := 1778) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_12
  simp only [← c108621, ← c108622, ← c108623] at e_1778_12
  have e_1778_13 := arithEq_of_rows h (row := 1778) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_13
  simp only [← c108624, ← c108625, k2, ← c108626] at e_1778_13
  have hr := e_1778_13
  simp only [e_1778_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4393)) := by
  have c108627 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108628 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108629 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108630 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108631 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108632 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_14 := arithEq_of_rows h (row := 1778) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_14
  simp only [← c108627, ← c108628, ← c108629] at e_1778_14
  have e_1778_15 := arithEq_of_rows h (row := 1778) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_15
  simp only [← c108630, ← c108631, k2, ← c108632] at e_1778_15
  have hr := e_1778_15
  simp only [e_1778_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4394)) := by
  have c108633 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108634 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108635 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108636 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108637 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108638 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_16 := arithEq_of_rows h (row := 1778) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_16
  simp only [← c108633, ← c108634, ← c108635] at e_1778_16
  have e_1778_17 := arithEq_of_rows h (row := 1778) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_17
  simp only [← c108636, ← c108637, k2, ← c108638] at e_1778_17
  have hr := e_1778_17
  simp only [e_1778_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1778 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4395)) := by
  have c108639 := (publicBatchWrapper86_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108640 := (publicBatchWrapper86_copies3395 a h).1
  have c108641 := (publicBatchWrapper86_copies3395 a h).2.1
  have c108642 := (publicBatchWrapper86_copies3395 a h).2.2.1
  have c108643 := (publicBatchWrapper86_copies3395 a h).2.2.2.1
  have c108644 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1778_18 := arithEq_of_rows h (row := 1778) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_18
  simp only [← c108639, ← c108640, ← c108641] at e_1778_18
  have e_1778_19 := arithEq_of_rows h (row := 1778) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_19
  simp only [← c108642, ← c108643, k2, ← c108644] at e_1778_19
  have hr := e_1778_19
  simp only [e_1778_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4396)) := by
  have c108645 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.1
  have c108646 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.1
  have c108647 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.1
  have c108648 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.1
  have c108649 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.1
  have c108650 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_0 := arithEq_of_rows h (row := 1779) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_0
  simp only [← c108645, ← c108646, ← c108647] at e_1779_0
  have e_1779_1 := arithEq_of_rows h (row := 1779) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_1
  simp only [← c108648, ← c108649, k2, ← c108650] at e_1779_1
  have hr := e_1779_1
  simp only [e_1779_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4397)) := by
  have c108651 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108652 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108653 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108654 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108655 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108656 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_2 := arithEq_of_rows h (row := 1779) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_2
  simp only [← c108651, ← c108652, ← c108653] at e_1779_2
  have e_1779_3 := arithEq_of_rows h (row := 1779) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_3
  simp only [← c108654, ← c108655, k2, ← c108656] at e_1779_3
  have hr := e_1779_3
  simp only [e_1779_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4398)) := by
  have c108657 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108658 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108659 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108660 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108661 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108662 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_4 := arithEq_of_rows h (row := 1779) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_4
  simp only [← c108657, ← c108658, ← c108659] at e_1779_4
  have e_1779_5 := arithEq_of_rows h (row := 1779) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_5
  simp only [← c108660, ← c108661, k2, ← c108662] at e_1779_5
  have hr := e_1779_5
  simp only [e_1779_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4399)) := by
  have c108663 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108664 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108665 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108666 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108667 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108668 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_6 := arithEq_of_rows h (row := 1779) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_6
  simp only [← c108663, ← c108664, ← c108665] at e_1779_6
  have e_1779_7 := arithEq_of_rows h (row := 1779) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_7
  simp only [← c108666, ← c108667, k2, ← c108668] at e_1779_7
  have hr := e_1779_7
  simp only [e_1779_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4400)) := by
  have c108669 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108670 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108671 := (publicBatchWrapper86_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108672 := (publicBatchWrapper86_copies3396 a h).1
  have c108673 := (publicBatchWrapper86_copies3396 a h).2.1
  have c108674 := (publicBatchWrapper86_copies3396 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_8 := arithEq_of_rows h (row := 1779) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_8
  simp only [← c108669, ← c108670, ← c108671] at e_1779_8
  have e_1779_9 := arithEq_of_rows h (row := 1779) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_9
  simp only [← c108672, ← c108673, k2, ← c108674] at e_1779_9
  have hr := e_1779_9
  simp only [e_1779_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4401)) := by
  have c108675 := (publicBatchWrapper86_copies3396 a h).2.2.2.1
  have c108676 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.1
  have c108677 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.1
  have c108678 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.1
  have c108679 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.1
  have c108680 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_10 := arithEq_of_rows h (row := 1779) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_10
  simp only [← c108675, ← c108676, ← c108677] at e_1779_10
  have e_1779_11 := arithEq_of_rows h (row := 1779) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_11
  simp only [← c108678, ← c108679, k2, ← c108680] at e_1779_11
  have hr := e_1779_11
  simp only [e_1779_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4570)) := by
  have c108681 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.1
  have c108682 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108683 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108684 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108685 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108686 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_12 := arithEq_of_rows h (row := 1779) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_12
  simp only [← c108681, ← c108682, ← c108683] at e_1779_12
  have e_1779_13 := arithEq_of_rows h (row := 1779) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_13
  simp only [← c108684, ← c108685, k2, ← c108686] at e_1779_13
  have hr := e_1779_13
  simp only [e_1779_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4571)) := by
  have c108687 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108688 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108689 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108690 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108691 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108692 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_14 := arithEq_of_rows h (row := 1779) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_14
  simp only [← c108687, ← c108688, ← c108689] at e_1779_14
  have e_1779_15 := arithEq_of_rows h (row := 1779) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_15
  simp only [← c108690, ← c108691, k2, ← c108692] at e_1779_15
  have hr := e_1779_15
  simp only [e_1779_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4572)) := by
  have c108693 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108694 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108695 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108696 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108697 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108698 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_16 := arithEq_of_rows h (row := 1779) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_16
  simp only [← c108693, ← c108694, ← c108695] at e_1779_16
  have e_1779_17 := arithEq_of_rows h (row := 1779) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_17
  simp only [← c108696, ← c108697, k2, ← c108698] at e_1779_17
  have hr := e_1779_17
  simp only [e_1779_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1779 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4573)) := by
  have c108699 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108700 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108701 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108702 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108703 := (publicBatchWrapper86_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108704 := (publicBatchWrapper86_copies3397 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1779_18 := arithEq_of_rows h (row := 1779) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_18
  simp only [← c108699, ← c108700, ← c108701] at e_1779_18
  have e_1779_19 := arithEq_of_rows h (row := 1779) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_19
  simp only [← c108702, ← c108703, k2, ← c108704] at e_1779_19
  have hr := e_1779_19
  simp only [e_1779_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4574)) := by
  have c108705 := (publicBatchWrapper86_copies3397 a h).2.1
  have c108706 := (publicBatchWrapper86_copies3397 a h).2.2.1
  have c108707 := (publicBatchWrapper86_copies3397 a h).2.2.2.1
  have c108708 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.1
  have c108709 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.1
  have c108710 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_0 := arithEq_of_rows h (row := 1780) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_0
  simp only [← c108705, ← c108706, ← c108707] at e_1780_0
  have e_1780_1 := arithEq_of_rows h (row := 1780) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_1
  simp only [← c108708, ← c108709, k2, ← c108710] at e_1780_1
  have hr := e_1780_1
  simp only [e_1780_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4575)) := by
  have c108711 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.1
  have c108712 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.1
  have c108713 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.1
  have c108714 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108715 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108716 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_2 := arithEq_of_rows h (row := 1780) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_2
  simp only [← c108711, ← c108712, ← c108713] at e_1780_2
  have e_1780_3 := arithEq_of_rows h (row := 1780) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_3
  simp only [← c108714, ← c108715, k2, ← c108716] at e_1780_3
  have hr := e_1780_3
  simp only [e_1780_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4576)) := by
  have c108717 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108718 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108719 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108720 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108721 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108722 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_4 := arithEq_of_rows h (row := 1780) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_4
  simp only [← c108717, ← c108718, ← c108719] at e_1780_4
  have e_1780_5 := arithEq_of_rows h (row := 1780) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_5
  simp only [← c108720, ← c108721, k2, ← c108722] at e_1780_5
  have hr := e_1780_5
  simp only [e_1780_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4577)) := by
  have c108723 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108724 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108725 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108726 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108727 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108728 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_6 := arithEq_of_rows h (row := 1780) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_6
  simp only [← c108723, ← c108724, ← c108725] at e_1780_6
  have e_1780_7 := arithEq_of_rows h (row := 1780) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_7
  simp only [← c108726, ← c108727, k2, ← c108728] at e_1780_7
  have hr := e_1780_7
  simp only [e_1780_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4578)) := by
  have c108729 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108730 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108731 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108732 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108733 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108734 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_8 := arithEq_of_rows h (row := 1780) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_8
  simp only [← c108729, ← c108730, ← c108731] at e_1780_8
  have e_1780_9 := arithEq_of_rows h (row := 1780) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_9
  simp only [← c108732, ← c108733, k2, ← c108734] at e_1780_9
  have hr := e_1780_9
  simp only [e_1780_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4579)) := by
  have c108735 := (publicBatchWrapper86_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108736 := (publicBatchWrapper86_copies3398 a h).1
  have c108737 := (publicBatchWrapper86_copies3398 a h).2.1
  have c108738 := (publicBatchWrapper86_copies3398 a h).2.2.1
  have c108739 := (publicBatchWrapper86_copies3398 a h).2.2.2.1
  have c108740 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_10 := arithEq_of_rows h (row := 1780) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_10
  simp only [← c108735, ← c108736, ← c108737] at e_1780_10
  have e_1780_11 := arithEq_of_rows h (row := 1780) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_11
  simp only [← c108738, ← c108739, k2, ← c108740] at e_1780_11
  have hr := e_1780_11
  simp only [e_1780_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4580)) := by
  have c108741 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.1
  have c108742 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.1
  have c108743 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.1
  have c108744 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.1
  have c108745 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.1
  have c108746 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_12 := arithEq_of_rows h (row := 1780) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_12
  simp only [← c108741, ← c108742, ← c108743] at e_1780_12
  have e_1780_13 := arithEq_of_rows h (row := 1780) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_13
  simp only [← c108744, ← c108745, k2, ← c108746] at e_1780_13
  have hr := e_1780_13
  simp only [e_1780_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4581)) := by
  have c108747 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108748 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108749 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108750 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108751 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108752 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_14 := arithEq_of_rows h (row := 1780) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_14
  simp only [← c108747, ← c108748, ← c108749] at e_1780_14
  have e_1780_15 := arithEq_of_rows h (row := 1780) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_15
  simp only [← c108750, ← c108751, k2, ← c108752] at e_1780_15
  have hr := e_1780_15
  simp only [e_1780_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4582)) := by
  have c108753 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108754 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108755 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108756 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108757 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108758 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_16 := arithEq_of_rows h (row := 1780) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_16
  simp only [← c108753, ← c108754, ← c108755] at e_1780_16
  have e_1780_17 := arithEq_of_rows h (row := 1780) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_17
  simp only [← c108756, ← c108757, k2, ← c108758] at e_1780_17
  have hr := e_1780_17
  simp only [e_1780_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1780 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4583)) := by
  have c108759 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108760 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108761 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108762 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108763 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108764 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1780_18 := arithEq_of_rows h (row := 1780) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_18
  simp only [← c108759, ← c108760, ← c108761] at e_1780_18
  have e_1780_19 := arithEq_of_rows h (row := 1780) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_19
  simp only [← c108762, ← c108763, k2, ← c108764] at e_1780_19
  have hr := e_1780_19
  simp only [e_1780_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4584)) := by
  have c108765 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108766 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108767 := (publicBatchWrapper86_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108768 := (publicBatchWrapper86_copies3399 a h).1
  have c108769 := (publicBatchWrapper86_copies3399 a h).2.1
  have c108770 := (publicBatchWrapper86_copies3399 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_0 := arithEq_of_rows h (row := 1781) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_0
  simp only [← c108765, ← c108766, ← c108767] at e_1781_0
  have e_1781_1 := arithEq_of_rows h (row := 1781) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_1
  simp only [← c108768, ← c108769, k2, ← c108770] at e_1781_1
  have hr := e_1781_1
  simp only [e_1781_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4585)) := by
  have c108771 := (publicBatchWrapper86_copies3399 a h).2.2.2.1
  have c108772 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.1
  have c108773 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.1
  have c108774 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.1
  have c108775 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.1
  have c108776 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_2 := arithEq_of_rows h (row := 1781) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_2
  simp only [← c108771, ← c108772, ← c108773] at e_1781_2
  have e_1781_3 := arithEq_of_rows h (row := 1781) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_3
  simp only [← c108774, ← c108775, k2, ← c108776] at e_1781_3
  have hr := e_1781_3
  simp only [e_1781_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4586)) := by
  have c108777 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.1
  have c108778 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108779 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108780 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108781 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108782 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_4 := arithEq_of_rows h (row := 1781) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_4
  simp only [← c108777, ← c108778, ← c108779] at e_1781_4
  have e_1781_5 := arithEq_of_rows h (row := 1781) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_5
  simp only [← c108780, ← c108781, k2, ← c108782] at e_1781_5
  have hr := e_1781_5
  simp only [e_1781_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4587)) := by
  have c108783 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108784 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108785 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108786 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108787 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108788 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_6 := arithEq_of_rows h (row := 1781) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_6
  simp only [← c108783, ← c108784, ← c108785] at e_1781_6
  have e_1781_7 := arithEq_of_rows h (row := 1781) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_7
  simp only [← c108786, ← c108787, k2, ← c108788] at e_1781_7
  have hr := e_1781_7
  simp only [e_1781_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4588)) := by
  have c108789 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108790 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108791 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108792 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108793 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108794 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_8 := arithEq_of_rows h (row := 1781) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_8
  simp only [← c108789, ← c108790, ← c108791] at e_1781_8
  have e_1781_9 := arithEq_of_rows h (row := 1781) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_9
  simp only [← c108792, ← c108793, k2, ← c108794] at e_1781_9
  have hr := e_1781_9
  simp only [e_1781_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4589)) := by
  have c108795 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108796 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108797 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108798 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108799 := (publicBatchWrapper86_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108800 := (publicBatchWrapper86_copies3400 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_10 := arithEq_of_rows h (row := 1781) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_10
  simp only [← c108795, ← c108796, ← c108797] at e_1781_10
  have e_1781_11 := arithEq_of_rows h (row := 1781) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_11
  simp only [← c108798, ← c108799, k2, ← c108800] at e_1781_11
  have hr := e_1781_11
  simp only [e_1781_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4590)) := by
  have c108801 := (publicBatchWrapper86_copies3400 a h).2.1
  have c108802 := (publicBatchWrapper86_copies3400 a h).2.2.1
  have c108803 := (publicBatchWrapper86_copies3400 a h).2.2.2.1
  have c108804 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.1
  have c108805 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.1
  have c108806 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_12 := arithEq_of_rows h (row := 1781) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_12
  simp only [← c108801, ← c108802, ← c108803] at e_1781_12
  have e_1781_13 := arithEq_of_rows h (row := 1781) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_13
  simp only [← c108804, ← c108805, k2, ← c108806] at e_1781_13
  have hr := e_1781_13
  simp only [e_1781_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4591)) := by
  have c108807 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.1
  have c108808 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.1
  have c108809 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.1
  have c108810 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108811 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108812 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_14 := arithEq_of_rows h (row := 1781) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_14
  simp only [← c108807, ← c108808, ← c108809] at e_1781_14
  have e_1781_15 := arithEq_of_rows h (row := 1781) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_15
  simp only [← c108810, ← c108811, k2, ← c108812] at e_1781_15
  have hr := e_1781_15
  simp only [e_1781_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4592)) := by
  have c108813 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108814 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108815 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108816 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108817 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108818 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_16 := arithEq_of_rows h (row := 1781) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_16
  simp only [← c108813, ← c108814, ← c108815] at e_1781_16
  have e_1781_17 := arithEq_of_rows h (row := 1781) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_17
  simp only [← c108816, ← c108817, k2, ← c108818] at e_1781_17
  have hr := e_1781_17
  simp only [e_1781_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1781 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4593)) := by
  have c108819 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108820 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108821 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108822 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108823 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108824 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1781_18 := arithEq_of_rows h (row := 1781) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_18
  simp only [← c108819, ← c108820, ← c108821] at e_1781_18
  have e_1781_19 := arithEq_of_rows h (row := 1781) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_19
  simp only [← c108822, ← c108823, k2, ← c108824] at e_1781_19
  have hr := e_1781_19
  simp only [e_1781_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4594)) := by
  have c108825 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108826 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108827 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108828 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108829 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108830 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_0 := arithEq_of_rows h (row := 1782) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_0
  simp only [← c108825, ← c108826, ← c108827] at e_1782_0
  have e_1782_1 := arithEq_of_rows h (row := 1782) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_1
  simp only [← c108828, ← c108829, k2, ← c108830] at e_1782_1
  have hr := e_1782_1
  simp only [e_1782_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4595)) := by
  have c108831 := (publicBatchWrapper86_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108832 := (publicBatchWrapper86_copies3401 a h).1
  have c108833 := (publicBatchWrapper86_copies3401 a h).2.1
  have c108834 := (publicBatchWrapper86_copies3401 a h).2.2.1
  have c108835 := (publicBatchWrapper86_copies3401 a h).2.2.2.1
  have c108836 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_2 := arithEq_of_rows h (row := 1782) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_2
  simp only [← c108831, ← c108832, ← c108833] at e_1782_2
  have e_1782_3 := arithEq_of_rows h (row := 1782) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_3
  simp only [← c108834, ← c108835, k2, ← c108836] at e_1782_3
  have hr := e_1782_3
  simp only [e_1782_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4596)) := by
  have c108837 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.1
  have c108838 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.1
  have c108839 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.1
  have c108840 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.1
  have c108841 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.1
  have c108842 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_4 := arithEq_of_rows h (row := 1782) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_4
  simp only [← c108837, ← c108838, ← c108839] at e_1782_4
  have e_1782_5 := arithEq_of_rows h (row := 1782) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_5
  simp only [← c108840, ← c108841, k2, ← c108842] at e_1782_5
  have hr := e_1782_5
  simp only [e_1782_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4597)) := by
  have c108843 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108844 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108845 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108846 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108847 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108848 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_6 := arithEq_of_rows h (row := 1782) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_6
  simp only [← c108843, ← c108844, ← c108845] at e_1782_6
  have e_1782_7 := arithEq_of_rows h (row := 1782) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_7
  simp only [← c108846, ← c108847, k2, ← c108848] at e_1782_7
  have hr := e_1782_7
  simp only [e_1782_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4598)) := by
  have c108849 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108850 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108851 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108852 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108853 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108854 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_8 := arithEq_of_rows h (row := 1782) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_8
  simp only [← c108849, ← c108850, ← c108851] at e_1782_8
  have e_1782_9 := arithEq_of_rows h (row := 1782) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_9
  simp only [← c108852, ← c108853, k2, ← c108854] at e_1782_9
  have hr := e_1782_9
  simp only [e_1782_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4599)) := by
  have c108855 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108856 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108857 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108858 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108859 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108860 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_10 := arithEq_of_rows h (row := 1782) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_10
  simp only [← c108855, ← c108856, ← c108857] at e_1782_10
  have e_1782_11 := arithEq_of_rows h (row := 1782) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_11
  simp only [← c108858, ← c108859, k2, ← c108860] at e_1782_11
  have hr := e_1782_11
  simp only [e_1782_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4600)) := by
  have c108861 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108862 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108863 := (publicBatchWrapper86_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108864 := (publicBatchWrapper86_copies3402 a h).1
  have c108865 := (publicBatchWrapper86_copies3402 a h).2.1
  have c108866 := (publicBatchWrapper86_copies3402 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_12 := arithEq_of_rows h (row := 1782) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_12
  simp only [← c108861, ← c108862, ← c108863] at e_1782_12
  have e_1782_13 := arithEq_of_rows h (row := 1782) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_13
  simp only [← c108864, ← c108865, k2, ← c108866] at e_1782_13
  have hr := e_1782_13
  simp only [e_1782_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4601)) := by
  have c108867 := (publicBatchWrapper86_copies3402 a h).2.2.2.1
  have c108868 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.1
  have c108869 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.1
  have c108870 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.1
  have c108871 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.1
  have c108872 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_14 := arithEq_of_rows h (row := 1782) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_14
  simp only [← c108867, ← c108868, ← c108869] at e_1782_14
  have e_1782_15 := arithEq_of_rows h (row := 1782) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_15
  simp only [← c108870, ← c108871, k2, ← c108872] at e_1782_15
  have hr := e_1782_15
  simp only [e_1782_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4770)) := by
  have c108873 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.1
  have c108874 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108875 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108876 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108877 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108878 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_16 := arithEq_of_rows h (row := 1782) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_16
  simp only [← c108873, ← c108874, ← c108875] at e_1782_16
  have e_1782_17 := arithEq_of_rows h (row := 1782) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_17
  simp only [← c108876, ← c108877, k2, ← c108878] at e_1782_17
  have hr := e_1782_17
  simp only [e_1782_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1782 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4771)) := by
  have c108879 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108880 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108881 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108882 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108883 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108884 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1782_18 := arithEq_of_rows h (row := 1782) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_18
  simp only [← c108879, ← c108880, ← c108881] at e_1782_18
  have e_1782_19 := arithEq_of_rows h (row := 1782) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_19
  simp only [← c108882, ← c108883, k2, ← c108884] at e_1782_19
  have hr := e_1782_19
  simp only [e_1782_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4772)) := by
  have c108885 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108886 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108887 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108888 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108889 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108890 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_0 := arithEq_of_rows h (row := 1783) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_0
  simp only [← c108885, ← c108886, ← c108887] at e_1783_0
  have e_1783_1 := arithEq_of_rows h (row := 1783) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_1
  simp only [← c108888, ← c108889, k2, ← c108890] at e_1783_1
  have hr := e_1783_1
  simp only [e_1783_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4773)) := by
  have c108891 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108892 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108893 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108894 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108895 := (publicBatchWrapper86_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108896 := (publicBatchWrapper86_copies3403 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_2 := arithEq_of_rows h (row := 1783) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_2
  simp only [← c108891, ← c108892, ← c108893] at e_1783_2
  have e_1783_3 := arithEq_of_rows h (row := 1783) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_3
  simp only [← c108894, ← c108895, k2, ← c108896] at e_1783_3
  have hr := e_1783_3
  simp only [e_1783_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4774)) := by
  have c108897 := (publicBatchWrapper86_copies3403 a h).2.1
  have c108898 := (publicBatchWrapper86_copies3403 a h).2.2.1
  have c108899 := (publicBatchWrapper86_copies3403 a h).2.2.2.1
  have c108900 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.1
  have c108901 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.1
  have c108902 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_4 := arithEq_of_rows h (row := 1783) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_4
  simp only [← c108897, ← c108898, ← c108899] at e_1783_4
  have e_1783_5 := arithEq_of_rows h (row := 1783) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_5
  simp only [← c108900, ← c108901, k2, ← c108902] at e_1783_5
  have hr := e_1783_5
  simp only [e_1783_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4775)) := by
  have c108903 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.1
  have c108904 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.1
  have c108905 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.1
  have c108906 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108907 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108908 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_6 := arithEq_of_rows h (row := 1783) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_6
  simp only [← c108903, ← c108904, ← c108905] at e_1783_6
  have e_1783_7 := arithEq_of_rows h (row := 1783) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_7
  simp only [← c108906, ← c108907, k2, ← c108908] at e_1783_7
  have hr := e_1783_7
  simp only [e_1783_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4776)) := by
  have c108909 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108910 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108911 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108912 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108913 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108914 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_8 := arithEq_of_rows h (row := 1783) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_8
  simp only [← c108909, ← c108910, ← c108911] at e_1783_8
  have e_1783_9 := arithEq_of_rows h (row := 1783) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_9
  simp only [← c108912, ← c108913, k2, ← c108914] at e_1783_9
  have hr := e_1783_9
  simp only [e_1783_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4777)) := by
  have c108915 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108916 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108917 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108918 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108919 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108920 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_10 := arithEq_of_rows h (row := 1783) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_10
  simp only [← c108915, ← c108916, ← c108917] at e_1783_10
  have e_1783_11 := arithEq_of_rows h (row := 1783) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_11
  simp only [← c108918, ← c108919, k2, ← c108920] at e_1783_11
  have hr := e_1783_11
  simp only [e_1783_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4778)) := by
  have c108921 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108922 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108923 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108924 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108925 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108926 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_12 := arithEq_of_rows h (row := 1783) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_12
  simp only [← c108921, ← c108922, ← c108923] at e_1783_12
  have e_1783_13 := arithEq_of_rows h (row := 1783) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_13
  simp only [← c108924, ← c108925, k2, ← c108926] at e_1783_13
  have hr := e_1783_13
  simp only [e_1783_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4779)) := by
  have c108927 := (publicBatchWrapper86_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108928 := (publicBatchWrapper86_copies3404 a h).1
  have c108929 := (publicBatchWrapper86_copies3404 a h).2.1
  have c108930 := (publicBatchWrapper86_copies3404 a h).2.2.1
  have c108931 := (publicBatchWrapper86_copies3404 a h).2.2.2.1
  have c108932 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_14 := arithEq_of_rows h (row := 1783) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_14
  simp only [← c108927, ← c108928, ← c108929] at e_1783_14
  have e_1783_15 := arithEq_of_rows h (row := 1783) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_15
  simp only [← c108930, ← c108931, k2, ← c108932] at e_1783_15
  have hr := e_1783_15
  simp only [e_1783_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4780)) := by
  have c108933 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.1
  have c108934 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.1
  have c108935 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.1
  have c108936 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.1
  have c108937 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.1
  have c108938 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_16 := arithEq_of_rows h (row := 1783) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_16
  simp only [← c108933, ← c108934, ← c108935] at e_1783_16
  have e_1783_17 := arithEq_of_rows h (row := 1783) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_17
  simp only [← c108936, ← c108937, k2, ← c108938] at e_1783_17
  have hr := e_1783_17
  simp only [e_1783_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1783 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4781)) := by
  have c108939 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108940 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108941 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108942 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108943 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108944 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1783_18 := arithEq_of_rows h (row := 1783) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_18
  simp only [← c108939, ← c108940, ← c108941] at e_1783_18
  have e_1783_19 := arithEq_of_rows h (row := 1783) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_19
  simp only [← c108942, ← c108943, k2, ← c108944] at e_1783_19
  have hr := e_1783_19
  simp only [e_1783_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4782)) := by
  have c108945 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108946 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108947 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108948 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108949 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108950 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_0 := arithEq_of_rows h (row := 1784) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_0
  simp only [← c108945, ← c108946, ← c108947] at e_1784_0
  have e_1784_1 := arithEq_of_rows h (row := 1784) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_1
  simp only [← c108948, ← c108949, k2, ← c108950] at e_1784_1
  have hr := e_1784_1
  simp only [e_1784_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4783)) := by
  have c108951 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108952 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108953 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108954 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108955 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108956 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_2 := arithEq_of_rows h (row := 1784) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_2
  simp only [← c108951, ← c108952, ← c108953] at e_1784_2
  have e_1784_3 := arithEq_of_rows h (row := 1784) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_3
  simp only [← c108954, ← c108955, k2, ← c108956] at e_1784_3
  have hr := e_1784_3
  simp only [e_1784_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4784)) := by
  have c108957 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108958 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108959 := (publicBatchWrapper86_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108960 := (publicBatchWrapper86_copies3405 a h).1
  have c108961 := (publicBatchWrapper86_copies3405 a h).2.1
  have c108962 := (publicBatchWrapper86_copies3405 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_4 := arithEq_of_rows h (row := 1784) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_4
  simp only [← c108957, ← c108958, ← c108959] at e_1784_4
  have e_1784_5 := arithEq_of_rows h (row := 1784) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_5
  simp only [← c108960, ← c108961, k2, ← c108962] at e_1784_5
  have hr := e_1784_5
  simp only [e_1784_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4785)) := by
  have c108963 := (publicBatchWrapper86_copies3405 a h).2.2.2.1
  have c108964 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.1
  have c108965 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.1
  have c108966 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.1
  have c108967 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.1
  have c108968 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_6 := arithEq_of_rows h (row := 1784) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_6
  simp only [← c108963, ← c108964, ← c108965] at e_1784_6
  have e_1784_7 := arithEq_of_rows h (row := 1784) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_7
  simp only [← c108966, ← c108967, k2, ← c108968] at e_1784_7
  have hr := e_1784_7
  simp only [e_1784_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4786)) := by
  have c108969 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.1
  have c108970 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108971 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108972 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108973 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108974 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_8 := arithEq_of_rows h (row := 1784) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_8
  simp only [← c108969, ← c108970, ← c108971] at e_1784_8
  have e_1784_9 := arithEq_of_rows h (row := 1784) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_9
  simp only [← c108972, ← c108973, k2, ← c108974] at e_1784_9
  have hr := e_1784_9
  simp only [e_1784_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4787)) := by
  have c108975 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108976 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108977 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108978 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108979 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108980 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_10 := arithEq_of_rows h (row := 1784) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_10
  simp only [← c108975, ← c108976, ← c108977] at e_1784_10
  have e_1784_11 := arithEq_of_rows h (row := 1784) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_11
  simp only [← c108978, ← c108979, k2, ← c108980] at e_1784_11
  have hr := e_1784_11
  simp only [e_1784_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4788)) := by
  have c108981 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108982 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108983 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108984 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108985 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108986 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_12 := arithEq_of_rows h (row := 1784) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_12
  simp only [← c108981, ← c108982, ← c108983] at e_1784_12
  have e_1784_13 := arithEq_of_rows h (row := 1784) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_13
  simp only [← c108984, ← c108985, k2, ← c108986] at e_1784_13
  have hr := e_1784_13
  simp only [e_1784_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4789)) := by
  have c108987 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108988 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108989 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108990 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108991 := (publicBatchWrapper86_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108992 := (publicBatchWrapper86_copies3406 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_14 := arithEq_of_rows h (row := 1784) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_14
  simp only [← c108987, ← c108988, ← c108989] at e_1784_14
  have e_1784_15 := arithEq_of_rows h (row := 1784) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_15
  simp only [← c108990, ← c108991, k2, ← c108992] at e_1784_15
  have hr := e_1784_15
  simp only [e_1784_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4790)) := by
  have c108993 := (publicBatchWrapper86_copies3406 a h).2.1
  have c108994 := (publicBatchWrapper86_copies3406 a h).2.2.1
  have c108995 := (publicBatchWrapper86_copies3406 a h).2.2.2.1
  have c108996 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.1
  have c108997 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.1
  have c108998 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_16 := arithEq_of_rows h (row := 1784) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_16
  simp only [← c108993, ← c108994, ← c108995] at e_1784_16
  have e_1784_17 := arithEq_of_rows h (row := 1784) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_17
  simp only [← c108996, ← c108997, k2, ← c108998] at e_1784_17
  have hr := e_1784_17
  simp only [e_1784_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1784 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4791)) := by
  have c108999 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.1
  have c109000 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.1
  have c109001 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.1
  have c109002 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109003 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109004 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1784_18 := arithEq_of_rows h (row := 1784) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_18
  simp only [← c108999, ← c109000, ← c109001] at e_1784_18
  have e_1784_19 := arithEq_of_rows h (row := 1784) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_19
  simp only [← c109002, ← c109003, k2, ← c109004] at e_1784_19
  have hr := e_1784_19
  simp only [e_1784_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4792)) := by
  have c109005 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109006 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109007 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109008 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109009 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109010 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_0 := arithEq_of_rows h (row := 1785) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_0
  simp only [← c109005, ← c109006, ← c109007] at e_1785_0
  have e_1785_1 := arithEq_of_rows h (row := 1785) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_1
  simp only [← c109008, ← c109009, k2, ← c109010] at e_1785_1
  have hr := e_1785_1
  simp only [e_1785_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4793)) := by
  have c109011 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109012 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109013 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109014 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109015 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109016 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_2 := arithEq_of_rows h (row := 1785) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_2
  simp only [← c109011, ← c109012, ← c109013] at e_1785_2
  have e_1785_3 := arithEq_of_rows h (row := 1785) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_3
  simp only [← c109014, ← c109015, k2, ← c109016] at e_1785_3
  have hr := e_1785_3
  simp only [e_1785_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4794)) := by
  have c109017 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109018 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109019 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109020 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109021 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109022 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_4 := arithEq_of_rows h (row := 1785) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_4
  simp only [← c109017, ← c109018, ← c109019] at e_1785_4
  have e_1785_5 := arithEq_of_rows h (row := 1785) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_5
  simp only [← c109020, ← c109021, k2, ← c109022] at e_1785_5
  have hr := e_1785_5
  simp only [e_1785_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4795)) := by
  have c109023 := (publicBatchWrapper86_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109024 := (publicBatchWrapper86_copies3407 a h).1
  have c109025 := (publicBatchWrapper86_copies3407 a h).2.1
  have c109026 := (publicBatchWrapper86_copies3407 a h).2.2.1
  have c109027 := (publicBatchWrapper86_copies3407 a h).2.2.2.1
  have c109028 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_6 := arithEq_of_rows h (row := 1785) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_6
  simp only [← c109023, ← c109024, ← c109025] at e_1785_6
  have e_1785_7 := arithEq_of_rows h (row := 1785) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_7
  simp only [← c109026, ← c109027, k2, ← c109028] at e_1785_7
  have hr := e_1785_7
  simp only [e_1785_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4796)) := by
  have c109029 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.1
  have c109030 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.1
  have c109031 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.1
  have c109032 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.1
  have c109033 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.1
  have c109034 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_8 := arithEq_of_rows h (row := 1785) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_8
  simp only [← c109029, ← c109030, ← c109031] at e_1785_8
  have e_1785_9 := arithEq_of_rows h (row := 1785) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_9
  simp only [← c109032, ← c109033, k2, ← c109034] at e_1785_9
  have hr := e_1785_9
  simp only [e_1785_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4797)) := by
  have c109035 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109036 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109037 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109038 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109039 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109040 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_10 := arithEq_of_rows h (row := 1785) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_10
  simp only [← c109035, ← c109036, ← c109037] at e_1785_10
  have e_1785_11 := arithEq_of_rows h (row := 1785) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_11
  simp only [← c109038, ← c109039, k2, ← c109040] at e_1785_11
  have hr := e_1785_11
  simp only [e_1785_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4798)) := by
  have c109041 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109042 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109043 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109044 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109045 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109046 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_12 := arithEq_of_rows h (row := 1785) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_12
  simp only [← c109041, ← c109042, ← c109043] at e_1785_12
  have e_1785_13 := arithEq_of_rows h (row := 1785) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_13
  simp only [← c109044, ← c109045, k2, ← c109046] at e_1785_13
  have hr := e_1785_13
  simp only [e_1785_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4799)) := by
  have c109047 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109048 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109049 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109050 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109051 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109052 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_14 := arithEq_of_rows h (row := 1785) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_14
  simp only [← c109047, ← c109048, ← c109049] at e_1785_14
  have e_1785_15 := arithEq_of_rows h (row := 1785) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_15
  simp only [← c109050, ← c109051, k2, ← c109052] at e_1785_15
  have hr := e_1785_15
  simp only [e_1785_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4800)) := by
  have c109053 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109054 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109055 := (publicBatchWrapper86_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109056 := (publicBatchWrapper86_copies3408 a h).1
  have c109057 := (publicBatchWrapper86_copies3408 a h).2.1
  have c109058 := (publicBatchWrapper86_copies3408 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_16 := arithEq_of_rows h (row := 1785) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_16
  simp only [← c109053, ← c109054, ← c109055] at e_1785_16
  have e_1785_17 := arithEq_of_rows h (row := 1785) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_17
  simp only [← c109056, ← c109057, k2, ← c109058] at e_1785_17
  have hr := e_1785_17
  simp only [e_1785_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1785 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4801)) := by
  have c109059 := (publicBatchWrapper86_copies3408 a h).2.2.2.1
  have c109060 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.1
  have c109061 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.1
  have c109062 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.1
  have c109063 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.1
  have c109064 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1785_18 := arithEq_of_rows h (row := 1785) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_18
  simp only [← c109059, ← c109060, ← c109061] at e_1785_18
  have e_1785_19 := arithEq_of_rows h (row := 1785) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_19
  simp only [← c109062, ← c109063, k2, ← c109064] at e_1785_19
  have hr := e_1785_19
  simp only [e_1785_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 7) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4970)) := by
  have c109065 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.1
  have c109066 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109067 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109068 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109069 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109070 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_0 := arithEq_of_rows h (row := 1786) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_0
  simp only [← c109065, ← c109066, ← c109067] at e_1786_0
  have e_1786_1 := arithEq_of_rows h (row := 1786) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_1
  simp only [← c109068, ← c109069, k2, ← c109070] at e_1786_1
  have hr := e_1786_1
  simp only [e_1786_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 15) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4971)) := by
  have c109071 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109072 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109073 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109074 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109075 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109076 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_2 := arithEq_of_rows h (row := 1786) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_2
  simp only [← c109071, ← c109072, ← c109073] at e_1786_2
  have e_1786_3 := arithEq_of_rows h (row := 1786) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_3
  simp only [← c109074, ← c109075, k2, ← c109076] at e_1786_3
  have hr := e_1786_3
  simp only [e_1786_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 23) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4972)) := by
  have c109077 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109078 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109079 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109080 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109081 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109082 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_4 := arithEq_of_rows h (row := 1786) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_4
  simp only [← c109077, ← c109078, ← c109079] at e_1786_4
  have e_1786_5 := arithEq_of_rows h (row := 1786) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_5
  simp only [← c109080, ← c109081, k2, ← c109082] at e_1786_5
  have hr := e_1786_5
  simp only [e_1786_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 31) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4973)) := by
  have c109083 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109084 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109085 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109086 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109087 := (publicBatchWrapper86_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109088 := (publicBatchWrapper86_copies3409 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_6 := arithEq_of_rows h (row := 1786) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_6
  simp only [← c109083, ← c109084, ← c109085] at e_1786_6
  have e_1786_7 := arithEq_of_rows h (row := 1786) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_7
  simp only [← c109086, ← c109087, k2, ← c109088] at e_1786_7
  have hr := e_1786_7
  simp only [e_1786_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 39) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4974)) := by
  have c109089 := (publicBatchWrapper86_copies3409 a h).2.1
  have c109090 := (publicBatchWrapper86_copies3409 a h).2.2.1
  have c109091 := (publicBatchWrapper86_copies3409 a h).2.2.2.1
  have c109092 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.1
  have c109093 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.1
  have c109094 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_8 := arithEq_of_rows h (row := 1786) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_8
  simp only [← c109089, ← c109090, ← c109091] at e_1786_8
  have e_1786_9 := arithEq_of_rows h (row := 1786) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_9
  simp only [← c109092, ← c109093, k2, ← c109094] at e_1786_9
  have hr := e_1786_9
  simp only [e_1786_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 47) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4975)) := by
  have c109095 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.1
  have c109096 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.1
  have c109097 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.1
  have c109098 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109099 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109100 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_10 := arithEq_of_rows h (row := 1786) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_10
  simp only [← c109095, ← c109096, ← c109097] at e_1786_10
  have e_1786_11 := arithEq_of_rows h (row := 1786) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_11
  simp only [← c109098, ← c109099, k2, ← c109100] at e_1786_11
  have hr := e_1786_11
  simp only [e_1786_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 55) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4976)) := by
  have c109101 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109102 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109103 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109104 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109105 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109106 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_12 := arithEq_of_rows h (row := 1786) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_12
  simp only [← c109101, ← c109102, ← c109103] at e_1786_12
  have e_1786_13 := arithEq_of_rows h (row := 1786) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_13
  simp only [← c109104, ← c109105, k2, ← c109106] at e_1786_13
  have hr := e_1786_13
  simp only [e_1786_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 63) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4977)) := by
  have c109107 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109108 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109109 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109110 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109111 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109112 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_14 := arithEq_of_rows h (row := 1786) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_14
  simp only [← c109107, ← c109108, ← c109109] at e_1786_14
  have e_1786_15 := arithEq_of_rows h (row := 1786) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_15
  simp only [← c109110, ← c109111, k2, ← c109112] at e_1786_15
  have hr := e_1786_15
  simp only [e_1786_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 71) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4978)) := by
  have c109113 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109114 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109115 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109116 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109117 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109118 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_16 := arithEq_of_rows h (row := 1786) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_16
  simp only [← c109113, ← c109114, ← c109115] at e_1786_16
  have e_1786_17 := arithEq_of_rows h (row := 1786) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_17
  simp only [← c109116, ← c109117, k2, ← c109118] at e_1786_17
  have hr := e_1786_17
  simp only [e_1786_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1786 79) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4979)) := by
  have c109119 := (publicBatchWrapper86_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109120 := (publicBatchWrapper86_copies3410 a h).1
  have c109121 := (publicBatchWrapper86_copies3410 a h).2.1
  have c109122 := (publicBatchWrapper86_copies3410 a h).2.2.1
  have c109123 := (publicBatchWrapper86_copies3410 a h).2.2.2.1
  have c109124 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1786_18 := arithEq_of_rows h (row := 1786) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_18
  simp only [← c109119, ← c109120, ← c109121] at e_1786_18
  have e_1786_19 := arithEq_of_rows h (row := 1786) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_19
  simp only [← c109122, ← c109123, k2, ← c109124] at e_1786_19
  have hr := e_1786_19
  simp only [e_1786_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 7) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4980)) := by
  have c109125 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.1
  have c109126 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.1
  have c109127 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.1
  have c109128 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.1
  have c109129 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.1
  have c109130 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_0 := arithEq_of_rows h (row := 1787) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_0
  simp only [← c109125, ← c109126, ← c109127] at e_1787_0
  have e_1787_1 := arithEq_of_rows h (row := 1787) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_1
  simp only [← c109128, ← c109129, k2, ← c109130] at e_1787_1
  have hr := e_1787_1
  simp only [e_1787_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 15) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4981)) := by
  have c109131 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109132 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109133 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109134 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109135 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109136 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_2 := arithEq_of_rows h (row := 1787) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_2
  simp only [← c109131, ← c109132, ← c109133] at e_1787_2
  have e_1787_3 := arithEq_of_rows h (row := 1787) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_3
  simp only [← c109134, ← c109135, k2, ← c109136] at e_1787_3
  have hr := e_1787_3
  simp only [e_1787_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 23) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4982)) := by
  have c109137 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109138 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109139 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109140 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109141 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109142 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_4 := arithEq_of_rows h (row := 1787) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_4
  simp only [← c109137, ← c109138, ← c109139] at e_1787_4
  have e_1787_5 := arithEq_of_rows h (row := 1787) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_5
  simp only [← c109140, ← c109141, k2, ← c109142] at e_1787_5
  have hr := e_1787_5
  simp only [e_1787_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 31) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4983)) := by
  have c109143 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109144 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109145 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109146 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109147 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109148 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_6 := arithEq_of_rows h (row := 1787) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_6
  simp only [← c109143, ← c109144, ← c109145] at e_1787_6
  have e_1787_7 := arithEq_of_rows h (row := 1787) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_7
  simp only [← c109146, ← c109147, k2, ← c109148] at e_1787_7
  have hr := e_1787_7
  simp only [e_1787_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 39) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4984)) := by
  have c109149 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109150 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109151 := (publicBatchWrapper86_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109152 := (publicBatchWrapper86_copies3411 a h).1
  have c109153 := (publicBatchWrapper86_copies3411 a h).2.1
  have c109154 := (publicBatchWrapper86_copies3411 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_8 := arithEq_of_rows h (row := 1787) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_8
  simp only [← c109149, ← c109150, ← c109151] at e_1787_8
  have e_1787_9 := arithEq_of_rows h (row := 1787) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_9
  simp only [← c109152, ← c109153, k2, ← c109154] at e_1787_9
  have hr := e_1787_9
  simp only [e_1787_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 47) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4985)) := by
  have c109155 := (publicBatchWrapper86_copies3411 a h).2.2.2.1
  have c109156 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.1
  have c109157 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.1
  have c109158 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.1
  have c109159 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.1
  have c109160 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_10 := arithEq_of_rows h (row := 1787) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_10
  simp only [← c109155, ← c109156, ← c109157] at e_1787_10
  have e_1787_11 := arithEq_of_rows h (row := 1787) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_11
  simp only [← c109158, ← c109159, k2, ← c109160] at e_1787_11
  have hr := e_1787_11
  simp only [e_1787_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 55) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4986)) := by
  have c109161 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.1
  have c109162 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109163 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109164 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109165 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109166 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_12 := arithEq_of_rows h (row := 1787) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_12
  simp only [← c109161, ← c109162, ← c109163] at e_1787_12
  have e_1787_13 := arithEq_of_rows h (row := 1787) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_13
  simp only [← c109164, ← c109165, k2, ← c109166] at e_1787_13
  have hr := e_1787_13
  simp only [e_1787_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 63) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4987)) := by
  have c109167 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109168 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109169 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109170 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109171 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109172 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_14 := arithEq_of_rows h (row := 1787) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_14
  simp only [← c109167, ← c109168, ← c109169] at e_1787_14
  have e_1787_15 := arithEq_of_rows h (row := 1787) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_15
  simp only [← c109170, ← c109171, k2, ← c109172] at e_1787_15
  have hr := e_1787_15
  simp only [e_1787_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 71) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4988)) := by
  have c109173 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109174 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109175 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109176 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109177 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109178 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_16 := arithEq_of_rows h (row := 1787) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_16
  simp only [← c109173, ← c109174, ← c109175] at e_1787_16
  have e_1787_17 := arithEq_of_rows h (row := 1787) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_17
  simp only [← c109176, ← c109177, k2, ← c109178] at e_1787_17
  have hr := e_1787_17
  simp only [e_1787_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1787 79) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4989)) := by
  have c109179 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109180 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109181 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109182 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109183 := (publicBatchWrapper86_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109184 := (publicBatchWrapper86_copies3412 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1787_18 := arithEq_of_rows h (row := 1787) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_18
  simp only [← c109179, ← c109180, ← c109181] at e_1787_18
  have e_1787_19 := arithEq_of_rows h (row := 1787) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_19
  simp only [← c109182, ← c109183, k2, ← c109184] at e_1787_19
  have hr := e_1787_19
  simp only [e_1787_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 7) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4990)) := by
  have c109185 := (publicBatchWrapper86_copies3412 a h).2.1
  have c109186 := (publicBatchWrapper86_copies3412 a h).2.2.1
  have c109187 := (publicBatchWrapper86_copies3412 a h).2.2.2.1
  have c109188 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.1
  have c109189 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.1
  have c109190 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_0 := arithEq_of_rows h (row := 1788) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_0
  simp only [← c109185, ← c109186, ← c109187] at e_1788_0
  have e_1788_1 := arithEq_of_rows h (row := 1788) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_1
  simp only [← c109188, ← c109189, k2, ← c109190] at e_1788_1
  have hr := e_1788_1
  simp only [e_1788_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 15) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4991)) := by
  have c109191 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.1
  have c109192 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.1
  have c109193 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.1
  have c109194 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109195 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109196 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_2 := arithEq_of_rows h (row := 1788) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_2
  simp only [← c109191, ← c109192, ← c109193] at e_1788_2
  have e_1788_3 := arithEq_of_rows h (row := 1788) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_3
  simp only [← c109194, ← c109195, k2, ← c109196] at e_1788_3
  have hr := e_1788_3
  simp only [e_1788_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 23) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4992)) := by
  have c109197 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109198 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109199 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109200 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109201 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109202 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_4 := arithEq_of_rows h (row := 1788) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_4
  simp only [← c109197, ← c109198, ← c109199] at e_1788_4
  have e_1788_5 := arithEq_of_rows h (row := 1788) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_5
  simp only [← c109200, ← c109201, k2, ← c109202] at e_1788_5
  have hr := e_1788_5
  simp only [e_1788_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 31) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4993)) := by
  have c109203 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109204 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109205 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109206 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109207 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109208 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_6 := arithEq_of_rows h (row := 1788) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_6
  simp only [← c109203, ← c109204, ← c109205] at e_1788_6
  have e_1788_7 := arithEq_of_rows h (row := 1788) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_7
  simp only [← c109206, ← c109207, k2, ← c109208] at e_1788_7
  have hr := e_1788_7
  simp only [e_1788_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 39) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4994)) := by
  have c109209 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109210 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109211 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109212 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109213 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109214 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_8 := arithEq_of_rows h (row := 1788) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_8
  simp only [← c109209, ← c109210, ← c109211] at e_1788_8
  have e_1788_9 := arithEq_of_rows h (row := 1788) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_9
  simp only [← c109212, ← c109213, k2, ← c109214] at e_1788_9
  have hr := e_1788_9
  simp only [e_1788_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 47) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4995)) := by
  have c109215 := (publicBatchWrapper86_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109216 := (publicBatchWrapper86_copies3413 a h).1
  have c109217 := (publicBatchWrapper86_copies3413 a h).2.1
  have c109218 := (publicBatchWrapper86_copies3413 a h).2.2.1
  have c109219 := (publicBatchWrapper86_copies3413 a h).2.2.2.1
  have c109220 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_10 := arithEq_of_rows h (row := 1788) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_10
  simp only [← c109215, ← c109216, ← c109217] at e_1788_10
  have e_1788_11 := arithEq_of_rows h (row := 1788) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_11
  simp only [← c109218, ← c109219, k2, ← c109220] at e_1788_11
  have hr := e_1788_11
  simp only [e_1788_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 55) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4996)) := by
  have c109221 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.1
  have c109222 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.1
  have c109223 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.1
  have c109224 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.1
  have c109225 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.1
  have c109226 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_12 := arithEq_of_rows h (row := 1788) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_12
  simp only [← c109221, ← c109222, ← c109223] at e_1788_12
  have e_1788_13 := arithEq_of_rows h (row := 1788) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_13
  simp only [← c109224, ← c109225, k2, ← c109226] at e_1788_13
  have hr := e_1788_13
  simp only [e_1788_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 63) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4997)) := by
  have c109227 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109228 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109229 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109230 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109231 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109232 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_14 := arithEq_of_rows h (row := 1788) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_14
  simp only [← c109227, ← c109228, ← c109229] at e_1788_14
  have e_1788_15 := arithEq_of_rows h (row := 1788) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_15
  simp only [← c109230, ← c109231, k2, ← c109232] at e_1788_15
  have hr := e_1788_15
  simp only [e_1788_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 71) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4998)) := by
  have c109233 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109234 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109235 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109236 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109237 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109238 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_16 := arithEq_of_rows h (row := 1788) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_16
  simp only [← c109233, ← c109234, ← c109235] at e_1788_16
  have e_1788_17 := arithEq_of_rows h (row := 1788) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_17
  simp only [← c109236, ← c109237, k2, ← c109238] at e_1788_17
  have hr := e_1788_17
  simp only [e_1788_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 79) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4999)) := by
  have c109239 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109240 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109241 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109242 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109243 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109244 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_18 := arithEq_of_rows h (row := 1788) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_18
  simp only [← c109239, ← c109240, ← c109241] at e_1788_18
  have e_1788_19 := arithEq_of_rows h (row := 1788) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_19
  simp only [← c109242, ← c109243, k2, ← c109244] at e_1788_19
  have hr := e_1788_19
  simp only [e_1788_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 7) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 5000)) := by
  have c109245 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109246 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109247 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109248 := (publicBatchWrapper86_copies3414 a h).1
  have c109249 := (publicBatchWrapper86_copies3414 a h).2.1
  have c109250 := (publicBatchWrapper86_copies3414 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_0 := arithEq_of_rows h (row := 1789) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_0
  simp only [← c109245, ← c109246, ← c109247] at e_1789_0
  have e_1789_1 := arithEq_of_rows h (row := 1789) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_1
  simp only [← c109248, ← c109249, k2, ← c109250] at e_1789_1
  have hr := e_1789_1
  simp only [e_1789_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 15) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 5001)) := by
  have c109251 := (publicBatchWrapper86_copies3414 a h).2.2.2.1
  have c109252 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.1
  have c109253 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.1
  have c109254 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.1
  have c109255 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.1
  have c109256 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_2 := arithEq_of_rows h (row := 1789) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_2
  simp only [← c109251, ← c109252, ← c109253] at e_1789_2
  have e_1789_3 := arithEq_of_rows h (row := 1789) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_3
  simp only [← c109254, ← c109255, k2, ← c109256] at e_1789_3
  have hr := e_1789_3
  simp only [e_1789_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5170)) := by
  have c109257 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.1
  have c109258 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109259 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109260 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109261 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109262 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_4 := arithEq_of_rows h (row := 1789) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_4
  simp only [← c109257, ← c109258, ← c109259] at e_1789_4
  have e_1789_5 := arithEq_of_rows h (row := 1789) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_5
  simp only [← c109260, ← c109261, k2, ← c109262] at e_1789_5
  have hr := e_1789_5
  simp only [e_1789_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5171)) := by
  have c109263 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109264 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109265 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109266 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109267 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109268 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_6 := arithEq_of_rows h (row := 1789) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_6
  simp only [← c109263, ← c109264, ← c109265] at e_1789_6
  have e_1789_7 := arithEq_of_rows h (row := 1789) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_7
  simp only [← c109266, ← c109267, k2, ← c109268] at e_1789_7
  have hr := e_1789_7
  simp only [e_1789_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5172)) := by
  have c109269 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109270 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109271 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109272 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109273 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109274 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_8 := arithEq_of_rows h (row := 1789) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_8
  simp only [← c109269, ← c109270, ← c109271] at e_1789_8
  have e_1789_9 := arithEq_of_rows h (row := 1789) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_9
  simp only [← c109272, ← c109273, k2, ← c109274] at e_1789_9
  have hr := e_1789_9
  simp only [e_1789_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5173)) := by
  have c109275 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109276 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109277 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109278 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109279 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109280 := (publicBatchWrapper86_copies3415 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_10 := arithEq_of_rows h (row := 1789) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_10
  simp only [← c109275, ← c109276, ← c109277] at e_1789_10
  have e_1789_11 := arithEq_of_rows h (row := 1789) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_11
  simp only [← c109278, ← c109279, k2, ← c109280] at e_1789_11
  have hr := e_1789_11
  simp only [e_1789_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5174)) := by
  have c109281 := (publicBatchWrapper86_copies3415 a h).2.1
  have c109282 := (publicBatchWrapper86_copies3415 a h).2.2.1
  have c109283 := (publicBatchWrapper86_copies3415 a h).2.2.2.1
  have c109284 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.1
  have c109285 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.1
  have c109286 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_12 := arithEq_of_rows h (row := 1789) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_12
  simp only [← c109281, ← c109282, ← c109283] at e_1789_12
  have e_1789_13 := arithEq_of_rows h (row := 1789) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_13
  simp only [← c109284, ← c109285, k2, ← c109286] at e_1789_13
  have hr := e_1789_13
  simp only [e_1789_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5175)) := by
  have c109287 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.1
  have c109288 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.1
  have c109289 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.1
  have c109290 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109291 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109292 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_14 := arithEq_of_rows h (row := 1789) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_14
  simp only [← c109287, ← c109288, ← c109289] at e_1789_14
  have e_1789_15 := arithEq_of_rows h (row := 1789) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_15
  simp only [← c109290, ← c109291, k2, ← c109292] at e_1789_15
  have hr := e_1789_15
  simp only [e_1789_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5176)) := by
  have c109293 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109294 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109295 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109296 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109297 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109298 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_16 := arithEq_of_rows h (row := 1789) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_16
  simp only [← c109293, ← c109294, ← c109295] at e_1789_16
  have e_1789_17 := arithEq_of_rows h (row := 1789) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_17
  simp only [← c109296, ← c109297, k2, ← c109298] at e_1789_17
  have hr := e_1789_17
  simp only [e_1789_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5177)) := by
  have c109299 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109300 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109301 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109302 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109303 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109304 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_18 := arithEq_of_rows h (row := 1789) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_18
  simp only [← c109299, ← c109300, ← c109301] at e_1789_18
  have e_1789_19 := arithEq_of_rows h (row := 1789) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_19
  simp only [← c109302, ← c109303, k2, ← c109304] at e_1789_19
  have hr := e_1789_19
  simp only [e_1789_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5178)) := by
  have c109305 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109306 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109307 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109308 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109309 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109310 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_0 := arithEq_of_rows h (row := 1790) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_0
  simp only [← c109305, ← c109306, ← c109307] at e_1790_0
  have e_1790_1 := arithEq_of_rows h (row := 1790) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_1
  simp only [← c109308, ← c109309, k2, ← c109310] at e_1790_1
  have hr := e_1790_1
  simp only [e_1790_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5179)) := by
  have c109311 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109312 := (publicBatchWrapper86_copies3416 a h).1
  have c109313 := (publicBatchWrapper86_copies3416 a h).2.1
  have c109314 := (publicBatchWrapper86_copies3416 a h).2.2.1
  have c109315 := (publicBatchWrapper86_copies3416 a h).2.2.2.1
  have c109316 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_2 := arithEq_of_rows h (row := 1790) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_2
  simp only [← c109311, ← c109312, ← c109313] at e_1790_2
  have e_1790_3 := arithEq_of_rows h (row := 1790) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_3
  simp only [← c109314, ← c109315, k2, ← c109316] at e_1790_3
  have hr := e_1790_3
  simp only [e_1790_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5180)) := by
  have c109317 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.1
  have c109318 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.1
  have c109319 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.1
  have c109320 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.1
  have c109321 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.1
  have c109322 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_4 := arithEq_of_rows h (row := 1790) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_4
  simp only [← c109317, ← c109318, ← c109319] at e_1790_4
  have e_1790_5 := arithEq_of_rows h (row := 1790) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_5
  simp only [← c109320, ← c109321, k2, ← c109322] at e_1790_5
  have hr := e_1790_5
  simp only [e_1790_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5181)) := by
  have c109323 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109324 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109325 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109326 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109327 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109328 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_6 := arithEq_of_rows h (row := 1790) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_6
  simp only [← c109323, ← c109324, ← c109325] at e_1790_6
  have e_1790_7 := arithEq_of_rows h (row := 1790) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_7
  simp only [← c109326, ← c109327, k2, ← c109328] at e_1790_7
  have hr := e_1790_7
  simp only [e_1790_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5182)) := by
  have c109329 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109330 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109331 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109332 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109333 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109334 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_8 := arithEq_of_rows h (row := 1790) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_8
  simp only [← c109329, ← c109330, ← c109331] at e_1790_8
  have e_1790_9 := arithEq_of_rows h (row := 1790) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_9
  simp only [← c109332, ← c109333, k2, ← c109334] at e_1790_9
  have hr := e_1790_9
  simp only [e_1790_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5183)) := by
  have c109335 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109336 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109337 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109338 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109339 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109340 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_10 := arithEq_of_rows h (row := 1790) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_10
  simp only [← c109335, ← c109336, ← c109337] at e_1790_10
  have e_1790_11 := arithEq_of_rows h (row := 1790) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_11
  simp only [← c109338, ← c109339, k2, ← c109340] at e_1790_11
  have hr := e_1790_11
  simp only [e_1790_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5184)) := by
  have c109341 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109342 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109343 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109344 := (publicBatchWrapper86_copies3417 a h).1
  have c109345 := (publicBatchWrapper86_copies3417 a h).2.1
  have c109346 := (publicBatchWrapper86_copies3417 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_12 := arithEq_of_rows h (row := 1790) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_12
  simp only [← c109341, ← c109342, ← c109343] at e_1790_12
  have e_1790_13 := arithEq_of_rows h (row := 1790) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_13
  simp only [← c109344, ← c109345, k2, ← c109346] at e_1790_13
  have hr := e_1790_13
  simp only [e_1790_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5185)) := by
  have c109347 := (publicBatchWrapper86_copies3417 a h).2.2.2.1
  have c109348 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.1
  have c109349 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.1
  have c109350 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.1
  have c109351 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.1
  have c109352 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_14 := arithEq_of_rows h (row := 1790) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_14
  simp only [← c109347, ← c109348, ← c109349] at e_1790_14
  have e_1790_15 := arithEq_of_rows h (row := 1790) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_15
  simp only [← c109350, ← c109351, k2, ← c109352] at e_1790_15
  have hr := e_1790_15
  simp only [e_1790_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5186)) := by
  have c109353 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.1
  have c109354 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109355 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109356 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109357 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109358 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_16 := arithEq_of_rows h (row := 1790) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_16
  simp only [← c109353, ← c109354, ← c109355] at e_1790_16
  have e_1790_17 := arithEq_of_rows h (row := 1790) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_17
  simp only [← c109356, ← c109357, k2, ← c109358] at e_1790_17
  have hr := e_1790_17
  simp only [e_1790_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5187)) := by
  have c109359 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109360 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109361 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109362 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109363 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109364 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_18 := arithEq_of_rows h (row := 1790) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_18
  simp only [← c109359, ← c109360, ← c109361] at e_1790_18
  have e_1790_19 := arithEq_of_rows h (row := 1790) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_19
  simp only [← c109362, ← c109363, k2, ← c109364] at e_1790_19
  have hr := e_1790_19
  simp only [e_1790_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5188)) := by
  have c109365 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109366 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109367 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109368 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109369 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109370 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_0 := arithEq_of_rows h (row := 1791) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_0
  simp only [← c109365, ← c109366, ← c109367] at e_1791_0
  have e_1791_1 := arithEq_of_rows h (row := 1791) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_1
  simp only [← c109368, ← c109369, k2, ← c109370] at e_1791_1
  have hr := e_1791_1
  simp only [e_1791_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5189)) := by
  have c109371 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109372 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109373 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109374 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109375 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109376 := (publicBatchWrapper86_copies3418 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_2 := arithEq_of_rows h (row := 1791) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_2
  simp only [← c109371, ← c109372, ← c109373] at e_1791_2
  have e_1791_3 := arithEq_of_rows h (row := 1791) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_3
  simp only [← c109374, ← c109375, k2, ← c109376] at e_1791_3
  have hr := e_1791_3
  simp only [e_1791_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5190)) := by
  have c109377 := (publicBatchWrapper86_copies3418 a h).2.1
  have c109378 := (publicBatchWrapper86_copies3418 a h).2.2.1
  have c109379 := (publicBatchWrapper86_copies3418 a h).2.2.2.1
  have c109380 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.1
  have c109381 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.1
  have c109382 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_4 := arithEq_of_rows h (row := 1791) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_4
  simp only [← c109377, ← c109378, ← c109379] at e_1791_4
  have e_1791_5 := arithEq_of_rows h (row := 1791) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_5
  simp only [← c109380, ← c109381, k2, ← c109382] at e_1791_5
  have hr := e_1791_5
  simp only [e_1791_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5191)) := by
  have c109383 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.1
  have c109384 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.1
  have c109385 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.1
  have c109386 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109387 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109388 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_6 := arithEq_of_rows h (row := 1791) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_6
  simp only [← c109383, ← c109384, ← c109385] at e_1791_6
  have e_1791_7 := arithEq_of_rows h (row := 1791) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_7
  simp only [← c109386, ← c109387, k2, ← c109388] at e_1791_7
  have hr := e_1791_7
  simp only [e_1791_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5192)) := by
  have c109389 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109390 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109391 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109392 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109393 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109394 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_8 := arithEq_of_rows h (row := 1791) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_8
  simp only [← c109389, ← c109390, ← c109391] at e_1791_8
  have e_1791_9 := arithEq_of_rows h (row := 1791) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_9
  simp only [← c109392, ← c109393, k2, ← c109394] at e_1791_9
  have hr := e_1791_9
  simp only [e_1791_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5193)) := by
  have c109395 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109396 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109397 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109398 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109399 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109400 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_10 := arithEq_of_rows h (row := 1791) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_10
  simp only [← c109395, ← c109396, ← c109397] at e_1791_10
  have e_1791_11 := arithEq_of_rows h (row := 1791) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_11
  simp only [← c109398, ← c109399, k2, ← c109400] at e_1791_11
  have hr := e_1791_11
  simp only [e_1791_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5194)) := by
  have c109401 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109402 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109403 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109404 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109405 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109406 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_12 := arithEq_of_rows h (row := 1791) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_12
  simp only [← c109401, ← c109402, ← c109403] at e_1791_12
  have e_1791_13 := arithEq_of_rows h (row := 1791) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_13
  simp only [← c109404, ← c109405, k2, ← c109406] at e_1791_13
  have hr := e_1791_13
  simp only [e_1791_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5195)) := by
  have c109407 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109408 := (publicBatchWrapper86_copies3419 a h).1
  have c109409 := (publicBatchWrapper86_copies3419 a h).2.1
  have c109410 := (publicBatchWrapper86_copies3419 a h).2.2.1
  have c109411 := (publicBatchWrapper86_copies3419 a h).2.2.2.1
  have c109412 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_14 := arithEq_of_rows h (row := 1791) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_14
  simp only [← c109407, ← c109408, ← c109409] at e_1791_14
  have e_1791_15 := arithEq_of_rows h (row := 1791) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_15
  simp only [← c109410, ← c109411, k2, ← c109412] at e_1791_15
  have hr := e_1791_15
  simp only [e_1791_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5196)) := by
  have c109413 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.1
  have c109414 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.1
  have c109415 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.1
  have c109416 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.1
  have c109417 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.1
  have c109418 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_16 := arithEq_of_rows h (row := 1791) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_16
  simp only [← c109413, ← c109414, ← c109415] at e_1791_16
  have e_1791_17 := arithEq_of_rows h (row := 1791) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_17
  simp only [← c109416, ← c109417, k2, ← c109418] at e_1791_17
  have hr := e_1791_17
  simp only [e_1791_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5197)) := by
  have c109419 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109420 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109421 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109422 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109423 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109424 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_18 := arithEq_of_rows h (row := 1791) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_18
  simp only [← c109419, ← c109420, ← c109421] at e_1791_18
  have e_1791_19 := arithEq_of_rows h (row := 1791) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_19
  simp only [← c109422, ← c109423, k2, ← c109424] at e_1791_19
  have hr := e_1791_19
  simp only [e_1791_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5198)) := by
  have c109425 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109426 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109427 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109428 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109429 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109430 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_0 := arithEq_of_rows h (row := 1792) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_0
  simp only [← c109425, ← c109426, ← c109427] at e_1792_0
  have e_1792_1 := arithEq_of_rows h (row := 1792) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_1
  simp only [← c109428, ← c109429, k2, ← c109430] at e_1792_1
  have hr := e_1792_1
  simp only [e_1792_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5199)) := by
  have c109431 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109432 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109433 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109434 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109435 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109436 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_2 := arithEq_of_rows h (row := 1792) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_2
  simp only [← c109431, ← c109432, ← c109433] at e_1792_2
  have e_1792_3 := arithEq_of_rows h (row := 1792) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_3
  simp only [← c109434, ← c109435, k2, ← c109436] at e_1792_3
  have hr := e_1792_3
  simp only [e_1792_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5200)) := by
  have c109437 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109438 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109439 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109440 := (publicBatchWrapper86_copies3420 a h).1
  have c109441 := (publicBatchWrapper86_copies3420 a h).2.1
  have c109442 := (publicBatchWrapper86_copies3420 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_4 := arithEq_of_rows h (row := 1792) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_4
  simp only [← c109437, ← c109438, ← c109439] at e_1792_4
  have e_1792_5 := arithEq_of_rows h (row := 1792) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_5
  simp only [← c109440, ← c109441, k2, ← c109442] at e_1792_5
  have hr := e_1792_5
  simp only [e_1792_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5201)) := by
  have c109443 := (publicBatchWrapper86_copies3420 a h).2.2.2.1
  have c109444 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.1
  have c109445 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.1
  have c109446 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.1
  have c109447 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.1
  have c109448 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_6 := arithEq_of_rows h (row := 1792) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_6
  simp only [← c109443, ← c109444, ← c109445] at e_1792_6
  have e_1792_7 := arithEq_of_rows h (row := 1792) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_7
  simp only [← c109446, ← c109447, k2, ← c109448] at e_1792_7
  have hr := e_1792_7
  simp only [e_1792_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5370)) := by
  have c109449 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.1
  have c109450 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109451 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109452 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109453 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109454 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_8 := arithEq_of_rows h (row := 1792) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_8
  simp only [← c109449, ← c109450, ← c109451] at e_1792_8
  have e_1792_9 := arithEq_of_rows h (row := 1792) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_9
  simp only [← c109452, ← c109453, k2, ← c109454] at e_1792_9
  have hr := e_1792_9
  simp only [e_1792_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5371)) := by
  have c109455 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109456 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109457 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109458 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109459 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109460 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_10 := arithEq_of_rows h (row := 1792) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_10
  simp only [← c109455, ← c109456, ← c109457] at e_1792_10
  have e_1792_11 := arithEq_of_rows h (row := 1792) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_11
  simp only [← c109458, ← c109459, k2, ← c109460] at e_1792_11
  have hr := e_1792_11
  simp only [e_1792_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5372)) := by
  have c109461 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109462 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109463 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109464 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109465 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109466 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_12 := arithEq_of_rows h (row := 1792) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_12
  simp only [← c109461, ← c109462, ← c109463] at e_1792_12
  have e_1792_13 := arithEq_of_rows h (row := 1792) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_13
  simp only [← c109464, ← c109465, k2, ← c109466] at e_1792_13
  have hr := e_1792_13
  simp only [e_1792_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5373)) := by
  have c109467 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109468 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109469 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109470 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109471 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109472 := (publicBatchWrapper86_copies3421 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_14 := arithEq_of_rows h (row := 1792) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_14
  simp only [← c109467, ← c109468, ← c109469] at e_1792_14
  have e_1792_15 := arithEq_of_rows h (row := 1792) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_15
  simp only [← c109470, ← c109471, k2, ← c109472] at e_1792_15
  have hr := e_1792_15
  simp only [e_1792_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5374)) := by
  have c109473 := (publicBatchWrapper86_copies3421 a h).2.1
  have c109474 := (publicBatchWrapper86_copies3421 a h).2.2.1
  have c109475 := (publicBatchWrapper86_copies3421 a h).2.2.2.1
  have c109476 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.1
  have c109477 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.1
  have c109478 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_16 := arithEq_of_rows h (row := 1792) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_16
  simp only [← c109473, ← c109474, ← c109475] at e_1792_16
  have e_1792_17 := arithEq_of_rows h (row := 1792) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_17
  simp only [← c109476, ← c109477, k2, ← c109478] at e_1792_17
  have hr := e_1792_17
  simp only [e_1792_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5375)) := by
  have c109479 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.1
  have c109480 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.1
  have c109481 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.1
  have c109482 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109483 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109484 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_18 := arithEq_of_rows h (row := 1792) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_18
  simp only [← c109479, ← c109480, ← c109481] at e_1792_18
  have e_1792_19 := arithEq_of_rows h (row := 1792) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_19
  simp only [← c109482, ← c109483, k2, ← c109484] at e_1792_19
  have hr := e_1792_19
  simp only [e_1792_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5376)) := by
  have c109485 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109486 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109487 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109488 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109489 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109490 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_0 := arithEq_of_rows h (row := 1793) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_0
  simp only [← c109485, ← c109486, ← c109487] at e_1793_0
  have e_1793_1 := arithEq_of_rows h (row := 1793) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_1
  simp only [← c109488, ← c109489, k2, ← c109490] at e_1793_1
  have hr := e_1793_1
  simp only [e_1793_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5377)) := by
  have c109491 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109492 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109493 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109494 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109495 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109496 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_2 := arithEq_of_rows h (row := 1793) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_2
  simp only [← c109491, ← c109492, ← c109493] at e_1793_2
  have e_1793_3 := arithEq_of_rows h (row := 1793) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_3
  simp only [← c109494, ← c109495, k2, ← c109496] at e_1793_3
  have hr := e_1793_3
  simp only [e_1793_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5378)) := by
  have c109497 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109498 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109499 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109500 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109501 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109502 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_4 := arithEq_of_rows h (row := 1793) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_4
  simp only [← c109497, ← c109498, ← c109499] at e_1793_4
  have e_1793_5 := arithEq_of_rows h (row := 1793) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_5
  simp only [← c109500, ← c109501, k2, ← c109502] at e_1793_5
  have hr := e_1793_5
  simp only [e_1793_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5379)) := by
  have c109503 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109504 := (publicBatchWrapper86_copies3422 a h).1
  have c109505 := (publicBatchWrapper86_copies3422 a h).2.1
  have c109506 := (publicBatchWrapper86_copies3422 a h).2.2.1
  have c109507 := (publicBatchWrapper86_copies3422 a h).2.2.2.1
  have c109508 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_6 := arithEq_of_rows h (row := 1793) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_6
  simp only [← c109503, ← c109504, ← c109505] at e_1793_6
  have e_1793_7 := arithEq_of_rows h (row := 1793) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_7
  simp only [← c109506, ← c109507, k2, ← c109508] at e_1793_7
  have hr := e_1793_7
  simp only [e_1793_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5380)) := by
  have c109509 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.1
  have c109510 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.1
  have c109511 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.1
  have c109512 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.1
  have c109513 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.1
  have c109514 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_8 := arithEq_of_rows h (row := 1793) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_8
  simp only [← c109509, ← c109510, ← c109511] at e_1793_8
  have e_1793_9 := arithEq_of_rows h (row := 1793) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_9
  simp only [← c109512, ← c109513, k2, ← c109514] at e_1793_9
  have hr := e_1793_9
  simp only [e_1793_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5381)) := by
  have c109515 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109516 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109517 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109518 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109519 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109520 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_10 := arithEq_of_rows h (row := 1793) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_10
  simp only [← c109515, ← c109516, ← c109517] at e_1793_10
  have e_1793_11 := arithEq_of_rows h (row := 1793) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_11
  simp only [← c109518, ← c109519, k2, ← c109520] at e_1793_11
  have hr := e_1793_11
  simp only [e_1793_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5382)) := by
  have c109521 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109522 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109523 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109524 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109525 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109526 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_12 := arithEq_of_rows h (row := 1793) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_12
  simp only [← c109521, ← c109522, ← c109523] at e_1793_12
  have e_1793_13 := arithEq_of_rows h (row := 1793) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_13
  simp only [← c109524, ← c109525, k2, ← c109526] at e_1793_13
  have hr := e_1793_13
  simp only [e_1793_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5383)) := by
  have c109527 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109528 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109529 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109530 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109531 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109532 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_14 := arithEq_of_rows h (row := 1793) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_14
  simp only [← c109527, ← c109528, ← c109529] at e_1793_14
  have e_1793_15 := arithEq_of_rows h (row := 1793) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_15
  simp only [← c109530, ← c109531, k2, ← c109532] at e_1793_15
  have hr := e_1793_15
  simp only [e_1793_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5384)) := by
  have c109533 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109534 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109535 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109536 := (publicBatchWrapper86_copies3423 a h).1
  have c109537 := (publicBatchWrapper86_copies3423 a h).2.1
  have c109538 := (publicBatchWrapper86_copies3423 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_16 := arithEq_of_rows h (row := 1793) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_16
  simp only [← c109533, ← c109534, ← c109535] at e_1793_16
  have e_1793_17 := arithEq_of_rows h (row := 1793) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_17
  simp only [← c109536, ← c109537, k2, ← c109538] at e_1793_17
  have hr := e_1793_17
  simp only [e_1793_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5385)) := by
  have c109539 := (publicBatchWrapper86_copies3423 a h).2.2.2.1
  have c109540 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.1
  have c109541 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.1
  have c109542 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.1
  have c109543 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.1
  have c109544 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_18 := arithEq_of_rows h (row := 1793) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_18
  simp only [← c109539, ← c109540, ← c109541] at e_1793_18
  have e_1793_19 := arithEq_of_rows h (row := 1793) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_19
  simp only [← c109542, ← c109543, k2, ← c109544] at e_1793_19
  have hr := e_1793_19
  simp only [e_1793_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5386)) := by
  have c109545 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.1
  have c109546 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109547 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109548 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109549 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109550 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_0 := arithEq_of_rows h (row := 1794) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_0
  simp only [← c109545, ← c109546, ← c109547] at e_1794_0
  have e_1794_1 := arithEq_of_rows h (row := 1794) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_1
  simp only [← c109548, ← c109549, k2, ← c109550] at e_1794_1
  have hr := e_1794_1
  simp only [e_1794_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5387)) := by
  have c109551 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109552 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109553 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109554 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109555 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109556 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_2 := arithEq_of_rows h (row := 1794) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_2
  simp only [← c109551, ← c109552, ← c109553] at e_1794_2
  have e_1794_3 := arithEq_of_rows h (row := 1794) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_3
  simp only [← c109554, ← c109555, k2, ← c109556] at e_1794_3
  have hr := e_1794_3
  simp only [e_1794_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5388)) := by
  have c109557 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109558 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109559 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109560 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109561 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109562 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_4 := arithEq_of_rows h (row := 1794) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_4
  simp only [← c109557, ← c109558, ← c109559] at e_1794_4
  have e_1794_5 := arithEq_of_rows h (row := 1794) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_5
  simp only [← c109560, ← c109561, k2, ← c109562] at e_1794_5
  have hr := e_1794_5
  simp only [e_1794_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5389)) := by
  have c109563 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109564 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109565 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109566 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109567 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109568 := (publicBatchWrapper86_copies3424 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_6 := arithEq_of_rows h (row := 1794) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_6
  simp only [← c109563, ← c109564, ← c109565] at e_1794_6
  have e_1794_7 := arithEq_of_rows h (row := 1794) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_7
  simp only [← c109566, ← c109567, k2, ← c109568] at e_1794_7
  have hr := e_1794_7
  simp only [e_1794_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5390)) := by
  have c109569 := (publicBatchWrapper86_copies3424 a h).2.1
  have c109570 := (publicBatchWrapper86_copies3424 a h).2.2.1
  have c109571 := (publicBatchWrapper86_copies3424 a h).2.2.2.1
  have c109572 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.1
  have c109573 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.1
  have c109574 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_8 := arithEq_of_rows h (row := 1794) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_8
  simp only [← c109569, ← c109570, ← c109571] at e_1794_8
  have e_1794_9 := arithEq_of_rows h (row := 1794) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_9
  simp only [← c109572, ← c109573, k2, ← c109574] at e_1794_9
  have hr := e_1794_9
  simp only [e_1794_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5391)) := by
  have c109575 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.1
  have c109576 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.1
  have c109577 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.1
  have c109578 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109579 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109580 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_10 := arithEq_of_rows h (row := 1794) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_10
  simp only [← c109575, ← c109576, ← c109577] at e_1794_10
  have e_1794_11 := arithEq_of_rows h (row := 1794) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_11
  simp only [← c109578, ← c109579, k2, ← c109580] at e_1794_11
  have hr := e_1794_11
  simp only [e_1794_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5392)) := by
  have c109581 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109582 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109583 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109584 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109585 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109586 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_12 := arithEq_of_rows h (row := 1794) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_12
  simp only [← c109581, ← c109582, ← c109583] at e_1794_12
  have e_1794_13 := arithEq_of_rows h (row := 1794) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_13
  simp only [← c109584, ← c109585, k2, ← c109586] at e_1794_13
  have hr := e_1794_13
  simp only [e_1794_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5393)) := by
  have c109587 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109588 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109589 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109590 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109591 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109592 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_14 := arithEq_of_rows h (row := 1794) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_14
  simp only [← c109587, ← c109588, ← c109589] at e_1794_14
  have e_1794_15 := arithEq_of_rows h (row := 1794) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_15
  simp only [← c109590, ← c109591, k2, ← c109592] at e_1794_15
  have hr := e_1794_15
  simp only [e_1794_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5394)) := by
  have c109593 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109594 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109595 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109596 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109597 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109598 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_16 := arithEq_of_rows h (row := 1794) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_16
  simp only [← c109593, ← c109594, ← c109595] at e_1794_16
  have e_1794_17 := arithEq_of_rows h (row := 1794) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_17
  simp only [← c109596, ← c109597, k2, ← c109598] at e_1794_17
  have hr := e_1794_17
  simp only [e_1794_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5395)) := by
  have c109599 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109600 := (publicBatchWrapper86_copies3425 a h).1
  have c109601 := (publicBatchWrapper86_copies3425 a h).2.1
  have c109602 := (publicBatchWrapper86_copies3425 a h).2.2.1
  have c109603 := (publicBatchWrapper86_copies3425 a h).2.2.2.1
  have c109604 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_18 := arithEq_of_rows h (row := 1794) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_18
  simp only [← c109599, ← c109600, ← c109601] at e_1794_18
  have e_1794_19 := arithEq_of_rows h (row := 1794) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_19
  simp only [← c109602, ← c109603, k2, ← c109604] at e_1794_19
  have hr := e_1794_19
  simp only [e_1794_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5396)) := by
  have c109605 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.1
  have c109606 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.1
  have c109607 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.1
  have c109608 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.1
  have c109609 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.1
  have c109610 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_0 := arithEq_of_rows h (row := 1795) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_0
  simp only [← c109605, ← c109606, ← c109607] at e_1795_0
  have e_1795_1 := arithEq_of_rows h (row := 1795) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_1
  simp only [← c109608, ← c109609, k2, ← c109610] at e_1795_1
  have hr := e_1795_1
  simp only [e_1795_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5397)) := by
  have c109611 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109612 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109613 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109614 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109615 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109616 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_2 := arithEq_of_rows h (row := 1795) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_2
  simp only [← c109611, ← c109612, ← c109613] at e_1795_2
  have e_1795_3 := arithEq_of_rows h (row := 1795) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_3
  simp only [← c109614, ← c109615, k2, ← c109616] at e_1795_3
  have hr := e_1795_3
  simp only [e_1795_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5398)) := by
  have c109617 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109618 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109619 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109620 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109621 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109622 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_4 := arithEq_of_rows h (row := 1795) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_4
  simp only [← c109617, ← c109618, ← c109619] at e_1795_4
  have e_1795_5 := arithEq_of_rows h (row := 1795) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_5
  simp only [← c109620, ← c109621, k2, ← c109622] at e_1795_5
  have hr := e_1795_5
  simp only [e_1795_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5399)) := by
  have c109623 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109624 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109625 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109626 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109627 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109628 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_6 := arithEq_of_rows h (row := 1795) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_6
  simp only [← c109623, ← c109624, ← c109625] at e_1795_6
  have e_1795_7 := arithEq_of_rows h (row := 1795) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_7
  simp only [← c109626, ← c109627, k2, ← c109628] at e_1795_7
  have hr := e_1795_7
  simp only [e_1795_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5400)) := by
  have c109629 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109630 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109631 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109632 := (publicBatchWrapper86_copies3426 a h).1
  have c109633 := (publicBatchWrapper86_copies3426 a h).2.1
  have c109634 := (publicBatchWrapper86_copies3426 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_8 := arithEq_of_rows h (row := 1795) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_8
  simp only [← c109629, ← c109630, ← c109631] at e_1795_8
  have e_1795_9 := arithEq_of_rows h (row := 1795) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_9
  simp only [← c109632, ← c109633, k2, ← c109634] at e_1795_9
  have hr := e_1795_9
  simp only [e_1795_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5401)) := by
  have c109635 := (publicBatchWrapper86_copies3426 a h).2.2.2.1
  have c109636 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.1
  have c109637 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.1
  have c109638 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.1
  have c109639 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.1
  have c109640 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_10 := arithEq_of_rows h (row := 1795) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_10
  simp only [← c109635, ← c109636, ← c109637] at e_1795_10
  have e_1795_11 := arithEq_of_rows h (row := 1795) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_11
  simp only [← c109638, ← c109639, k2, ← c109640] at e_1795_11
  have hr := e_1795_11
  simp only [e_1795_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5570)) := by
  have c109641 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.1
  have c109642 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109643 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109644 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109645 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109646 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_12 := arithEq_of_rows h (row := 1795) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_12
  simp only [← c109641, ← c109642, ← c109643] at e_1795_12
  have e_1795_13 := arithEq_of_rows h (row := 1795) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_13
  simp only [← c109644, ← c109645, k2, ← c109646] at e_1795_13
  have hr := e_1795_13
  simp only [e_1795_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5571)) := by
  have c109647 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109648 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109649 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109650 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109651 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109652 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_14 := arithEq_of_rows h (row := 1795) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_14
  simp only [← c109647, ← c109648, ← c109649] at e_1795_14
  have e_1795_15 := arithEq_of_rows h (row := 1795) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_15
  simp only [← c109650, ← c109651, k2, ← c109652] at e_1795_15
  have hr := e_1795_15
  simp only [e_1795_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5572)) := by
  have c109653 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109654 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109655 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109656 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109657 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109658 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_16 := arithEq_of_rows h (row := 1795) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_16
  simp only [← c109653, ← c109654, ← c109655] at e_1795_16
  have e_1795_17 := arithEq_of_rows h (row := 1795) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_17
  simp only [← c109656, ← c109657, k2, ← c109658] at e_1795_17
  have hr := e_1795_17
  simp only [e_1795_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5573)) := by
  have c109659 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109660 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109661 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109662 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109663 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109664 := (publicBatchWrapper86_copies3427 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_18 := arithEq_of_rows h (row := 1795) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_18
  simp only [← c109659, ← c109660, ← c109661] at e_1795_18
  have e_1795_19 := arithEq_of_rows h (row := 1795) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_19
  simp only [← c109662, ← c109663, k2, ← c109664] at e_1795_19
  have hr := e_1795_19
  simp only [e_1795_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5574)) := by
  have c109665 := (publicBatchWrapper86_copies3427 a h).2.1
  have c109666 := (publicBatchWrapper86_copies3427 a h).2.2.1
  have c109667 := (publicBatchWrapper86_copies3427 a h).2.2.2.1
  have c109668 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.1
  have c109669 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.1
  have c109670 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_0 := arithEq_of_rows h (row := 1796) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_0
  simp only [← c109665, ← c109666, ← c109667] at e_1796_0
  have e_1796_1 := arithEq_of_rows h (row := 1796) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_1
  simp only [← c109668, ← c109669, k2, ← c109670] at e_1796_1
  have hr := e_1796_1
  simp only [e_1796_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5575)) := by
  have c109671 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.1
  have c109672 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.1
  have c109673 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.1
  have c109674 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109675 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109676 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_2 := arithEq_of_rows h (row := 1796) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_2
  simp only [← c109671, ← c109672, ← c109673] at e_1796_2
  have e_1796_3 := arithEq_of_rows h (row := 1796) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_3
  simp only [← c109674, ← c109675, k2, ← c109676] at e_1796_3
  have hr := e_1796_3
  simp only [e_1796_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5576)) := by
  have c109677 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109678 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109679 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109680 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109681 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109682 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_4 := arithEq_of_rows h (row := 1796) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_4
  simp only [← c109677, ← c109678, ← c109679] at e_1796_4
  have e_1796_5 := arithEq_of_rows h (row := 1796) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_5
  simp only [← c109680, ← c109681, k2, ← c109682] at e_1796_5
  have hr := e_1796_5
  simp only [e_1796_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5577)) := by
  have c109683 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109684 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109685 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109686 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109687 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109688 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_6 := arithEq_of_rows h (row := 1796) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_6
  simp only [← c109683, ← c109684, ← c109685] at e_1796_6
  have e_1796_7 := arithEq_of_rows h (row := 1796) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_7
  simp only [← c109686, ← c109687, k2, ← c109688] at e_1796_7
  have hr := e_1796_7
  simp only [e_1796_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5578)) := by
  have c109689 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109690 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109691 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109692 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109693 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109694 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_8 := arithEq_of_rows h (row := 1796) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_8
  simp only [← c109689, ← c109690, ← c109691] at e_1796_8
  have e_1796_9 := arithEq_of_rows h (row := 1796) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_9
  simp only [← c109692, ← c109693, k2, ← c109694] at e_1796_9
  have hr := e_1796_9
  simp only [e_1796_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5579)) := by
  have c109695 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109696 := (publicBatchWrapper86_copies3428 a h).1
  have c109697 := (publicBatchWrapper86_copies3428 a h).2.1
  have c109698 := (publicBatchWrapper86_copies3428 a h).2.2.1
  have c109699 := (publicBatchWrapper86_copies3428 a h).2.2.2.1
  have c109700 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_10 := arithEq_of_rows h (row := 1796) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_10
  simp only [← c109695, ← c109696, ← c109697] at e_1796_10
  have e_1796_11 := arithEq_of_rows h (row := 1796) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_11
  simp only [← c109698, ← c109699, k2, ← c109700] at e_1796_11
  have hr := e_1796_11
  simp only [e_1796_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5580)) := by
  have c109701 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.1
  have c109702 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.1
  have c109703 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.1
  have c109704 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.1
  have c109705 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.1
  have c109706 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_12 := arithEq_of_rows h (row := 1796) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_12
  simp only [← c109701, ← c109702, ← c109703] at e_1796_12
  have e_1796_13 := arithEq_of_rows h (row := 1796) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_13
  simp only [← c109704, ← c109705, k2, ← c109706] at e_1796_13
  have hr := e_1796_13
  simp only [e_1796_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5581)) := by
  have c109707 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109708 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109709 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109710 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109711 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109712 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_14 := arithEq_of_rows h (row := 1796) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_14
  simp only [← c109707, ← c109708, ← c109709] at e_1796_14
  have e_1796_15 := arithEq_of_rows h (row := 1796) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_15
  simp only [← c109710, ← c109711, k2, ← c109712] at e_1796_15
  have hr := e_1796_15
  simp only [e_1796_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5582)) := by
  have c109713 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109714 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109715 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109716 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109717 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109718 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_16 := arithEq_of_rows h (row := 1796) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_16
  simp only [← c109713, ← c109714, ← c109715] at e_1796_16
  have e_1796_17 := arithEq_of_rows h (row := 1796) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_17
  simp only [← c109716, ← c109717, k2, ← c109718] at e_1796_17
  have hr := e_1796_17
  simp only [e_1796_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5583)) := by
  have c109719 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109720 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109721 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109722 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109723 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109724 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_18 := arithEq_of_rows h (row := 1796) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_18
  simp only [← c109719, ← c109720, ← c109721] at e_1796_18
  have e_1796_19 := arithEq_of_rows h (row := 1796) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_19
  simp only [← c109722, ← c109723, k2, ← c109724] at e_1796_19
  have hr := e_1796_19
  simp only [e_1796_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5584)) := by
  have c109725 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109726 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109727 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109728 := (publicBatchWrapper86_copies3429 a h).1
  have c109729 := (publicBatchWrapper86_copies3429 a h).2.1
  have c109730 := (publicBatchWrapper86_copies3429 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_0 := arithEq_of_rows h (row := 1797) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_0
  simp only [← c109725, ← c109726, ← c109727] at e_1797_0
  have e_1797_1 := arithEq_of_rows h (row := 1797) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_1
  simp only [← c109728, ← c109729, k2, ← c109730] at e_1797_1
  have hr := e_1797_1
  simp only [e_1797_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5585)) := by
  have c109731 := (publicBatchWrapper86_copies3429 a h).2.2.2.1
  have c109732 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.1
  have c109733 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.1
  have c109734 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.1
  have c109735 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.1
  have c109736 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_2 := arithEq_of_rows h (row := 1797) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_2
  simp only [← c109731, ← c109732, ← c109733] at e_1797_2
  have e_1797_3 := arithEq_of_rows h (row := 1797) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_3
  simp only [← c109734, ← c109735, k2, ← c109736] at e_1797_3
  have hr := e_1797_3
  simp only [e_1797_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5586)) := by
  have c109737 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.1
  have c109738 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109739 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109740 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109741 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109742 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_4 := arithEq_of_rows h (row := 1797) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_4
  simp only [← c109737, ← c109738, ← c109739] at e_1797_4
  have e_1797_5 := arithEq_of_rows h (row := 1797) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_5
  simp only [← c109740, ← c109741, k2, ← c109742] at e_1797_5
  have hr := e_1797_5
  simp only [e_1797_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5587)) := by
  have c109743 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109744 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109745 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109746 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109747 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109748 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_6 := arithEq_of_rows h (row := 1797) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_6
  simp only [← c109743, ← c109744, ← c109745] at e_1797_6
  have e_1797_7 := arithEq_of_rows h (row := 1797) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_7
  simp only [← c109746, ← c109747, k2, ← c109748] at e_1797_7
  have hr := e_1797_7
  simp only [e_1797_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5588)) := by
  have c109749 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109750 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109751 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109752 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109753 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109754 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_8 := arithEq_of_rows h (row := 1797) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_8
  simp only [← c109749, ← c109750, ← c109751] at e_1797_8
  have e_1797_9 := arithEq_of_rows h (row := 1797) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_9
  simp only [← c109752, ← c109753, k2, ← c109754] at e_1797_9
  have hr := e_1797_9
  simp only [e_1797_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5589)) := by
  have c109755 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109756 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109757 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109758 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109759 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109760 := (publicBatchWrapper86_copies3430 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_10 := arithEq_of_rows h (row := 1797) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_10
  simp only [← c109755, ← c109756, ← c109757] at e_1797_10
  have e_1797_11 := arithEq_of_rows h (row := 1797) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_11
  simp only [← c109758, ← c109759, k2, ← c109760] at e_1797_11
  have hr := e_1797_11
  simp only [e_1797_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5590)) := by
  have c109761 := (publicBatchWrapper86_copies3430 a h).2.1
  have c109762 := (publicBatchWrapper86_copies3430 a h).2.2.1
  have c109763 := (publicBatchWrapper86_copies3430 a h).2.2.2.1
  have c109764 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.1
  have c109765 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.1
  have c109766 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_12 := arithEq_of_rows h (row := 1797) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_12
  simp only [← c109761, ← c109762, ← c109763] at e_1797_12
  have e_1797_13 := arithEq_of_rows h (row := 1797) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_13
  simp only [← c109764, ← c109765, k2, ← c109766] at e_1797_13
  have hr := e_1797_13
  simp only [e_1797_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5591)) := by
  have c109767 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.1
  have c109768 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.1
  have c109769 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.1
  have c109770 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109771 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109772 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_14 := arithEq_of_rows h (row := 1797) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_14
  simp only [← c109767, ← c109768, ← c109769] at e_1797_14
  have e_1797_15 := arithEq_of_rows h (row := 1797) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_15
  simp only [← c109770, ← c109771, k2, ← c109772] at e_1797_15
  have hr := e_1797_15
  simp only [e_1797_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5592)) := by
  have c109773 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109774 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109775 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109776 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109777 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109778 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_16 := arithEq_of_rows h (row := 1797) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_16
  simp only [← c109773, ← c109774, ← c109775] at e_1797_16
  have e_1797_17 := arithEq_of_rows h (row := 1797) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_17
  simp only [← c109776, ← c109777, k2, ← c109778] at e_1797_17
  have hr := e_1797_17
  simp only [e_1797_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5593)) := by
  have c109779 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109780 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109781 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109782 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109783 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109784 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_18 := arithEq_of_rows h (row := 1797) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_18
  simp only [← c109779, ← c109780, ← c109781] at e_1797_18
  have e_1797_19 := arithEq_of_rows h (row := 1797) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_19
  simp only [← c109782, ← c109783, k2, ← c109784] at e_1797_19
  have hr := e_1797_19
  simp only [e_1797_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5594)) := by
  have c109785 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109786 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109787 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109788 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109789 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109790 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_0 := arithEq_of_rows h (row := 1798) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_0
  simp only [← c109785, ← c109786, ← c109787] at e_1798_0
  have e_1798_1 := arithEq_of_rows h (row := 1798) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_1
  simp only [← c109788, ← c109789, k2, ← c109790] at e_1798_1
  have hr := e_1798_1
  simp only [e_1798_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5595)) := by
  have c109791 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109792 := (publicBatchWrapper86_copies3431 a h).1
  have c109793 := (publicBatchWrapper86_copies3431 a h).2.1
  have c109794 := (publicBatchWrapper86_copies3431 a h).2.2.1
  have c109795 := (publicBatchWrapper86_copies3431 a h).2.2.2.1
  have c109796 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_2 := arithEq_of_rows h (row := 1798) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_2
  simp only [← c109791, ← c109792, ← c109793] at e_1798_2
  have e_1798_3 := arithEq_of_rows h (row := 1798) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_3
  simp only [← c109794, ← c109795, k2, ← c109796] at e_1798_3
  have hr := e_1798_3
  simp only [e_1798_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5596)) := by
  have c109797 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.1
  have c109798 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.1
  have c109799 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.1
  have c109800 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.1
  have c109801 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.1
  have c109802 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_4 := arithEq_of_rows h (row := 1798) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_4
  simp only [← c109797, ← c109798, ← c109799] at e_1798_4
  have e_1798_5 := arithEq_of_rows h (row := 1798) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_5
  simp only [← c109800, ← c109801, k2, ← c109802] at e_1798_5
  have hr := e_1798_5
  simp only [e_1798_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5597)) := by
  have c109803 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109804 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109805 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109806 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109807 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109808 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_6 := arithEq_of_rows h (row := 1798) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_6
  simp only [← c109803, ← c109804, ← c109805] at e_1798_6
  have e_1798_7 := arithEq_of_rows h (row := 1798) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_7
  simp only [← c109806, ← c109807, k2, ← c109808] at e_1798_7
  have hr := e_1798_7
  simp only [e_1798_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5598)) := by
  have c109809 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109810 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109811 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109812 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109813 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109814 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_8 := arithEq_of_rows h (row := 1798) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_8
  simp only [← c109809, ← c109810, ← c109811] at e_1798_8
  have e_1798_9 := arithEq_of_rows h (row := 1798) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_9
  simp only [← c109812, ← c109813, k2, ← c109814] at e_1798_9
  have hr := e_1798_9
  simp only [e_1798_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5599)) := by
  have c109815 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109816 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109817 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109818 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109819 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109820 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_10 := arithEq_of_rows h (row := 1798) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_10
  simp only [← c109815, ← c109816, ← c109817] at e_1798_10
  have e_1798_11 := arithEq_of_rows h (row := 1798) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_11
  simp only [← c109818, ← c109819, k2, ← c109820] at e_1798_11
  have hr := e_1798_11
  simp only [e_1798_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5600)) := by
  have c109821 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109822 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109823 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109824 := (publicBatchWrapper86_copies3432 a h).1
  have c109825 := (publicBatchWrapper86_copies3432 a h).2.1
  have c109826 := (publicBatchWrapper86_copies3432 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_12 := arithEq_of_rows h (row := 1798) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_12
  simp only [← c109821, ← c109822, ← c109823] at e_1798_12
  have e_1798_13 := arithEq_of_rows h (row := 1798) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_13
  simp only [← c109824, ← c109825, k2, ← c109826] at e_1798_13
  have hr := e_1798_13
  simp only [e_1798_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5601)) := by
  have c109827 := (publicBatchWrapper86_copies3432 a h).2.2.2.1
  have c109828 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.1
  have c109829 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.1
  have c109830 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.1
  have c109831 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.1
  have c109832 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_14 := arithEq_of_rows h (row := 1798) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_14
  simp only [← c109827, ← c109828, ← c109829] at e_1798_14
  have e_1798_15 := arithEq_of_rows h (row := 1798) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_15
  simp only [← c109830, ← c109831, k2, ← c109832] at e_1798_15
  have hr := e_1798_15
  simp only [e_1798_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5770)) := by
  have c109833 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.1
  have c109834 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109835 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109836 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109837 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109838 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_16 := arithEq_of_rows h (row := 1798) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_16
  simp only [← c109833, ← c109834, ← c109835] at e_1798_16
  have e_1798_17 := arithEq_of_rows h (row := 1798) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_17
  simp only [← c109836, ← c109837, k2, ← c109838] at e_1798_17
  have hr := e_1798_17
  simp only [e_1798_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5771)) := by
  have c109839 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109840 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109841 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109842 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109843 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109844 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_18 := arithEq_of_rows h (row := 1798) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_18
  simp only [← c109839, ← c109840, ← c109841] at e_1798_18
  have e_1798_19 := arithEq_of_rows h (row := 1798) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_19
  simp only [← c109842, ← c109843, k2, ← c109844] at e_1798_19
  have hr := e_1798_19
  simp only [e_1798_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5772)) := by
  have c109845 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109846 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109847 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109848 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109849 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109850 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_0 := arithEq_of_rows h (row := 1799) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_0
  simp only [← c109845, ← c109846, ← c109847] at e_1799_0
  have e_1799_1 := arithEq_of_rows h (row := 1799) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_1
  simp only [← c109848, ← c109849, k2, ← c109850] at e_1799_1
  have hr := e_1799_1
  simp only [e_1799_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5773)) := by
  have c109851 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109852 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109853 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109854 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109855 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109856 := (publicBatchWrapper86_copies3433 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_2 := arithEq_of_rows h (row := 1799) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_2
  simp only [← c109851, ← c109852, ← c109853] at e_1799_2
  have e_1799_3 := arithEq_of_rows h (row := 1799) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_3
  simp only [← c109854, ← c109855, k2, ← c109856] at e_1799_3
  have hr := e_1799_3
  simp only [e_1799_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5774)) := by
  have c109857 := (publicBatchWrapper86_copies3433 a h).2.1
  have c109858 := (publicBatchWrapper86_copies3433 a h).2.2.1
  have c109859 := (publicBatchWrapper86_copies3433 a h).2.2.2.1
  have c109860 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.1
  have c109861 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.1
  have c109862 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_4 := arithEq_of_rows h (row := 1799) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_4
  simp only [← c109857, ← c109858, ← c109859] at e_1799_4
  have e_1799_5 := arithEq_of_rows h (row := 1799) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_5
  simp only [← c109860, ← c109861, k2, ← c109862] at e_1799_5
  have hr := e_1799_5
  simp only [e_1799_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5775)) := by
  have c109863 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.1
  have c109864 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.1
  have c109865 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.1
  have c109866 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109867 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109868 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_6 := arithEq_of_rows h (row := 1799) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_6
  simp only [← c109863, ← c109864, ← c109865] at e_1799_6
  have e_1799_7 := arithEq_of_rows h (row := 1799) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_7
  simp only [← c109866, ← c109867, k2, ← c109868] at e_1799_7
  have hr := e_1799_7
  simp only [e_1799_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5776)) := by
  have c109869 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109870 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109871 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109872 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109873 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109874 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_8 := arithEq_of_rows h (row := 1799) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_8
  simp only [← c109869, ← c109870, ← c109871] at e_1799_8
  have e_1799_9 := arithEq_of_rows h (row := 1799) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_9
  simp only [← c109872, ← c109873, k2, ← c109874] at e_1799_9
  have hr := e_1799_9
  simp only [e_1799_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5777)) := by
  have c109875 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109876 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109877 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109878 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109879 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109880 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_10 := arithEq_of_rows h (row := 1799) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_10
  simp only [← c109875, ← c109876, ← c109877] at e_1799_10
  have e_1799_11 := arithEq_of_rows h (row := 1799) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_11
  simp only [← c109878, ← c109879, k2, ← c109880] at e_1799_11
  have hr := e_1799_11
  simp only [e_1799_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5778)) := by
  have c109881 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109882 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109883 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109884 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109885 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109886 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_12 := arithEq_of_rows h (row := 1799) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_12
  simp only [← c109881, ← c109882, ← c109883] at e_1799_12
  have e_1799_13 := arithEq_of_rows h (row := 1799) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_13
  simp only [← c109884, ← c109885, k2, ← c109886] at e_1799_13
  have hr := e_1799_13
  simp only [e_1799_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5779)) := by
  have c109887 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109888 := (publicBatchWrapper86_copies3434 a h).1
  have c109889 := (publicBatchWrapper86_copies3434 a h).2.1
  have c109890 := (publicBatchWrapper86_copies3434 a h).2.2.1
  have c109891 := (publicBatchWrapper86_copies3434 a h).2.2.2.1
  have c109892 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_14 := arithEq_of_rows h (row := 1799) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_14
  simp only [← c109887, ← c109888, ← c109889] at e_1799_14
  have e_1799_15 := arithEq_of_rows h (row := 1799) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_15
  simp only [← c109890, ← c109891, k2, ← c109892] at e_1799_15
  have hr := e_1799_15
  simp only [e_1799_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5780)) := by
  have c109893 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.1
  have c109894 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.1
  have c109895 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.1
  have c109896 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.1
  have c109897 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.1
  have c109898 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_16 := arithEq_of_rows h (row := 1799) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_16
  simp only [← c109893, ← c109894, ← c109895] at e_1799_16
  have e_1799_17 := arithEq_of_rows h (row := 1799) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_17
  simp only [← c109896, ← c109897, k2, ← c109898] at e_1799_17
  have hr := e_1799_17
  simp only [e_1799_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5781)) := by
  have c109899 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109900 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109901 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109902 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109903 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109904 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_18 := arithEq_of_rows h (row := 1799) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_18
  simp only [← c109899, ← c109900, ← c109901] at e_1799_18
  have e_1799_19 := arithEq_of_rows h (row := 1799) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_19
  simp only [← c109902, ← c109903, k2, ← c109904] at e_1799_19
  have hr := e_1799_19
  simp only [e_1799_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5782)) := by
  have c109905 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109906 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109907 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109908 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109909 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109910 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_0 := arithEq_of_rows h (row := 1800) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_0
  simp only [← c109905, ← c109906, ← c109907] at e_1800_0
  have e_1800_1 := arithEq_of_rows h (row := 1800) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_1
  simp only [← c109908, ← c109909, k2, ← c109910] at e_1800_1
  have hr := e_1800_1
  simp only [e_1800_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5783)) := by
  have c109911 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109912 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109913 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109914 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109915 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109916 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_2 := arithEq_of_rows h (row := 1800) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_2
  simp only [← c109911, ← c109912, ← c109913] at e_1800_2
  have e_1800_3 := arithEq_of_rows h (row := 1800) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_3
  simp only [← c109914, ← c109915, k2, ← c109916] at e_1800_3
  have hr := e_1800_3
  simp only [e_1800_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5784)) := by
  have c109917 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109918 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109919 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109920 := (publicBatchWrapper86_copies3435 a h).1
  have c109921 := (publicBatchWrapper86_copies3435 a h).2.1
  have c109922 := (publicBatchWrapper86_copies3435 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_4 := arithEq_of_rows h (row := 1800) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_4
  simp only [← c109917, ← c109918, ← c109919] at e_1800_4
  have e_1800_5 := arithEq_of_rows h (row := 1800) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_5
  simp only [← c109920, ← c109921, k2, ← c109922] at e_1800_5
  have hr := e_1800_5
  simp only [e_1800_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5785)) := by
  have c109923 := (publicBatchWrapper86_copies3435 a h).2.2.2.1
  have c109924 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.1
  have c109925 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.1
  have c109926 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.1
  have c109927 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.1
  have c109928 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_6 := arithEq_of_rows h (row := 1800) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_6
  simp only [← c109923, ← c109924, ← c109925] at e_1800_6
  have e_1800_7 := arithEq_of_rows h (row := 1800) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_7
  simp only [← c109926, ← c109927, k2, ← c109928] at e_1800_7
  have hr := e_1800_7
  simp only [e_1800_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5786)) := by
  have c109929 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.1
  have c109930 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109931 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109932 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109933 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109934 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_8 := arithEq_of_rows h (row := 1800) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_8
  simp only [← c109929, ← c109930, ← c109931] at e_1800_8
  have e_1800_9 := arithEq_of_rows h (row := 1800) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_9
  simp only [← c109932, ← c109933, k2, ← c109934] at e_1800_9
  have hr := e_1800_9
  simp only [e_1800_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5787)) := by
  have c109935 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109936 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109937 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109938 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109939 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109940 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_10 := arithEq_of_rows h (row := 1800) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_10
  simp only [← c109935, ← c109936, ← c109937] at e_1800_10
  have e_1800_11 := arithEq_of_rows h (row := 1800) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_11
  simp only [← c109938, ← c109939, k2, ← c109940] at e_1800_11
  have hr := e_1800_11
  simp only [e_1800_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5788)) := by
  have c109941 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109942 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109943 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109944 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109945 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109946 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_12 := arithEq_of_rows h (row := 1800) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_12
  simp only [← c109941, ← c109942, ← c109943] at e_1800_12
  have e_1800_13 := arithEq_of_rows h (row := 1800) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_13
  simp only [← c109944, ← c109945, k2, ← c109946] at e_1800_13
  have hr := e_1800_13
  simp only [e_1800_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5789)) := by
  have c109947 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109948 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109949 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109950 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109951 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109952 := (publicBatchWrapper86_copies3436 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_14 := arithEq_of_rows h (row := 1800) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_14
  simp only [← c109947, ← c109948, ← c109949] at e_1800_14
  have e_1800_15 := arithEq_of_rows h (row := 1800) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_15
  simp only [← c109950, ← c109951, k2, ← c109952] at e_1800_15
  have hr := e_1800_15
  simp only [e_1800_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5790)) := by
  have c109953 := (publicBatchWrapper86_copies3436 a h).2.1
  have c109954 := (publicBatchWrapper86_copies3436 a h).2.2.1
  have c109955 := (publicBatchWrapper86_copies3436 a h).2.2.2.1
  have c109956 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.1
  have c109957 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.1
  have c109958 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_16 := arithEq_of_rows h (row := 1800) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_16
  simp only [← c109953, ← c109954, ← c109955] at e_1800_16
  have e_1800_17 := arithEq_of_rows h (row := 1800) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_17
  simp only [← c109956, ← c109957, k2, ← c109958] at e_1800_17
  have hr := e_1800_17
  simp only [e_1800_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5791)) := by
  have c109959 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.1
  have c109960 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.1
  have c109961 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.1
  have c109962 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109963 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109964 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_18 := arithEq_of_rows h (row := 1800) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_18
  simp only [← c109959, ← c109960, ← c109961] at e_1800_18
  have e_1800_19 := arithEq_of_rows h (row := 1800) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_19
  simp only [← c109962, ← c109963, k2, ← c109964] at e_1800_19
  have hr := e_1800_19
  simp only [e_1800_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5792)) := by
  have c109965 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109966 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109967 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109968 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109969 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109970 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_0 := arithEq_of_rows h (row := 1801) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_0
  simp only [← c109965, ← c109966, ← c109967] at e_1801_0
  have e_1801_1 := arithEq_of_rows h (row := 1801) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_1
  simp only [← c109968, ← c109969, k2, ← c109970] at e_1801_1
  have hr := e_1801_1
  simp only [e_1801_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5793)) := by
  have c109971 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109972 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109973 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109974 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109975 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109976 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_2 := arithEq_of_rows h (row := 1801) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_2
  simp only [← c109971, ← c109972, ← c109973] at e_1801_2
  have e_1801_3 := arithEq_of_rows h (row := 1801) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_3
  simp only [← c109974, ← c109975, k2, ← c109976] at e_1801_3
  have hr := e_1801_3
  simp only [e_1801_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5794)) := by
  have c109977 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109978 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109979 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109980 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109981 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109982 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_4 := arithEq_of_rows h (row := 1801) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_4
  simp only [← c109977, ← c109978, ← c109979] at e_1801_4
  have e_1801_5 := arithEq_of_rows h (row := 1801) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_5
  simp only [← c109980, ← c109981, k2, ← c109982] at e_1801_5
  have hr := e_1801_5
  simp only [e_1801_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5795)) := by
  have c109983 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109984 := (publicBatchWrapper86_copies3437 a h).1
  have c109985 := (publicBatchWrapper86_copies3437 a h).2.1
  have c109986 := (publicBatchWrapper86_copies3437 a h).2.2.1
  have c109987 := (publicBatchWrapper86_copies3437 a h).2.2.2.1
  have c109988 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_6 := arithEq_of_rows h (row := 1801) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_6
  simp only [← c109983, ← c109984, ← c109985] at e_1801_6
  have e_1801_7 := arithEq_of_rows h (row := 1801) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_7
  simp only [← c109986, ← c109987, k2, ← c109988] at e_1801_7
  have hr := e_1801_7
  simp only [e_1801_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5796)) := by
  have c109989 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.1
  have c109990 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.1
  have c109991 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.1
  have c109992 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.1
  have c109993 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.1
  have c109994 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_8 := arithEq_of_rows h (row := 1801) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_8
  simp only [← c109989, ← c109990, ← c109991] at e_1801_8
  have e_1801_9 := arithEq_of_rows h (row := 1801) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_9
  simp only [← c109992, ← c109993, k2, ← c109994] at e_1801_9
  have hr := e_1801_9
  simp only [e_1801_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5797)) := by
  have c109995 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109996 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109997 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109998 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109999 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110000 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_10 := arithEq_of_rows h (row := 1801) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_10
  simp only [← c109995, ← c109996, ← c109997] at e_1801_10
  have e_1801_11 := arithEq_of_rows h (row := 1801) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_11
  simp only [← c109998, ← c109999, k2, ← c110000] at e_1801_11
  have hr := e_1801_11
  simp only [e_1801_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5798)) := by
  have c110001 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110002 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110003 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110004 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110005 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110006 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_12 := arithEq_of_rows h (row := 1801) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_12
  simp only [← c110001, ← c110002, ← c110003] at e_1801_12
  have e_1801_13 := arithEq_of_rows h (row := 1801) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_13
  simp only [← c110004, ← c110005, k2, ← c110006] at e_1801_13
  have hr := e_1801_13
  simp only [e_1801_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5799)) := by
  have c110007 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110008 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110009 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110010 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110011 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110012 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_14 := arithEq_of_rows h (row := 1801) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_14
  simp only [← c110007, ← c110008, ← c110009] at e_1801_14
  have e_1801_15 := arithEq_of_rows h (row := 1801) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_15
  simp only [← c110010, ← c110011, k2, ← c110012] at e_1801_15
  have hr := e_1801_15
  simp only [e_1801_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5800)) := by
  have c110013 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110014 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110015 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110016 := (publicBatchWrapper86_copies3438 a h).1
  have c110017 := (publicBatchWrapper86_copies3438 a h).2.1
  have c110018 := (publicBatchWrapper86_copies3438 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_16 := arithEq_of_rows h (row := 1801) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_16
  simp only [← c110013, ← c110014, ← c110015] at e_1801_16
  have e_1801_17 := arithEq_of_rows h (row := 1801) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_17
  simp only [← c110016, ← c110017, k2, ← c110018] at e_1801_17
  have hr := e_1801_17
  simp only [e_1801_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5801)) := by
  have c110019 := (publicBatchWrapper86_copies3438 a h).2.2.2.1
  have c110020 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.1
  have c110021 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.1
  have c110022 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.1
  have c110023 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.1
  have c110024 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_18 := arithEq_of_rows h (row := 1801) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_18
  simp only [← c110019, ← c110020, ← c110021] at e_1801_18
  have e_1801_19 := arithEq_of_rows h (row := 1801) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_19
  simp only [← c110022, ← c110023, k2, ← c110024] at e_1801_19
  have hr := e_1801_19
  simp only [e_1801_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5970)) := by
  have c110025 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.1
  have c110026 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110027 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110028 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110029 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110030 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_0 := arithEq_of_rows h (row := 1802) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_0
  simp only [← c110025, ← c110026, ← c110027] at e_1802_0
  have e_1802_1 := arithEq_of_rows h (row := 1802) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_1
  simp only [← c110028, ← c110029, k2, ← c110030] at e_1802_1
  have hr := e_1802_1
  simp only [e_1802_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5971)) := by
  have c110031 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110032 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110033 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110034 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110035 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110036 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_2 := arithEq_of_rows h (row := 1802) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_2
  simp only [← c110031, ← c110032, ← c110033] at e_1802_2
  have e_1802_3 := arithEq_of_rows h (row := 1802) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_3
  simp only [← c110034, ← c110035, k2, ← c110036] at e_1802_3
  have hr := e_1802_3
  simp only [e_1802_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5972)) := by
  have c110037 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110038 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110039 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110040 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110041 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110042 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_4 := arithEq_of_rows h (row := 1802) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_4
  simp only [← c110037, ← c110038, ← c110039] at e_1802_4
  have e_1802_5 := arithEq_of_rows h (row := 1802) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_5
  simp only [← c110040, ← c110041, k2, ← c110042] at e_1802_5
  have hr := e_1802_5
  simp only [e_1802_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5973)) := by
  have c110043 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110044 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110045 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110046 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110047 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110048 := (publicBatchWrapper86_copies3439 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_6 := arithEq_of_rows h (row := 1802) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_6
  simp only [← c110043, ← c110044, ← c110045] at e_1802_6
  have e_1802_7 := arithEq_of_rows h (row := 1802) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_7
  simp only [← c110046, ← c110047, k2, ← c110048] at e_1802_7
  have hr := e_1802_7
  simp only [e_1802_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5974)) := by
  have c110049 := (publicBatchWrapper86_copies3439 a h).2.1
  have c110050 := (publicBatchWrapper86_copies3439 a h).2.2.1
  have c110051 := (publicBatchWrapper86_copies3439 a h).2.2.2.1
  have c110052 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.1
  have c110053 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.1
  have c110054 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_8 := arithEq_of_rows h (row := 1802) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_8
  simp only [← c110049, ← c110050, ← c110051] at e_1802_8
  have e_1802_9 := arithEq_of_rows h (row := 1802) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_9
  simp only [← c110052, ← c110053, k2, ← c110054] at e_1802_9
  have hr := e_1802_9
  simp only [e_1802_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5975)) := by
  have c110055 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.1
  have c110056 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.1
  have c110057 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.1
  have c110058 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110059 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110060 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_10 := arithEq_of_rows h (row := 1802) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_10
  simp only [← c110055, ← c110056, ← c110057] at e_1802_10
  have e_1802_11 := arithEq_of_rows h (row := 1802) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_11
  simp only [← c110058, ← c110059, k2, ← c110060] at e_1802_11
  have hr := e_1802_11
  simp only [e_1802_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5976)) := by
  have c110061 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110062 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110063 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110064 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110065 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110066 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_12 := arithEq_of_rows h (row := 1802) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_12
  simp only [← c110061, ← c110062, ← c110063] at e_1802_12
  have e_1802_13 := arithEq_of_rows h (row := 1802) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_13
  simp only [← c110064, ← c110065, k2, ← c110066] at e_1802_13
  have hr := e_1802_13
  simp only [e_1802_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5977)) := by
  have c110067 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110068 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110069 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110070 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110071 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110072 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_14 := arithEq_of_rows h (row := 1802) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_14
  simp only [← c110067, ← c110068, ← c110069] at e_1802_14
  have e_1802_15 := arithEq_of_rows h (row := 1802) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_15
  simp only [← c110070, ← c110071, k2, ← c110072] at e_1802_15
  have hr := e_1802_15
  simp only [e_1802_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5978)) := by
  have c110073 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110074 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110075 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110076 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110077 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110078 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_16 := arithEq_of_rows h (row := 1802) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_16
  simp only [← c110073, ← c110074, ← c110075] at e_1802_16
  have e_1802_17 := arithEq_of_rows h (row := 1802) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_17
  simp only [← c110076, ← c110077, k2, ← c110078] at e_1802_17
  have hr := e_1802_17
  simp only [e_1802_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5979)) := by
  have c110079 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110080 := (publicBatchWrapper86_copies3440 a h).1
  have c110081 := (publicBatchWrapper86_copies3440 a h).2.1
  have c110082 := (publicBatchWrapper86_copies3440 a h).2.2.1
  have c110083 := (publicBatchWrapper86_copies3440 a h).2.2.2.1
  have c110084 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_18 := arithEq_of_rows h (row := 1802) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_18
  simp only [← c110079, ← c110080, ← c110081] at e_1802_18
  have e_1802_19 := arithEq_of_rows h (row := 1802) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_19
  simp only [← c110082, ← c110083, k2, ← c110084] at e_1802_19
  have hr := e_1802_19
  simp only [e_1802_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5980)) := by
  have c110085 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.1
  have c110086 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.1
  have c110087 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.1
  have c110088 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.1
  have c110089 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.1
  have c110090 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_0 := arithEq_of_rows h (row := 1803) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_0
  simp only [← c110085, ← c110086, ← c110087] at e_1803_0
  have e_1803_1 := arithEq_of_rows h (row := 1803) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_1
  simp only [← c110088, ← c110089, k2, ← c110090] at e_1803_1
  have hr := e_1803_1
  simp only [e_1803_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5981)) := by
  have c110091 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110092 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110093 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110094 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110095 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110096 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_2 := arithEq_of_rows h (row := 1803) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_2
  simp only [← c110091, ← c110092, ← c110093] at e_1803_2
  have e_1803_3 := arithEq_of_rows h (row := 1803) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_3
  simp only [← c110094, ← c110095, k2, ← c110096] at e_1803_3
  have hr := e_1803_3
  simp only [e_1803_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5982)) := by
  have c110097 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110098 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110099 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110100 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110101 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110102 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_4 := arithEq_of_rows h (row := 1803) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_4
  simp only [← c110097, ← c110098, ← c110099] at e_1803_4
  have e_1803_5 := arithEq_of_rows h (row := 1803) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_5
  simp only [← c110100, ← c110101, k2, ← c110102] at e_1803_5
  have hr := e_1803_5
  simp only [e_1803_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5983)) := by
  have c110103 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110104 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110105 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110106 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110107 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110108 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_6 := arithEq_of_rows h (row := 1803) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_6
  simp only [← c110103, ← c110104, ← c110105] at e_1803_6
  have e_1803_7 := arithEq_of_rows h (row := 1803) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_7
  simp only [← c110106, ← c110107, k2, ← c110108] at e_1803_7
  have hr := e_1803_7
  simp only [e_1803_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5984)) := by
  have c110109 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110110 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110111 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110112 := (publicBatchWrapper86_copies3441 a h).1
  have c110113 := (publicBatchWrapper86_copies3441 a h).2.1
  have c110114 := (publicBatchWrapper86_copies3441 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_8 := arithEq_of_rows h (row := 1803) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_8
  simp only [← c110109, ← c110110, ← c110111] at e_1803_8
  have e_1803_9 := arithEq_of_rows h (row := 1803) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_9
  simp only [← c110112, ← c110113, k2, ← c110114] at e_1803_9
  have hr := e_1803_9
  simp only [e_1803_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5985)) := by
  have c110115 := (publicBatchWrapper86_copies3441 a h).2.2.2.1
  have c110116 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.1
  have c110117 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.1
  have c110118 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.1
  have c110119 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.1
  have c110120 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_10 := arithEq_of_rows h (row := 1803) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_10
  simp only [← c110115, ← c110116, ← c110117] at e_1803_10
  have e_1803_11 := arithEq_of_rows h (row := 1803) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_11
  simp only [← c110118, ← c110119, k2, ← c110120] at e_1803_11
  have hr := e_1803_11
  simp only [e_1803_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5986)) := by
  have c110121 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.1
  have c110122 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110123 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110124 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110125 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110126 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_12 := arithEq_of_rows h (row := 1803) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_12
  simp only [← c110121, ← c110122, ← c110123] at e_1803_12
  have e_1803_13 := arithEq_of_rows h (row := 1803) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_13
  simp only [← c110124, ← c110125, k2, ← c110126] at e_1803_13
  have hr := e_1803_13
  simp only [e_1803_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5987)) := by
  have c110127 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110128 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110129 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110130 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110131 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110132 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_14 := arithEq_of_rows h (row := 1803) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_14
  simp only [← c110127, ← c110128, ← c110129] at e_1803_14
  have e_1803_15 := arithEq_of_rows h (row := 1803) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_15
  simp only [← c110130, ← c110131, k2, ← c110132] at e_1803_15
  have hr := e_1803_15
  simp only [e_1803_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5988)) := by
  have c110133 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110134 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110135 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110136 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110137 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110138 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_16 := arithEq_of_rows h (row := 1803) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_16
  simp only [← c110133, ← c110134, ← c110135] at e_1803_16
  have e_1803_17 := arithEq_of_rows h (row := 1803) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_17
  simp only [← c110136, ← c110137, k2, ← c110138] at e_1803_17
  have hr := e_1803_17
  simp only [e_1803_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5989)) := by
  have c110139 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110140 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110141 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110142 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110143 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110144 := (publicBatchWrapper86_copies3442 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_18 := arithEq_of_rows h (row := 1803) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_18
  simp only [← c110139, ← c110140, ← c110141] at e_1803_18
  have e_1803_19 := arithEq_of_rows h (row := 1803) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_19
  simp only [← c110142, ← c110143, k2, ← c110144] at e_1803_19
  have hr := e_1803_19
  simp only [e_1803_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5990)) := by
  have c110145 := (publicBatchWrapper86_copies3442 a h).2.1
  have c110146 := (publicBatchWrapper86_copies3442 a h).2.2.1
  have c110147 := (publicBatchWrapper86_copies3442 a h).2.2.2.1
  have c110148 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.1
  have c110149 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.1
  have c110150 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_0 := arithEq_of_rows h (row := 1804) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_0
  simp only [← c110145, ← c110146, ← c110147] at e_1804_0
  have e_1804_1 := arithEq_of_rows h (row := 1804) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_1
  simp only [← c110148, ← c110149, k2, ← c110150] at e_1804_1
  have hr := e_1804_1
  simp only [e_1804_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5991)) := by
  have c110151 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.1
  have c110152 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.1
  have c110153 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.1
  have c110154 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110155 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110156 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_2 := arithEq_of_rows h (row := 1804) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_2
  simp only [← c110151, ← c110152, ← c110153] at e_1804_2
  have e_1804_3 := arithEq_of_rows h (row := 1804) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_3
  simp only [← c110154, ← c110155, k2, ← c110156] at e_1804_3
  have hr := e_1804_3
  simp only [e_1804_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5992)) := by
  have c110157 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110158 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110159 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110160 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110161 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110162 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_4 := arithEq_of_rows h (row := 1804) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_4
  simp only [← c110157, ← c110158, ← c110159] at e_1804_4
  have e_1804_5 := arithEq_of_rows h (row := 1804) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_5
  simp only [← c110160, ← c110161, k2, ← c110162] at e_1804_5
  have hr := e_1804_5
  simp only [e_1804_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5993)) := by
  have c110163 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110164 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110165 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110166 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110167 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110168 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_6 := arithEq_of_rows h (row := 1804) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_6
  simp only [← c110163, ← c110164, ← c110165] at e_1804_6
  have e_1804_7 := arithEq_of_rows h (row := 1804) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_7
  simp only [← c110166, ← c110167, k2, ← c110168] at e_1804_7
  have hr := e_1804_7
  simp only [e_1804_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5994)) := by
  have c110169 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110170 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110171 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110172 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110173 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110174 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_8 := arithEq_of_rows h (row := 1804) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_8
  simp only [← c110169, ← c110170, ← c110171] at e_1804_8
  have e_1804_9 := arithEq_of_rows h (row := 1804) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_9
  simp only [← c110172, ← c110173, k2, ← c110174] at e_1804_9
  have hr := e_1804_9
  simp only [e_1804_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5995)) := by
  have c110175 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110176 := (publicBatchWrapper86_copies3443 a h).1
  have c110177 := (publicBatchWrapper86_copies3443 a h).2.1
  have c110178 := (publicBatchWrapper86_copies3443 a h).2.2.1
  have c110179 := (publicBatchWrapper86_copies3443 a h).2.2.2.1
  have c110180 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_10 := arithEq_of_rows h (row := 1804) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_10
  simp only [← c110175, ← c110176, ← c110177] at e_1804_10
  have e_1804_11 := arithEq_of_rows h (row := 1804) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_11
  simp only [← c110178, ← c110179, k2, ← c110180] at e_1804_11
  have hr := e_1804_11
  simp only [e_1804_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5996)) := by
  have c110181 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.1
  have c110182 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.1
  have c110183 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.1
  have c110184 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.1
  have c110185 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.1
  have c110186 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_12 := arithEq_of_rows h (row := 1804) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_12
  simp only [← c110181, ← c110182, ← c110183] at e_1804_12
  have e_1804_13 := arithEq_of_rows h (row := 1804) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_13
  simp only [← c110184, ← c110185, k2, ← c110186] at e_1804_13
  have hr := e_1804_13
  simp only [e_1804_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5997)) := by
  have c110187 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110188 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110189 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110190 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110191 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110192 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_14 := arithEq_of_rows h (row := 1804) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_14
  simp only [← c110187, ← c110188, ← c110189] at e_1804_14
  have e_1804_15 := arithEq_of_rows h (row := 1804) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_15
  simp only [← c110190, ← c110191, k2, ← c110192] at e_1804_15
  have hr := e_1804_15
  simp only [e_1804_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5998)) := by
  have c110193 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110194 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110195 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110196 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110197 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110198 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_16 := arithEq_of_rows h (row := 1804) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_16
  simp only [← c110193, ← c110194, ← c110195] at e_1804_16
  have e_1804_17 := arithEq_of_rows h (row := 1804) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_17
  simp only [← c110196, ← c110197, k2, ← c110198] at e_1804_17
  have hr := e_1804_17
  simp only [e_1804_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5999)) := by
  have c110199 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110200 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110201 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110202 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110203 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110204 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_18 := arithEq_of_rows h (row := 1804) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_18
  simp only [← c110199, ← c110200, ← c110201] at e_1804_18
  have e_1804_19 := arithEq_of_rows h (row := 1804) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_19
  simp only [← c110202, ← c110203, k2, ← c110204] at e_1804_19
  have hr := e_1804_19
  simp only [e_1804_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 6000)) := by
  have c110205 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110206 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110207 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110208 := (publicBatchWrapper86_copies3444 a h).1
  have c110209 := (publicBatchWrapper86_copies3444 a h).2.1
  have c110210 := (publicBatchWrapper86_copies3444 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_0 := arithEq_of_rows h (row := 1805) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_0
  simp only [← c110205, ← c110206, ← c110207] at e_1805_0
  have e_1805_1 := arithEq_of_rows h (row := 1805) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_1
  simp only [← c110208, ← c110209, k2, ← c110210] at e_1805_1
  have hr := e_1805_1
  simp only [e_1805_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 6001)) := by
  have c110211 := (publicBatchWrapper86_copies3444 a h).2.2.2.1
  have c110212 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.1
  have c110213 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.1
  have c110214 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.1
  have c110215 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.1
  have c110216 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_2 := arithEq_of_rows h (row := 1805) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_2
  simp only [← c110211, ← c110212, ← c110213] at e_1805_2
  have e_1805_3 := arithEq_of_rows h (row := 1805) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_3
  simp only [← c110214, ← c110215, k2, ← c110216] at e_1805_3
  have hr := e_1805_3
  simp only [e_1805_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6170)) := by
  have c110217 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.1
  have c110218 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110219 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110220 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110221 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110222 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_4 := arithEq_of_rows h (row := 1805) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_4
  simp only [← c110217, ← c110218, ← c110219] at e_1805_4
  have e_1805_5 := arithEq_of_rows h (row := 1805) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_5
  simp only [← c110220, ← c110221, k2, ← c110222] at e_1805_5
  have hr := e_1805_5
  simp only [e_1805_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6171)) := by
  have c110223 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110224 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110225 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110226 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110227 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110228 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_6 := arithEq_of_rows h (row := 1805) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_6
  simp only [← c110223, ← c110224, ← c110225] at e_1805_6
  have e_1805_7 := arithEq_of_rows h (row := 1805) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_7
  simp only [← c110226, ← c110227, k2, ← c110228] at e_1805_7
  have hr := e_1805_7
  simp only [e_1805_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6172)) := by
  have c110229 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110230 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110231 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110232 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110233 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110234 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_8 := arithEq_of_rows h (row := 1805) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_8
  simp only [← c110229, ← c110230, ← c110231] at e_1805_8
  have e_1805_9 := arithEq_of_rows h (row := 1805) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_9
  simp only [← c110232, ← c110233, k2, ← c110234] at e_1805_9
  have hr := e_1805_9
  simp only [e_1805_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6173)) := by
  have c110235 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110236 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110237 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110238 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110239 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110240 := (publicBatchWrapper86_copies3445 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_10 := arithEq_of_rows h (row := 1805) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_10
  simp only [← c110235, ← c110236, ← c110237] at e_1805_10
  have e_1805_11 := arithEq_of_rows h (row := 1805) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_11
  simp only [← c110238, ← c110239, k2, ← c110240] at e_1805_11
  have hr := e_1805_11
  simp only [e_1805_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6174)) := by
  have c110241 := (publicBatchWrapper86_copies3445 a h).2.1
  have c110242 := (publicBatchWrapper86_copies3445 a h).2.2.1
  have c110243 := (publicBatchWrapper86_copies3445 a h).2.2.2.1
  have c110244 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.1
  have c110245 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.1
  have c110246 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_12 := arithEq_of_rows h (row := 1805) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_12
  simp only [← c110241, ← c110242, ← c110243] at e_1805_12
  have e_1805_13 := arithEq_of_rows h (row := 1805) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_13
  simp only [← c110244, ← c110245, k2, ← c110246] at e_1805_13
  have hr := e_1805_13
  simp only [e_1805_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6175)) := by
  have c110247 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.1
  have c110248 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.1
  have c110249 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.1
  have c110250 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110251 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110252 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_14 := arithEq_of_rows h (row := 1805) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_14
  simp only [← c110247, ← c110248, ← c110249] at e_1805_14
  have e_1805_15 := arithEq_of_rows h (row := 1805) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_15
  simp only [← c110250, ← c110251, k2, ← c110252] at e_1805_15
  have hr := e_1805_15
  simp only [e_1805_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6176)) := by
  have c110253 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110254 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110255 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110256 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110257 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110258 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_16 := arithEq_of_rows h (row := 1805) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_16
  simp only [← c110253, ← c110254, ← c110255] at e_1805_16
  have e_1805_17 := arithEq_of_rows h (row := 1805) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_17
  simp only [← c110256, ← c110257, k2, ← c110258] at e_1805_17
  have hr := e_1805_17
  simp only [e_1805_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6177)) := by
  have c110259 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110260 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110261 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110262 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110263 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110264 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_18 := arithEq_of_rows h (row := 1805) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_18
  simp only [← c110259, ← c110260, ← c110261] at e_1805_18
  have e_1805_19 := arithEq_of_rows h (row := 1805) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_19
  simp only [← c110262, ← c110263, k2, ← c110264] at e_1805_19
  have hr := e_1805_19
  simp only [e_1805_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6178)) := by
  have c110265 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110266 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110267 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110268 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110269 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110270 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_0 := arithEq_of_rows h (row := 1806) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_0
  simp only [← c110265, ← c110266, ← c110267] at e_1806_0
  have e_1806_1 := arithEq_of_rows h (row := 1806) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_1
  simp only [← c110268, ← c110269, k2, ← c110270] at e_1806_1
  have hr := e_1806_1
  simp only [e_1806_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6179)) := by
  have c110271 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110272 := (publicBatchWrapper86_copies3446 a h).1
  have c110273 := (publicBatchWrapper86_copies3446 a h).2.1
  have c110274 := (publicBatchWrapper86_copies3446 a h).2.2.1
  have c110275 := (publicBatchWrapper86_copies3446 a h).2.2.2.1
  have c110276 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_2 := arithEq_of_rows h (row := 1806) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_2
  simp only [← c110271, ← c110272, ← c110273] at e_1806_2
  have e_1806_3 := arithEq_of_rows h (row := 1806) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_3
  simp only [← c110274, ← c110275, k2, ← c110276] at e_1806_3
  have hr := e_1806_3
  simp only [e_1806_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6180)) := by
  have c110277 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.1
  have c110278 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.1
  have c110279 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.1
  have c110280 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.1
  have c110281 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.1
  have c110282 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_4 := arithEq_of_rows h (row := 1806) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_4
  simp only [← c110277, ← c110278, ← c110279] at e_1806_4
  have e_1806_5 := arithEq_of_rows h (row := 1806) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_5
  simp only [← c110280, ← c110281, k2, ← c110282] at e_1806_5
  have hr := e_1806_5
  simp only [e_1806_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6181)) := by
  have c110283 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110284 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110285 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110286 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110287 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110288 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_6 := arithEq_of_rows h (row := 1806) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_6
  simp only [← c110283, ← c110284, ← c110285] at e_1806_6
  have e_1806_7 := arithEq_of_rows h (row := 1806) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_7
  simp only [← c110286, ← c110287, k2, ← c110288] at e_1806_7
  have hr := e_1806_7
  simp only [e_1806_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6182)) := by
  have c110289 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110290 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110291 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110292 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110293 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110294 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_8 := arithEq_of_rows h (row := 1806) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_8
  simp only [← c110289, ← c110290, ← c110291] at e_1806_8
  have e_1806_9 := arithEq_of_rows h (row := 1806) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_9
  simp only [← c110292, ← c110293, k2, ← c110294] at e_1806_9
  have hr := e_1806_9
  simp only [e_1806_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6183)) := by
  have c110295 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110296 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110297 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110298 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110299 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110300 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_10 := arithEq_of_rows h (row := 1806) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_10
  simp only [← c110295, ← c110296, ← c110297] at e_1806_10
  have e_1806_11 := arithEq_of_rows h (row := 1806) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_11
  simp only [← c110298, ← c110299, k2, ← c110300] at e_1806_11
  have hr := e_1806_11
  simp only [e_1806_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6184)) := by
  have c110301 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110302 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110303 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110304 := (publicBatchWrapper86_copies3447 a h).1
  have c110305 := (publicBatchWrapper86_copies3447 a h).2.1
  have c110306 := (publicBatchWrapper86_copies3447 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_12 := arithEq_of_rows h (row := 1806) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_12
  simp only [← c110301, ← c110302, ← c110303] at e_1806_12
  have e_1806_13 := arithEq_of_rows h (row := 1806) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_13
  simp only [← c110304, ← c110305, k2, ← c110306] at e_1806_13
  have hr := e_1806_13
  simp only [e_1806_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6185)) := by
  have c110307 := (publicBatchWrapper86_copies3447 a h).2.2.2.1
  have c110308 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.1
  have c110309 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.1
  have c110310 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.1
  have c110311 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.1
  have c110312 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_14 := arithEq_of_rows h (row := 1806) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_14
  simp only [← c110307, ← c110308, ← c110309] at e_1806_14
  have e_1806_15 := arithEq_of_rows h (row := 1806) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_15
  simp only [← c110310, ← c110311, k2, ← c110312] at e_1806_15
  have hr := e_1806_15
  simp only [e_1806_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6186)) := by
  have c110313 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.1
  have c110314 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110315 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110316 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110317 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110318 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_16 := arithEq_of_rows h (row := 1806) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_16
  simp only [← c110313, ← c110314, ← c110315] at e_1806_16
  have e_1806_17 := arithEq_of_rows h (row := 1806) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_17
  simp only [← c110316, ← c110317, k2, ← c110318] at e_1806_17
  have hr := e_1806_17
  simp only [e_1806_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6187)) := by
  have c110319 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110320 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110321 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110322 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110323 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110324 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_18 := arithEq_of_rows h (row := 1806) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_18
  simp only [← c110319, ← c110320, ← c110321] at e_1806_18
  have e_1806_19 := arithEq_of_rows h (row := 1806) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_19
  simp only [← c110322, ← c110323, k2, ← c110324] at e_1806_19
  have hr := e_1806_19
  simp only [e_1806_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6188)) := by
  have c110325 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110326 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110327 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110328 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110329 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110330 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_0 := arithEq_of_rows h (row := 1807) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_0
  simp only [← c110325, ← c110326, ← c110327] at e_1807_0
  have e_1807_1 := arithEq_of_rows h (row := 1807) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_1
  simp only [← c110328, ← c110329, k2, ← c110330] at e_1807_1
  have hr := e_1807_1
  simp only [e_1807_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6189)) := by
  have c110331 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110332 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110333 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110334 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110335 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110336 := (publicBatchWrapper86_copies3448 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_2 := arithEq_of_rows h (row := 1807) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_2
  simp only [← c110331, ← c110332, ← c110333] at e_1807_2
  have e_1807_3 := arithEq_of_rows h (row := 1807) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_3
  simp only [← c110334, ← c110335, k2, ← c110336] at e_1807_3
  have hr := e_1807_3
  simp only [e_1807_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6190)) := by
  have c110337 := (publicBatchWrapper86_copies3448 a h).2.1
  have c110338 := (publicBatchWrapper86_copies3448 a h).2.2.1
  have c110339 := (publicBatchWrapper86_copies3448 a h).2.2.2.1
  have c110340 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.1
  have c110341 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.1
  have c110342 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_4 := arithEq_of_rows h (row := 1807) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_4
  simp only [← c110337, ← c110338, ← c110339] at e_1807_4
  have e_1807_5 := arithEq_of_rows h (row := 1807) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_5
  simp only [← c110340, ← c110341, k2, ← c110342] at e_1807_5
  have hr := e_1807_5
  simp only [e_1807_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6191)) := by
  have c110343 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.1
  have c110344 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.1
  have c110345 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.1
  have c110346 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110347 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110348 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_6 := arithEq_of_rows h (row := 1807) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_6
  simp only [← c110343, ← c110344, ← c110345] at e_1807_6
  have e_1807_7 := arithEq_of_rows h (row := 1807) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_7
  simp only [← c110346, ← c110347, k2, ← c110348] at e_1807_7
  have hr := e_1807_7
  simp only [e_1807_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6192)) := by
  have c110349 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110350 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110351 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110352 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110353 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110354 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_8 := arithEq_of_rows h (row := 1807) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_8
  simp only [← c110349, ← c110350, ← c110351] at e_1807_8
  have e_1807_9 := arithEq_of_rows h (row := 1807) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_9
  simp only [← c110352, ← c110353, k2, ← c110354] at e_1807_9
  have hr := e_1807_9
  simp only [e_1807_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6193)) := by
  have c110355 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110356 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110357 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110358 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110359 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110360 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_10 := arithEq_of_rows h (row := 1807) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_10
  simp only [← c110355, ← c110356, ← c110357] at e_1807_10
  have e_1807_11 := arithEq_of_rows h (row := 1807) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_11
  simp only [← c110358, ← c110359, k2, ← c110360] at e_1807_11
  have hr := e_1807_11
  simp only [e_1807_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6194)) := by
  have c110361 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110362 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110363 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110364 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110365 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110366 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_12 := arithEq_of_rows h (row := 1807) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_12
  simp only [← c110361, ← c110362, ← c110363] at e_1807_12
  have e_1807_13 := arithEq_of_rows h (row := 1807) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_13
  simp only [← c110364, ← c110365, k2, ← c110366] at e_1807_13
  have hr := e_1807_13
  simp only [e_1807_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6195)) := by
  have c110367 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110368 := (publicBatchWrapper86_copies3449 a h).1
  have c110369 := (publicBatchWrapper86_copies3449 a h).2.1
  have c110370 := (publicBatchWrapper86_copies3449 a h).2.2.1
  have c110371 := (publicBatchWrapper86_copies3449 a h).2.2.2.1
  have c110372 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_14 := arithEq_of_rows h (row := 1807) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_14
  simp only [← c110367, ← c110368, ← c110369] at e_1807_14
  have e_1807_15 := arithEq_of_rows h (row := 1807) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_15
  simp only [← c110370, ← c110371, k2, ← c110372] at e_1807_15
  have hr := e_1807_15
  simp only [e_1807_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6196)) := by
  have c110373 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.1
  have c110374 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.1
  have c110375 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.1
  have c110376 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.1
  have c110377 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.1
  have c110378 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_16 := arithEq_of_rows h (row := 1807) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_16
  simp only [← c110373, ← c110374, ← c110375] at e_1807_16
  have e_1807_17 := arithEq_of_rows h (row := 1807) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_17
  simp only [← c110376, ← c110377, k2, ← c110378] at e_1807_17
  have hr := e_1807_17
  simp only [e_1807_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6197)) := by
  have c110379 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110380 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110381 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110382 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110383 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110384 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_18 := arithEq_of_rows h (row := 1807) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_18
  simp only [← c110379, ← c110380, ← c110381] at e_1807_18
  have e_1807_19 := arithEq_of_rows h (row := 1807) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_19
  simp only [← c110382, ← c110383, k2, ← c110384] at e_1807_19
  have hr := e_1807_19
  simp only [e_1807_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6198)) := by
  have c110385 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110386 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110387 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110388 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110389 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110390 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_0 := arithEq_of_rows h (row := 1808) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_0
  simp only [← c110385, ← c110386, ← c110387] at e_1808_0
  have e_1808_1 := arithEq_of_rows h (row := 1808) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_1
  simp only [← c110388, ← c110389, k2, ← c110390] at e_1808_1
  have hr := e_1808_1
  simp only [e_1808_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6199)) := by
  have c110391 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110392 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110393 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110394 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110395 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110396 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_2 := arithEq_of_rows h (row := 1808) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_2
  simp only [← c110391, ← c110392, ← c110393] at e_1808_2
  have e_1808_3 := arithEq_of_rows h (row := 1808) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_3
  simp only [← c110394, ← c110395, k2, ← c110396] at e_1808_3
  have hr := e_1808_3
  simp only [e_1808_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6200)) := by
  have c110397 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110398 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110399 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110400 := (publicBatchWrapper86_copies3450 a h).1
  have c110401 := (publicBatchWrapper86_copies3450 a h).2.1
  have c110402 := (publicBatchWrapper86_copies3450 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_4 := arithEq_of_rows h (row := 1808) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_4
  simp only [← c110397, ← c110398, ← c110399] at e_1808_4
  have e_1808_5 := arithEq_of_rows h (row := 1808) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_5
  simp only [← c110400, ← c110401, k2, ← c110402] at e_1808_5
  have hr := e_1808_5
  simp only [e_1808_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6201)) := by
  have c110403 := (publicBatchWrapper86_copies3450 a h).2.2.2.1
  have c110404 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.1
  have c110405 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.1
  have c110406 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.1
  have c110407 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.1
  have c110408 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_6 := arithEq_of_rows h (row := 1808) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_6
  simp only [← c110403, ← c110404, ← c110405] at e_1808_6
  have e_1808_7 := arithEq_of_rows h (row := 1808) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_7
  simp only [← c110406, ← c110407, k2, ← c110408] at e_1808_7
  have hr := e_1808_7
  simp only [e_1808_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6370)) := by
  have c110409 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.1
  have c110410 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110411 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110412 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110413 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110414 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_8 := arithEq_of_rows h (row := 1808) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_8
  simp only [← c110409, ← c110410, ← c110411] at e_1808_8
  have e_1808_9 := arithEq_of_rows h (row := 1808) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_9
  simp only [← c110412, ← c110413, k2, ← c110414] at e_1808_9
  have hr := e_1808_9
  simp only [e_1808_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6371)) := by
  have c110415 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110416 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110417 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110418 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110419 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110420 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_10 := arithEq_of_rows h (row := 1808) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_10
  simp only [← c110415, ← c110416, ← c110417] at e_1808_10
  have e_1808_11 := arithEq_of_rows h (row := 1808) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_11
  simp only [← c110418, ← c110419, k2, ← c110420] at e_1808_11
  have hr := e_1808_11
  simp only [e_1808_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6372)) := by
  have c110421 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110422 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110423 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110424 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110425 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110426 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_12 := arithEq_of_rows h (row := 1808) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_12
  simp only [← c110421, ← c110422, ← c110423] at e_1808_12
  have e_1808_13 := arithEq_of_rows h (row := 1808) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_13
  simp only [← c110424, ← c110425, k2, ← c110426] at e_1808_13
  have hr := e_1808_13
  simp only [e_1808_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6373)) := by
  have c110427 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110428 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110429 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110430 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110431 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110432 := (publicBatchWrapper86_copies3451 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_14 := arithEq_of_rows h (row := 1808) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_14
  simp only [← c110427, ← c110428, ← c110429] at e_1808_14
  have e_1808_15 := arithEq_of_rows h (row := 1808) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_15
  simp only [← c110430, ← c110431, k2, ← c110432] at e_1808_15
  have hr := e_1808_15
  simp only [e_1808_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6374)) := by
  have c110433 := (publicBatchWrapper86_copies3451 a h).2.1
  have c110434 := (publicBatchWrapper86_copies3451 a h).2.2.1
  have c110435 := (publicBatchWrapper86_copies3451 a h).2.2.2.1
  have c110436 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.1
  have c110437 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.1
  have c110438 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_16 := arithEq_of_rows h (row := 1808) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_16
  simp only [← c110433, ← c110434, ← c110435] at e_1808_16
  have e_1808_17 := arithEq_of_rows h (row := 1808) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_17
  simp only [← c110436, ← c110437, k2, ← c110438] at e_1808_17
  have hr := e_1808_17
  simp only [e_1808_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6375)) := by
  have c110439 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.1
  have c110440 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.1
  have c110441 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.1
  have c110442 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110443 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110444 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_18 := arithEq_of_rows h (row := 1808) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_18
  simp only [← c110439, ← c110440, ← c110441] at e_1808_18
  have e_1808_19 := arithEq_of_rows h (row := 1808) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_19
  simp only [← c110442, ← c110443, k2, ← c110444] at e_1808_19
  have hr := e_1808_19
  simp only [e_1808_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6376)) := by
  have c110445 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110446 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110447 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110448 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110449 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110450 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_0 := arithEq_of_rows h (row := 1809) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_0
  simp only [← c110445, ← c110446, ← c110447] at e_1809_0
  have e_1809_1 := arithEq_of_rows h (row := 1809) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_1
  simp only [← c110448, ← c110449, k2, ← c110450] at e_1809_1
  have hr := e_1809_1
  simp only [e_1809_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6377)) := by
  have c110451 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110452 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110453 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110454 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110455 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110456 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_2 := arithEq_of_rows h (row := 1809) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_2
  simp only [← c110451, ← c110452, ← c110453] at e_1809_2
  have e_1809_3 := arithEq_of_rows h (row := 1809) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_3
  simp only [← c110454, ← c110455, k2, ← c110456] at e_1809_3
  have hr := e_1809_3
  simp only [e_1809_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6378)) := by
  have c110457 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110458 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110459 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110460 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110461 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110462 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_4 := arithEq_of_rows h (row := 1809) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_4
  simp only [← c110457, ← c110458, ← c110459] at e_1809_4
  have e_1809_5 := arithEq_of_rows h (row := 1809) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_5
  simp only [← c110460, ← c110461, k2, ← c110462] at e_1809_5
  have hr := e_1809_5
  simp only [e_1809_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6379)) := by
  have c110463 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110464 := (publicBatchWrapper86_copies3452 a h).1
  have c110465 := (publicBatchWrapper86_copies3452 a h).2.1
  have c110466 := (publicBatchWrapper86_copies3452 a h).2.2.1
  have c110467 := (publicBatchWrapper86_copies3452 a h).2.2.2.1
  have c110468 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_6 := arithEq_of_rows h (row := 1809) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_6
  simp only [← c110463, ← c110464, ← c110465] at e_1809_6
  have e_1809_7 := arithEq_of_rows h (row := 1809) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_7
  simp only [← c110466, ← c110467, k2, ← c110468] at e_1809_7
  have hr := e_1809_7
  simp only [e_1809_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6380)) := by
  have c110469 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.1
  have c110470 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.1
  have c110471 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.1
  have c110472 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.1
  have c110473 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.1
  have c110474 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_8 := arithEq_of_rows h (row := 1809) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_8
  simp only [← c110469, ← c110470, ← c110471] at e_1809_8
  have e_1809_9 := arithEq_of_rows h (row := 1809) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_9
  simp only [← c110472, ← c110473, k2, ← c110474] at e_1809_9
  have hr := e_1809_9
  simp only [e_1809_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6381)) := by
  have c110475 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110476 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110477 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110478 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110479 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110480 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_10 := arithEq_of_rows h (row := 1809) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_10
  simp only [← c110475, ← c110476, ← c110477] at e_1809_10
  have e_1809_11 := arithEq_of_rows h (row := 1809) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_11
  simp only [← c110478, ← c110479, k2, ← c110480] at e_1809_11
  have hr := e_1809_11
  simp only [e_1809_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6382)) := by
  have c110481 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110482 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110483 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110484 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110485 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110486 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_12 := arithEq_of_rows h (row := 1809) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_12
  simp only [← c110481, ← c110482, ← c110483] at e_1809_12
  have e_1809_13 := arithEq_of_rows h (row := 1809) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_13
  simp only [← c110484, ← c110485, k2, ← c110486] at e_1809_13
  have hr := e_1809_13
  simp only [e_1809_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6383)) := by
  have c110487 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110488 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110489 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110490 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110491 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110492 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_14 := arithEq_of_rows h (row := 1809) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_14
  simp only [← c110487, ← c110488, ← c110489] at e_1809_14
  have e_1809_15 := arithEq_of_rows h (row := 1809) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_15
  simp only [← c110490, ← c110491, k2, ← c110492] at e_1809_15
  have hr := e_1809_15
  simp only [e_1809_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6384)) := by
  have c110493 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110494 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110495 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110496 := (publicBatchWrapper86_copies3453 a h).1
  have c110497 := (publicBatchWrapper86_copies3453 a h).2.1
  have c110498 := (publicBatchWrapper86_copies3453 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_16 := arithEq_of_rows h (row := 1809) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_16
  simp only [← c110493, ← c110494, ← c110495] at e_1809_16
  have e_1809_17 := arithEq_of_rows h (row := 1809) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_17
  simp only [← c110496, ← c110497, k2, ← c110498] at e_1809_17
  have hr := e_1809_17
  simp only [e_1809_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6385)) := by
  have c110499 := (publicBatchWrapper86_copies3453 a h).2.2.2.1
  have c110500 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.1
  have c110501 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.1
  have c110502 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.1
  have c110503 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.1
  have c110504 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_18 := arithEq_of_rows h (row := 1809) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_18
  simp only [← c110499, ← c110500, ← c110501] at e_1809_18
  have e_1809_19 := arithEq_of_rows h (row := 1809) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_19
  simp only [← c110502, ← c110503, k2, ← c110504] at e_1809_19
  have hr := e_1809_19
  simp only [e_1809_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6386)) := by
  have c110505 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.1
  have c110506 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110507 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110508 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110509 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110510 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_0 := arithEq_of_rows h (row := 1810) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_0
  simp only [← c110505, ← c110506, ← c110507] at e_1810_0
  have e_1810_1 := arithEq_of_rows h (row := 1810) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_1
  simp only [← c110508, ← c110509, k2, ← c110510] at e_1810_1
  have hr := e_1810_1
  simp only [e_1810_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6387)) := by
  have c110511 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110512 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110513 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110514 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110515 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110516 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_2 := arithEq_of_rows h (row := 1810) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_2
  simp only [← c110511, ← c110512, ← c110513] at e_1810_2
  have e_1810_3 := arithEq_of_rows h (row := 1810) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_3
  simp only [← c110514, ← c110515, k2, ← c110516] at e_1810_3
  have hr := e_1810_3
  simp only [e_1810_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6388)) := by
  have c110517 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110518 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110519 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110520 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110521 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110522 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_4 := arithEq_of_rows h (row := 1810) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_4
  simp only [← c110517, ← c110518, ← c110519] at e_1810_4
  have e_1810_5 := arithEq_of_rows h (row := 1810) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_5
  simp only [← c110520, ← c110521, k2, ← c110522] at e_1810_5
  have hr := e_1810_5
  simp only [e_1810_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6389)) := by
  have c110523 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110524 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110525 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110526 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110527 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110528 := (publicBatchWrapper86_copies3454 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_6 := arithEq_of_rows h (row := 1810) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_6
  simp only [← c110523, ← c110524, ← c110525] at e_1810_6
  have e_1810_7 := arithEq_of_rows h (row := 1810) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_7
  simp only [← c110526, ← c110527, k2, ← c110528] at e_1810_7
  have hr := e_1810_7
  simp only [e_1810_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6390)) := by
  have c110529 := (publicBatchWrapper86_copies3454 a h).2.1
  have c110530 := (publicBatchWrapper86_copies3454 a h).2.2.1
  have c110531 := (publicBatchWrapper86_copies3454 a h).2.2.2.1
  have c110532 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.1
  have c110533 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.1
  have c110534 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_8 := arithEq_of_rows h (row := 1810) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_8
  simp only [← c110529, ← c110530, ← c110531] at e_1810_8
  have e_1810_9 := arithEq_of_rows h (row := 1810) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_9
  simp only [← c110532, ← c110533, k2, ← c110534] at e_1810_9
  have hr := e_1810_9
  simp only [e_1810_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6391)) := by
  have c110535 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.1
  have c110536 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.1
  have c110537 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.1
  have c110538 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110539 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110540 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_10 := arithEq_of_rows h (row := 1810) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_10
  simp only [← c110535, ← c110536, ← c110537] at e_1810_10
  have e_1810_11 := arithEq_of_rows h (row := 1810) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_11
  simp only [← c110538, ← c110539, k2, ← c110540] at e_1810_11
  have hr := e_1810_11
  simp only [e_1810_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6392)) := by
  have c110541 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110542 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110543 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110544 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110545 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110546 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_12 := arithEq_of_rows h (row := 1810) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_12
  simp only [← c110541, ← c110542, ← c110543] at e_1810_12
  have e_1810_13 := arithEq_of_rows h (row := 1810) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_13
  simp only [← c110544, ← c110545, k2, ← c110546] at e_1810_13
  have hr := e_1810_13
  simp only [e_1810_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6393)) := by
  have c110547 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110548 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110549 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110550 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110551 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110552 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_14 := arithEq_of_rows h (row := 1810) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_14
  simp only [← c110547, ← c110548, ← c110549] at e_1810_14
  have e_1810_15 := arithEq_of_rows h (row := 1810) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_15
  simp only [← c110550, ← c110551, k2, ← c110552] at e_1810_15
  have hr := e_1810_15
  simp only [e_1810_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6394)) := by
  have c110553 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110554 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110555 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110556 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110557 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110558 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_16 := arithEq_of_rows h (row := 1810) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_16
  simp only [← c110553, ← c110554, ← c110555] at e_1810_16
  have e_1810_17 := arithEq_of_rows h (row := 1810) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_17
  simp only [← c110556, ← c110557, k2, ← c110558] at e_1810_17
  have hr := e_1810_17
  simp only [e_1810_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6395)) := by
  have c110559 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110560 := (publicBatchWrapper86_copies3455 a h).1
  have c110561 := (publicBatchWrapper86_copies3455 a h).2.1
  have c110562 := (publicBatchWrapper86_copies3455 a h).2.2.1
  have c110563 := (publicBatchWrapper86_copies3455 a h).2.2.2.1
  have c110564 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_18 := arithEq_of_rows h (row := 1810) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_18
  simp only [← c110559, ← c110560, ← c110561] at e_1810_18
  have e_1810_19 := arithEq_of_rows h (row := 1810) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_19
  simp only [← c110562, ← c110563, k2, ← c110564] at e_1810_19
  have hr := e_1810_19
  simp only [e_1810_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6396)) := by
  have c110565 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.1
  have c110566 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.1
  have c110567 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.1
  have c110568 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.1
  have c110569 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.1
  have c110570 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_0 := arithEq_of_rows h (row := 1811) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_0
  simp only [← c110565, ← c110566, ← c110567] at e_1811_0
  have e_1811_1 := arithEq_of_rows h (row := 1811) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_1
  simp only [← c110568, ← c110569, k2, ← c110570] at e_1811_1
  have hr := e_1811_1
  simp only [e_1811_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6397)) := by
  have c110571 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110572 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110573 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110574 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110575 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110576 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_2 := arithEq_of_rows h (row := 1811) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_2
  simp only [← c110571, ← c110572, ← c110573] at e_1811_2
  have e_1811_3 := arithEq_of_rows h (row := 1811) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_3
  simp only [← c110574, ← c110575, k2, ← c110576] at e_1811_3
  have hr := e_1811_3
  simp only [e_1811_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6398)) := by
  have c110577 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110578 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110579 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110580 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110581 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110582 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_4 := arithEq_of_rows h (row := 1811) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_4
  simp only [← c110577, ← c110578, ← c110579] at e_1811_4
  have e_1811_5 := arithEq_of_rows h (row := 1811) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_5
  simp only [← c110580, ← c110581, k2, ← c110582] at e_1811_5
  have hr := e_1811_5
  simp only [e_1811_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6399)) := by
  have c110583 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110584 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110585 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110586 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110587 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110588 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_6 := arithEq_of_rows h (row := 1811) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_6
  simp only [← c110583, ← c110584, ← c110585] at e_1811_6
  have e_1811_7 := arithEq_of_rows h (row := 1811) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_7
  simp only [← c110586, ← c110587, k2, ← c110588] at e_1811_7
  have hr := e_1811_7
  simp only [e_1811_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6400)) := by
  have c110589 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110590 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110591 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110592 := (publicBatchWrapper86_copies3456 a h).1
  have c110593 := (publicBatchWrapper86_copies3456 a h).2.1
  have c110594 := (publicBatchWrapper86_copies3456 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_8 := arithEq_of_rows h (row := 1811) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_8
  simp only [← c110589, ← c110590, ← c110591] at e_1811_8
  have e_1811_9 := arithEq_of_rows h (row := 1811) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_9
  simp only [← c110592, ← c110593, k2, ← c110594] at e_1811_9
  have hr := e_1811_9
  simp only [e_1811_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6401)) := by
  have c110595 := (publicBatchWrapper86_copies3456 a h).2.2.2.1
  have c110596 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.1
  have c110597 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.1
  have c110598 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.1
  have c110599 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.1
  have c110600 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_10 := arithEq_of_rows h (row := 1811) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_10
  simp only [← c110595, ← c110596, ← c110597] at e_1811_10
  have e_1811_11 := arithEq_of_rows h (row := 1811) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_11
  simp only [← c110598, ← c110599, k2, ← c110600] at e_1811_11
  have hr := e_1811_11
  simp only [e_1811_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6570)) := by
  have c110601 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.1
  have c110602 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110603 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110604 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110605 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110606 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_12 := arithEq_of_rows h (row := 1811) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_12
  simp only [← c110601, ← c110602, ← c110603] at e_1811_12
  have e_1811_13 := arithEq_of_rows h (row := 1811) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_13
  simp only [← c110604, ← c110605, k2, ← c110606] at e_1811_13
  have hr := e_1811_13
  simp only [e_1811_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6571)) := by
  have c110607 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110608 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110609 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110610 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110611 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110612 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_14 := arithEq_of_rows h (row := 1811) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_14
  simp only [← c110607, ← c110608, ← c110609] at e_1811_14
  have e_1811_15 := arithEq_of_rows h (row := 1811) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_15
  simp only [← c110610, ← c110611, k2, ← c110612] at e_1811_15
  have hr := e_1811_15
  simp only [e_1811_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6572)) := by
  have c110613 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110614 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110615 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110616 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110617 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110618 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_16 := arithEq_of_rows h (row := 1811) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_16
  simp only [← c110613, ← c110614, ← c110615] at e_1811_16
  have e_1811_17 := arithEq_of_rows h (row := 1811) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_17
  simp only [← c110616, ← c110617, k2, ← c110618] at e_1811_17
  have hr := e_1811_17
  simp only [e_1811_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6573)) := by
  have c110619 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110620 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110621 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110622 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110623 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110624 := (publicBatchWrapper86_copies3457 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_18 := arithEq_of_rows h (row := 1811) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_18
  simp only [← c110619, ← c110620, ← c110621] at e_1811_18
  have e_1811_19 := arithEq_of_rows h (row := 1811) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_19
  simp only [← c110622, ← c110623, k2, ← c110624] at e_1811_19
  have hr := e_1811_19
  simp only [e_1811_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6574)) := by
  have c110625 := (publicBatchWrapper86_copies3457 a h).2.1
  have c110626 := (publicBatchWrapper86_copies3457 a h).2.2.1
  have c110627 := (publicBatchWrapper86_copies3457 a h).2.2.2.1
  have c110628 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.1
  have c110629 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.1
  have c110630 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_0 := arithEq_of_rows h (row := 1812) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_0
  simp only [← c110625, ← c110626, ← c110627] at e_1812_0
  have e_1812_1 := arithEq_of_rows h (row := 1812) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_1
  simp only [← c110628, ← c110629, k2, ← c110630] at e_1812_1
  have hr := e_1812_1
  simp only [e_1812_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6575)) := by
  have c110631 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.1
  have c110632 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.1
  have c110633 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.1
  have c110634 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110635 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110636 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_2 := arithEq_of_rows h (row := 1812) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_2
  simp only [← c110631, ← c110632, ← c110633] at e_1812_2
  have e_1812_3 := arithEq_of_rows h (row := 1812) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_3
  simp only [← c110634, ← c110635, k2, ← c110636] at e_1812_3
  have hr := e_1812_3
  simp only [e_1812_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6576)) := by
  have c110637 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110638 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110639 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110640 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110641 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110642 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_4 := arithEq_of_rows h (row := 1812) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_4
  simp only [← c110637, ← c110638, ← c110639] at e_1812_4
  have e_1812_5 := arithEq_of_rows h (row := 1812) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_5
  simp only [← c110640, ← c110641, k2, ← c110642] at e_1812_5
  have hr := e_1812_5
  simp only [e_1812_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6577)) := by
  have c110643 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110644 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110645 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110646 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110647 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110648 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_6 := arithEq_of_rows h (row := 1812) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_6
  simp only [← c110643, ← c110644, ← c110645] at e_1812_6
  have e_1812_7 := arithEq_of_rows h (row := 1812) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_7
  simp only [← c110646, ← c110647, k2, ← c110648] at e_1812_7
  have hr := e_1812_7
  simp only [e_1812_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6578)) := by
  have c110649 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110650 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110651 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110652 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110653 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110654 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_8 := arithEq_of_rows h (row := 1812) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_8
  simp only [← c110649, ← c110650, ← c110651] at e_1812_8
  have e_1812_9 := arithEq_of_rows h (row := 1812) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_9
  simp only [← c110652, ← c110653, k2, ← c110654] at e_1812_9
  have hr := e_1812_9
  simp only [e_1812_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6579)) := by
  have c110655 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110656 := (publicBatchWrapper86_copies3458 a h).1
  have c110657 := (publicBatchWrapper86_copies3458 a h).2.1
  have c110658 := (publicBatchWrapper86_copies3458 a h).2.2.1
  have c110659 := (publicBatchWrapper86_copies3458 a h).2.2.2.1
  have c110660 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_10 := arithEq_of_rows h (row := 1812) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_10
  simp only [← c110655, ← c110656, ← c110657] at e_1812_10
  have e_1812_11 := arithEq_of_rows h (row := 1812) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_11
  simp only [← c110658, ← c110659, k2, ← c110660] at e_1812_11
  have hr := e_1812_11
  simp only [e_1812_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6580)) := by
  have c110661 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.1
  have c110662 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.1
  have c110663 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.1
  have c110664 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.1
  have c110665 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.1
  have c110666 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_12 := arithEq_of_rows h (row := 1812) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_12
  simp only [← c110661, ← c110662, ← c110663] at e_1812_12
  have e_1812_13 := arithEq_of_rows h (row := 1812) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_13
  simp only [← c110664, ← c110665, k2, ← c110666] at e_1812_13
  have hr := e_1812_13
  simp only [e_1812_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6581)) := by
  have c110667 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110668 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110669 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110670 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110671 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110672 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_14 := arithEq_of_rows h (row := 1812) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_14
  simp only [← c110667, ← c110668, ← c110669] at e_1812_14
  have e_1812_15 := arithEq_of_rows h (row := 1812) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_15
  simp only [← c110670, ← c110671, k2, ← c110672] at e_1812_15
  have hr := e_1812_15
  simp only [e_1812_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6582)) := by
  have c110673 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110674 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110675 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110676 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110677 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110678 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_16 := arithEq_of_rows h (row := 1812) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_16
  simp only [← c110673, ← c110674, ← c110675] at e_1812_16
  have e_1812_17 := arithEq_of_rows h (row := 1812) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_17
  simp only [← c110676, ← c110677, k2, ← c110678] at e_1812_17
  have hr := e_1812_17
  simp only [e_1812_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6583)) := by
  have c110679 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110680 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110681 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110682 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110683 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110684 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_18 := arithEq_of_rows h (row := 1812) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_18
  simp only [← c110679, ← c110680, ← c110681] at e_1812_18
  have e_1812_19 := arithEq_of_rows h (row := 1812) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_19
  simp only [← c110682, ← c110683, k2, ← c110684] at e_1812_19
  have hr := e_1812_19
  simp only [e_1812_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6584)) := by
  have c110685 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110686 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110687 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110688 := (publicBatchWrapper86_copies3459 a h).1
  have c110689 := (publicBatchWrapper86_copies3459 a h).2.1
  have c110690 := (publicBatchWrapper86_copies3459 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_0 := arithEq_of_rows h (row := 1813) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_0
  simp only [← c110685, ← c110686, ← c110687] at e_1813_0
  have e_1813_1 := arithEq_of_rows h (row := 1813) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_1
  simp only [← c110688, ← c110689, k2, ← c110690] at e_1813_1
  have hr := e_1813_1
  simp only [e_1813_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6585)) := by
  have c110691 := (publicBatchWrapper86_copies3459 a h).2.2.2.1
  have c110692 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.1
  have c110693 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.1
  have c110694 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.1
  have c110695 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.1
  have c110696 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_2 := arithEq_of_rows h (row := 1813) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_2
  simp only [← c110691, ← c110692, ← c110693] at e_1813_2
  have e_1813_3 := arithEq_of_rows h (row := 1813) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_3
  simp only [← c110694, ← c110695, k2, ← c110696] at e_1813_3
  have hr := e_1813_3
  simp only [e_1813_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6586)) := by
  have c110697 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.1
  have c110698 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110699 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110700 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110701 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110702 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_4 := arithEq_of_rows h (row := 1813) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_4
  simp only [← c110697, ← c110698, ← c110699] at e_1813_4
  have e_1813_5 := arithEq_of_rows h (row := 1813) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_5
  simp only [← c110700, ← c110701, k2, ← c110702] at e_1813_5
  have hr := e_1813_5
  simp only [e_1813_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6587)) := by
  have c110703 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110704 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110705 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110706 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110707 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110708 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_6 := arithEq_of_rows h (row := 1813) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_6
  simp only [← c110703, ← c110704, ← c110705] at e_1813_6
  have e_1813_7 := arithEq_of_rows h (row := 1813) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_7
  simp only [← c110706, ← c110707, k2, ← c110708] at e_1813_7
  have hr := e_1813_7
  simp only [e_1813_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6588)) := by
  have c110709 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110710 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110711 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110712 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110713 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110714 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_8 := arithEq_of_rows h (row := 1813) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_8
  simp only [← c110709, ← c110710, ← c110711] at e_1813_8
  have e_1813_9 := arithEq_of_rows h (row := 1813) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_9
  simp only [← c110712, ← c110713, k2, ← c110714] at e_1813_9
  have hr := e_1813_9
  simp only [e_1813_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6589)) := by
  have c110715 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110716 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110717 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110718 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110719 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110720 := (publicBatchWrapper86_copies3460 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_10 := arithEq_of_rows h (row := 1813) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_10
  simp only [← c110715, ← c110716, ← c110717] at e_1813_10
  have e_1813_11 := arithEq_of_rows h (row := 1813) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_11
  simp only [← c110718, ← c110719, k2, ← c110720] at e_1813_11
  have hr := e_1813_11
  simp only [e_1813_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6590)) := by
  have c110721 := (publicBatchWrapper86_copies3460 a h).2.1
  have c110722 := (publicBatchWrapper86_copies3460 a h).2.2.1
  have c110723 := (publicBatchWrapper86_copies3460 a h).2.2.2.1
  have c110724 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.1
  have c110725 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.1
  have c110726 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_12 := arithEq_of_rows h (row := 1813) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_12
  simp only [← c110721, ← c110722, ← c110723] at e_1813_12
  have e_1813_13 := arithEq_of_rows h (row := 1813) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_13
  simp only [← c110724, ← c110725, k2, ← c110726] at e_1813_13
  have hr := e_1813_13
  simp only [e_1813_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6591)) := by
  have c110727 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.1
  have c110728 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.1
  have c110729 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.1
  have c110730 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110731 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110732 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_14 := arithEq_of_rows h (row := 1813) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_14
  simp only [← c110727, ← c110728, ← c110729] at e_1813_14
  have e_1813_15 := arithEq_of_rows h (row := 1813) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_15
  simp only [← c110730, ← c110731, k2, ← c110732] at e_1813_15
  have hr := e_1813_15
  simp only [e_1813_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6592)) := by
  have c110733 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110734 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110735 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110736 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110737 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110738 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_16 := arithEq_of_rows h (row := 1813) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_16
  simp only [← c110733, ← c110734, ← c110735] at e_1813_16
  have e_1813_17 := arithEq_of_rows h (row := 1813) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_17
  simp only [← c110736, ← c110737, k2, ← c110738] at e_1813_17
  have hr := e_1813_17
  simp only [e_1813_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6593)) := by
  have c110739 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110740 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110741 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110742 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110743 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110744 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_18 := arithEq_of_rows h (row := 1813) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_18
  simp only [← c110739, ← c110740, ← c110741] at e_1813_18
  have e_1813_19 := arithEq_of_rows h (row := 1813) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_19
  simp only [← c110742, ← c110743, k2, ← c110744] at e_1813_19
  have hr := e_1813_19
  simp only [e_1813_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6594)) := by
  have c110745 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110746 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110747 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110748 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110749 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110750 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_0 := arithEq_of_rows h (row := 1814) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_0
  simp only [← c110745, ← c110746, ← c110747] at e_1814_0
  have e_1814_1 := arithEq_of_rows h (row := 1814) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_1
  simp only [← c110748, ← c110749, k2, ← c110750] at e_1814_1
  have hr := e_1814_1
  simp only [e_1814_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6595)) := by
  have c110751 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110752 := (publicBatchWrapper86_copies3461 a h).1
  have c110753 := (publicBatchWrapper86_copies3461 a h).2.1
  have c110754 := (publicBatchWrapper86_copies3461 a h).2.2.1
  have c110755 := (publicBatchWrapper86_copies3461 a h).2.2.2.1
  have c110756 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_2 := arithEq_of_rows h (row := 1814) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_2
  simp only [← c110751, ← c110752, ← c110753] at e_1814_2
  have e_1814_3 := arithEq_of_rows h (row := 1814) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_3
  simp only [← c110754, ← c110755, k2, ← c110756] at e_1814_3
  have hr := e_1814_3
  simp only [e_1814_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6596)) := by
  have c110757 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.1
  have c110758 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.1
  have c110759 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.1
  have c110760 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.1
  have c110761 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.1
  have c110762 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_4 := arithEq_of_rows h (row := 1814) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_4
  simp only [← c110757, ← c110758, ← c110759] at e_1814_4
  have e_1814_5 := arithEq_of_rows h (row := 1814) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_5
  simp only [← c110760, ← c110761, k2, ← c110762] at e_1814_5
  have hr := e_1814_5
  simp only [e_1814_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6597)) := by
  have c110763 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110764 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110765 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110766 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110767 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110768 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_6 := arithEq_of_rows h (row := 1814) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_6
  simp only [← c110763, ← c110764, ← c110765] at e_1814_6
  have e_1814_7 := arithEq_of_rows h (row := 1814) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_7
  simp only [← c110766, ← c110767, k2, ← c110768] at e_1814_7
  have hr := e_1814_7
  simp only [e_1814_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6598)) := by
  have c110769 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110770 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110771 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110772 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110773 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110774 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_8 := arithEq_of_rows h (row := 1814) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_8
  simp only [← c110769, ← c110770, ← c110771] at e_1814_8
  have e_1814_9 := arithEq_of_rows h (row := 1814) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_9
  simp only [← c110772, ← c110773, k2, ← c110774] at e_1814_9
  have hr := e_1814_9
  simp only [e_1814_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6599)) := by
  have c110775 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110776 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110777 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110778 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110779 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110780 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_10 := arithEq_of_rows h (row := 1814) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_10
  simp only [← c110775, ← c110776, ← c110777] at e_1814_10
  have e_1814_11 := arithEq_of_rows h (row := 1814) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_11
  simp only [← c110778, ← c110779, k2, ← c110780] at e_1814_11
  have hr := e_1814_11
  simp only [e_1814_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6600)) := by
  have c110781 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110782 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110783 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110784 := (publicBatchWrapper86_copies3462 a h).1
  have c110785 := (publicBatchWrapper86_copies3462 a h).2.1
  have c110786 := (publicBatchWrapper86_copies3462 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_12 := arithEq_of_rows h (row := 1814) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_12
  simp only [← c110781, ← c110782, ← c110783] at e_1814_12
  have e_1814_13 := arithEq_of_rows h (row := 1814) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_13
  simp only [← c110784, ← c110785, k2, ← c110786] at e_1814_13
  have hr := e_1814_13
  simp only [e_1814_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6601)) := by
  have c110787 := (publicBatchWrapper86_copies3462 a h).2.2.2.1
  have c110788 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.1
  have c110789 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.1
  have c110790 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.1
  have c110791 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.1
  have c110792 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_14 := arithEq_of_rows h (row := 1814) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_14
  simp only [← c110787, ← c110788, ← c110789] at e_1814_14
  have e_1814_15 := arithEq_of_rows h (row := 1814) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_15
  simp only [← c110790, ← c110791, k2, ← c110792] at e_1814_15
  have hr := e_1814_15
  simp only [e_1814_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6770)) := by
  have c110793 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.1
  have c110794 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110795 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110796 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110797 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110798 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_16 := arithEq_of_rows h (row := 1814) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_16
  simp only [← c110793, ← c110794, ← c110795] at e_1814_16
  have e_1814_17 := arithEq_of_rows h (row := 1814) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_17
  simp only [← c110796, ← c110797, k2, ← c110798] at e_1814_17
  have hr := e_1814_17
  simp only [e_1814_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6771)) := by
  have c110799 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110800 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110801 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110802 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110803 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110804 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_18 := arithEq_of_rows h (row := 1814) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_18
  simp only [← c110799, ← c110800, ← c110801] at e_1814_18
  have e_1814_19 := arithEq_of_rows h (row := 1814) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_19
  simp only [← c110802, ← c110803, k2, ← c110804] at e_1814_19
  have hr := e_1814_19
  simp only [e_1814_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6772)) := by
  have c110805 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110806 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110807 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110808 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110809 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110810 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_0 := arithEq_of_rows h (row := 1815) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_0
  simp only [← c110805, ← c110806, ← c110807] at e_1815_0
  have e_1815_1 := arithEq_of_rows h (row := 1815) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_1
  simp only [← c110808, ← c110809, k2, ← c110810] at e_1815_1
  have hr := e_1815_1
  simp only [e_1815_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6773)) := by
  have c110811 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110812 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110813 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110814 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110815 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110816 := (publicBatchWrapper86_copies3463 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_2 := arithEq_of_rows h (row := 1815) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_2
  simp only [← c110811, ← c110812, ← c110813] at e_1815_2
  have e_1815_3 := arithEq_of_rows h (row := 1815) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_3
  simp only [← c110814, ← c110815, k2, ← c110816] at e_1815_3
  have hr := e_1815_3
  simp only [e_1815_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6774)) := by
  have c110817 := (publicBatchWrapper86_copies3463 a h).2.1
  have c110818 := (publicBatchWrapper86_copies3463 a h).2.2.1
  have c110819 := (publicBatchWrapper86_copies3463 a h).2.2.2.1
  have c110820 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.1
  have c110821 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.1
  have c110822 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_4 := arithEq_of_rows h (row := 1815) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_4
  simp only [← c110817, ← c110818, ← c110819] at e_1815_4
  have e_1815_5 := arithEq_of_rows h (row := 1815) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_5
  simp only [← c110820, ← c110821, k2, ← c110822] at e_1815_5
  have hr := e_1815_5
  simp only [e_1815_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6775)) := by
  have c110823 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.1
  have c110824 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.1
  have c110825 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.1
  have c110826 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110827 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110828 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_6 := arithEq_of_rows h (row := 1815) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_6
  simp only [← c110823, ← c110824, ← c110825] at e_1815_6
  have e_1815_7 := arithEq_of_rows h (row := 1815) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_7
  simp only [← c110826, ← c110827, k2, ← c110828] at e_1815_7
  have hr := e_1815_7
  simp only [e_1815_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6776)) := by
  have c110829 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110830 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110831 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110832 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110833 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110834 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_8 := arithEq_of_rows h (row := 1815) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_8
  simp only [← c110829, ← c110830, ← c110831] at e_1815_8
  have e_1815_9 := arithEq_of_rows h (row := 1815) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_9
  simp only [← c110832, ← c110833, k2, ← c110834] at e_1815_9
  have hr := e_1815_9
  simp only [e_1815_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6777)) := by
  have c110835 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110836 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110837 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110838 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110839 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110840 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_10 := arithEq_of_rows h (row := 1815) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_10
  simp only [← c110835, ← c110836, ← c110837] at e_1815_10
  have e_1815_11 := arithEq_of_rows h (row := 1815) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_11
  simp only [← c110838, ← c110839, k2, ← c110840] at e_1815_11
  have hr := e_1815_11
  simp only [e_1815_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6778)) := by
  have c110841 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110842 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110843 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110844 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110845 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110846 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_12 := arithEq_of_rows h (row := 1815) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_12
  simp only [← c110841, ← c110842, ← c110843] at e_1815_12
  have e_1815_13 := arithEq_of_rows h (row := 1815) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_13
  simp only [← c110844, ← c110845, k2, ← c110846] at e_1815_13
  have hr := e_1815_13
  simp only [e_1815_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6779)) := by
  have c110847 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110848 := (publicBatchWrapper86_copies3464 a h).1
  have c110849 := (publicBatchWrapper86_copies3464 a h).2.1
  have c110850 := (publicBatchWrapper86_copies3464 a h).2.2.1
  have c110851 := (publicBatchWrapper86_copies3464 a h).2.2.2.1
  have c110852 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_14 := arithEq_of_rows h (row := 1815) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_14
  simp only [← c110847, ← c110848, ← c110849] at e_1815_14
  have e_1815_15 := arithEq_of_rows h (row := 1815) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_15
  simp only [← c110850, ← c110851, k2, ← c110852] at e_1815_15
  have hr := e_1815_15
  simp only [e_1815_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6780)) := by
  have c110853 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.1
  have c110854 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.1
  have c110855 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.1
  have c110856 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.1
  have c110857 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.1
  have c110858 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_16 := arithEq_of_rows h (row := 1815) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_16
  simp only [← c110853, ← c110854, ← c110855] at e_1815_16
  have e_1815_17 := arithEq_of_rows h (row := 1815) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_17
  simp only [← c110856, ← c110857, k2, ← c110858] at e_1815_17
  have hr := e_1815_17
  simp only [e_1815_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6781)) := by
  have c110859 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110860 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110861 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110862 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110863 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110864 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_18 := arithEq_of_rows h (row := 1815) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_18
  simp only [← c110859, ← c110860, ← c110861] at e_1815_18
  have e_1815_19 := arithEq_of_rows h (row := 1815) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_19
  simp only [← c110862, ← c110863, k2, ← c110864] at e_1815_19
  have hr := e_1815_19
  simp only [e_1815_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6782)) := by
  have c110865 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110866 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110867 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110868 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110869 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110870 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_0 := arithEq_of_rows h (row := 1816) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_0
  simp only [← c110865, ← c110866, ← c110867] at e_1816_0
  have e_1816_1 := arithEq_of_rows h (row := 1816) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_1
  simp only [← c110868, ← c110869, k2, ← c110870] at e_1816_1
  have hr := e_1816_1
  simp only [e_1816_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6783)) := by
  have c110871 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110872 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110873 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110874 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110875 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110876 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_2 := arithEq_of_rows h (row := 1816) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_2
  simp only [← c110871, ← c110872, ← c110873] at e_1816_2
  have e_1816_3 := arithEq_of_rows h (row := 1816) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_3
  simp only [← c110874, ← c110875, k2, ← c110876] at e_1816_3
  have hr := e_1816_3
  simp only [e_1816_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6784)) := by
  have c110877 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110878 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110879 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110880 := (publicBatchWrapper86_copies3465 a h).1
  have c110881 := (publicBatchWrapper86_copies3465 a h).2.1
  have c110882 := (publicBatchWrapper86_copies3465 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_4 := arithEq_of_rows h (row := 1816) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_4
  simp only [← c110877, ← c110878, ← c110879] at e_1816_4
  have e_1816_5 := arithEq_of_rows h (row := 1816) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_5
  simp only [← c110880, ← c110881, k2, ← c110882] at e_1816_5
  have hr := e_1816_5
  simp only [e_1816_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6785)) := by
  have c110883 := (publicBatchWrapper86_copies3465 a h).2.2.2.1
  have c110884 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.1
  have c110885 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.1
  have c110886 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.1
  have c110887 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.1
  have c110888 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_6 := arithEq_of_rows h (row := 1816) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_6
  simp only [← c110883, ← c110884, ← c110885] at e_1816_6
  have e_1816_7 := arithEq_of_rows h (row := 1816) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_7
  simp only [← c110886, ← c110887, k2, ← c110888] at e_1816_7
  have hr := e_1816_7
  simp only [e_1816_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6786)) := by
  have c110889 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.1
  have c110890 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110891 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110892 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110893 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110894 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_8 := arithEq_of_rows h (row := 1816) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_8
  simp only [← c110889, ← c110890, ← c110891] at e_1816_8
  have e_1816_9 := arithEq_of_rows h (row := 1816) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_9
  simp only [← c110892, ← c110893, k2, ← c110894] at e_1816_9
  have hr := e_1816_9
  simp only [e_1816_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6787)) := by
  have c110895 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110896 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110897 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110898 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110899 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110900 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_10 := arithEq_of_rows h (row := 1816) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_10
  simp only [← c110895, ← c110896, ← c110897] at e_1816_10
  have e_1816_11 := arithEq_of_rows h (row := 1816) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_11
  simp only [← c110898, ← c110899, k2, ← c110900] at e_1816_11
  have hr := e_1816_11
  simp only [e_1816_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6788)) := by
  have c110901 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110902 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110903 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110904 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110905 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110906 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_12 := arithEq_of_rows h (row := 1816) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_12
  simp only [← c110901, ← c110902, ← c110903] at e_1816_12
  have e_1816_13 := arithEq_of_rows h (row := 1816) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_13
  simp only [← c110904, ← c110905, k2, ← c110906] at e_1816_13
  have hr := e_1816_13
  simp only [e_1816_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6789)) := by
  have c110907 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110908 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110909 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110910 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110911 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110912 := (publicBatchWrapper86_copies3466 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_14 := arithEq_of_rows h (row := 1816) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_14
  simp only [← c110907, ← c110908, ← c110909] at e_1816_14
  have e_1816_15 := arithEq_of_rows h (row := 1816) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_15
  simp only [← c110910, ← c110911, k2, ← c110912] at e_1816_15
  have hr := e_1816_15
  simp only [e_1816_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6790)) := by
  have c110913 := (publicBatchWrapper86_copies3466 a h).2.1
  have c110914 := (publicBatchWrapper86_copies3466 a h).2.2.1
  have c110915 := (publicBatchWrapper86_copies3466 a h).2.2.2.1
  have c110916 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.1
  have c110917 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.1
  have c110918 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_16 := arithEq_of_rows h (row := 1816) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_16
  simp only [← c110913, ← c110914, ← c110915] at e_1816_16
  have e_1816_17 := arithEq_of_rows h (row := 1816) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_17
  simp only [← c110916, ← c110917, k2, ← c110918] at e_1816_17
  have hr := e_1816_17
  simp only [e_1816_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6791)) := by
  have c110919 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.1
  have c110920 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.1
  have c110921 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.1
  have c110922 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110923 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110924 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_18 := arithEq_of_rows h (row := 1816) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_18
  simp only [← c110919, ← c110920, ← c110921] at e_1816_18
  have e_1816_19 := arithEq_of_rows h (row := 1816) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_19
  simp only [← c110922, ← c110923, k2, ← c110924] at e_1816_19
  have hr := e_1816_19
  simp only [e_1816_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6792)) := by
  have c110925 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110926 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110927 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110928 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110929 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110930 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_0 := arithEq_of_rows h (row := 1817) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_0
  simp only [← c110925, ← c110926, ← c110927] at e_1817_0
  have e_1817_1 := arithEq_of_rows h (row := 1817) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_1
  simp only [← c110928, ← c110929, k2, ← c110930] at e_1817_1
  have hr := e_1817_1
  simp only [e_1817_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6793)) := by
  have c110931 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110932 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110933 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110934 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110935 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110936 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_2 := arithEq_of_rows h (row := 1817) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_2
  simp only [← c110931, ← c110932, ← c110933] at e_1817_2
  have e_1817_3 := arithEq_of_rows h (row := 1817) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_3
  simp only [← c110934, ← c110935, k2, ← c110936] at e_1817_3
  have hr := e_1817_3
  simp only [e_1817_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6794)) := by
  have c110937 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110938 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110939 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110940 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110941 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110942 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_4 := arithEq_of_rows h (row := 1817) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_4
  simp only [← c110937, ← c110938, ← c110939] at e_1817_4
  have e_1817_5 := arithEq_of_rows h (row := 1817) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_5
  simp only [← c110940, ← c110941, k2, ← c110942] at e_1817_5
  have hr := e_1817_5
  simp only [e_1817_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6795)) := by
  have c110943 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110944 := (publicBatchWrapper86_copies3467 a h).1
  have c110945 := (publicBatchWrapper86_copies3467 a h).2.1
  have c110946 := (publicBatchWrapper86_copies3467 a h).2.2.1
  have c110947 := (publicBatchWrapper86_copies3467 a h).2.2.2.1
  have c110948 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_6 := arithEq_of_rows h (row := 1817) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_6
  simp only [← c110943, ← c110944, ← c110945] at e_1817_6
  have e_1817_7 := arithEq_of_rows h (row := 1817) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_7
  simp only [← c110946, ← c110947, k2, ← c110948] at e_1817_7
  have hr := e_1817_7
  simp only [e_1817_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6796)) := by
  have c110949 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.1
  have c110950 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.1
  have c110951 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.1
  have c110952 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.1
  have c110953 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.1
  have c110954 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_8 := arithEq_of_rows h (row := 1817) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_8
  simp only [← c110949, ← c110950, ← c110951] at e_1817_8
  have e_1817_9 := arithEq_of_rows h (row := 1817) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_9
  simp only [← c110952, ← c110953, k2, ← c110954] at e_1817_9
  have hr := e_1817_9
  simp only [e_1817_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6797)) := by
  have c110955 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110956 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110957 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110958 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110959 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110960 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_10 := arithEq_of_rows h (row := 1817) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_10
  simp only [← c110955, ← c110956, ← c110957] at e_1817_10
  have e_1817_11 := arithEq_of_rows h (row := 1817) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_11
  simp only [← c110958, ← c110959, k2, ← c110960] at e_1817_11
  have hr := e_1817_11
  simp only [e_1817_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6798)) := by
  have c110961 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110962 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110963 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110964 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110965 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110966 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_12 := arithEq_of_rows h (row := 1817) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_12
  simp only [← c110961, ← c110962, ← c110963] at e_1817_12
  have e_1817_13 := arithEq_of_rows h (row := 1817) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_13
  simp only [← c110964, ← c110965, k2, ← c110966] at e_1817_13
  have hr := e_1817_13
  simp only [e_1817_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6799)) := by
  have c110967 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110968 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110969 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110970 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110971 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110972 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_14 := arithEq_of_rows h (row := 1817) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_14
  simp only [← c110967, ← c110968, ← c110969] at e_1817_14
  have e_1817_15 := arithEq_of_rows h (row := 1817) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_15
  simp only [← c110970, ← c110971, k2, ← c110972] at e_1817_15
  have hr := e_1817_15
  simp only [e_1817_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6800)) := by
  have c110973 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110974 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110975 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110976 := (publicBatchWrapper86_copies3468 a h).1
  have c110977 := (publicBatchWrapper86_copies3468 a h).2.1
  have c110978 := (publicBatchWrapper86_copies3468 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_16 := arithEq_of_rows h (row := 1817) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_16
  simp only [← c110973, ← c110974, ← c110975] at e_1817_16
  have e_1817_17 := arithEq_of_rows h (row := 1817) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_17
  simp only [← c110976, ← c110977, k2, ← c110978] at e_1817_17
  have hr := e_1817_17
  simp only [e_1817_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6801)) := by
  have c110979 := (publicBatchWrapper86_copies3468 a h).2.2.2.1
  have c110980 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.1
  have c110981 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.1
  have c110982 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.1
  have c110983 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.1
  have c110984 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_18 := arithEq_of_rows h (row := 1817) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_18
  simp only [← c110979, ← c110980, ← c110981] at e_1817_18
  have e_1817_19 := arithEq_of_rows h (row := 1817) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_19
  simp only [← c110982, ← c110983, k2, ← c110984] at e_1817_19
  have hr := e_1817_19
  simp only [e_1817_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6970)) := by
  have c110985 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.1
  have c110986 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110987 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110988 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110989 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110990 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_0 := arithEq_of_rows h (row := 1818) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_0
  simp only [← c110985, ← c110986, ← c110987] at e_1818_0
  have e_1818_1 := arithEq_of_rows h (row := 1818) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_1
  simp only [← c110988, ← c110989, k2, ← c110990] at e_1818_1
  have hr := e_1818_1
  simp only [e_1818_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6971)) := by
  have c110991 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110992 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110993 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110994 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110995 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110996 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_2 := arithEq_of_rows h (row := 1818) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_2
  simp only [← c110991, ← c110992, ← c110993] at e_1818_2
  have e_1818_3 := arithEq_of_rows h (row := 1818) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_3
  simp only [← c110994, ← c110995, k2, ← c110996] at e_1818_3
  have hr := e_1818_3
  simp only [e_1818_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6972)) := by
  have c110997 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110998 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110999 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111000 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111001 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111002 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_4 := arithEq_of_rows h (row := 1818) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_4
  simp only [← c110997, ← c110998, ← c110999] at e_1818_4
  have e_1818_5 := arithEq_of_rows h (row := 1818) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_5
  simp only [← c111000, ← c111001, k2, ← c111002] at e_1818_5
  have hr := e_1818_5
  simp only [e_1818_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6973)) := by
  have c111003 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111004 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111005 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111006 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111007 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111008 := (publicBatchWrapper86_copies3469 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_6 := arithEq_of_rows h (row := 1818) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_6
  simp only [← c111003, ← c111004, ← c111005] at e_1818_6
  have e_1818_7 := arithEq_of_rows h (row := 1818) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_7
  simp only [← c111006, ← c111007, k2, ← c111008] at e_1818_7
  have hr := e_1818_7
  simp only [e_1818_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6974)) := by
  have c111009 := (publicBatchWrapper86_copies3469 a h).2.1
  have c111010 := (publicBatchWrapper86_copies3469 a h).2.2.1
  have c111011 := (publicBatchWrapper86_copies3469 a h).2.2.2.1
  have c111012 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.1
  have c111013 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.1
  have c111014 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_8 := arithEq_of_rows h (row := 1818) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_8
  simp only [← c111009, ← c111010, ← c111011] at e_1818_8
  have e_1818_9 := arithEq_of_rows h (row := 1818) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_9
  simp only [← c111012, ← c111013, k2, ← c111014] at e_1818_9
  have hr := e_1818_9
  simp only [e_1818_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6975)) := by
  have c111015 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.1
  have c111016 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.1
  have c111017 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.1
  have c111018 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111019 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111020 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_10 := arithEq_of_rows h (row := 1818) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_10
  simp only [← c111015, ← c111016, ← c111017] at e_1818_10
  have e_1818_11 := arithEq_of_rows h (row := 1818) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_11
  simp only [← c111018, ← c111019, k2, ← c111020] at e_1818_11
  have hr := e_1818_11
  simp only [e_1818_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6976)) := by
  have c111021 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111022 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111023 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111024 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111025 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111026 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_12 := arithEq_of_rows h (row := 1818) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_12
  simp only [← c111021, ← c111022, ← c111023] at e_1818_12
  have e_1818_13 := arithEq_of_rows h (row := 1818) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_13
  simp only [← c111024, ← c111025, k2, ← c111026] at e_1818_13
  have hr := e_1818_13
  simp only [e_1818_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6977)) := by
  have c111027 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111028 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111029 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111030 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111031 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111032 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_14 := arithEq_of_rows h (row := 1818) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_14
  simp only [← c111027, ← c111028, ← c111029] at e_1818_14
  have e_1818_15 := arithEq_of_rows h (row := 1818) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_15
  simp only [← c111030, ← c111031, k2, ← c111032] at e_1818_15
  have hr := e_1818_15
  simp only [e_1818_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
