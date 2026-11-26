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

theorem publicBatchWrapper86_f9974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8820)) := by
  have c64197 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.1
  have c64198 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.1
  have c64199 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.1
  have c64200 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.1
  have c64201 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.1
  have c64202 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_4 := arithEq_of_rows h (row := 1038) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_4
  simp only [← c64197, ← c64198, ← c64199] at e_1038_4
  have e_1038_5 := arithEq_of_rows h (row := 1038) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_5
  simp only [← c64200, ← c64201, k2, ← c64202] at e_1038_5
  have hr := e_1038_5
  simp only [e_1038_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8821)) := by
  have c64203 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64204 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64205 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64206 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64207 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64208 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_6 := arithEq_of_rows h (row := 1038) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_6
  simp only [← c64203, ← c64204, ← c64205] at e_1038_6
  have e_1038_7 := arithEq_of_rows h (row := 1038) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_7
  simp only [← c64206, ← c64207, k2, ← c64208] at e_1038_7
  have hr := e_1038_7
  simp only [e_1038_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8822)) := by
  have c64209 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64210 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64211 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64212 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64213 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64214 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_8 := arithEq_of_rows h (row := 1038) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_8
  simp only [← c64209, ← c64210, ← c64211] at e_1038_8
  have e_1038_9 := arithEq_of_rows h (row := 1038) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_9
  simp only [← c64212, ← c64213, k2, ← c64214] at e_1038_9
  have hr := e_1038_9
  simp only [e_1038_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8823)) := by
  have c64215 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64216 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64217 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64218 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64219 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64220 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_10 := arithEq_of_rows h (row := 1038) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_10
  simp only [← c64215, ← c64216, ← c64217] at e_1038_10
  have e_1038_11 := arithEq_of_rows h (row := 1038) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_11
  simp only [← c64218, ← c64219, k2, ← c64220] at e_1038_11
  have hr := e_1038_11
  simp only [e_1038_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8824)) := by
  have c64221 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64222 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64223 := (publicBatchWrapper86_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64224 := (publicBatchWrapper86_copies2007 a h).1
  have c64225 := (publicBatchWrapper86_copies2007 a h).2.1
  have c64226 := (publicBatchWrapper86_copies2007 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_12 := arithEq_of_rows h (row := 1038) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_12
  simp only [← c64221, ← c64222, ← c64223] at e_1038_12
  have e_1038_13 := arithEq_of_rows h (row := 1038) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_13
  simp only [← c64224, ← c64225, k2, ← c64226] at e_1038_13
  have hr := e_1038_13
  simp only [e_1038_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8825)) := by
  have c64227 := (publicBatchWrapper86_copies2007 a h).2.2.2.1
  have c64228 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.1
  have c64229 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.1
  have c64230 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.1
  have c64231 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.1
  have c64232 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_14 := arithEq_of_rows h (row := 1038) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_14
  simp only [← c64227, ← c64228, ← c64229] at e_1038_14
  have e_1038_15 := arithEq_of_rows h (row := 1038) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_15
  simp only [← c64230, ← c64231, k2, ← c64232] at e_1038_15
  have hr := e_1038_15
  simp only [e_1038_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8826)) := by
  have c64233 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.1
  have c64234 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64235 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64236 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64237 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64238 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_16 := arithEq_of_rows h (row := 1038) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_16
  simp only [← c64233, ← c64234, ← c64235] at e_1038_16
  have e_1038_17 := arithEq_of_rows h (row := 1038) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_17
  simp only [← c64236, ← c64237, k2, ← c64238] at e_1038_17
  have hr := e_1038_17
  simp only [e_1038_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1038 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8827)) := by
  have c64239 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64240 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64241 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64242 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64243 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64244 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1038_18 := arithEq_of_rows h (row := 1038) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_18
  simp only [← c64239, ← c64240, ← c64241] at e_1038_18
  have e_1038_19 := arithEq_of_rows h (row := 1038) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_19
  simp only [← c64242, ← c64243, k2, ← c64244] at e_1038_19
  have hr := e_1038_19
  simp only [e_1038_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8828)) := by
  have c64245 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64246 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64247 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64248 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64249 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64250 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_0 := arithEq_of_rows h (row := 1039) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_0
  simp only [← c64245, ← c64246, ← c64247] at e_1039_0
  have e_1039_1 := arithEq_of_rows h (row := 1039) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_1
  simp only [← c64248, ← c64249, k2, ← c64250] at e_1039_1
  have hr := e_1039_1
  simp only [e_1039_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8829)) := by
  have c64251 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64252 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64253 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64254 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64255 := (publicBatchWrapper86_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64256 := (publicBatchWrapper86_copies2008 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_2 := arithEq_of_rows h (row := 1039) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_2
  simp only [← c64251, ← c64252, ← c64253] at e_1039_2
  have e_1039_3 := arithEq_of_rows h (row := 1039) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_3
  simp only [← c64254, ← c64255, k2, ← c64256] at e_1039_3
  have hr := e_1039_3
  simp only [e_1039_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8830)) := by
  have c64257 := (publicBatchWrapper86_copies2008 a h).2.1
  have c64258 := (publicBatchWrapper86_copies2008 a h).2.2.1
  have c64259 := (publicBatchWrapper86_copies2008 a h).2.2.2.1
  have c64260 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.1
  have c64261 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.1
  have c64262 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_4 := arithEq_of_rows h (row := 1039) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_4
  simp only [← c64257, ← c64258, ← c64259] at e_1039_4
  have e_1039_5 := arithEq_of_rows h (row := 1039) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_5
  simp only [← c64260, ← c64261, k2, ← c64262] at e_1039_5
  have hr := e_1039_5
  simp only [e_1039_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8831)) := by
  have c64263 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.1
  have c64264 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.1
  have c64265 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.1
  have c64266 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64267 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64268 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_6 := arithEq_of_rows h (row := 1039) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_6
  simp only [← c64263, ← c64264, ← c64265] at e_1039_6
  have e_1039_7 := arithEq_of_rows h (row := 1039) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_7
  simp only [← c64266, ← c64267, k2, ← c64268] at e_1039_7
  have hr := e_1039_7
  simp only [e_1039_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8832)) := by
  have c64269 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64270 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64271 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64272 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64273 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64274 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_8 := arithEq_of_rows h (row := 1039) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_8
  simp only [← c64269, ← c64270, ← c64271] at e_1039_8
  have e_1039_9 := arithEq_of_rows h (row := 1039) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_9
  simp only [← c64272, ← c64273, k2, ← c64274] at e_1039_9
  have hr := e_1039_9
  simp only [e_1039_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8833)) := by
  have c64275 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64276 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64277 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64278 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64279 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64280 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_10 := arithEq_of_rows h (row := 1039) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_10
  simp only [← c64275, ← c64276, ← c64277] at e_1039_10
  have e_1039_11 := arithEq_of_rows h (row := 1039) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_11
  simp only [← c64278, ← c64279, k2, ← c64280] at e_1039_11
  have hr := e_1039_11
  simp only [e_1039_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8834)) := by
  have c64281 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64282 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64283 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64284 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64285 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64286 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_12 := arithEq_of_rows h (row := 1039) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_12
  simp only [← c64281, ← c64282, ← c64283] at e_1039_12
  have e_1039_13 := arithEq_of_rows h (row := 1039) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_13
  simp only [← c64284, ← c64285, k2, ← c64286] at e_1039_13
  have hr := e_1039_13
  simp only [e_1039_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8835)) := by
  have c64287 := (publicBatchWrapper86_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64288 := (publicBatchWrapper86_copies2009 a h).1
  have c64289 := (publicBatchWrapper86_copies2009 a h).2.1
  have c64290 := (publicBatchWrapper86_copies2009 a h).2.2.1
  have c64291 := (publicBatchWrapper86_copies2009 a h).2.2.2.1
  have c64292 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_14 := arithEq_of_rows h (row := 1039) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_14
  simp only [← c64287, ← c64288, ← c64289] at e_1039_14
  have e_1039_15 := arithEq_of_rows h (row := 1039) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_15
  simp only [← c64290, ← c64291, k2, ← c64292] at e_1039_15
  have hr := e_1039_15
  simp only [e_1039_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8836)) := by
  have c64293 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.1
  have c64294 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.1
  have c64295 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.1
  have c64296 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.1
  have c64297 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.1
  have c64298 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_16 := arithEq_of_rows h (row := 1039) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_16
  simp only [← c64293, ← c64294, ← c64295] at e_1039_16
  have e_1039_17 := arithEq_of_rows h (row := 1039) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_17
  simp only [← c64296, ← c64297, k2, ← c64298] at e_1039_17
  have hr := e_1039_17
  simp only [e_1039_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1039 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8837)) := by
  have c64299 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64300 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64301 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64302 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64303 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64304 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1039_18 := arithEq_of_rows h (row := 1039) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_18
  simp only [← c64299, ← c64300, ← c64301] at e_1039_18
  have e_1039_19 := arithEq_of_rows h (row := 1039) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_19
  simp only [← c64302, ← c64303, k2, ← c64304] at e_1039_19
  have hr := e_1039_19
  simp only [e_1039_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8838)) := by
  have c64305 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64306 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64307 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64308 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64309 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64310 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_0 := arithEq_of_rows h (row := 1040) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_0
  simp only [← c64305, ← c64306, ← c64307] at e_1040_0
  have e_1040_1 := arithEq_of_rows h (row := 1040) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_1
  simp only [← c64308, ← c64309, k2, ← c64310] at e_1040_1
  have hr := e_1040_1
  simp only [e_1040_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8839)) := by
  have c64311 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64312 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64313 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64314 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64315 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64316 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_2 := arithEq_of_rows h (row := 1040) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_2
  simp only [← c64311, ← c64312, ← c64313] at e_1040_2
  have e_1040_3 := arithEq_of_rows h (row := 1040) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_3
  simp only [← c64314, ← c64315, k2, ← c64316] at e_1040_3
  have hr := e_1040_3
  simp only [e_1040_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8840)) := by
  have c64317 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64318 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64319 := (publicBatchWrapper86_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64320 := (publicBatchWrapper86_copies2010 a h).1
  have c64321 := (publicBatchWrapper86_copies2010 a h).2.1
  have c64322 := (publicBatchWrapper86_copies2010 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_4 := arithEq_of_rows h (row := 1040) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_4
  simp only [← c64317, ← c64318, ← c64319] at e_1040_4
  have e_1040_5 := arithEq_of_rows h (row := 1040) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_5
  simp only [← c64320, ← c64321, k2, ← c64322] at e_1040_5
  have hr := e_1040_5
  simp only [e_1040_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8841)) := by
  have c64323 := (publicBatchWrapper86_copies2010 a h).2.2.2.1
  have c64324 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.1
  have c64325 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.1
  have c64326 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.1
  have c64327 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.1
  have c64328 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_6 := arithEq_of_rows h (row := 1040) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_6
  simp only [← c64323, ← c64324, ← c64325] at e_1040_6
  have e_1040_7 := arithEq_of_rows h (row := 1040) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_7
  simp only [← c64326, ← c64327, k2, ← c64328] at e_1040_7
  have hr := e_1040_7
  simp only [e_1040_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8842)) := by
  have c64329 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.1
  have c64330 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64331 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64332 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64333 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64334 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_8 := arithEq_of_rows h (row := 1040) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_8
  simp only [← c64329, ← c64330, ← c64331] at e_1040_8
  have e_1040_9 := arithEq_of_rows h (row := 1040) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_9
  simp only [← c64332, ← c64333, k2, ← c64334] at e_1040_9
  have hr := e_1040_9
  simp only [e_1040_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8843)) := by
  have c64335 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64336 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64337 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64338 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64339 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64340 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_10 := arithEq_of_rows h (row := 1040) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_10
  simp only [← c64335, ← c64336, ← c64337] at e_1040_10
  have e_1040_11 := arithEq_of_rows h (row := 1040) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_11
  simp only [← c64338, ← c64339, k2, ← c64340] at e_1040_11
  have hr := e_1040_11
  simp only [e_1040_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8844)) := by
  have c64341 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64342 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64343 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64344 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64345 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64346 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_12 := arithEq_of_rows h (row := 1040) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_12
  simp only [← c64341, ← c64342, ← c64343] at e_1040_12
  have e_1040_13 := arithEq_of_rows h (row := 1040) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_13
  simp only [← c64344, ← c64345, k2, ← c64346] at e_1040_13
  have hr := e_1040_13
  simp only [e_1040_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8845)) := by
  have c64347 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64348 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64349 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64350 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64351 := (publicBatchWrapper86_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64352 := (publicBatchWrapper86_copies2011 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_14 := arithEq_of_rows h (row := 1040) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_14
  simp only [← c64347, ← c64348, ← c64349] at e_1040_14
  have e_1040_15 := arithEq_of_rows h (row := 1040) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_15
  simp only [← c64350, ← c64351, k2, ← c64352] at e_1040_15
  have hr := e_1040_15
  simp only [e_1040_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8846)) := by
  have c64353 := (publicBatchWrapper86_copies2011 a h).2.1
  have c64354 := (publicBatchWrapper86_copies2011 a h).2.2.1
  have c64355 := (publicBatchWrapper86_copies2011 a h).2.2.2.1
  have c64356 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.1
  have c64357 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.1
  have c64358 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_16 := arithEq_of_rows h (row := 1040) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_16
  simp only [← c64353, ← c64354, ← c64355] at e_1040_16
  have e_1040_17 := arithEq_of_rows h (row := 1040) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_17
  simp only [← c64356, ← c64357, k2, ← c64358] at e_1040_17
  have hr := e_1040_17
  simp only [e_1040_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1040 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8847)) := by
  have c64359 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.1
  have c64360 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.1
  have c64361 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.1
  have c64362 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64363 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64364 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1040_18 := arithEq_of_rows h (row := 1040) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_18
  simp only [← c64359, ← c64360, ← c64361] at e_1040_18
  have e_1040_19 := arithEq_of_rows h (row := 1040) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_19
  simp only [← c64362, ← c64363, k2, ← c64364] at e_1040_19
  have hr := e_1040_19
  simp only [e_1040_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8848)) := by
  have c64365 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64366 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64367 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64368 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64369 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64370 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_0 := arithEq_of_rows h (row := 1041) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_0
  simp only [← c64365, ← c64366, ← c64367] at e_1041_0
  have e_1041_1 := arithEq_of_rows h (row := 1041) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_1
  simp only [← c64368, ← c64369, k2, ← c64370] at e_1041_1
  have hr := e_1041_1
  simp only [e_1041_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8849)) := by
  have c64371 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64372 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64373 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64374 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64375 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64376 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_2 := arithEq_of_rows h (row := 1041) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_2
  simp only [← c64371, ← c64372, ← c64373] at e_1041_2
  have e_1041_3 := arithEq_of_rows h (row := 1041) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_3
  simp only [← c64374, ← c64375, k2, ← c64376] at e_1041_3
  have hr := e_1041_3
  simp only [e_1041_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8850)) := by
  have c64377 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64378 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64379 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64380 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64381 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64382 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_4 := arithEq_of_rows h (row := 1041) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_4
  simp only [← c64377, ← c64378, ← c64379] at e_1041_4
  have e_1041_5 := arithEq_of_rows h (row := 1041) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_5
  simp only [← c64380, ← c64381, k2, ← c64382] at e_1041_5
  have hr := e_1041_5
  simp only [e_1041_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8851)) := by
  have c64383 := (publicBatchWrapper86_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64384 := (publicBatchWrapper86_copies2012 a h).1
  have c64385 := (publicBatchWrapper86_copies2012 a h).2.1
  have c64386 := (publicBatchWrapper86_copies2012 a h).2.2.1
  have c64387 := (publicBatchWrapper86_copies2012 a h).2.2.2.1
  have c64388 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_6 := arithEq_of_rows h (row := 1041) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_6
  simp only [← c64383, ← c64384, ← c64385] at e_1041_6
  have e_1041_7 := arithEq_of_rows h (row := 1041) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_7
  simp only [← c64386, ← c64387, k2, ← c64388] at e_1041_7
  have hr := e_1041_7
  simp only [e_1041_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8852)) := by
  have c64389 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.1
  have c64390 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.1
  have c64391 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.1
  have c64392 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.1
  have c64393 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.1
  have c64394 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_8 := arithEq_of_rows h (row := 1041) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_8
  simp only [← c64389, ← c64390, ← c64391] at e_1041_8
  have e_1041_9 := arithEq_of_rows h (row := 1041) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_9
  simp only [← c64392, ← c64393, k2, ← c64394] at e_1041_9
  have hr := e_1041_9
  simp only [e_1041_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8853)) := by
  have c64395 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64396 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64397 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64398 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64399 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64400 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_10 := arithEq_of_rows h (row := 1041) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_10
  simp only [← c64395, ← c64396, ← c64397] at e_1041_10
  have e_1041_11 := arithEq_of_rows h (row := 1041) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_11
  simp only [← c64398, ← c64399, k2, ← c64400] at e_1041_11
  have hr := e_1041_11
  simp only [e_1041_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8854)) := by
  have c64401 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64402 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64403 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64404 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64405 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64406 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_12 := arithEq_of_rows h (row := 1041) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_12
  simp only [← c64401, ← c64402, ← c64403] at e_1041_12
  have e_1041_13 := arithEq_of_rows h (row := 1041) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_13
  simp only [← c64404, ← c64405, k2, ← c64406] at e_1041_13
  have hr := e_1041_13
  simp only [e_1041_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8855)) := by
  have c64407 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64408 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64409 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64410 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64411 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64412 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_14 := arithEq_of_rows h (row := 1041) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_14
  simp only [← c64407, ← c64408, ← c64409] at e_1041_14
  have e_1041_15 := arithEq_of_rows h (row := 1041) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_15
  simp only [← c64410, ← c64411, k2, ← c64412] at e_1041_15
  have hr := e_1041_15
  simp only [e_1041_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8856)) := by
  have c64413 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64414 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64415 := (publicBatchWrapper86_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64416 := (publicBatchWrapper86_copies2013 a h).1
  have c64417 := (publicBatchWrapper86_copies2013 a h).2.1
  have c64418 := (publicBatchWrapper86_copies2013 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_16 := arithEq_of_rows h (row := 1041) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_16
  simp only [← c64413, ← c64414, ← c64415] at e_1041_16
  have e_1041_17 := arithEq_of_rows h (row := 1041) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_17
  simp only [← c64416, ← c64417, k2, ← c64418] at e_1041_17
  have hr := e_1041_17
  simp only [e_1041_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1041 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8857)) := by
  have c64419 := (publicBatchWrapper86_copies2013 a h).2.2.2.1
  have c64420 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.1
  have c64421 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.1
  have c64422 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.1
  have c64423 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.1
  have c64424 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1041_18 := arithEq_of_rows h (row := 1041) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_18
  simp only [← c64419, ← c64420, ← c64421] at e_1041_18
  have e_1041_19 := arithEq_of_rows h (row := 1041) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_19
  simp only [← c64422, ← c64423, k2, ← c64424] at e_1041_19
  have hr := e_1041_19
  simp only [e_1041_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8858)) := by
  have c64425 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.1
  have c64426 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64427 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64428 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64429 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64430 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_0 := arithEq_of_rows h (row := 1042) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_0
  simp only [← c64425, ← c64426, ← c64427] at e_1042_0
  have e_1042_1 := arithEq_of_rows h (row := 1042) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_1
  simp only [← c64428, ← c64429, k2, ← c64430] at e_1042_1
  have hr := e_1042_1
  simp only [e_1042_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8859)) := by
  have c64431 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64432 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64433 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64434 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64435 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64436 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_2 := arithEq_of_rows h (row := 1042) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_2
  simp only [← c64431, ← c64432, ← c64433] at e_1042_2
  have e_1042_3 := arithEq_of_rows h (row := 1042) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_3
  simp only [← c64434, ← c64435, k2, ← c64436] at e_1042_3
  have hr := e_1042_3
  simp only [e_1042_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8860)) := by
  have c64437 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64438 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64439 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64440 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64441 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64442 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_4 := arithEq_of_rows h (row := 1042) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_4
  simp only [← c64437, ← c64438, ← c64439] at e_1042_4
  have e_1042_5 := arithEq_of_rows h (row := 1042) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_5
  simp only [← c64440, ← c64441, k2, ← c64442] at e_1042_5
  have hr := e_1042_5
  simp only [e_1042_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8861)) := by
  have c64443 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64444 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64445 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64446 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64447 := (publicBatchWrapper86_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64448 := (publicBatchWrapper86_copies2014 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_6 := arithEq_of_rows h (row := 1042) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_6
  simp only [← c64443, ← c64444, ← c64445] at e_1042_6
  have e_1042_7 := arithEq_of_rows h (row := 1042) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_7
  simp only [← c64446, ← c64447, k2, ← c64448] at e_1042_7
  have hr := e_1042_7
  simp only [e_1042_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8862)) := by
  have c64449 := (publicBatchWrapper86_copies2014 a h).2.1
  have c64450 := (publicBatchWrapper86_copies2014 a h).2.2.1
  have c64451 := (publicBatchWrapper86_copies2014 a h).2.2.2.1
  have c64452 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.1
  have c64453 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.1
  have c64454 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_8 := arithEq_of_rows h (row := 1042) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_8
  simp only [← c64449, ← c64450, ← c64451] at e_1042_8
  have e_1042_9 := arithEq_of_rows h (row := 1042) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_9
  simp only [← c64452, ← c64453, k2, ← c64454] at e_1042_9
  have hr := e_1042_9
  simp only [e_1042_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8863)) := by
  have c64455 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.1
  have c64456 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.1
  have c64457 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.1
  have c64458 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64459 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64460 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_10 := arithEq_of_rows h (row := 1042) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_10
  simp only [← c64455, ← c64456, ← c64457] at e_1042_10
  have e_1042_11 := arithEq_of_rows h (row := 1042) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_11
  simp only [← c64458, ← c64459, k2, ← c64460] at e_1042_11
  have hr := e_1042_11
  simp only [e_1042_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8864)) := by
  have c64461 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64462 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64463 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64464 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64465 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64466 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_12 := arithEq_of_rows h (row := 1042) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_12
  simp only [← c64461, ← c64462, ← c64463] at e_1042_12
  have e_1042_13 := arithEq_of_rows h (row := 1042) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_13
  simp only [← c64464, ← c64465, k2, ← c64466] at e_1042_13
  have hr := e_1042_13
  simp only [e_1042_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8865)) := by
  have c64467 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64468 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64469 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64470 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64471 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64472 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_14 := arithEq_of_rows h (row := 1042) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_14
  simp only [← c64467, ← c64468, ← c64469] at e_1042_14
  have e_1042_15 := arithEq_of_rows h (row := 1042) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_15
  simp only [← c64470, ← c64471, k2, ← c64472] at e_1042_15
  have hr := e_1042_15
  simp only [e_1042_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8866)) := by
  have c64473 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64474 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64475 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64476 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64477 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64478 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_16 := arithEq_of_rows h (row := 1042) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_16
  simp only [← c64473, ← c64474, ← c64475] at e_1042_16
  have e_1042_17 := arithEq_of_rows h (row := 1042) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_17
  simp only [← c64476, ← c64477, k2, ← c64478] at e_1042_17
  have hr := e_1042_17
  simp only [e_1042_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1042 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8867)) := by
  have c64479 := (publicBatchWrapper86_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64480 := (publicBatchWrapper86_copies2015 a h).1
  have c64481 := (publicBatchWrapper86_copies2015 a h).2.1
  have c64482 := (publicBatchWrapper86_copies2015 a h).2.2.1
  have c64483 := (publicBatchWrapper86_copies2015 a h).2.2.2.1
  have c64484 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1042_18 := arithEq_of_rows h (row := 1042) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_18
  simp only [← c64479, ← c64480, ← c64481] at e_1042_18
  have e_1042_19 := arithEq_of_rows h (row := 1042) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_19
  simp only [← c64482, ← c64483, k2, ← c64484] at e_1042_19
  have hr := e_1042_19
  simp only [e_1042_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8868)) := by
  have c64485 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.1
  have c64486 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.1
  have c64487 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.1
  have c64488 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.1
  have c64489 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.1
  have c64490 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_0 := arithEq_of_rows h (row := 1043) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_0
  simp only [← c64485, ← c64486, ← c64487] at e_1043_0
  have e_1043_1 := arithEq_of_rows h (row := 1043) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_1
  simp only [← c64488, ← c64489, k2, ← c64490] at e_1043_1
  have hr := e_1043_1
  simp only [e_1043_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8869)) := by
  have c64491 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64492 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64493 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64494 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64495 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64496 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_2 := arithEq_of_rows h (row := 1043) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_2
  simp only [← c64491, ← c64492, ← c64493] at e_1043_2
  have e_1043_3 := arithEq_of_rows h (row := 1043) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_3
  simp only [← c64494, ← c64495, k2, ← c64496] at e_1043_3
  have hr := e_1043_3
  simp only [e_1043_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8870)) := by
  have c64497 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64498 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64499 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64500 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64501 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64502 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_4 := arithEq_of_rows h (row := 1043) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_4
  simp only [← c64497, ← c64498, ← c64499] at e_1043_4
  have e_1043_5 := arithEq_of_rows h (row := 1043) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_5
  simp only [← c64500, ← c64501, k2, ← c64502] at e_1043_5
  have hr := e_1043_5
  simp only [e_1043_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8871)) := by
  have c64503 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64504 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64505 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64506 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64507 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64508 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_6 := arithEq_of_rows h (row := 1043) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_6
  simp only [← c64503, ← c64504, ← c64505] at e_1043_6
  have e_1043_7 := arithEq_of_rows h (row := 1043) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_7
  simp only [← c64506, ← c64507, k2, ← c64508] at e_1043_7
  have hr := e_1043_7
  simp only [e_1043_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8872)) := by
  have c64509 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64510 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64511 := (publicBatchWrapper86_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64512 := (publicBatchWrapper86_copies2016 a h).1
  have c64513 := (publicBatchWrapper86_copies2016 a h).2.1
  have c64514 := (publicBatchWrapper86_copies2016 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_8 := arithEq_of_rows h (row := 1043) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_8
  simp only [← c64509, ← c64510, ← c64511] at e_1043_8
  have e_1043_9 := arithEq_of_rows h (row := 1043) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_9
  simp only [← c64512, ← c64513, k2, ← c64514] at e_1043_9
  have hr := e_1043_9
  simp only [e_1043_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8873)) := by
  have c64515 := (publicBatchWrapper86_copies2016 a h).2.2.2.1
  have c64516 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.1
  have c64517 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.1
  have c64518 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.1
  have c64519 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.1
  have c64520 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_10 := arithEq_of_rows h (row := 1043) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_10
  simp only [← c64515, ← c64516, ← c64517] at e_1043_10
  have e_1043_11 := arithEq_of_rows h (row := 1043) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_11
  simp only [← c64518, ← c64519, k2, ← c64520] at e_1043_11
  have hr := e_1043_11
  simp only [e_1043_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8874)) := by
  have c64521 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.1
  have c64522 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64523 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64524 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64525 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64526 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_12 := arithEq_of_rows h (row := 1043) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_12
  simp only [← c64521, ← c64522, ← c64523] at e_1043_12
  have e_1043_13 := arithEq_of_rows h (row := 1043) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_13
  simp only [← c64524, ← c64525, k2, ← c64526] at e_1043_13
  have hr := e_1043_13
  simp only [e_1043_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8875)) := by
  have c64527 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64528 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64529 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64530 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64531 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64532 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_14 := arithEq_of_rows h (row := 1043) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_14
  simp only [← c64527, ← c64528, ← c64529] at e_1043_14
  have e_1043_15 := arithEq_of_rows h (row := 1043) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_15
  simp only [← c64530, ← c64531, k2, ← c64532] at e_1043_15
  have hr := e_1043_15
  simp only [e_1043_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8876)) := by
  have c64533 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64534 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64535 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64536 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64537 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64538 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_16 := arithEq_of_rows h (row := 1043) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_16
  simp only [← c64533, ← c64534, ← c64535] at e_1043_16
  have e_1043_17 := arithEq_of_rows h (row := 1043) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_17
  simp only [← c64536, ← c64537, k2, ← c64538] at e_1043_17
  have hr := e_1043_17
  simp only [e_1043_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1043 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8877)) := by
  have c64539 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64540 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64541 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64542 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64543 := (publicBatchWrapper86_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64544 := (publicBatchWrapper86_copies2017 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1043_18 := arithEq_of_rows h (row := 1043) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_18
  simp only [← c64539, ← c64540, ← c64541] at e_1043_18
  have e_1043_19 := arithEq_of_rows h (row := 1043) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_19
  simp only [← c64542, ← c64543, k2, ← c64544] at e_1043_19
  have hr := e_1043_19
  simp only [e_1043_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8878)) := by
  have c64545 := (publicBatchWrapper86_copies2017 a h).2.1
  have c64546 := (publicBatchWrapper86_copies2017 a h).2.2.1
  have c64547 := (publicBatchWrapper86_copies2017 a h).2.2.2.1
  have c64548 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.1
  have c64549 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.1
  have c64550 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_0 := arithEq_of_rows h (row := 1044) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_0
  simp only [← c64545, ← c64546, ← c64547] at e_1044_0
  have e_1044_1 := arithEq_of_rows h (row := 1044) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_1
  simp only [← c64548, ← c64549, k2, ← c64550] at e_1044_1
  have hr := e_1044_1
  simp only [e_1044_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8879)) := by
  have c64551 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.1
  have c64552 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.1
  have c64553 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.1
  have c64554 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64555 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64556 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_2 := arithEq_of_rows h (row := 1044) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_2
  simp only [← c64551, ← c64552, ← c64553] at e_1044_2
  have e_1044_3 := arithEq_of_rows h (row := 1044) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_3
  simp only [← c64554, ← c64555, k2, ← c64556] at e_1044_3
  have hr := e_1044_3
  simp only [e_1044_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8880)) := by
  have c64557 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64558 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64559 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64560 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64561 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64562 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_4 := arithEq_of_rows h (row := 1044) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_4
  simp only [← c64557, ← c64558, ← c64559] at e_1044_4
  have e_1044_5 := arithEq_of_rows h (row := 1044) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_5
  simp only [← c64560, ← c64561, k2, ← c64562] at e_1044_5
  have hr := e_1044_5
  simp only [e_1044_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8881)) := by
  have c64563 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64564 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64565 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64566 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64567 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64568 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_6 := arithEq_of_rows h (row := 1044) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_6
  simp only [← c64563, ← c64564, ← c64565] at e_1044_6
  have e_1044_7 := arithEq_of_rows h (row := 1044) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_7
  simp only [← c64566, ← c64567, k2, ← c64568] at e_1044_7
  have hr := e_1044_7
  simp only [e_1044_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8882)) := by
  have c64569 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64570 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64571 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64572 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64573 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64574 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_8 := arithEq_of_rows h (row := 1044) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_8
  simp only [← c64569, ← c64570, ← c64571] at e_1044_8
  have e_1044_9 := arithEq_of_rows h (row := 1044) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_9
  simp only [← c64572, ← c64573, k2, ← c64574] at e_1044_9
  have hr := e_1044_9
  simp only [e_1044_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8883)) := by
  have c64575 := (publicBatchWrapper86_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64576 := (publicBatchWrapper86_copies2018 a h).1
  have c64577 := (publicBatchWrapper86_copies2018 a h).2.1
  have c64578 := (publicBatchWrapper86_copies2018 a h).2.2.1
  have c64579 := (publicBatchWrapper86_copies2018 a h).2.2.2.1
  have c64580 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_10 := arithEq_of_rows h (row := 1044) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_10
  simp only [← c64575, ← c64576, ← c64577] at e_1044_10
  have e_1044_11 := arithEq_of_rows h (row := 1044) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_11
  simp only [← c64578, ← c64579, k2, ← c64580] at e_1044_11
  have hr := e_1044_11
  simp only [e_1044_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8884)) := by
  have c64581 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.1
  have c64582 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.1
  have c64583 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.1
  have c64584 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.1
  have c64585 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.1
  have c64586 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_12 := arithEq_of_rows h (row := 1044) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_12
  simp only [← c64581, ← c64582, ← c64583] at e_1044_12
  have e_1044_13 := arithEq_of_rows h (row := 1044) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_13
  simp only [← c64584, ← c64585, k2, ← c64586] at e_1044_13
  have hr := e_1044_13
  simp only [e_1044_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8885)) := by
  have c64587 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64588 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64589 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64590 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64591 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64592 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_14 := arithEq_of_rows h (row := 1044) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_14
  simp only [← c64587, ← c64588, ← c64589] at e_1044_14
  have e_1044_15 := arithEq_of_rows h (row := 1044) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_15
  simp only [← c64590, ← c64591, k2, ← c64592] at e_1044_15
  have hr := e_1044_15
  simp only [e_1044_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8886)) := by
  have c64593 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64594 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64595 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64596 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64597 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64598 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_16 := arithEq_of_rows h (row := 1044) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_16
  simp only [← c64593, ← c64594, ← c64595] at e_1044_16
  have e_1044_17 := arithEq_of_rows h (row := 1044) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_17
  simp only [← c64596, ← c64597, k2, ← c64598] at e_1044_17
  have hr := e_1044_17
  simp only [e_1044_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1044 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8887)) := by
  have c64599 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64600 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64601 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64602 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64603 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64604 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1044_18 := arithEq_of_rows h (row := 1044) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_18
  simp only [← c64599, ← c64600, ← c64601] at e_1044_18
  have e_1044_19 := arithEq_of_rows h (row := 1044) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_19
  simp only [← c64602, ← c64603, k2, ← c64604] at e_1044_19
  have hr := e_1044_19
  simp only [e_1044_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8888)) := by
  have c64605 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64606 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64607 := (publicBatchWrapper86_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64608 := (publicBatchWrapper86_copies2019 a h).1
  have c64609 := (publicBatchWrapper86_copies2019 a h).2.1
  have c64610 := (publicBatchWrapper86_copies2019 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_0 := arithEq_of_rows h (row := 1045) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_0
  simp only [← c64605, ← c64606, ← c64607] at e_1045_0
  have e_1045_1 := arithEq_of_rows h (row := 1045) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_1
  simp only [← c64608, ← c64609, k2, ← c64610] at e_1045_1
  have hr := e_1045_1
  simp only [e_1045_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8889)) := by
  have c64611 := (publicBatchWrapper86_copies2019 a h).2.2.2.1
  have c64612 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.1
  have c64613 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.1
  have c64614 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.1
  have c64615 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.1
  have c64616 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_2 := arithEq_of_rows h (row := 1045) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_2
  simp only [← c64611, ← c64612, ← c64613] at e_1045_2
  have e_1045_3 := arithEq_of_rows h (row := 1045) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_3
  simp only [← c64614, ← c64615, k2, ← c64616] at e_1045_3
  have hr := e_1045_3
  simp only [e_1045_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8890)) := by
  have c64617 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.1
  have c64618 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64619 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64620 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64621 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64622 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_4 := arithEq_of_rows h (row := 1045) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_4
  simp only [← c64617, ← c64618, ← c64619] at e_1045_4
  have e_1045_5 := arithEq_of_rows h (row := 1045) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_5
  simp only [← c64620, ← c64621, k2, ← c64622] at e_1045_5
  have hr := e_1045_5
  simp only [e_1045_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8891)) := by
  have c64623 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64624 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64625 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64626 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64627 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64628 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_6 := arithEq_of_rows h (row := 1045) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_6
  simp only [← c64623, ← c64624, ← c64625] at e_1045_6
  have e_1045_7 := arithEq_of_rows h (row := 1045) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_7
  simp only [← c64626, ← c64627, k2, ← c64628] at e_1045_7
  have hr := e_1045_7
  simp only [e_1045_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8892)) := by
  have c64629 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64630 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64631 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64632 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64633 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64634 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_8 := arithEq_of_rows h (row := 1045) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_8
  simp only [← c64629, ← c64630, ← c64631] at e_1045_8
  have e_1045_9 := arithEq_of_rows h (row := 1045) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_9
  simp only [← c64632, ← c64633, k2, ← c64634] at e_1045_9
  have hr := e_1045_9
  simp only [e_1045_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8893)) := by
  have c64635 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64636 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64637 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64638 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64639 := (publicBatchWrapper86_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64640 := (publicBatchWrapper86_copies2020 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_10 := arithEq_of_rows h (row := 1045) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_10
  simp only [← c64635, ← c64636, ← c64637] at e_1045_10
  have e_1045_11 := arithEq_of_rows h (row := 1045) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_11
  simp only [← c64638, ← c64639, k2, ← c64640] at e_1045_11
  have hr := e_1045_11
  simp only [e_1045_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8894)) := by
  have c64641 := (publicBatchWrapper86_copies2020 a h).2.1
  have c64642 := (publicBatchWrapper86_copies2020 a h).2.2.1
  have c64643 := (publicBatchWrapper86_copies2020 a h).2.2.2.1
  have c64644 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.1
  have c64645 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.1
  have c64646 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_12 := arithEq_of_rows h (row := 1045) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_12
  simp only [← c64641, ← c64642, ← c64643] at e_1045_12
  have e_1045_13 := arithEq_of_rows h (row := 1045) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_13
  simp only [← c64644, ← c64645, k2, ← c64646] at e_1045_13
  have hr := e_1045_13
  simp only [e_1045_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8895)) := by
  have c64647 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.1
  have c64648 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.1
  have c64649 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.1
  have c64650 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64651 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64652 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_14 := arithEq_of_rows h (row := 1045) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_14
  simp only [← c64647, ← c64648, ← c64649] at e_1045_14
  have e_1045_15 := arithEq_of_rows h (row := 1045) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_15
  simp only [← c64650, ← c64651, k2, ← c64652] at e_1045_15
  have hr := e_1045_15
  simp only [e_1045_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8896)) := by
  have c64653 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64654 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64655 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64656 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64657 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64658 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_16 := arithEq_of_rows h (row := 1045) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_16
  simp only [← c64653, ← c64654, ← c64655] at e_1045_16
  have e_1045_17 := arithEq_of_rows h (row := 1045) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_17
  simp only [← c64656, ← c64657, k2, ← c64658] at e_1045_17
  have hr := e_1045_17
  simp only [e_1045_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1045 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8897)) := by
  have c64659 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64660 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64661 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64662 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64663 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64664 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1045_18 := arithEq_of_rows h (row := 1045) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_18
  simp only [← c64659, ← c64660, ← c64661] at e_1045_18
  have e_1045_19 := arithEq_of_rows h (row := 1045) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_19
  simp only [← c64662, ← c64663, k2, ← c64664] at e_1045_19
  have hr := e_1045_19
  simp only [e_1045_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8898)) := by
  have c64665 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64666 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64667 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64668 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64669 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64670 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_0 := arithEq_of_rows h (row := 1046) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_0
  simp only [← c64665, ← c64666, ← c64667] at e_1046_0
  have e_1046_1 := arithEq_of_rows h (row := 1046) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_1
  simp only [← c64668, ← c64669, k2, ← c64670] at e_1046_1
  have hr := e_1046_1
  simp only [e_1046_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8899)) := by
  have c64671 := (publicBatchWrapper86_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64672 := (publicBatchWrapper86_copies2021 a h).1
  have c64673 := (publicBatchWrapper86_copies2021 a h).2.1
  have c64674 := (publicBatchWrapper86_copies2021 a h).2.2.1
  have c64675 := (publicBatchWrapper86_copies2021 a h).2.2.2.1
  have c64676 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_2 := arithEq_of_rows h (row := 1046) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_2
  simp only [← c64671, ← c64672, ← c64673] at e_1046_2
  have e_1046_3 := arithEq_of_rows h (row := 1046) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_3
  simp only [← c64674, ← c64675, k2, ← c64676] at e_1046_3
  have hr := e_1046_3
  simp only [e_1046_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8900)) := by
  have c64677 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.1
  have c64678 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.1
  have c64679 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.1
  have c64680 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.1
  have c64681 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.1
  have c64682 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_4 := arithEq_of_rows h (row := 1046) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_4
  simp only [← c64677, ← c64678, ← c64679] at e_1046_4
  have e_1046_5 := arithEq_of_rows h (row := 1046) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_5
  simp only [← c64680, ← c64681, k2, ← c64682] at e_1046_5
  have hr := e_1046_5
  simp only [e_1046_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8901)) := by
  have c64683 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64684 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64685 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64686 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64687 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64688 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_6 := arithEq_of_rows h (row := 1046) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_6
  simp only [← c64683, ← c64684, ← c64685] at e_1046_6
  have e_1046_7 := arithEq_of_rows h (row := 1046) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_7
  simp only [← c64686, ← c64687, k2, ← c64688] at e_1046_7
  have hr := e_1046_7
  simp only [e_1046_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8902)) := by
  have c64689 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64690 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64691 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64692 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64693 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64694 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_8 := arithEq_of_rows h (row := 1046) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_8
  simp only [← c64689, ← c64690, ← c64691] at e_1046_8
  have e_1046_9 := arithEq_of_rows h (row := 1046) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_9
  simp only [← c64692, ← c64693, k2, ← c64694] at e_1046_9
  have hr := e_1046_9
  simp only [e_1046_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8903)) := by
  have c64695 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64696 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64697 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64698 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64699 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64700 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_10 := arithEq_of_rows h (row := 1046) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_10
  simp only [← c64695, ← c64696, ← c64697] at e_1046_10
  have e_1046_11 := arithEq_of_rows h (row := 1046) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_11
  simp only [← c64698, ← c64699, k2, ← c64700] at e_1046_11
  have hr := e_1046_11
  simp only [e_1046_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8904)) := by
  have c64701 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64702 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64703 := (publicBatchWrapper86_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64704 := (publicBatchWrapper86_copies2022 a h).1
  have c64705 := (publicBatchWrapper86_copies2022 a h).2.1
  have c64706 := (publicBatchWrapper86_copies2022 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_12 := arithEq_of_rows h (row := 1046) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_12
  simp only [← c64701, ← c64702, ← c64703] at e_1046_12
  have e_1046_13 := arithEq_of_rows h (row := 1046) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_13
  simp only [← c64704, ← c64705, k2, ← c64706] at e_1046_13
  have hr := e_1046_13
  simp only [e_1046_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8905)) := by
  have c64707 := (publicBatchWrapper86_copies2022 a h).2.2.2.1
  have c64708 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.1
  have c64709 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.1
  have c64710 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.1
  have c64711 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.1
  have c64712 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_14 := arithEq_of_rows h (row := 1046) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_14
  simp only [← c64707, ← c64708, ← c64709] at e_1046_14
  have e_1046_15 := arithEq_of_rows h (row := 1046) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_15
  simp only [← c64710, ← c64711, k2, ← c64712] at e_1046_15
  have hr := e_1046_15
  simp only [e_1046_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8906)) := by
  have c64713 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.1
  have c64714 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64715 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64716 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64717 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64718 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_16 := arithEq_of_rows h (row := 1046) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_16
  simp only [← c64713, ← c64714, ← c64715] at e_1046_16
  have e_1046_17 := arithEq_of_rows h (row := 1046) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_17
  simp only [← c64716, ← c64717, k2, ← c64718] at e_1046_17
  have hr := e_1046_17
  simp only [e_1046_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1046 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8907)) := by
  have c64719 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64720 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64721 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64722 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64723 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64724 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1046_18 := arithEq_of_rows h (row := 1046) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_18
  simp only [← c64719, ← c64720, ← c64721] at e_1046_18
  have e_1046_19 := arithEq_of_rows h (row := 1046) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_19
  simp only [← c64722, ← c64723, k2, ← c64724] at e_1046_19
  have hr := e_1046_19
  simp only [e_1046_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8908)) := by
  have c64725 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64726 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64727 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64728 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64729 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64730 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_0 := arithEq_of_rows h (row := 1047) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_0
  simp only [← c64725, ← c64726, ← c64727] at e_1047_0
  have e_1047_1 := arithEq_of_rows h (row := 1047) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_1
  simp only [← c64728, ← c64729, k2, ← c64730] at e_1047_1
  have hr := e_1047_1
  simp only [e_1047_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8909)) := by
  have c64731 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64732 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64733 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64734 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64735 := (publicBatchWrapper86_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64736 := (publicBatchWrapper86_copies2023 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_2 := arithEq_of_rows h (row := 1047) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_2
  simp only [← c64731, ← c64732, ← c64733] at e_1047_2
  have e_1047_3 := arithEq_of_rows h (row := 1047) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_3
  simp only [← c64734, ← c64735, k2, ← c64736] at e_1047_3
  have hr := e_1047_3
  simp only [e_1047_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8910)) := by
  have c64737 := (publicBatchWrapper86_copies2023 a h).2.1
  have c64738 := (publicBatchWrapper86_copies2023 a h).2.2.1
  have c64739 := (publicBatchWrapper86_copies2023 a h).2.2.2.1
  have c64740 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.1
  have c64741 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.1
  have c64742 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_4 := arithEq_of_rows h (row := 1047) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_4
  simp only [← c64737, ← c64738, ← c64739] at e_1047_4
  have e_1047_5 := arithEq_of_rows h (row := 1047) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_5
  simp only [← c64740, ← c64741, k2, ← c64742] at e_1047_5
  have hr := e_1047_5
  simp only [e_1047_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8911)) := by
  have c64743 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.1
  have c64744 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.1
  have c64745 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.1
  have c64746 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64747 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64748 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_6 := arithEq_of_rows h (row := 1047) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_6
  simp only [← c64743, ← c64744, ← c64745] at e_1047_6
  have e_1047_7 := arithEq_of_rows h (row := 1047) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_7
  simp only [← c64746, ← c64747, k2, ← c64748] at e_1047_7
  have hr := e_1047_7
  simp only [e_1047_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8912)) := by
  have c64749 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64750 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64751 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64752 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64753 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64754 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_8 := arithEq_of_rows h (row := 1047) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_8
  simp only [← c64749, ← c64750, ← c64751] at e_1047_8
  have e_1047_9 := arithEq_of_rows h (row := 1047) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_9
  simp only [← c64752, ← c64753, k2, ← c64754] at e_1047_9
  have hr := e_1047_9
  simp only [e_1047_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8913)) := by
  have c64755 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64756 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64757 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64758 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64759 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64760 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_10 := arithEq_of_rows h (row := 1047) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_10
  simp only [← c64755, ← c64756, ← c64757] at e_1047_10
  have e_1047_11 := arithEq_of_rows h (row := 1047) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_11
  simp only [← c64758, ← c64759, k2, ← c64760] at e_1047_11
  have hr := e_1047_11
  simp only [e_1047_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8914)) := by
  have c64761 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64762 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64763 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64764 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64765 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64766 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_12 := arithEq_of_rows h (row := 1047) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_12
  simp only [← c64761, ← c64762, ← c64763] at e_1047_12
  have e_1047_13 := arithEq_of_rows h (row := 1047) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_13
  simp only [← c64764, ← c64765, k2, ← c64766] at e_1047_13
  have hr := e_1047_13
  simp only [e_1047_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8915)) := by
  have c64767 := (publicBatchWrapper86_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64768 := (publicBatchWrapper86_copies2024 a h).1
  have c64769 := (publicBatchWrapper86_copies2024 a h).2.1
  have c64770 := (publicBatchWrapper86_copies2024 a h).2.2.1
  have c64771 := (publicBatchWrapper86_copies2024 a h).2.2.2.1
  have c64772 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_14 := arithEq_of_rows h (row := 1047) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_14
  simp only [← c64767, ← c64768, ← c64769] at e_1047_14
  have e_1047_15 := arithEq_of_rows h (row := 1047) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_15
  simp only [← c64770, ← c64771, k2, ← c64772] at e_1047_15
  have hr := e_1047_15
  simp only [e_1047_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8916)) := by
  have c64773 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.1
  have c64774 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.1
  have c64775 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.1
  have c64776 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.1
  have c64777 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.1
  have c64778 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_16 := arithEq_of_rows h (row := 1047) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_16
  simp only [← c64773, ← c64774, ← c64775] at e_1047_16
  have e_1047_17 := arithEq_of_rows h (row := 1047) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_17
  simp only [← c64776, ← c64777, k2, ← c64778] at e_1047_17
  have hr := e_1047_17
  simp only [e_1047_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1047 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8917)) := by
  have c64779 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64780 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64781 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64782 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64783 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64784 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1047_18 := arithEq_of_rows h (row := 1047) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_18
  simp only [← c64779, ← c64780, ← c64781] at e_1047_18
  have e_1047_19 := arithEq_of_rows h (row := 1047) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_19
  simp only [← c64782, ← c64783, k2, ← c64784] at e_1047_19
  have hr := e_1047_19
  simp only [e_1047_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8918)) := by
  have c64785 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64786 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64787 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64788 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64789 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64790 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_0 := arithEq_of_rows h (row := 1048) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_0
  simp only [← c64785, ← c64786, ← c64787] at e_1048_0
  have e_1048_1 := arithEq_of_rows h (row := 1048) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_1
  simp only [← c64788, ← c64789, k2, ← c64790] at e_1048_1
  have hr := e_1048_1
  simp only [e_1048_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8919)) := by
  have c64791 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64792 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64793 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64794 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64795 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64796 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_2 := arithEq_of_rows h (row := 1048) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_2
  simp only [← c64791, ← c64792, ← c64793] at e_1048_2
  have e_1048_3 := arithEq_of_rows h (row := 1048) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_3
  simp only [← c64794, ← c64795, k2, ← c64796] at e_1048_3
  have hr := e_1048_3
  simp only [e_1048_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8920)) := by
  have c64797 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64798 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64799 := (publicBatchWrapper86_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64800 := (publicBatchWrapper86_copies2025 a h).1
  have c64801 := (publicBatchWrapper86_copies2025 a h).2.1
  have c64802 := (publicBatchWrapper86_copies2025 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_4 := arithEq_of_rows h (row := 1048) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_4
  simp only [← c64797, ← c64798, ← c64799] at e_1048_4
  have e_1048_5 := arithEq_of_rows h (row := 1048) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_5
  simp only [← c64800, ← c64801, k2, ← c64802] at e_1048_5
  have hr := e_1048_5
  simp only [e_1048_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8921)) := by
  have c64803 := (publicBatchWrapper86_copies2025 a h).2.2.2.1
  have c64804 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.1
  have c64805 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.1
  have c64806 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.1
  have c64807 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.1
  have c64808 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_6 := arithEq_of_rows h (row := 1048) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_6
  simp only [← c64803, ← c64804, ← c64805] at e_1048_6
  have e_1048_7 := arithEq_of_rows h (row := 1048) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_7
  simp only [← c64806, ← c64807, k2, ← c64808] at e_1048_7
  have hr := e_1048_7
  simp only [e_1048_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8922)) := by
  have c64809 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.1
  have c64810 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64811 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64812 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64813 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64814 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_8 := arithEq_of_rows h (row := 1048) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_8
  simp only [← c64809, ← c64810, ← c64811] at e_1048_8
  have e_1048_9 := arithEq_of_rows h (row := 1048) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_9
  simp only [← c64812, ← c64813, k2, ← c64814] at e_1048_9
  have hr := e_1048_9
  simp only [e_1048_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8923)) := by
  have c64815 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64816 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64817 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64818 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64819 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64820 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_10 := arithEq_of_rows h (row := 1048) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_10
  simp only [← c64815, ← c64816, ← c64817] at e_1048_10
  have e_1048_11 := arithEq_of_rows h (row := 1048) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_11
  simp only [← c64818, ← c64819, k2, ← c64820] at e_1048_11
  have hr := e_1048_11
  simp only [e_1048_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8924)) := by
  have c64821 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64822 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64823 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64824 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64825 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64826 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_12 := arithEq_of_rows h (row := 1048) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_12
  simp only [← c64821, ← c64822, ← c64823] at e_1048_12
  have e_1048_13 := arithEq_of_rows h (row := 1048) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_13
  simp only [← c64824, ← c64825, k2, ← c64826] at e_1048_13
  have hr := e_1048_13
  simp only [e_1048_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8925)) := by
  have c64827 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64828 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64829 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64830 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64831 := (publicBatchWrapper86_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64832 := (publicBatchWrapper86_copies2026 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_14 := arithEq_of_rows h (row := 1048) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_14
  simp only [← c64827, ← c64828, ← c64829] at e_1048_14
  have e_1048_15 := arithEq_of_rows h (row := 1048) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_15
  simp only [← c64830, ← c64831, k2, ← c64832] at e_1048_15
  have hr := e_1048_15
  simp only [e_1048_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8926)) := by
  have c64833 := (publicBatchWrapper86_copies2026 a h).2.1
  have c64834 := (publicBatchWrapper86_copies2026 a h).2.2.1
  have c64835 := (publicBatchWrapper86_copies2026 a h).2.2.2.1
  have c64836 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.1
  have c64837 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.1
  have c64838 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_16 := arithEq_of_rows h (row := 1048) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_16
  simp only [← c64833, ← c64834, ← c64835] at e_1048_16
  have e_1048_17 := arithEq_of_rows h (row := 1048) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_17
  simp only [← c64836, ← c64837, k2, ← c64838] at e_1048_17
  have hr := e_1048_17
  simp only [e_1048_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1048 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8927)) := by
  have c64839 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.1
  have c64840 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.1
  have c64841 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.1
  have c64842 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64843 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64844 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1048_18 := arithEq_of_rows h (row := 1048) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_18
  simp only [← c64839, ← c64840, ← c64841] at e_1048_18
  have e_1048_19 := arithEq_of_rows h (row := 1048) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_19
  simp only [← c64842, ← c64843, k2, ← c64844] at e_1048_19
  have hr := e_1048_19
  simp only [e_1048_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8928)) := by
  have c64845 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64846 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64847 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64848 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64849 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64850 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_0 := arithEq_of_rows h (row := 1049) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_0
  simp only [← c64845, ← c64846, ← c64847] at e_1049_0
  have e_1049_1 := arithEq_of_rows h (row := 1049) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_1
  simp only [← c64848, ← c64849, k2, ← c64850] at e_1049_1
  have hr := e_1049_1
  simp only [e_1049_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8929)) := by
  have c64851 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64852 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64853 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64854 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64855 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64856 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_2 := arithEq_of_rows h (row := 1049) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_2
  simp only [← c64851, ← c64852, ← c64853] at e_1049_2
  have e_1049_3 := arithEq_of_rows h (row := 1049) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_3
  simp only [← c64854, ← c64855, k2, ← c64856] at e_1049_3
  have hr := e_1049_3
  simp only [e_1049_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8930)) := by
  have c64857 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64858 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64859 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64860 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64861 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64862 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_4 := arithEq_of_rows h (row := 1049) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_4
  simp only [← c64857, ← c64858, ← c64859] at e_1049_4
  have e_1049_5 := arithEq_of_rows h (row := 1049) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_5
  simp only [← c64860, ← c64861, k2, ← c64862] at e_1049_5
  have hr := e_1049_5
  simp only [e_1049_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8931)) := by
  have c64863 := (publicBatchWrapper86_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64864 := (publicBatchWrapper86_copies2027 a h).1
  have c64865 := (publicBatchWrapper86_copies2027 a h).2.1
  have c64866 := (publicBatchWrapper86_copies2027 a h).2.2.1
  have c64867 := (publicBatchWrapper86_copies2027 a h).2.2.2.1
  have c64868 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_6 := arithEq_of_rows h (row := 1049) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_6
  simp only [← c64863, ← c64864, ← c64865] at e_1049_6
  have e_1049_7 := arithEq_of_rows h (row := 1049) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_7
  simp only [← c64866, ← c64867, k2, ← c64868] at e_1049_7
  have hr := e_1049_7
  simp only [e_1049_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8932)) := by
  have c64869 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.1
  have c64870 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.1
  have c64871 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.1
  have c64872 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.1
  have c64873 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.1
  have c64874 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_8 := arithEq_of_rows h (row := 1049) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_8
  simp only [← c64869, ← c64870, ← c64871] at e_1049_8
  have e_1049_9 := arithEq_of_rows h (row := 1049) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_9
  simp only [← c64872, ← c64873, k2, ← c64874] at e_1049_9
  have hr := e_1049_9
  simp only [e_1049_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8933)) := by
  have c64875 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64876 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64877 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64878 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64879 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64880 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_10 := arithEq_of_rows h (row := 1049) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_10
  simp only [← c64875, ← c64876, ← c64877] at e_1049_10
  have e_1049_11 := arithEq_of_rows h (row := 1049) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_11
  simp only [← c64878, ← c64879, k2, ← c64880] at e_1049_11
  have hr := e_1049_11
  simp only [e_1049_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8934)) := by
  have c64881 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64882 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64883 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64884 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64885 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64886 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_12 := arithEq_of_rows h (row := 1049) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_12
  simp only [← c64881, ← c64882, ← c64883] at e_1049_12
  have e_1049_13 := arithEq_of_rows h (row := 1049) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_13
  simp only [← c64884, ← c64885, k2, ← c64886] at e_1049_13
  have hr := e_1049_13
  simp only [e_1049_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8935)) := by
  have c64887 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64888 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64889 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64890 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64891 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64892 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_14 := arithEq_of_rows h (row := 1049) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_14
  simp only [← c64887, ← c64888, ← c64889] at e_1049_14
  have e_1049_15 := arithEq_of_rows h (row := 1049) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_15
  simp only [← c64890, ← c64891, k2, ← c64892] at e_1049_15
  have hr := e_1049_15
  simp only [e_1049_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8936)) := by
  have c64893 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64894 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64895 := (publicBatchWrapper86_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64896 := (publicBatchWrapper86_copies2028 a h).1
  have c64897 := (publicBatchWrapper86_copies2028 a h).2.1
  have c64898 := (publicBatchWrapper86_copies2028 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_16 := arithEq_of_rows h (row := 1049) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_16
  simp only [← c64893, ← c64894, ← c64895] at e_1049_16
  have e_1049_17 := arithEq_of_rows h (row := 1049) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_17
  simp only [← c64896, ← c64897, k2, ← c64898] at e_1049_17
  have hr := e_1049_17
  simp only [e_1049_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1049 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8937)) := by
  have c64899 := (publicBatchWrapper86_copies2028 a h).2.2.2.1
  have c64900 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.1
  have c64901 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.1
  have c64902 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.1
  have c64903 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.1
  have c64904 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1049_18 := arithEq_of_rows h (row := 1049) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_18
  simp only [← c64899, ← c64900, ← c64901] at e_1049_18
  have e_1049_19 := arithEq_of_rows h (row := 1049) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_19
  simp only [← c64902, ← c64903, k2, ← c64904] at e_1049_19
  have hr := e_1049_19
  simp only [e_1049_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8938)) := by
  have c64905 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.1
  have c64906 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64907 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64908 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64909 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64910 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_0 := arithEq_of_rows h (row := 1050) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_0
  simp only [← c64905, ← c64906, ← c64907] at e_1050_0
  have e_1050_1 := arithEq_of_rows h (row := 1050) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_1
  simp only [← c64908, ← c64909, k2, ← c64910] at e_1050_1
  have hr := e_1050_1
  simp only [e_1050_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8939)) := by
  have c64911 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64912 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64913 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64914 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64915 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64916 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_2 := arithEq_of_rows h (row := 1050) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_2
  simp only [← c64911, ← c64912, ← c64913] at e_1050_2
  have e_1050_3 := arithEq_of_rows h (row := 1050) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_3
  simp only [← c64914, ← c64915, k2, ← c64916] at e_1050_3
  have hr := e_1050_3
  simp only [e_1050_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8940)) := by
  have c64917 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64918 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64919 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64920 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64921 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64922 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_4 := arithEq_of_rows h (row := 1050) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_4
  simp only [← c64917, ← c64918, ← c64919] at e_1050_4
  have e_1050_5 := arithEq_of_rows h (row := 1050) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_5
  simp only [← c64920, ← c64921, k2, ← c64922] at e_1050_5
  have hr := e_1050_5
  simp only [e_1050_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8941)) := by
  have c64923 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64924 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64925 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64926 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64927 := (publicBatchWrapper86_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64928 := (publicBatchWrapper86_copies2029 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_6 := arithEq_of_rows h (row := 1050) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_6
  simp only [← c64923, ← c64924, ← c64925] at e_1050_6
  have e_1050_7 := arithEq_of_rows h (row := 1050) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_7
  simp only [← c64926, ← c64927, k2, ← c64928] at e_1050_7
  have hr := e_1050_7
  simp only [e_1050_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8942)) := by
  have c64929 := (publicBatchWrapper86_copies2029 a h).2.1
  have c64930 := (publicBatchWrapper86_copies2029 a h).2.2.1
  have c64931 := (publicBatchWrapper86_copies2029 a h).2.2.2.1
  have c64932 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.1
  have c64933 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.1
  have c64934 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_8 := arithEq_of_rows h (row := 1050) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_8
  simp only [← c64929, ← c64930, ← c64931] at e_1050_8
  have e_1050_9 := arithEq_of_rows h (row := 1050) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_9
  simp only [← c64932, ← c64933, k2, ← c64934] at e_1050_9
  have hr := e_1050_9
  simp only [e_1050_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8943)) := by
  have c64935 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.1
  have c64936 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.1
  have c64937 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.1
  have c64938 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64939 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64940 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_10 := arithEq_of_rows h (row := 1050) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_10
  simp only [← c64935, ← c64936, ← c64937] at e_1050_10
  have e_1050_11 := arithEq_of_rows h (row := 1050) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_11
  simp only [← c64938, ← c64939, k2, ← c64940] at e_1050_11
  have hr := e_1050_11
  simp only [e_1050_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8944)) := by
  have c64941 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64942 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64943 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64944 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64945 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64946 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_12 := arithEq_of_rows h (row := 1050) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_12
  simp only [← c64941, ← c64942, ← c64943] at e_1050_12
  have e_1050_13 := arithEq_of_rows h (row := 1050) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_13
  simp only [← c64944, ← c64945, k2, ← c64946] at e_1050_13
  have hr := e_1050_13
  simp only [e_1050_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8945)) := by
  have c64947 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64948 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64949 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64950 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64951 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64952 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_14 := arithEq_of_rows h (row := 1050) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_14
  simp only [← c64947, ← c64948, ← c64949] at e_1050_14
  have e_1050_15 := arithEq_of_rows h (row := 1050) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_15
  simp only [← c64950, ← c64951, k2, ← c64952] at e_1050_15
  have hr := e_1050_15
  simp only [e_1050_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8946)) := by
  have c64953 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64954 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64955 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64956 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64957 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64958 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_16 := arithEq_of_rows h (row := 1050) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_16
  simp only [← c64953, ← c64954, ← c64955] at e_1050_16
  have e_1050_17 := arithEq_of_rows h (row := 1050) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_17
  simp only [← c64956, ← c64957, k2, ← c64958] at e_1050_17
  have hr := e_1050_17
  simp only [e_1050_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1050 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8947)) := by
  have c64959 := (publicBatchWrapper86_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64960 := (publicBatchWrapper86_copies2030 a h).1
  have c64961 := (publicBatchWrapper86_copies2030 a h).2.1
  have c64962 := (publicBatchWrapper86_copies2030 a h).2.2.1
  have c64963 := (publicBatchWrapper86_copies2030 a h).2.2.2.1
  have c64964 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1050_18 := arithEq_of_rows h (row := 1050) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_18
  simp only [← c64959, ← c64960, ← c64961] at e_1050_18
  have e_1050_19 := arithEq_of_rows h (row := 1050) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_19
  simp only [← c64962, ← c64963, k2, ← c64964] at e_1050_19
  have hr := e_1050_19
  simp only [e_1050_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8948)) := by
  have c64965 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.1
  have c64966 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.1
  have c64967 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.1
  have c64968 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.1
  have c64969 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.1
  have c64970 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_0 := arithEq_of_rows h (row := 1051) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_0
  simp only [← c64965, ← c64966, ← c64967] at e_1051_0
  have e_1051_1 := arithEq_of_rows h (row := 1051) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_1
  simp only [← c64968, ← c64969, k2, ← c64970] at e_1051_1
  have hr := e_1051_1
  simp only [e_1051_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8949)) := by
  have c64971 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64972 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64973 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64974 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64975 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64976 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_2 := arithEq_of_rows h (row := 1051) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_2
  simp only [← c64971, ← c64972, ← c64973] at e_1051_2
  have e_1051_3 := arithEq_of_rows h (row := 1051) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_3
  simp only [← c64974, ← c64975, k2, ← c64976] at e_1051_3
  have hr := e_1051_3
  simp only [e_1051_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8950)) := by
  have c64977 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64978 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64979 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64980 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64981 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64982 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_4 := arithEq_of_rows h (row := 1051) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_4
  simp only [← c64977, ← c64978, ← c64979] at e_1051_4
  have e_1051_5 := arithEq_of_rows h (row := 1051) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_5
  simp only [← c64980, ← c64981, k2, ← c64982] at e_1051_5
  have hr := e_1051_5
  simp only [e_1051_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8951)) := by
  have c64983 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64984 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64985 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64986 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64987 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64988 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_6 := arithEq_of_rows h (row := 1051) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_6
  simp only [← c64983, ← c64984, ← c64985] at e_1051_6
  have e_1051_7 := arithEq_of_rows h (row := 1051) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_7
  simp only [← c64986, ← c64987, k2, ← c64988] at e_1051_7
  have hr := e_1051_7
  simp only [e_1051_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8952)) := by
  have c64989 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64990 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64991 := (publicBatchWrapper86_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64992 := (publicBatchWrapper86_copies2031 a h).1
  have c64993 := (publicBatchWrapper86_copies2031 a h).2.1
  have c64994 := (publicBatchWrapper86_copies2031 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_8 := arithEq_of_rows h (row := 1051) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_8
  simp only [← c64989, ← c64990, ← c64991] at e_1051_8
  have e_1051_9 := arithEq_of_rows h (row := 1051) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_9
  simp only [← c64992, ← c64993, k2, ← c64994] at e_1051_9
  have hr := e_1051_9
  simp only [e_1051_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8953)) := by
  have c64995 := (publicBatchWrapper86_copies2031 a h).2.2.2.1
  have c64996 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.1
  have c64997 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.1
  have c64998 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.1
  have c64999 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.1
  have c65000 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_10 := arithEq_of_rows h (row := 1051) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_10
  simp only [← c64995, ← c64996, ← c64997] at e_1051_10
  have e_1051_11 := arithEq_of_rows h (row := 1051) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_11
  simp only [← c64998, ← c64999, k2, ← c65000] at e_1051_11
  have hr := e_1051_11
  simp only [e_1051_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8954)) := by
  have c65001 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.1
  have c65002 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65003 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65004 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65005 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65006 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_12 := arithEq_of_rows h (row := 1051) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_12
  simp only [← c65001, ← c65002, ← c65003] at e_1051_12
  have e_1051_13 := arithEq_of_rows h (row := 1051) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_13
  simp only [← c65004, ← c65005, k2, ← c65006] at e_1051_13
  have hr := e_1051_13
  simp only [e_1051_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8955)) := by
  have c65007 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65008 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65009 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65010 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65011 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65012 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_14 := arithEq_of_rows h (row := 1051) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_14
  simp only [← c65007, ← c65008, ← c65009] at e_1051_14
  have e_1051_15 := arithEq_of_rows h (row := 1051) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_15
  simp only [← c65010, ← c65011, k2, ← c65012] at e_1051_15
  have hr := e_1051_15
  simp only [e_1051_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8956)) := by
  have c65013 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65014 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65015 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65016 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65017 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65018 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_16 := arithEq_of_rows h (row := 1051) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_16
  simp only [← c65013, ← c65014, ← c65015] at e_1051_16
  have e_1051_17 := arithEq_of_rows h (row := 1051) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_17
  simp only [← c65016, ← c65017, k2, ← c65018] at e_1051_17
  have hr := e_1051_17
  simp only [e_1051_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1051 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8957)) := by
  have c65019 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65020 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65021 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65022 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65023 := (publicBatchWrapper86_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65024 := (publicBatchWrapper86_copies2032 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1051_18 := arithEq_of_rows h (row := 1051) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_18
  simp only [← c65019, ← c65020, ← c65021] at e_1051_18
  have e_1051_19 := arithEq_of_rows h (row := 1051) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_19
  simp only [← c65022, ← c65023, k2, ← c65024] at e_1051_19
  have hr := e_1051_19
  simp only [e_1051_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8958)) := by
  have c65025 := (publicBatchWrapper86_copies2032 a h).2.1
  have c65026 := (publicBatchWrapper86_copies2032 a h).2.2.1
  have c65027 := (publicBatchWrapper86_copies2032 a h).2.2.2.1
  have c65028 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.1
  have c65029 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.1
  have c65030 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_0 := arithEq_of_rows h (row := 1052) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_0
  simp only [← c65025, ← c65026, ← c65027] at e_1052_0
  have e_1052_1 := arithEq_of_rows h (row := 1052) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_1
  simp only [← c65028, ← c65029, k2, ← c65030] at e_1052_1
  have hr := e_1052_1
  simp only [e_1052_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8959)) := by
  have c65031 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.1
  have c65032 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.1
  have c65033 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.1
  have c65034 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65035 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65036 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_2 := arithEq_of_rows h (row := 1052) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_2
  simp only [← c65031, ← c65032, ← c65033] at e_1052_2
  have e_1052_3 := arithEq_of_rows h (row := 1052) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_3
  simp only [← c65034, ← c65035, k2, ← c65036] at e_1052_3
  have hr := e_1052_3
  simp only [e_1052_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8960)) := by
  have c65037 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65038 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65039 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65040 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65041 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65042 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_4 := arithEq_of_rows h (row := 1052) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_4
  simp only [← c65037, ← c65038, ← c65039] at e_1052_4
  have e_1052_5 := arithEq_of_rows h (row := 1052) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_5
  simp only [← c65040, ← c65041, k2, ← c65042] at e_1052_5
  have hr := e_1052_5
  simp only [e_1052_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8961)) := by
  have c65043 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65044 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65045 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65046 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65047 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65048 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_6 := arithEq_of_rows h (row := 1052) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_6
  simp only [← c65043, ← c65044, ← c65045] at e_1052_6
  have e_1052_7 := arithEq_of_rows h (row := 1052) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_7
  simp only [← c65046, ← c65047, k2, ← c65048] at e_1052_7
  have hr := e_1052_7
  simp only [e_1052_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8962)) := by
  have c65049 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65050 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65051 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65052 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65053 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65054 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_8 := arithEq_of_rows h (row := 1052) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_8
  simp only [← c65049, ← c65050, ← c65051] at e_1052_8
  have e_1052_9 := arithEq_of_rows h (row := 1052) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_9
  simp only [← c65052, ← c65053, k2, ← c65054] at e_1052_9
  have hr := e_1052_9
  simp only [e_1052_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8963)) := by
  have c65055 := (publicBatchWrapper86_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65056 := (publicBatchWrapper86_copies2033 a h).1
  have c65057 := (publicBatchWrapper86_copies2033 a h).2.1
  have c65058 := (publicBatchWrapper86_copies2033 a h).2.2.1
  have c65059 := (publicBatchWrapper86_copies2033 a h).2.2.2.1
  have c65060 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_10 := arithEq_of_rows h (row := 1052) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_10
  simp only [← c65055, ← c65056, ← c65057] at e_1052_10
  have e_1052_11 := arithEq_of_rows h (row := 1052) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_11
  simp only [← c65058, ← c65059, k2, ← c65060] at e_1052_11
  have hr := e_1052_11
  simp only [e_1052_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8964)) := by
  have c65061 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.1
  have c65062 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.1
  have c65063 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.1
  have c65064 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.1
  have c65065 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.1
  have c65066 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_12 := arithEq_of_rows h (row := 1052) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_12
  simp only [← c65061, ← c65062, ← c65063] at e_1052_12
  have e_1052_13 := arithEq_of_rows h (row := 1052) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_13
  simp only [← c65064, ← c65065, k2, ← c65066] at e_1052_13
  have hr := e_1052_13
  simp only [e_1052_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8965)) := by
  have c65067 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65068 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65069 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65070 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65071 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65072 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_14 := arithEq_of_rows h (row := 1052) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_14
  simp only [← c65067, ← c65068, ← c65069] at e_1052_14
  have e_1052_15 := arithEq_of_rows h (row := 1052) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_15
  simp only [← c65070, ← c65071, k2, ← c65072] at e_1052_15
  have hr := e_1052_15
  simp only [e_1052_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8966)) := by
  have c65073 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65074 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65075 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65076 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65077 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65078 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_16 := arithEq_of_rows h (row := 1052) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_16
  simp only [← c65073, ← c65074, ← c65075] at e_1052_16
  have e_1052_17 := arithEq_of_rows h (row := 1052) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_17
  simp only [← c65076, ← c65077, k2, ← c65078] at e_1052_17
  have hr := e_1052_17
  simp only [e_1052_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1052 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8967)) := by
  have c65079 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65080 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65081 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65082 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65083 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65084 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1052_18 := arithEq_of_rows h (row := 1052) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_18
  simp only [← c65079, ← c65080, ← c65081] at e_1052_18
  have e_1052_19 := arithEq_of_rows h (row := 1052) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_19
  simp only [← c65082, ← c65083, k2, ← c65084] at e_1052_19
  have hr := e_1052_19
  simp only [e_1052_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8968)) := by
  have c65085 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65086 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65087 := (publicBatchWrapper86_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65088 := (publicBatchWrapper86_copies2034 a h).1
  have c65089 := (publicBatchWrapper86_copies2034 a h).2.1
  have c65090 := (publicBatchWrapper86_copies2034 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_0 := arithEq_of_rows h (row := 1053) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_0
  simp only [← c65085, ← c65086, ← c65087] at e_1053_0
  have e_1053_1 := arithEq_of_rows h (row := 1053) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_1
  simp only [← c65088, ← c65089, k2, ← c65090] at e_1053_1
  have hr := e_1053_1
  simp only [e_1053_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8969)) := by
  have c65091 := (publicBatchWrapper86_copies2034 a h).2.2.2.1
  have c65092 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.1
  have c65093 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.1
  have c65094 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.1
  have c65095 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.1
  have c65096 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_2 := arithEq_of_rows h (row := 1053) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_2
  simp only [← c65091, ← c65092, ← c65093] at e_1053_2
  have e_1053_3 := arithEq_of_rows h (row := 1053) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_3
  simp only [← c65094, ← c65095, k2, ← c65096] at e_1053_3
  have hr := e_1053_3
  simp only [e_1053_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9010)) := by
  have c65097 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.1
  have c65098 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65099 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65100 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65101 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65102 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_4 := arithEq_of_rows h (row := 1053) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_4
  simp only [← c65097, ← c65098, ← c65099] at e_1053_4
  have e_1053_5 := arithEq_of_rows h (row := 1053) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_5
  simp only [← c65100, ← c65101, k2, ← c65102] at e_1053_5
  have hr := e_1053_5
  simp only [e_1053_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9011)) := by
  have c65103 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65104 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65105 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65106 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65107 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65108 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_6 := arithEq_of_rows h (row := 1053) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_6
  simp only [← c65103, ← c65104, ← c65105] at e_1053_6
  have e_1053_7 := arithEq_of_rows h (row := 1053) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_7
  simp only [← c65106, ← c65107, k2, ← c65108] at e_1053_7
  have hr := e_1053_7
  simp only [e_1053_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9012)) := by
  have c65109 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65110 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65111 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65112 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65113 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65114 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_8 := arithEq_of_rows h (row := 1053) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_8
  simp only [← c65109, ← c65110, ← c65111] at e_1053_8
  have e_1053_9 := arithEq_of_rows h (row := 1053) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_9
  simp only [← c65112, ← c65113, k2, ← c65114] at e_1053_9
  have hr := e_1053_9
  simp only [e_1053_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9013)) := by
  have c65115 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65116 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65117 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65118 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65119 := (publicBatchWrapper86_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65120 := (publicBatchWrapper86_copies2035 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_10 := arithEq_of_rows h (row := 1053) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_10
  simp only [← c65115, ← c65116, ← c65117] at e_1053_10
  have e_1053_11 := arithEq_of_rows h (row := 1053) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_11
  simp only [← c65118, ← c65119, k2, ← c65120] at e_1053_11
  have hr := e_1053_11
  simp only [e_1053_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9014)) := by
  have c65121 := (publicBatchWrapper86_copies2035 a h).2.1
  have c65122 := (publicBatchWrapper86_copies2035 a h).2.2.1
  have c65123 := (publicBatchWrapper86_copies2035 a h).2.2.2.1
  have c65124 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.1
  have c65125 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.1
  have c65126 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_12 := arithEq_of_rows h (row := 1053) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_12
  simp only [← c65121, ← c65122, ← c65123] at e_1053_12
  have e_1053_13 := arithEq_of_rows h (row := 1053) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_13
  simp only [← c65124, ← c65125, k2, ← c65126] at e_1053_13
  have hr := e_1053_13
  simp only [e_1053_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9015)) := by
  have c65127 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.1
  have c65128 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.1
  have c65129 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.1
  have c65130 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65131 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65132 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_14 := arithEq_of_rows h (row := 1053) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_14
  simp only [← c65127, ← c65128, ← c65129] at e_1053_14
  have e_1053_15 := arithEq_of_rows h (row := 1053) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_15
  simp only [← c65130, ← c65131, k2, ← c65132] at e_1053_15
  have hr := e_1053_15
  simp only [e_1053_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9016)) := by
  have c65133 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65134 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65135 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65136 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65137 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65138 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_16 := arithEq_of_rows h (row := 1053) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_16
  simp only [← c65133, ← c65134, ← c65135] at e_1053_16
  have e_1053_17 := arithEq_of_rows h (row := 1053) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_17
  simp only [← c65136, ← c65137, k2, ← c65138] at e_1053_17
  have hr := e_1053_17
  simp only [e_1053_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1053 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9017)) := by
  have c65139 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65140 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65141 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65142 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65143 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65144 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1053_18 := arithEq_of_rows h (row := 1053) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_18
  simp only [← c65139, ← c65140, ← c65141] at e_1053_18
  have e_1053_19 := arithEq_of_rows h (row := 1053) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_19
  simp only [← c65142, ← c65143, k2, ← c65144] at e_1053_19
  have hr := e_1053_19
  simp only [e_1053_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9018)) := by
  have c65145 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65146 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65147 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65148 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65149 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65150 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_0 := arithEq_of_rows h (row := 1054) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_0
  simp only [← c65145, ← c65146, ← c65147] at e_1054_0
  have e_1054_1 := arithEq_of_rows h (row := 1054) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_1
  simp only [← c65148, ← c65149, k2, ← c65150] at e_1054_1
  have hr := e_1054_1
  simp only [e_1054_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9019)) := by
  have c65151 := (publicBatchWrapper86_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65152 := (publicBatchWrapper86_copies2036 a h).1
  have c65153 := (publicBatchWrapper86_copies2036 a h).2.1
  have c65154 := (publicBatchWrapper86_copies2036 a h).2.2.1
  have c65155 := (publicBatchWrapper86_copies2036 a h).2.2.2.1
  have c65156 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_2 := arithEq_of_rows h (row := 1054) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_2
  simp only [← c65151, ← c65152, ← c65153] at e_1054_2
  have e_1054_3 := arithEq_of_rows h (row := 1054) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_3
  simp only [← c65154, ← c65155, k2, ← c65156] at e_1054_3
  have hr := e_1054_3
  simp only [e_1054_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9020)) := by
  have c65157 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.1
  have c65158 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.1
  have c65159 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.1
  have c65160 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.1
  have c65161 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.1
  have c65162 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_4 := arithEq_of_rows h (row := 1054) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_4
  simp only [← c65157, ← c65158, ← c65159] at e_1054_4
  have e_1054_5 := arithEq_of_rows h (row := 1054) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_5
  simp only [← c65160, ← c65161, k2, ← c65162] at e_1054_5
  have hr := e_1054_5
  simp only [e_1054_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9021)) := by
  have c65163 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65164 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65165 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65166 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65167 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65168 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_6 := arithEq_of_rows h (row := 1054) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_6
  simp only [← c65163, ← c65164, ← c65165] at e_1054_6
  have e_1054_7 := arithEq_of_rows h (row := 1054) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_7
  simp only [← c65166, ← c65167, k2, ← c65168] at e_1054_7
  have hr := e_1054_7
  simp only [e_1054_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9022)) := by
  have c65169 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65170 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65171 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65172 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65173 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65174 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_8 := arithEq_of_rows h (row := 1054) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_8
  simp only [← c65169, ← c65170, ← c65171] at e_1054_8
  have e_1054_9 := arithEq_of_rows h (row := 1054) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_9
  simp only [← c65172, ← c65173, k2, ← c65174] at e_1054_9
  have hr := e_1054_9
  simp only [e_1054_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9023)) := by
  have c65175 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65176 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65177 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65178 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65179 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65180 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_10 := arithEq_of_rows h (row := 1054) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_10
  simp only [← c65175, ← c65176, ← c65177] at e_1054_10
  have e_1054_11 := arithEq_of_rows h (row := 1054) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_11
  simp only [← c65178, ← c65179, k2, ← c65180] at e_1054_11
  have hr := e_1054_11
  simp only [e_1054_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9024)) := by
  have c65181 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65182 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65183 := (publicBatchWrapper86_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65184 := (publicBatchWrapper86_copies2037 a h).1
  have c65185 := (publicBatchWrapper86_copies2037 a h).2.1
  have c65186 := (publicBatchWrapper86_copies2037 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_12 := arithEq_of_rows h (row := 1054) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_12
  simp only [← c65181, ← c65182, ← c65183] at e_1054_12
  have e_1054_13 := arithEq_of_rows h (row := 1054) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_13
  simp only [← c65184, ← c65185, k2, ← c65186] at e_1054_13
  have hr := e_1054_13
  simp only [e_1054_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9025)) := by
  have c65187 := (publicBatchWrapper86_copies2037 a h).2.2.2.1
  have c65188 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.1
  have c65189 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.1
  have c65190 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.1
  have c65191 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.1
  have c65192 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_14 := arithEq_of_rows h (row := 1054) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_14
  simp only [← c65187, ← c65188, ← c65189] at e_1054_14
  have e_1054_15 := arithEq_of_rows h (row := 1054) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_15
  simp only [← c65190, ← c65191, k2, ← c65192] at e_1054_15
  have hr := e_1054_15
  simp only [e_1054_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9026)) := by
  have c65193 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.1
  have c65194 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65195 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65196 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65197 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65198 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_16 := arithEq_of_rows h (row := 1054) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_16
  simp only [← c65193, ← c65194, ← c65195] at e_1054_16
  have e_1054_17 := arithEq_of_rows h (row := 1054) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_17
  simp only [← c65196, ← c65197, k2, ← c65198] at e_1054_17
  have hr := e_1054_17
  simp only [e_1054_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1054 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9027)) := by
  have c65199 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65200 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65201 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65202 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65203 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65204 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1054_18 := arithEq_of_rows h (row := 1054) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_18
  simp only [← c65199, ← c65200, ← c65201] at e_1054_18
  have e_1054_19 := arithEq_of_rows h (row := 1054) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_19
  simp only [← c65202, ← c65203, k2, ← c65204] at e_1054_19
  have hr := e_1054_19
  simp only [e_1054_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9028)) := by
  have c65205 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65206 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65207 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65208 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65209 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65210 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_0 := arithEq_of_rows h (row := 1055) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_0
  simp only [← c65205, ← c65206, ← c65207] at e_1055_0
  have e_1055_1 := arithEq_of_rows h (row := 1055) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_1
  simp only [← c65208, ← c65209, k2, ← c65210] at e_1055_1
  have hr := e_1055_1
  simp only [e_1055_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9029)) := by
  have c65211 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65212 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65213 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65214 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65215 := (publicBatchWrapper86_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65216 := (publicBatchWrapper86_copies2038 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_2 := arithEq_of_rows h (row := 1055) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_2
  simp only [← c65211, ← c65212, ← c65213] at e_1055_2
  have e_1055_3 := arithEq_of_rows h (row := 1055) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_3
  simp only [← c65214, ← c65215, k2, ← c65216] at e_1055_3
  have hr := e_1055_3
  simp only [e_1055_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9030)) := by
  have c65217 := (publicBatchWrapper86_copies2038 a h).2.1
  have c65218 := (publicBatchWrapper86_copies2038 a h).2.2.1
  have c65219 := (publicBatchWrapper86_copies2038 a h).2.2.2.1
  have c65220 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.1
  have c65221 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.1
  have c65222 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_4 := arithEq_of_rows h (row := 1055) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_4
  simp only [← c65217, ← c65218, ← c65219] at e_1055_4
  have e_1055_5 := arithEq_of_rows h (row := 1055) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_5
  simp only [← c65220, ← c65221, k2, ← c65222] at e_1055_5
  have hr := e_1055_5
  simp only [e_1055_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9031)) := by
  have c65223 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.1
  have c65224 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.1
  have c65225 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.1
  have c65226 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65227 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65228 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_6 := arithEq_of_rows h (row := 1055) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_6
  simp only [← c65223, ← c65224, ← c65225] at e_1055_6
  have e_1055_7 := arithEq_of_rows h (row := 1055) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_7
  simp only [← c65226, ← c65227, k2, ← c65228] at e_1055_7
  have hr := e_1055_7
  simp only [e_1055_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9032)) := by
  have c65229 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65230 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65231 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65232 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65233 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65234 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_8 := arithEq_of_rows h (row := 1055) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_8
  simp only [← c65229, ← c65230, ← c65231] at e_1055_8
  have e_1055_9 := arithEq_of_rows h (row := 1055) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_9
  simp only [← c65232, ← c65233, k2, ← c65234] at e_1055_9
  have hr := e_1055_9
  simp only [e_1055_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9033)) := by
  have c65235 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65236 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65237 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65238 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65239 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65240 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_10 := arithEq_of_rows h (row := 1055) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_10
  simp only [← c65235, ← c65236, ← c65237] at e_1055_10
  have e_1055_11 := arithEq_of_rows h (row := 1055) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_11
  simp only [← c65238, ← c65239, k2, ← c65240] at e_1055_11
  have hr := e_1055_11
  simp only [e_1055_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9034)) := by
  have c65241 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65242 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65243 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65244 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65245 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65246 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_12 := arithEq_of_rows h (row := 1055) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_12
  simp only [← c65241, ← c65242, ← c65243] at e_1055_12
  have e_1055_13 := arithEq_of_rows h (row := 1055) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_13
  simp only [← c65244, ← c65245, k2, ← c65246] at e_1055_13
  have hr := e_1055_13
  simp only [e_1055_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9035)) := by
  have c65247 := (publicBatchWrapper86_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65248 := (publicBatchWrapper86_copies2039 a h).1
  have c65249 := (publicBatchWrapper86_copies2039 a h).2.1
  have c65250 := (publicBatchWrapper86_copies2039 a h).2.2.1
  have c65251 := (publicBatchWrapper86_copies2039 a h).2.2.2.1
  have c65252 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_14 := arithEq_of_rows h (row := 1055) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_14
  simp only [← c65247, ← c65248, ← c65249] at e_1055_14
  have e_1055_15 := arithEq_of_rows h (row := 1055) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_15
  simp only [← c65250, ← c65251, k2, ← c65252] at e_1055_15
  have hr := e_1055_15
  simp only [e_1055_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9036)) := by
  have c65253 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.1
  have c65254 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.1
  have c65255 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.1
  have c65256 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.1
  have c65257 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.1
  have c65258 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_16 := arithEq_of_rows h (row := 1055) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_16
  simp only [← c65253, ← c65254, ← c65255] at e_1055_16
  have e_1055_17 := arithEq_of_rows h (row := 1055) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_17
  simp only [← c65256, ← c65257, k2, ← c65258] at e_1055_17
  have hr := e_1055_17
  simp only [e_1055_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1055 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9037)) := by
  have c65259 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65260 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65261 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65262 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65263 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65264 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1055_18 := arithEq_of_rows h (row := 1055) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_18
  simp only [← c65259, ← c65260, ← c65261] at e_1055_18
  have e_1055_19 := arithEq_of_rows h (row := 1055) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_19
  simp only [← c65262, ← c65263, k2, ← c65264] at e_1055_19
  have hr := e_1055_19
  simp only [e_1055_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9038)) := by
  have c65265 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65266 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65267 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65268 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65269 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65270 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_0 := arithEq_of_rows h (row := 1056) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_0
  simp only [← c65265, ← c65266, ← c65267] at e_1056_0
  have e_1056_1 := arithEq_of_rows h (row := 1056) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_1
  simp only [← c65268, ← c65269, k2, ← c65270] at e_1056_1
  have hr := e_1056_1
  simp only [e_1056_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9039)) := by
  have c65271 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65272 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65273 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65274 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65275 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65276 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_2 := arithEq_of_rows h (row := 1056) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_2
  simp only [← c65271, ← c65272, ← c65273] at e_1056_2
  have e_1056_3 := arithEq_of_rows h (row := 1056) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_3
  simp only [← c65274, ← c65275, k2, ← c65276] at e_1056_3
  have hr := e_1056_3
  simp only [e_1056_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9040)) := by
  have c65277 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65278 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65279 := (publicBatchWrapper86_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65280 := (publicBatchWrapper86_copies2040 a h).1
  have c65281 := (publicBatchWrapper86_copies2040 a h).2.1
  have c65282 := (publicBatchWrapper86_copies2040 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_4 := arithEq_of_rows h (row := 1056) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_4
  simp only [← c65277, ← c65278, ← c65279] at e_1056_4
  have e_1056_5 := arithEq_of_rows h (row := 1056) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_5
  simp only [← c65280, ← c65281, k2, ← c65282] at e_1056_5
  have hr := e_1056_5
  simp only [e_1056_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9041)) := by
  have c65283 := (publicBatchWrapper86_copies2040 a h).2.2.2.1
  have c65284 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.1
  have c65285 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.1
  have c65286 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.1
  have c65287 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.1
  have c65288 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_6 := arithEq_of_rows h (row := 1056) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_6
  simp only [← c65283, ← c65284, ← c65285] at e_1056_6
  have e_1056_7 := arithEq_of_rows h (row := 1056) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_7
  simp only [← c65286, ← c65287, k2, ← c65288] at e_1056_7
  have hr := e_1056_7
  simp only [e_1056_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9042)) := by
  have c65289 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.1
  have c65290 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65291 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65292 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65293 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65294 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_8 := arithEq_of_rows h (row := 1056) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_8
  simp only [← c65289, ← c65290, ← c65291] at e_1056_8
  have e_1056_9 := arithEq_of_rows h (row := 1056) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_9
  simp only [← c65292, ← c65293, k2, ← c65294] at e_1056_9
  have hr := e_1056_9
  simp only [e_1056_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9043)) := by
  have c65295 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65296 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65297 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65298 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65299 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65300 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_10 := arithEq_of_rows h (row := 1056) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_10
  simp only [← c65295, ← c65296, ← c65297] at e_1056_10
  have e_1056_11 := arithEq_of_rows h (row := 1056) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_11
  simp only [← c65298, ← c65299, k2, ← c65300] at e_1056_11
  have hr := e_1056_11
  simp only [e_1056_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9044)) := by
  have c65301 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65302 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65303 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65304 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65305 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65306 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_12 := arithEq_of_rows h (row := 1056) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_12
  simp only [← c65301, ← c65302, ← c65303] at e_1056_12
  have e_1056_13 := arithEq_of_rows h (row := 1056) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_13
  simp only [← c65304, ← c65305, k2, ← c65306] at e_1056_13
  have hr := e_1056_13
  simp only [e_1056_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9045)) := by
  have c65307 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65308 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65309 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65310 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65311 := (publicBatchWrapper86_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65312 := (publicBatchWrapper86_copies2041 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_14 := arithEq_of_rows h (row := 1056) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_14
  simp only [← c65307, ← c65308, ← c65309] at e_1056_14
  have e_1056_15 := arithEq_of_rows h (row := 1056) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_15
  simp only [← c65310, ← c65311, k2, ← c65312] at e_1056_15
  have hr := e_1056_15
  simp only [e_1056_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9046)) := by
  have c65313 := (publicBatchWrapper86_copies2041 a h).2.1
  have c65314 := (publicBatchWrapper86_copies2041 a h).2.2.1
  have c65315 := (publicBatchWrapper86_copies2041 a h).2.2.2.1
  have c65316 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.1
  have c65317 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.1
  have c65318 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_16 := arithEq_of_rows h (row := 1056) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_16
  simp only [← c65313, ← c65314, ← c65315] at e_1056_16
  have e_1056_17 := arithEq_of_rows h (row := 1056) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_17
  simp only [← c65316, ← c65317, k2, ← c65318] at e_1056_17
  have hr := e_1056_17
  simp only [e_1056_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1056 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9047)) := by
  have c65319 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.1
  have c65320 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.1
  have c65321 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.1
  have c65322 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65323 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65324 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1056_18 := arithEq_of_rows h (row := 1056) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_18
  simp only [← c65319, ← c65320, ← c65321] at e_1056_18
  have e_1056_19 := arithEq_of_rows h (row := 1056) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_19
  simp only [← c65322, ← c65323, k2, ← c65324] at e_1056_19
  have hr := e_1056_19
  simp only [e_1056_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9048)) := by
  have c65325 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65326 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65327 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65328 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65329 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65330 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_0 := arithEq_of_rows h (row := 1057) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_0
  simp only [← c65325, ← c65326, ← c65327] at e_1057_0
  have e_1057_1 := arithEq_of_rows h (row := 1057) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_1
  simp only [← c65328, ← c65329, k2, ← c65330] at e_1057_1
  have hr := e_1057_1
  simp only [e_1057_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9049)) := by
  have c65331 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65332 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65333 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65334 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65335 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65336 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_2 := arithEq_of_rows h (row := 1057) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_2
  simp only [← c65331, ← c65332, ← c65333] at e_1057_2
  have e_1057_3 := arithEq_of_rows h (row := 1057) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_3
  simp only [← c65334, ← c65335, k2, ← c65336] at e_1057_3
  have hr := e_1057_3
  simp only [e_1057_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9050)) := by
  have c65337 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65338 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65339 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65340 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65341 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65342 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_4 := arithEq_of_rows h (row := 1057) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_4
  simp only [← c65337, ← c65338, ← c65339] at e_1057_4
  have e_1057_5 := arithEq_of_rows h (row := 1057) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_5
  simp only [← c65340, ← c65341, k2, ← c65342] at e_1057_5
  have hr := e_1057_5
  simp only [e_1057_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9051)) := by
  have c65343 := (publicBatchWrapper86_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65344 := (publicBatchWrapper86_copies2042 a h).1
  have c65345 := (publicBatchWrapper86_copies2042 a h).2.1
  have c65346 := (publicBatchWrapper86_copies2042 a h).2.2.1
  have c65347 := (publicBatchWrapper86_copies2042 a h).2.2.2.1
  have c65348 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_6 := arithEq_of_rows h (row := 1057) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_6
  simp only [← c65343, ← c65344, ← c65345] at e_1057_6
  have e_1057_7 := arithEq_of_rows h (row := 1057) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_7
  simp only [← c65346, ← c65347, k2, ← c65348] at e_1057_7
  have hr := e_1057_7
  simp only [e_1057_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9052)) := by
  have c65349 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.1
  have c65350 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.1
  have c65351 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.1
  have c65352 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.1
  have c65353 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.1
  have c65354 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_8 := arithEq_of_rows h (row := 1057) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_8
  simp only [← c65349, ← c65350, ← c65351] at e_1057_8
  have e_1057_9 := arithEq_of_rows h (row := 1057) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_9
  simp only [← c65352, ← c65353, k2, ← c65354] at e_1057_9
  have hr := e_1057_9
  simp only [e_1057_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9053)) := by
  have c65355 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65356 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65357 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65358 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65359 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65360 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_10 := arithEq_of_rows h (row := 1057) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_10
  simp only [← c65355, ← c65356, ← c65357] at e_1057_10
  have e_1057_11 := arithEq_of_rows h (row := 1057) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_11
  simp only [← c65358, ← c65359, k2, ← c65360] at e_1057_11
  have hr := e_1057_11
  simp only [e_1057_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9054)) := by
  have c65361 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65362 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65363 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65364 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65365 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65366 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_12 := arithEq_of_rows h (row := 1057) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_12
  simp only [← c65361, ← c65362, ← c65363] at e_1057_12
  have e_1057_13 := arithEq_of_rows h (row := 1057) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_13
  simp only [← c65364, ← c65365, k2, ← c65366] at e_1057_13
  have hr := e_1057_13
  simp only [e_1057_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9055)) := by
  have c65367 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65368 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65369 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65370 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65371 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65372 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_14 := arithEq_of_rows h (row := 1057) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_14
  simp only [← c65367, ← c65368, ← c65369] at e_1057_14
  have e_1057_15 := arithEq_of_rows h (row := 1057) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_15
  simp only [← c65370, ← c65371, k2, ← c65372] at e_1057_15
  have hr := e_1057_15
  simp only [e_1057_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9056)) := by
  have c65373 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65374 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65375 := (publicBatchWrapper86_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65376 := (publicBatchWrapper86_copies2043 a h).1
  have c65377 := (publicBatchWrapper86_copies2043 a h).2.1
  have c65378 := (publicBatchWrapper86_copies2043 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_16 := arithEq_of_rows h (row := 1057) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_16
  simp only [← c65373, ← c65374, ← c65375] at e_1057_16
  have e_1057_17 := arithEq_of_rows h (row := 1057) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_17
  simp only [← c65376, ← c65377, k2, ← c65378] at e_1057_17
  have hr := e_1057_17
  simp only [e_1057_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1057 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9057)) := by
  have c65379 := (publicBatchWrapper86_copies2043 a h).2.2.2.1
  have c65380 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.1
  have c65381 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.1
  have c65382 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.1
  have c65383 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.1
  have c65384 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1057_18 := arithEq_of_rows h (row := 1057) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_18
  simp only [← c65379, ← c65380, ← c65381] at e_1057_18
  have e_1057_19 := arithEq_of_rows h (row := 1057) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_19
  simp only [← c65382, ← c65383, k2, ← c65384] at e_1057_19
  have hr := e_1057_19
  simp only [e_1057_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9058)) := by
  have c65385 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.1
  have c65386 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65387 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65388 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65389 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65390 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_0 := arithEq_of_rows h (row := 1058) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_0
  simp only [← c65385, ← c65386, ← c65387] at e_1058_0
  have e_1058_1 := arithEq_of_rows h (row := 1058) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_1
  simp only [← c65388, ← c65389, k2, ← c65390] at e_1058_1
  have hr := e_1058_1
  simp only [e_1058_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9059)) := by
  have c65391 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65392 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65393 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65394 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65395 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65396 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_2 := arithEq_of_rows h (row := 1058) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_2
  simp only [← c65391, ← c65392, ← c65393] at e_1058_2
  have e_1058_3 := arithEq_of_rows h (row := 1058) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_3
  simp only [← c65394, ← c65395, k2, ← c65396] at e_1058_3
  have hr := e_1058_3
  simp only [e_1058_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9060)) := by
  have c65397 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65398 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65399 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65400 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65401 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65402 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_4 := arithEq_of_rows h (row := 1058) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_4
  simp only [← c65397, ← c65398, ← c65399] at e_1058_4
  have e_1058_5 := arithEq_of_rows h (row := 1058) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_5
  simp only [← c65400, ← c65401, k2, ← c65402] at e_1058_5
  have hr := e_1058_5
  simp only [e_1058_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9061)) := by
  have c65403 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65404 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65405 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65406 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65407 := (publicBatchWrapper86_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65408 := (publicBatchWrapper86_copies2044 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_6 := arithEq_of_rows h (row := 1058) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_6
  simp only [← c65403, ← c65404, ← c65405] at e_1058_6
  have e_1058_7 := arithEq_of_rows h (row := 1058) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_7
  simp only [← c65406, ← c65407, k2, ← c65408] at e_1058_7
  have hr := e_1058_7
  simp only [e_1058_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9062)) := by
  have c65409 := (publicBatchWrapper86_copies2044 a h).2.1
  have c65410 := (publicBatchWrapper86_copies2044 a h).2.2.1
  have c65411 := (publicBatchWrapper86_copies2044 a h).2.2.2.1
  have c65412 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.1
  have c65413 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.1
  have c65414 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_8 := arithEq_of_rows h (row := 1058) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_8
  simp only [← c65409, ← c65410, ← c65411] at e_1058_8
  have e_1058_9 := arithEq_of_rows h (row := 1058) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_9
  simp only [← c65412, ← c65413, k2, ← c65414] at e_1058_9
  have hr := e_1058_9
  simp only [e_1058_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9063)) := by
  have c65415 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.1
  have c65416 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.1
  have c65417 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.1
  have c65418 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65419 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65420 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_10 := arithEq_of_rows h (row := 1058) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_10
  simp only [← c65415, ← c65416, ← c65417] at e_1058_10
  have e_1058_11 := arithEq_of_rows h (row := 1058) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_11
  simp only [← c65418, ← c65419, k2, ← c65420] at e_1058_11
  have hr := e_1058_11
  simp only [e_1058_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9064)) := by
  have c65421 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65422 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65423 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65424 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65425 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65426 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_12 := arithEq_of_rows h (row := 1058) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_12
  simp only [← c65421, ← c65422, ← c65423] at e_1058_12
  have e_1058_13 := arithEq_of_rows h (row := 1058) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_13
  simp only [← c65424, ← c65425, k2, ← c65426] at e_1058_13
  have hr := e_1058_13
  simp only [e_1058_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9065)) := by
  have c65427 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65428 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65429 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65430 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65431 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65432 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_14 := arithEq_of_rows h (row := 1058) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_14
  simp only [← c65427, ← c65428, ← c65429] at e_1058_14
  have e_1058_15 := arithEq_of_rows h (row := 1058) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_15
  simp only [← c65430, ← c65431, k2, ← c65432] at e_1058_15
  have hr := e_1058_15
  simp only [e_1058_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9066)) := by
  have c65433 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65434 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65435 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65436 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65437 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65438 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_16 := arithEq_of_rows h (row := 1058) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_16
  simp only [← c65433, ← c65434, ← c65435] at e_1058_16
  have e_1058_17 := arithEq_of_rows h (row := 1058) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_17
  simp only [← c65436, ← c65437, k2, ← c65438] at e_1058_17
  have hr := e_1058_17
  simp only [e_1058_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1058 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9067)) := by
  have c65439 := (publicBatchWrapper86_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65440 := (publicBatchWrapper86_copies2045 a h).1
  have c65441 := (publicBatchWrapper86_copies2045 a h).2.1
  have c65442 := (publicBatchWrapper86_copies2045 a h).2.2.1
  have c65443 := (publicBatchWrapper86_copies2045 a h).2.2.2.1
  have c65444 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1058_18 := arithEq_of_rows h (row := 1058) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_18
  simp only [← c65439, ← c65440, ← c65441] at e_1058_18
  have e_1058_19 := arithEq_of_rows h (row := 1058) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_19
  simp only [← c65442, ← c65443, k2, ← c65444] at e_1058_19
  have hr := e_1058_19
  simp only [e_1058_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9068)) := by
  have c65445 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.1
  have c65446 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.1
  have c65447 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.1
  have c65448 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.1
  have c65449 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.1
  have c65450 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_0 := arithEq_of_rows h (row := 1059) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_0
  simp only [← c65445, ← c65446, ← c65447] at e_1059_0
  have e_1059_1 := arithEq_of_rows h (row := 1059) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_1
  simp only [← c65448, ← c65449, k2, ← c65450] at e_1059_1
  have hr := e_1059_1
  simp only [e_1059_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9069)) := by
  have c65451 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65452 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65453 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65454 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65455 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65456 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_2 := arithEq_of_rows h (row := 1059) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_2
  simp only [← c65451, ← c65452, ← c65453] at e_1059_2
  have e_1059_3 := arithEq_of_rows h (row := 1059) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_3
  simp only [← c65454, ← c65455, k2, ← c65456] at e_1059_3
  have hr := e_1059_3
  simp only [e_1059_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9070)) := by
  have c65457 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65458 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65459 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65460 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65461 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65462 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_4 := arithEq_of_rows h (row := 1059) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_4
  simp only [← c65457, ← c65458, ← c65459] at e_1059_4
  have e_1059_5 := arithEq_of_rows h (row := 1059) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_5
  simp only [← c65460, ← c65461, k2, ← c65462] at e_1059_5
  have hr := e_1059_5
  simp only [e_1059_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9071)) := by
  have c65463 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65464 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65465 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65466 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65467 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65468 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_6 := arithEq_of_rows h (row := 1059) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_6
  simp only [← c65463, ← c65464, ← c65465] at e_1059_6
  have e_1059_7 := arithEq_of_rows h (row := 1059) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_7
  simp only [← c65466, ← c65467, k2, ← c65468] at e_1059_7
  have hr := e_1059_7
  simp only [e_1059_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9072)) := by
  have c65469 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65470 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65471 := (publicBatchWrapper86_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65472 := (publicBatchWrapper86_copies2046 a h).1
  have c65473 := (publicBatchWrapper86_copies2046 a h).2.1
  have c65474 := (publicBatchWrapper86_copies2046 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_8 := arithEq_of_rows h (row := 1059) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_8
  simp only [← c65469, ← c65470, ← c65471] at e_1059_8
  have e_1059_9 := arithEq_of_rows h (row := 1059) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_9
  simp only [← c65472, ← c65473, k2, ← c65474] at e_1059_9
  have hr := e_1059_9
  simp only [e_1059_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9073)) := by
  have c65475 := (publicBatchWrapper86_copies2046 a h).2.2.2.1
  have c65476 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.1
  have c65477 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.1
  have c65478 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.1
  have c65479 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.1
  have c65480 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_10 := arithEq_of_rows h (row := 1059) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_10
  simp only [← c65475, ← c65476, ← c65477] at e_1059_10
  have e_1059_11 := arithEq_of_rows h (row := 1059) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_11
  simp only [← c65478, ← c65479, k2, ← c65480] at e_1059_11
  have hr := e_1059_11
  simp only [e_1059_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9074)) := by
  have c65481 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.1
  have c65482 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65483 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65484 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65485 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65486 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_12 := arithEq_of_rows h (row := 1059) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_12
  simp only [← c65481, ← c65482, ← c65483] at e_1059_12
  have e_1059_13 := arithEq_of_rows h (row := 1059) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_13
  simp only [← c65484, ← c65485, k2, ← c65486] at e_1059_13
  have hr := e_1059_13
  simp only [e_1059_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9075)) := by
  have c65487 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65488 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65489 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65490 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65491 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65492 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_14 := arithEq_of_rows h (row := 1059) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_14
  simp only [← c65487, ← c65488, ← c65489] at e_1059_14
  have e_1059_15 := arithEq_of_rows h (row := 1059) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_15
  simp only [← c65490, ← c65491, k2, ← c65492] at e_1059_15
  have hr := e_1059_15
  simp only [e_1059_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9076)) := by
  have c65493 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65494 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65495 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65496 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65497 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65498 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_16 := arithEq_of_rows h (row := 1059) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_16
  simp only [← c65493, ← c65494, ← c65495] at e_1059_16
  have e_1059_17 := arithEq_of_rows h (row := 1059) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_17
  simp only [← c65496, ← c65497, k2, ← c65498] at e_1059_17
  have hr := e_1059_17
  simp only [e_1059_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1059 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9077)) := by
  have c65499 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65500 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65501 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65502 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65503 := (publicBatchWrapper86_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65504 := (publicBatchWrapper86_copies2047 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1059_18 := arithEq_of_rows h (row := 1059) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_18
  simp only [← c65499, ← c65500, ← c65501] at e_1059_18
  have e_1059_19 := arithEq_of_rows h (row := 1059) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_19
  simp only [← c65502, ← c65503, k2, ← c65504] at e_1059_19
  have hr := e_1059_19
  simp only [e_1059_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9078)) := by
  have c65505 := (publicBatchWrapper86_copies2047 a h).2.1
  have c65506 := (publicBatchWrapper86_copies2047 a h).2.2.1
  have c65507 := (publicBatchWrapper86_copies2047 a h).2.2.2.1
  have c65508 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.1
  have c65509 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.1
  have c65510 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_0 := arithEq_of_rows h (row := 1060) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_0
  simp only [← c65505, ← c65506, ← c65507] at e_1060_0
  have e_1060_1 := arithEq_of_rows h (row := 1060) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_1
  simp only [← c65508, ← c65509, k2, ← c65510] at e_1060_1
  have hr := e_1060_1
  simp only [e_1060_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9079)) := by
  have c65511 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.1
  have c65512 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.1
  have c65513 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.1
  have c65514 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65515 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65516 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_2 := arithEq_of_rows h (row := 1060) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_2
  simp only [← c65511, ← c65512, ← c65513] at e_1060_2
  have e_1060_3 := arithEq_of_rows h (row := 1060) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_3
  simp only [← c65514, ← c65515, k2, ← c65516] at e_1060_3
  have hr := e_1060_3
  simp only [e_1060_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9080)) := by
  have c65517 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65518 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65519 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65520 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65521 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65522 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_4 := arithEq_of_rows h (row := 1060) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_4
  simp only [← c65517, ← c65518, ← c65519] at e_1060_4
  have e_1060_5 := arithEq_of_rows h (row := 1060) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_5
  simp only [← c65520, ← c65521, k2, ← c65522] at e_1060_5
  have hr := e_1060_5
  simp only [e_1060_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9081)) := by
  have c65523 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65524 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65525 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65526 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65527 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65528 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_6 := arithEq_of_rows h (row := 1060) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_6
  simp only [← c65523, ← c65524, ← c65525] at e_1060_6
  have e_1060_7 := arithEq_of_rows h (row := 1060) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_7
  simp only [← c65526, ← c65527, k2, ← c65528] at e_1060_7
  have hr := e_1060_7
  simp only [e_1060_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9082)) := by
  have c65529 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65530 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65531 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65532 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65533 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65534 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_8 := arithEq_of_rows h (row := 1060) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_8
  simp only [← c65529, ← c65530, ← c65531] at e_1060_8
  have e_1060_9 := arithEq_of_rows h (row := 1060) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_9
  simp only [← c65532, ← c65533, k2, ← c65534] at e_1060_9
  have hr := e_1060_9
  simp only [e_1060_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9083)) := by
  have c65535 := (publicBatchWrapper86_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65536 := (publicBatchWrapper86_copies2048 a h).1
  have c65537 := (publicBatchWrapper86_copies2048 a h).2.1
  have c65538 := (publicBatchWrapper86_copies2048 a h).2.2.1
  have c65539 := (publicBatchWrapper86_copies2048 a h).2.2.2.1
  have c65540 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_10 := arithEq_of_rows h (row := 1060) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_10
  simp only [← c65535, ← c65536, ← c65537] at e_1060_10
  have e_1060_11 := arithEq_of_rows h (row := 1060) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_11
  simp only [← c65538, ← c65539, k2, ← c65540] at e_1060_11
  have hr := e_1060_11
  simp only [e_1060_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9084)) := by
  have c65541 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.1
  have c65542 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.1
  have c65543 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.1
  have c65544 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.1
  have c65545 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.1
  have c65546 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_12 := arithEq_of_rows h (row := 1060) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_12
  simp only [← c65541, ← c65542, ← c65543] at e_1060_12
  have e_1060_13 := arithEq_of_rows h (row := 1060) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_13
  simp only [← c65544, ← c65545, k2, ← c65546] at e_1060_13
  have hr := e_1060_13
  simp only [e_1060_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9085)) := by
  have c65547 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65548 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65549 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65550 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65551 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65552 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_14 := arithEq_of_rows h (row := 1060) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_14
  simp only [← c65547, ← c65548, ← c65549] at e_1060_14
  have e_1060_15 := arithEq_of_rows h (row := 1060) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_15
  simp only [← c65550, ← c65551, k2, ← c65552] at e_1060_15
  have hr := e_1060_15
  simp only [e_1060_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9086)) := by
  have c65553 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65554 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65555 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65556 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65557 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65558 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_16 := arithEq_of_rows h (row := 1060) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_16
  simp only [← c65553, ← c65554, ← c65555] at e_1060_16
  have e_1060_17 := arithEq_of_rows h (row := 1060) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_17
  simp only [← c65556, ← c65557, k2, ← c65558] at e_1060_17
  have hr := e_1060_17
  simp only [e_1060_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1060 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9087)) := by
  have c65559 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65560 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65561 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65562 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65563 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65564 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1060_18 := arithEq_of_rows h (row := 1060) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_18
  simp only [← c65559, ← c65560, ← c65561] at e_1060_18
  have e_1060_19 := arithEq_of_rows h (row := 1060) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_19
  simp only [← c65562, ← c65563, k2, ← c65564] at e_1060_19
  have hr := e_1060_19
  simp only [e_1060_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9088)) := by
  have c65565 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65566 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65567 := (publicBatchWrapper86_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65568 := (publicBatchWrapper86_copies2049 a h).1
  have c65569 := (publicBatchWrapper86_copies2049 a h).2.1
  have c65570 := (publicBatchWrapper86_copies2049 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_0 := arithEq_of_rows h (row := 1061) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_0
  simp only [← c65565, ← c65566, ← c65567] at e_1061_0
  have e_1061_1 := arithEq_of_rows h (row := 1061) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_1
  simp only [← c65568, ← c65569, k2, ← c65570] at e_1061_1
  have hr := e_1061_1
  simp only [e_1061_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9089)) := by
  have c65571 := (publicBatchWrapper86_copies2049 a h).2.2.2.1
  have c65572 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.1
  have c65573 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.1
  have c65574 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.1
  have c65575 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.1
  have c65576 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_2 := arithEq_of_rows h (row := 1061) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_2
  simp only [← c65571, ← c65572, ← c65573] at e_1061_2
  have e_1061_3 := arithEq_of_rows h (row := 1061) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_3
  simp only [← c65574, ← c65575, k2, ← c65576] at e_1061_3
  have hr := e_1061_3
  simp only [e_1061_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9090)) := by
  have c65577 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.1
  have c65578 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65579 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65580 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65581 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65582 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_4 := arithEq_of_rows h (row := 1061) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_4
  simp only [← c65577, ← c65578, ← c65579] at e_1061_4
  have e_1061_5 := arithEq_of_rows h (row := 1061) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_5
  simp only [← c65580, ← c65581, k2, ← c65582] at e_1061_5
  have hr := e_1061_5
  simp only [e_1061_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9091)) := by
  have c65583 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65584 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65585 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65586 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65587 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65588 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_6 := arithEq_of_rows h (row := 1061) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_6
  simp only [← c65583, ← c65584, ← c65585] at e_1061_6
  have e_1061_7 := arithEq_of_rows h (row := 1061) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_7
  simp only [← c65586, ← c65587, k2, ← c65588] at e_1061_7
  have hr := e_1061_7
  simp only [e_1061_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9092)) := by
  have c65589 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65590 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65591 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65592 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65593 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65594 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_8 := arithEq_of_rows h (row := 1061) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_8
  simp only [← c65589, ← c65590, ← c65591] at e_1061_8
  have e_1061_9 := arithEq_of_rows h (row := 1061) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_9
  simp only [← c65592, ← c65593, k2, ← c65594] at e_1061_9
  have hr := e_1061_9
  simp only [e_1061_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9093)) := by
  have c65595 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65596 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65597 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65598 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65599 := (publicBatchWrapper86_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65600 := (publicBatchWrapper86_copies2050 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_10 := arithEq_of_rows h (row := 1061) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_10
  simp only [← c65595, ← c65596, ← c65597] at e_1061_10
  have e_1061_11 := arithEq_of_rows h (row := 1061) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_11
  simp only [← c65598, ← c65599, k2, ← c65600] at e_1061_11
  have hr := e_1061_11
  simp only [e_1061_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9094)) := by
  have c65601 := (publicBatchWrapper86_copies2050 a h).2.1
  have c65602 := (publicBatchWrapper86_copies2050 a h).2.2.1
  have c65603 := (publicBatchWrapper86_copies2050 a h).2.2.2.1
  have c65604 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.1
  have c65605 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.1
  have c65606 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_12 := arithEq_of_rows h (row := 1061) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_12
  simp only [← c65601, ← c65602, ← c65603] at e_1061_12
  have e_1061_13 := arithEq_of_rows h (row := 1061) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_13
  simp only [← c65604, ← c65605, k2, ← c65606] at e_1061_13
  have hr := e_1061_13
  simp only [e_1061_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9095)) := by
  have c65607 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.1
  have c65608 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.1
  have c65609 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.1
  have c65610 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65611 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65612 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_14 := arithEq_of_rows h (row := 1061) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_14
  simp only [← c65607, ← c65608, ← c65609] at e_1061_14
  have e_1061_15 := arithEq_of_rows h (row := 1061) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_15
  simp only [← c65610, ← c65611, k2, ← c65612] at e_1061_15
  have hr := e_1061_15
  simp only [e_1061_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9096)) := by
  have c65613 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65614 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65615 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65616 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65617 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65618 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_16 := arithEq_of_rows h (row := 1061) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_16
  simp only [← c65613, ← c65614, ← c65615] at e_1061_16
  have e_1061_17 := arithEq_of_rows h (row := 1061) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_17
  simp only [← c65616, ← c65617, k2, ← c65618] at e_1061_17
  have hr := e_1061_17
  simp only [e_1061_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1061 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9097)) := by
  have c65619 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65620 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65621 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65622 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65623 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65624 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1061_18 := arithEq_of_rows h (row := 1061) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_18
  simp only [← c65619, ← c65620, ← c65621] at e_1061_18
  have e_1061_19 := arithEq_of_rows h (row := 1061) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_19
  simp only [← c65622, ← c65623, k2, ← c65624] at e_1061_19
  have hr := e_1061_19
  simp only [e_1061_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9098)) := by
  have c65625 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65626 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65627 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65628 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65629 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65630 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_0 := arithEq_of_rows h (row := 1062) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_0
  simp only [← c65625, ← c65626, ← c65627] at e_1062_0
  have e_1062_1 := arithEq_of_rows h (row := 1062) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_1
  simp only [← c65628, ← c65629, k2, ← c65630] at e_1062_1
  have hr := e_1062_1
  simp only [e_1062_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9099)) := by
  have c65631 := (publicBatchWrapper86_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65632 := (publicBatchWrapper86_copies2051 a h).1
  have c65633 := (publicBatchWrapper86_copies2051 a h).2.1
  have c65634 := (publicBatchWrapper86_copies2051 a h).2.2.1
  have c65635 := (publicBatchWrapper86_copies2051 a h).2.2.2.1
  have c65636 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_2 := arithEq_of_rows h (row := 1062) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_2
  simp only [← c65631, ← c65632, ← c65633] at e_1062_2
  have e_1062_3 := arithEq_of_rows h (row := 1062) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_3
  simp only [← c65634, ← c65635, k2, ← c65636] at e_1062_3
  have hr := e_1062_3
  simp only [e_1062_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9100)) := by
  have c65637 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.1
  have c65638 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.1
  have c65639 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.1
  have c65640 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.1
  have c65641 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.1
  have c65642 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_4 := arithEq_of_rows h (row := 1062) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_4
  simp only [← c65637, ← c65638, ← c65639] at e_1062_4
  have e_1062_5 := arithEq_of_rows h (row := 1062) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_5
  simp only [← c65640, ← c65641, k2, ← c65642] at e_1062_5
  have hr := e_1062_5
  simp only [e_1062_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9101)) := by
  have c65643 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65644 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65645 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65646 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65647 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65648 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_6 := arithEq_of_rows h (row := 1062) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_6
  simp only [← c65643, ← c65644, ← c65645] at e_1062_6
  have e_1062_7 := arithEq_of_rows h (row := 1062) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_7
  simp only [← c65646, ← c65647, k2, ← c65648] at e_1062_7
  have hr := e_1062_7
  simp only [e_1062_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9102)) := by
  have c65649 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65650 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65651 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65652 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65653 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65654 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_8 := arithEq_of_rows h (row := 1062) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_8
  simp only [← c65649, ← c65650, ← c65651] at e_1062_8
  have e_1062_9 := arithEq_of_rows h (row := 1062) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_9
  simp only [← c65652, ← c65653, k2, ← c65654] at e_1062_9
  have hr := e_1062_9
  simp only [e_1062_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9103)) := by
  have c65655 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65656 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65657 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65658 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65659 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65660 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_10 := arithEq_of_rows h (row := 1062) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_10
  simp only [← c65655, ← c65656, ← c65657] at e_1062_10
  have e_1062_11 := arithEq_of_rows h (row := 1062) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_11
  simp only [← c65658, ← c65659, k2, ← c65660] at e_1062_11
  have hr := e_1062_11
  simp only [e_1062_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9104)) := by
  have c65661 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65662 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65663 := (publicBatchWrapper86_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65664 := (publicBatchWrapper86_copies2052 a h).1
  have c65665 := (publicBatchWrapper86_copies2052 a h).2.1
  have c65666 := (publicBatchWrapper86_copies2052 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_12 := arithEq_of_rows h (row := 1062) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_12
  simp only [← c65661, ← c65662, ← c65663] at e_1062_12
  have e_1062_13 := arithEq_of_rows h (row := 1062) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_13
  simp only [← c65664, ← c65665, k2, ← c65666] at e_1062_13
  have hr := e_1062_13
  simp only [e_1062_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9105)) := by
  have c65667 := (publicBatchWrapper86_copies2052 a h).2.2.2.1
  have c65668 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.1
  have c65669 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.1
  have c65670 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.1
  have c65671 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.1
  have c65672 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_14 := arithEq_of_rows h (row := 1062) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_14
  simp only [← c65667, ← c65668, ← c65669] at e_1062_14
  have e_1062_15 := arithEq_of_rows h (row := 1062) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_15
  simp only [← c65670, ← c65671, k2, ← c65672] at e_1062_15
  have hr := e_1062_15
  simp only [e_1062_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9106)) := by
  have c65673 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.1
  have c65674 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65675 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65676 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65677 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65678 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_16 := arithEq_of_rows h (row := 1062) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_16
  simp only [← c65673, ← c65674, ← c65675] at e_1062_16
  have e_1062_17 := arithEq_of_rows h (row := 1062) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_17
  simp only [← c65676, ← c65677, k2, ← c65678] at e_1062_17
  have hr := e_1062_17
  simp only [e_1062_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1062 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9107)) := by
  have c65679 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65680 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65681 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65682 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65683 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65684 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1062_18 := arithEq_of_rows h (row := 1062) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_18
  simp only [← c65679, ← c65680, ← c65681] at e_1062_18
  have e_1062_19 := arithEq_of_rows h (row := 1062) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_19
  simp only [← c65682, ← c65683, k2, ← c65684] at e_1062_19
  have hr := e_1062_19
  simp only [e_1062_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9108)) := by
  have c65685 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65686 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65687 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65688 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65689 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65690 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_0 := arithEq_of_rows h (row := 1063) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_0
  simp only [← c65685, ← c65686, ← c65687] at e_1063_0
  have e_1063_1 := arithEq_of_rows h (row := 1063) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_1
  simp only [← c65688, ← c65689, k2, ← c65690] at e_1063_1
  have hr := e_1063_1
  simp only [e_1063_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9109)) := by
  have c65691 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65692 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65693 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65694 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65695 := (publicBatchWrapper86_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65696 := (publicBatchWrapper86_copies2053 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_2 := arithEq_of_rows h (row := 1063) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_2
  simp only [← c65691, ← c65692, ← c65693] at e_1063_2
  have e_1063_3 := arithEq_of_rows h (row := 1063) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_3
  simp only [← c65694, ← c65695, k2, ← c65696] at e_1063_3
  have hr := e_1063_3
  simp only [e_1063_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9110)) := by
  have c65697 := (publicBatchWrapper86_copies2053 a h).2.1
  have c65698 := (publicBatchWrapper86_copies2053 a h).2.2.1
  have c65699 := (publicBatchWrapper86_copies2053 a h).2.2.2.1
  have c65700 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.1
  have c65701 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.1
  have c65702 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_4 := arithEq_of_rows h (row := 1063) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_4
  simp only [← c65697, ← c65698, ← c65699] at e_1063_4
  have e_1063_5 := arithEq_of_rows h (row := 1063) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_5
  simp only [← c65700, ← c65701, k2, ← c65702] at e_1063_5
  have hr := e_1063_5
  simp only [e_1063_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9111)) := by
  have c65703 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.1
  have c65704 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.1
  have c65705 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.1
  have c65706 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65707 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65708 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_6 := arithEq_of_rows h (row := 1063) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_6
  simp only [← c65703, ← c65704, ← c65705] at e_1063_6
  have e_1063_7 := arithEq_of_rows h (row := 1063) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_7
  simp only [← c65706, ← c65707, k2, ← c65708] at e_1063_7
  have hr := e_1063_7
  simp only [e_1063_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9112)) := by
  have c65709 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65710 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65711 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65712 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65713 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65714 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_8 := arithEq_of_rows h (row := 1063) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_8
  simp only [← c65709, ← c65710, ← c65711] at e_1063_8
  have e_1063_9 := arithEq_of_rows h (row := 1063) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_9
  simp only [← c65712, ← c65713, k2, ← c65714] at e_1063_9
  have hr := e_1063_9
  simp only [e_1063_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9113)) := by
  have c65715 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65716 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65717 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65718 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65719 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65720 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_10 := arithEq_of_rows h (row := 1063) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_10
  simp only [← c65715, ← c65716, ← c65717] at e_1063_10
  have e_1063_11 := arithEq_of_rows h (row := 1063) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_11
  simp only [← c65718, ← c65719, k2, ← c65720] at e_1063_11
  have hr := e_1063_11
  simp only [e_1063_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9114)) := by
  have c65721 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65722 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65723 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65724 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65725 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65726 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_12 := arithEq_of_rows h (row := 1063) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_12
  simp only [← c65721, ← c65722, ← c65723] at e_1063_12
  have e_1063_13 := arithEq_of_rows h (row := 1063) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_13
  simp only [← c65724, ← c65725, k2, ← c65726] at e_1063_13
  have hr := e_1063_13
  simp only [e_1063_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9115)) := by
  have c65727 := (publicBatchWrapper86_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65728 := (publicBatchWrapper86_copies2054 a h).1
  have c65729 := (publicBatchWrapper86_copies2054 a h).2.1
  have c65730 := (publicBatchWrapper86_copies2054 a h).2.2.1
  have c65731 := (publicBatchWrapper86_copies2054 a h).2.2.2.1
  have c65732 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_14 := arithEq_of_rows h (row := 1063) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_14
  simp only [← c65727, ← c65728, ← c65729] at e_1063_14
  have e_1063_15 := arithEq_of_rows h (row := 1063) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_15
  simp only [← c65730, ← c65731, k2, ← c65732] at e_1063_15
  have hr := e_1063_15
  simp only [e_1063_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9116)) := by
  have c65733 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.1
  have c65734 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.1
  have c65735 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.1
  have c65736 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.1
  have c65737 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.1
  have c65738 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_16 := arithEq_of_rows h (row := 1063) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_16
  simp only [← c65733, ← c65734, ← c65735] at e_1063_16
  have e_1063_17 := arithEq_of_rows h (row := 1063) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_17
  simp only [← c65736, ← c65737, k2, ← c65738] at e_1063_17
  have hr := e_1063_17
  simp only [e_1063_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1063 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9117)) := by
  have c65739 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65740 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65741 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65742 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65743 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65744 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1063_18 := arithEq_of_rows h (row := 1063) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_18
  simp only [← c65739, ← c65740, ← c65741] at e_1063_18
  have e_1063_19 := arithEq_of_rows h (row := 1063) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_19
  simp only [← c65742, ← c65743, k2, ← c65744] at e_1063_19
  have hr := e_1063_19
  simp only [e_1063_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9118)) := by
  have c65745 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65746 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65747 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65748 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65749 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65750 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_0 := arithEq_of_rows h (row := 1064) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_0
  simp only [← c65745, ← c65746, ← c65747] at e_1064_0
  have e_1064_1 := arithEq_of_rows h (row := 1064) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_1
  simp only [← c65748, ← c65749, k2, ← c65750] at e_1064_1
  have hr := e_1064_1
  simp only [e_1064_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9119)) := by
  have c65751 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65752 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65753 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65754 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65755 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65756 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_2 := arithEq_of_rows h (row := 1064) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_2
  simp only [← c65751, ← c65752, ← c65753] at e_1064_2
  have e_1064_3 := arithEq_of_rows h (row := 1064) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_3
  simp only [← c65754, ← c65755, k2, ← c65756] at e_1064_3
  have hr := e_1064_3
  simp only [e_1064_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9120)) := by
  have c65757 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65758 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65759 := (publicBatchWrapper86_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65760 := (publicBatchWrapper86_copies2055 a h).1
  have c65761 := (publicBatchWrapper86_copies2055 a h).2.1
  have c65762 := (publicBatchWrapper86_copies2055 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_4 := arithEq_of_rows h (row := 1064) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_4
  simp only [← c65757, ← c65758, ← c65759] at e_1064_4
  have e_1064_5 := arithEq_of_rows h (row := 1064) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_5
  simp only [← c65760, ← c65761, k2, ← c65762] at e_1064_5
  have hr := e_1064_5
  simp only [e_1064_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9121)) := by
  have c65763 := (publicBatchWrapper86_copies2055 a h).2.2.2.1
  have c65764 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.1
  have c65765 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.1
  have c65766 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.1
  have c65767 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.1
  have c65768 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_6 := arithEq_of_rows h (row := 1064) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_6
  simp only [← c65763, ← c65764, ← c65765] at e_1064_6
  have e_1064_7 := arithEq_of_rows h (row := 1064) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_7
  simp only [← c65766, ← c65767, k2, ← c65768] at e_1064_7
  have hr := e_1064_7
  simp only [e_1064_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9122)) := by
  have c65769 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.1
  have c65770 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65771 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65772 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65773 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65774 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_8 := arithEq_of_rows h (row := 1064) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_8
  simp only [← c65769, ← c65770, ← c65771] at e_1064_8
  have e_1064_9 := arithEq_of_rows h (row := 1064) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_9
  simp only [← c65772, ← c65773, k2, ← c65774] at e_1064_9
  have hr := e_1064_9
  simp only [e_1064_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9123)) := by
  have c65775 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65776 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65777 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65778 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65779 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65780 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_10 := arithEq_of_rows h (row := 1064) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_10
  simp only [← c65775, ← c65776, ← c65777] at e_1064_10
  have e_1064_11 := arithEq_of_rows h (row := 1064) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_11
  simp only [← c65778, ← c65779, k2, ← c65780] at e_1064_11
  have hr := e_1064_11
  simp only [e_1064_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9124)) := by
  have c65781 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65782 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65783 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65784 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65785 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65786 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_12 := arithEq_of_rows h (row := 1064) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_12
  simp only [← c65781, ← c65782, ← c65783] at e_1064_12
  have e_1064_13 := arithEq_of_rows h (row := 1064) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_13
  simp only [← c65784, ← c65785, k2, ← c65786] at e_1064_13
  have hr := e_1064_13
  simp only [e_1064_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9125)) := by
  have c65787 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65788 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65789 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65790 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65791 := (publicBatchWrapper86_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65792 := (publicBatchWrapper86_copies2056 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_14 := arithEq_of_rows h (row := 1064) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_14
  simp only [← c65787, ← c65788, ← c65789] at e_1064_14
  have e_1064_15 := arithEq_of_rows h (row := 1064) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_15
  simp only [← c65790, ← c65791, k2, ← c65792] at e_1064_15
  have hr := e_1064_15
  simp only [e_1064_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9126)) := by
  have c65793 := (publicBatchWrapper86_copies2056 a h).2.1
  have c65794 := (publicBatchWrapper86_copies2056 a h).2.2.1
  have c65795 := (publicBatchWrapper86_copies2056 a h).2.2.2.1
  have c65796 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.1
  have c65797 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.1
  have c65798 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_16 := arithEq_of_rows h (row := 1064) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_16
  simp only [← c65793, ← c65794, ← c65795] at e_1064_16
  have e_1064_17 := arithEq_of_rows h (row := 1064) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_17
  simp only [← c65796, ← c65797, k2, ← c65798] at e_1064_17
  have hr := e_1064_17
  simp only [e_1064_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1064 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9127)) := by
  have c65799 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.1
  have c65800 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.1
  have c65801 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.1
  have c65802 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65803 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65804 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1064_18 := arithEq_of_rows h (row := 1064) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_18
  simp only [← c65799, ← c65800, ← c65801] at e_1064_18
  have e_1064_19 := arithEq_of_rows h (row := 1064) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_19
  simp only [← c65802, ← c65803, k2, ← c65804] at e_1064_19
  have hr := e_1064_19
  simp only [e_1064_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9128)) := by
  have c65805 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65806 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65807 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65808 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65809 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65810 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_0 := arithEq_of_rows h (row := 1065) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_0
  simp only [← c65805, ← c65806, ← c65807] at e_1065_0
  have e_1065_1 := arithEq_of_rows h (row := 1065) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_1
  simp only [← c65808, ← c65809, k2, ← c65810] at e_1065_1
  have hr := e_1065_1
  simp only [e_1065_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9129)) := by
  have c65811 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65812 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65813 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65814 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65815 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65816 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_2 := arithEq_of_rows h (row := 1065) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_2
  simp only [← c65811, ← c65812, ← c65813] at e_1065_2
  have e_1065_3 := arithEq_of_rows h (row := 1065) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_3
  simp only [← c65814, ← c65815, k2, ← c65816] at e_1065_3
  have hr := e_1065_3
  simp only [e_1065_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9130)) := by
  have c65817 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65818 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65819 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65820 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65821 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65822 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_4 := arithEq_of_rows h (row := 1065) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_4
  simp only [← c65817, ← c65818, ← c65819] at e_1065_4
  have e_1065_5 := arithEq_of_rows h (row := 1065) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_5
  simp only [← c65820, ← c65821, k2, ← c65822] at e_1065_5
  have hr := e_1065_5
  simp only [e_1065_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9131)) := by
  have c65823 := (publicBatchWrapper86_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65824 := (publicBatchWrapper86_copies2057 a h).1
  have c65825 := (publicBatchWrapper86_copies2057 a h).2.1
  have c65826 := (publicBatchWrapper86_copies2057 a h).2.2.1
  have c65827 := (publicBatchWrapper86_copies2057 a h).2.2.2.1
  have c65828 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_6 := arithEq_of_rows h (row := 1065) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_6
  simp only [← c65823, ← c65824, ← c65825] at e_1065_6
  have e_1065_7 := arithEq_of_rows h (row := 1065) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_7
  simp only [← c65826, ← c65827, k2, ← c65828] at e_1065_7
  have hr := e_1065_7
  simp only [e_1065_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9132)) := by
  have c65829 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.1
  have c65830 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.1
  have c65831 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.1
  have c65832 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.1
  have c65833 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.1
  have c65834 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_8 := arithEq_of_rows h (row := 1065) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_8
  simp only [← c65829, ← c65830, ← c65831] at e_1065_8
  have e_1065_9 := arithEq_of_rows h (row := 1065) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_9
  simp only [← c65832, ← c65833, k2, ← c65834] at e_1065_9
  have hr := e_1065_9
  simp only [e_1065_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9133)) := by
  have c65835 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65836 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65837 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65838 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65839 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65840 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_10 := arithEq_of_rows h (row := 1065) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_10
  simp only [← c65835, ← c65836, ← c65837] at e_1065_10
  have e_1065_11 := arithEq_of_rows h (row := 1065) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_11
  simp only [← c65838, ← c65839, k2, ← c65840] at e_1065_11
  have hr := e_1065_11
  simp only [e_1065_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9134)) := by
  have c65841 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65842 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65843 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65844 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65845 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65846 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_12 := arithEq_of_rows h (row := 1065) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_12
  simp only [← c65841, ← c65842, ← c65843] at e_1065_12
  have e_1065_13 := arithEq_of_rows h (row := 1065) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_13
  simp only [← c65844, ← c65845, k2, ← c65846] at e_1065_13
  have hr := e_1065_13
  simp only [e_1065_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9135)) := by
  have c65847 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65848 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65849 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65850 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65851 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65852 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_14 := arithEq_of_rows h (row := 1065) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_14
  simp only [← c65847, ← c65848, ← c65849] at e_1065_14
  have e_1065_15 := arithEq_of_rows h (row := 1065) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_15
  simp only [← c65850, ← c65851, k2, ← c65852] at e_1065_15
  have hr := e_1065_15
  simp only [e_1065_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9136)) := by
  have c65853 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65854 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65855 := (publicBatchWrapper86_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65856 := (publicBatchWrapper86_copies2058 a h).1
  have c65857 := (publicBatchWrapper86_copies2058 a h).2.1
  have c65858 := (publicBatchWrapper86_copies2058 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_16 := arithEq_of_rows h (row := 1065) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_16
  simp only [← c65853, ← c65854, ← c65855] at e_1065_16
  have e_1065_17 := arithEq_of_rows h (row := 1065) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_17
  simp only [← c65856, ← c65857, k2, ← c65858] at e_1065_17
  have hr := e_1065_17
  simp only [e_1065_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1065 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9137)) := by
  have c65859 := (publicBatchWrapper86_copies2058 a h).2.2.2.1
  have c65860 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.1
  have c65861 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.1
  have c65862 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.1
  have c65863 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.1
  have c65864 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1065_18 := arithEq_of_rows h (row := 1065) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_18
  simp only [← c65859, ← c65860, ← c65861] at e_1065_18
  have e_1065_19 := arithEq_of_rows h (row := 1065) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_19
  simp only [← c65862, ← c65863, k2, ← c65864] at e_1065_19
  have hr := e_1065_19
  simp only [e_1065_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9138)) := by
  have c65865 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.1
  have c65866 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65867 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65868 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65869 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65870 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_0 := arithEq_of_rows h (row := 1066) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_0
  simp only [← c65865, ← c65866, ← c65867] at e_1066_0
  have e_1066_1 := arithEq_of_rows h (row := 1066) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_1
  simp only [← c65868, ← c65869, k2, ← c65870] at e_1066_1
  have hr := e_1066_1
  simp only [e_1066_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9139)) := by
  have c65871 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65872 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65873 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65874 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65875 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65876 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_2 := arithEq_of_rows h (row := 1066) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_2
  simp only [← c65871, ← c65872, ← c65873] at e_1066_2
  have e_1066_3 := arithEq_of_rows h (row := 1066) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_3
  simp only [← c65874, ← c65875, k2, ← c65876] at e_1066_3
  have hr := e_1066_3
  simp only [e_1066_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9140)) := by
  have c65877 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65878 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65879 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65880 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65881 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65882 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_4 := arithEq_of_rows h (row := 1066) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_4
  simp only [← c65877, ← c65878, ← c65879] at e_1066_4
  have e_1066_5 := arithEq_of_rows h (row := 1066) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_5
  simp only [← c65880, ← c65881, k2, ← c65882] at e_1066_5
  have hr := e_1066_5
  simp only [e_1066_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9141)) := by
  have c65883 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65884 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65885 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65886 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65887 := (publicBatchWrapper86_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65888 := (publicBatchWrapper86_copies2059 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_6 := arithEq_of_rows h (row := 1066) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_6
  simp only [← c65883, ← c65884, ← c65885] at e_1066_6
  have e_1066_7 := arithEq_of_rows h (row := 1066) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_7
  simp only [← c65886, ← c65887, k2, ← c65888] at e_1066_7
  have hr := e_1066_7
  simp only [e_1066_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9142)) := by
  have c65889 := (publicBatchWrapper86_copies2059 a h).2.1
  have c65890 := (publicBatchWrapper86_copies2059 a h).2.2.1
  have c65891 := (publicBatchWrapper86_copies2059 a h).2.2.2.1
  have c65892 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.1
  have c65893 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.1
  have c65894 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_8 := arithEq_of_rows h (row := 1066) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_8
  simp only [← c65889, ← c65890, ← c65891] at e_1066_8
  have e_1066_9 := arithEq_of_rows h (row := 1066) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_9
  simp only [← c65892, ← c65893, k2, ← c65894] at e_1066_9
  have hr := e_1066_9
  simp only [e_1066_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9143)) := by
  have c65895 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.1
  have c65896 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.1
  have c65897 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.1
  have c65898 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65899 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65900 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_10 := arithEq_of_rows h (row := 1066) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_10
  simp only [← c65895, ← c65896, ← c65897] at e_1066_10
  have e_1066_11 := arithEq_of_rows h (row := 1066) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_11
  simp only [← c65898, ← c65899, k2, ← c65900] at e_1066_11
  have hr := e_1066_11
  simp only [e_1066_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9144)) := by
  have c65901 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65902 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65903 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65904 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65905 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65906 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_12 := arithEq_of_rows h (row := 1066) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_12
  simp only [← c65901, ← c65902, ← c65903] at e_1066_12
  have e_1066_13 := arithEq_of_rows h (row := 1066) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_13
  simp only [← c65904, ← c65905, k2, ← c65906] at e_1066_13
  have hr := e_1066_13
  simp only [e_1066_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9145)) := by
  have c65907 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65908 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65909 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65910 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65911 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65912 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_14 := arithEq_of_rows h (row := 1066) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_14
  simp only [← c65907, ← c65908, ← c65909] at e_1066_14
  have e_1066_15 := arithEq_of_rows h (row := 1066) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_15
  simp only [← c65910, ← c65911, k2, ← c65912] at e_1066_15
  have hr := e_1066_15
  simp only [e_1066_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9146)) := by
  have c65913 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65914 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65915 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65916 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65917 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65918 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_16 := arithEq_of_rows h (row := 1066) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_16
  simp only [← c65913, ← c65914, ← c65915] at e_1066_16
  have e_1066_17 := arithEq_of_rows h (row := 1066) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_17
  simp only [← c65916, ← c65917, k2, ← c65918] at e_1066_17
  have hr := e_1066_17
  simp only [e_1066_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1066 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9147)) := by
  have c65919 := (publicBatchWrapper86_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65920 := (publicBatchWrapper86_copies2060 a h).1
  have c65921 := (publicBatchWrapper86_copies2060 a h).2.1
  have c65922 := (publicBatchWrapper86_copies2060 a h).2.2.1
  have c65923 := (publicBatchWrapper86_copies2060 a h).2.2.2.1
  have c65924 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1066_18 := arithEq_of_rows h (row := 1066) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_18
  simp only [← c65919, ← c65920, ← c65921] at e_1066_18
  have e_1066_19 := arithEq_of_rows h (row := 1066) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_19
  simp only [← c65922, ← c65923, k2, ← c65924] at e_1066_19
  have hr := e_1066_19
  simp only [e_1066_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9148)) := by
  have c65925 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.1
  have c65926 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.1
  have c65927 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.1
  have c65928 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.1
  have c65929 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.1
  have c65930 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_0 := arithEq_of_rows h (row := 1067) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_0
  simp only [← c65925, ← c65926, ← c65927] at e_1067_0
  have e_1067_1 := arithEq_of_rows h (row := 1067) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_1
  simp only [← c65928, ← c65929, k2, ← c65930] at e_1067_1
  have hr := e_1067_1
  simp only [e_1067_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9149)) := by
  have c65931 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65932 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65933 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65934 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65935 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65936 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_2 := arithEq_of_rows h (row := 1067) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_2
  simp only [← c65931, ← c65932, ← c65933] at e_1067_2
  have e_1067_3 := arithEq_of_rows h (row := 1067) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_3
  simp only [← c65934, ← c65935, k2, ← c65936] at e_1067_3
  have hr := e_1067_3
  simp only [e_1067_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9150)) := by
  have c65937 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65938 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65939 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65940 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65941 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65942 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_4 := arithEq_of_rows h (row := 1067) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_4
  simp only [← c65937, ← c65938, ← c65939] at e_1067_4
  have e_1067_5 := arithEq_of_rows h (row := 1067) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_5
  simp only [← c65940, ← c65941, k2, ← c65942] at e_1067_5
  have hr := e_1067_5
  simp only [e_1067_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9151)) := by
  have c65943 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65944 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65945 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65946 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65947 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65948 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_6 := arithEq_of_rows h (row := 1067) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_6
  simp only [← c65943, ← c65944, ← c65945] at e_1067_6
  have e_1067_7 := arithEq_of_rows h (row := 1067) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_7
  simp only [← c65946, ← c65947, k2, ← c65948] at e_1067_7
  have hr := e_1067_7
  simp only [e_1067_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9152)) := by
  have c65949 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65950 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65951 := (publicBatchWrapper86_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65952 := (publicBatchWrapper86_copies2061 a h).1
  have c65953 := (publicBatchWrapper86_copies2061 a h).2.1
  have c65954 := (publicBatchWrapper86_copies2061 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_8 := arithEq_of_rows h (row := 1067) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_8
  simp only [← c65949, ← c65950, ← c65951] at e_1067_8
  have e_1067_9 := arithEq_of_rows h (row := 1067) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_9
  simp only [← c65952, ← c65953, k2, ← c65954] at e_1067_9
  have hr := e_1067_9
  simp only [e_1067_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9153)) := by
  have c65955 := (publicBatchWrapper86_copies2061 a h).2.2.2.1
  have c65956 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.1
  have c65957 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.1
  have c65958 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.1
  have c65959 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.1
  have c65960 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_10 := arithEq_of_rows h (row := 1067) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_10
  simp only [← c65955, ← c65956, ← c65957] at e_1067_10
  have e_1067_11 := arithEq_of_rows h (row := 1067) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_11
  simp only [← c65958, ← c65959, k2, ← c65960] at e_1067_11
  have hr := e_1067_11
  simp only [e_1067_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9154)) := by
  have c65961 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.1
  have c65962 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65963 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65964 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65965 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65966 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_12 := arithEq_of_rows h (row := 1067) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_12
  simp only [← c65961, ← c65962, ← c65963] at e_1067_12
  have e_1067_13 := arithEq_of_rows h (row := 1067) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_13
  simp only [← c65964, ← c65965, k2, ← c65966] at e_1067_13
  have hr := e_1067_13
  simp only [e_1067_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9155)) := by
  have c65967 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65968 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65969 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65970 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65971 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65972 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_14 := arithEq_of_rows h (row := 1067) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_14
  simp only [← c65967, ← c65968, ← c65969] at e_1067_14
  have e_1067_15 := arithEq_of_rows h (row := 1067) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_15
  simp only [← c65970, ← c65971, k2, ← c65972] at e_1067_15
  have hr := e_1067_15
  simp only [e_1067_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9156)) := by
  have c65973 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65974 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65975 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65976 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65977 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65978 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_16 := arithEq_of_rows h (row := 1067) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_16
  simp only [← c65973, ← c65974, ← c65975] at e_1067_16
  have e_1067_17 := arithEq_of_rows h (row := 1067) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_17
  simp only [← c65976, ← c65977, k2, ← c65978] at e_1067_17
  have hr := e_1067_17
  simp only [e_1067_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1067 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9157)) := by
  have c65979 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65980 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65981 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65982 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65983 := (publicBatchWrapper86_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65984 := (publicBatchWrapper86_copies2062 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1067_18 := arithEq_of_rows h (row := 1067) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_18
  simp only [← c65979, ← c65980, ← c65981] at e_1067_18
  have e_1067_19 := arithEq_of_rows h (row := 1067) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_19
  simp only [← c65982, ← c65983, k2, ← c65984] at e_1067_19
  have hr := e_1067_19
  simp only [e_1067_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9158)) := by
  have c65985 := (publicBatchWrapper86_copies2062 a h).2.1
  have c65986 := (publicBatchWrapper86_copies2062 a h).2.2.1
  have c65987 := (publicBatchWrapper86_copies2062 a h).2.2.2.1
  have c65988 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.1
  have c65989 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.1
  have c65990 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_0 := arithEq_of_rows h (row := 1068) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_0
  simp only [← c65985, ← c65986, ← c65987] at e_1068_0
  have e_1068_1 := arithEq_of_rows h (row := 1068) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_1
  simp only [← c65988, ← c65989, k2, ← c65990] at e_1068_1
  have hr := e_1068_1
  simp only [e_1068_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9159)) := by
  have c65991 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.1
  have c65992 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.1
  have c65993 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.1
  have c65994 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65995 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65996 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_2 := arithEq_of_rows h (row := 1068) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_2
  simp only [← c65991, ← c65992, ← c65993] at e_1068_2
  have e_1068_3 := arithEq_of_rows h (row := 1068) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_3
  simp only [← c65994, ← c65995, k2, ← c65996] at e_1068_3
  have hr := e_1068_3
  simp only [e_1068_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9160)) := by
  have c65997 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65998 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65999 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66000 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66001 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66002 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_4 := arithEq_of_rows h (row := 1068) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_4
  simp only [← c65997, ← c65998, ← c65999] at e_1068_4
  have e_1068_5 := arithEq_of_rows h (row := 1068) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_5
  simp only [← c66000, ← c66001, k2, ← c66002] at e_1068_5
  have hr := e_1068_5
  simp only [e_1068_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9161)) := by
  have c66003 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66004 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66005 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66006 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66007 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66008 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_6 := arithEq_of_rows h (row := 1068) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_6
  simp only [← c66003, ← c66004, ← c66005] at e_1068_6
  have e_1068_7 := arithEq_of_rows h (row := 1068) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_7
  simp only [← c66006, ← c66007, k2, ← c66008] at e_1068_7
  have hr := e_1068_7
  simp only [e_1068_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9162)) := by
  have c66009 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66010 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66011 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66012 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66013 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66014 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_8 := arithEq_of_rows h (row := 1068) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_8
  simp only [← c66009, ← c66010, ← c66011] at e_1068_8
  have e_1068_9 := arithEq_of_rows h (row := 1068) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_9
  simp only [← c66012, ← c66013, k2, ← c66014] at e_1068_9
  have hr := e_1068_9
  simp only [e_1068_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9163)) := by
  have c66015 := (publicBatchWrapper86_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66016 := (publicBatchWrapper86_copies2063 a h).1
  have c66017 := (publicBatchWrapper86_copies2063 a h).2.1
  have c66018 := (publicBatchWrapper86_copies2063 a h).2.2.1
  have c66019 := (publicBatchWrapper86_copies2063 a h).2.2.2.1
  have c66020 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_10 := arithEq_of_rows h (row := 1068) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_10
  simp only [← c66015, ← c66016, ← c66017] at e_1068_10
  have e_1068_11 := arithEq_of_rows h (row := 1068) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_11
  simp only [← c66018, ← c66019, k2, ← c66020] at e_1068_11
  have hr := e_1068_11
  simp only [e_1068_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9164)) := by
  have c66021 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.1
  have c66022 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.1
  have c66023 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.1
  have c66024 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.1
  have c66025 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.1
  have c66026 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_12 := arithEq_of_rows h (row := 1068) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_12
  simp only [← c66021, ← c66022, ← c66023] at e_1068_12
  have e_1068_13 := arithEq_of_rows h (row := 1068) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_13
  simp only [← c66024, ← c66025, k2, ← c66026] at e_1068_13
  have hr := e_1068_13
  simp only [e_1068_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9165)) := by
  have c66027 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66028 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66029 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66030 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66031 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66032 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_14 := arithEq_of_rows h (row := 1068) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_14
  simp only [← c66027, ← c66028, ← c66029] at e_1068_14
  have e_1068_15 := arithEq_of_rows h (row := 1068) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_15
  simp only [← c66030, ← c66031, k2, ← c66032] at e_1068_15
  have hr := e_1068_15
  simp only [e_1068_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9166)) := by
  have c66033 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66034 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66035 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66036 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66037 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66038 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_16 := arithEq_of_rows h (row := 1068) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_16
  simp only [← c66033, ← c66034, ← c66035] at e_1068_16
  have e_1068_17 := arithEq_of_rows h (row := 1068) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_17
  simp only [← c66036, ← c66037, k2, ← c66038] at e_1068_17
  have hr := e_1068_17
  simp only [e_1068_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1068 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9167)) := by
  have c66039 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66040 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66041 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66042 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66043 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66044 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1068_18 := arithEq_of_rows h (row := 1068) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_18
  simp only [← c66039, ← c66040, ← c66041] at e_1068_18
  have e_1068_19 := arithEq_of_rows h (row := 1068) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_19
  simp only [← c66042, ← c66043, k2, ← c66044] at e_1068_19
  have hr := e_1068_19
  simp only [e_1068_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9168)) := by
  have c66045 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66046 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66047 := (publicBatchWrapper86_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66048 := (publicBatchWrapper86_copies2064 a h).1
  have c66049 := (publicBatchWrapper86_copies2064 a h).2.1
  have c66050 := (publicBatchWrapper86_copies2064 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_0 := arithEq_of_rows h (row := 1069) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_0
  simp only [← c66045, ← c66046, ← c66047] at e_1069_0
  have e_1069_1 := arithEq_of_rows h (row := 1069) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_1
  simp only [← c66048, ← c66049, k2, ← c66050] at e_1069_1
  have hr := e_1069_1
  simp only [e_1069_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9169)) := by
  have c66051 := (publicBatchWrapper86_copies2064 a h).2.2.2.1
  have c66052 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.1
  have c66053 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.1
  have c66054 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.1
  have c66055 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.1
  have c66056 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_2 := arithEq_of_rows h (row := 1069) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_2
  simp only [← c66051, ← c66052, ← c66053] at e_1069_2
  have e_1069_3 := arithEq_of_rows h (row := 1069) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_3
  simp only [← c66054, ← c66055, k2, ← c66056] at e_1069_3
  have hr := e_1069_3
  simp only [e_1069_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9210)) := by
  have c66057 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.1
  have c66058 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66059 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66060 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66061 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66062 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_4 := arithEq_of_rows h (row := 1069) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_4
  simp only [← c66057, ← c66058, ← c66059] at e_1069_4
  have e_1069_5 := arithEq_of_rows h (row := 1069) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_5
  simp only [← c66060, ← c66061, k2, ← c66062] at e_1069_5
  have hr := e_1069_5
  simp only [e_1069_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9211)) := by
  have c66063 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66064 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66065 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66066 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66067 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66068 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_6 := arithEq_of_rows h (row := 1069) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_6
  simp only [← c66063, ← c66064, ← c66065] at e_1069_6
  have e_1069_7 := arithEq_of_rows h (row := 1069) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_7
  simp only [← c66066, ← c66067, k2, ← c66068] at e_1069_7
  have hr := e_1069_7
  simp only [e_1069_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9212)) := by
  have c66069 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66070 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66071 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66072 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66073 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66074 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_8 := arithEq_of_rows h (row := 1069) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_8
  simp only [← c66069, ← c66070, ← c66071] at e_1069_8
  have e_1069_9 := arithEq_of_rows h (row := 1069) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_9
  simp only [← c66072, ← c66073, k2, ← c66074] at e_1069_9
  have hr := e_1069_9
  simp only [e_1069_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9213)) := by
  have c66075 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66076 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66077 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66078 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66079 := (publicBatchWrapper86_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66080 := (publicBatchWrapper86_copies2065 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_10 := arithEq_of_rows h (row := 1069) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_10
  simp only [← c66075, ← c66076, ← c66077] at e_1069_10
  have e_1069_11 := arithEq_of_rows h (row := 1069) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_11
  simp only [← c66078, ← c66079, k2, ← c66080] at e_1069_11
  have hr := e_1069_11
  simp only [e_1069_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9214)) := by
  have c66081 := (publicBatchWrapper86_copies2065 a h).2.1
  have c66082 := (publicBatchWrapper86_copies2065 a h).2.2.1
  have c66083 := (publicBatchWrapper86_copies2065 a h).2.2.2.1
  have c66084 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.1
  have c66085 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.1
  have c66086 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_12 := arithEq_of_rows h (row := 1069) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_12
  simp only [← c66081, ← c66082, ← c66083] at e_1069_12
  have e_1069_13 := arithEq_of_rows h (row := 1069) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_13
  simp only [← c66084, ← c66085, k2, ← c66086] at e_1069_13
  have hr := e_1069_13
  simp only [e_1069_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9215)) := by
  have c66087 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.1
  have c66088 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.1
  have c66089 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.1
  have c66090 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66091 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66092 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_14 := arithEq_of_rows h (row := 1069) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_14
  simp only [← c66087, ← c66088, ← c66089] at e_1069_14
  have e_1069_15 := arithEq_of_rows h (row := 1069) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_15
  simp only [← c66090, ← c66091, k2, ← c66092] at e_1069_15
  have hr := e_1069_15
  simp only [e_1069_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9216)) := by
  have c66093 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66094 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66095 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66096 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66097 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66098 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_16 := arithEq_of_rows h (row := 1069) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_16
  simp only [← c66093, ← c66094, ← c66095] at e_1069_16
  have e_1069_17 := arithEq_of_rows h (row := 1069) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_17
  simp only [← c66096, ← c66097, k2, ← c66098] at e_1069_17
  have hr := e_1069_17
  simp only [e_1069_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1069 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9217)) := by
  have c66099 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66100 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66101 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66102 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66103 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66104 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1069_18 := arithEq_of_rows h (row := 1069) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_18
  simp only [← c66099, ← c66100, ← c66101] at e_1069_18
  have e_1069_19 := arithEq_of_rows h (row := 1069) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_19
  simp only [← c66102, ← c66103, k2, ← c66104] at e_1069_19
  have hr := e_1069_19
  simp only [e_1069_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9218)) := by
  have c66105 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66106 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66107 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66108 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66109 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66110 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_0 := arithEq_of_rows h (row := 1070) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_0
  simp only [← c66105, ← c66106, ← c66107] at e_1070_0
  have e_1070_1 := arithEq_of_rows h (row := 1070) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_1
  simp only [← c66108, ← c66109, k2, ← c66110] at e_1070_1
  have hr := e_1070_1
  simp only [e_1070_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9219)) := by
  have c66111 := (publicBatchWrapper86_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66112 := (publicBatchWrapper86_copies2066 a h).1
  have c66113 := (publicBatchWrapper86_copies2066 a h).2.1
  have c66114 := (publicBatchWrapper86_copies2066 a h).2.2.1
  have c66115 := (publicBatchWrapper86_copies2066 a h).2.2.2.1
  have c66116 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_2 := arithEq_of_rows h (row := 1070) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_2
  simp only [← c66111, ← c66112, ← c66113] at e_1070_2
  have e_1070_3 := arithEq_of_rows h (row := 1070) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_3
  simp only [← c66114, ← c66115, k2, ← c66116] at e_1070_3
  have hr := e_1070_3
  simp only [e_1070_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9220)) := by
  have c66117 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.1
  have c66118 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.1
  have c66119 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.1
  have c66120 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.1
  have c66121 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.1
  have c66122 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_4 := arithEq_of_rows h (row := 1070) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_4
  simp only [← c66117, ← c66118, ← c66119] at e_1070_4
  have e_1070_5 := arithEq_of_rows h (row := 1070) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_5
  simp only [← c66120, ← c66121, k2, ← c66122] at e_1070_5
  have hr := e_1070_5
  simp only [e_1070_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9221)) := by
  have c66123 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66124 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66125 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66126 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66127 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66128 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_6 := arithEq_of_rows h (row := 1070) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_6
  simp only [← c66123, ← c66124, ← c66125] at e_1070_6
  have e_1070_7 := arithEq_of_rows h (row := 1070) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_7
  simp only [← c66126, ← c66127, k2, ← c66128] at e_1070_7
  have hr := e_1070_7
  simp only [e_1070_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9222)) := by
  have c66129 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66130 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66131 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66132 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66133 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66134 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_8 := arithEq_of_rows h (row := 1070) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_8
  simp only [← c66129, ← c66130, ← c66131] at e_1070_8
  have e_1070_9 := arithEq_of_rows h (row := 1070) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_9
  simp only [← c66132, ← c66133, k2, ← c66134] at e_1070_9
  have hr := e_1070_9
  simp only [e_1070_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9223)) := by
  have c66135 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66136 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66137 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66138 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66139 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66140 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_10 := arithEq_of_rows h (row := 1070) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_10
  simp only [← c66135, ← c66136, ← c66137] at e_1070_10
  have e_1070_11 := arithEq_of_rows h (row := 1070) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_11
  simp only [← c66138, ← c66139, k2, ← c66140] at e_1070_11
  have hr := e_1070_11
  simp only [e_1070_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9224)) := by
  have c66141 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66142 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66143 := (publicBatchWrapper86_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66144 := (publicBatchWrapper86_copies2067 a h).1
  have c66145 := (publicBatchWrapper86_copies2067 a h).2.1
  have c66146 := (publicBatchWrapper86_copies2067 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_12 := arithEq_of_rows h (row := 1070) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_12
  simp only [← c66141, ← c66142, ← c66143] at e_1070_12
  have e_1070_13 := arithEq_of_rows h (row := 1070) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_13
  simp only [← c66144, ← c66145, k2, ← c66146] at e_1070_13
  have hr := e_1070_13
  simp only [e_1070_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9225)) := by
  have c66147 := (publicBatchWrapper86_copies2067 a h).2.2.2.1
  have c66148 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.1
  have c66149 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.1
  have c66150 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.1
  have c66151 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.1
  have c66152 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_14 := arithEq_of_rows h (row := 1070) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_14
  simp only [← c66147, ← c66148, ← c66149] at e_1070_14
  have e_1070_15 := arithEq_of_rows h (row := 1070) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_15
  simp only [← c66150, ← c66151, k2, ← c66152] at e_1070_15
  have hr := e_1070_15
  simp only [e_1070_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9226)) := by
  have c66153 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.1
  have c66154 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66155 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66156 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66157 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66158 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_16 := arithEq_of_rows h (row := 1070) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_16
  simp only [← c66153, ← c66154, ← c66155] at e_1070_16
  have e_1070_17 := arithEq_of_rows h (row := 1070) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_17
  simp only [← c66156, ← c66157, k2, ← c66158] at e_1070_17
  have hr := e_1070_17
  simp only [e_1070_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1070 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9227)) := by
  have c66159 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66160 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66161 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66162 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66163 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66164 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1070_18 := arithEq_of_rows h (row := 1070) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_18
  simp only [← c66159, ← c66160, ← c66161] at e_1070_18
  have e_1070_19 := arithEq_of_rows h (row := 1070) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_19
  simp only [← c66162, ← c66163, k2, ← c66164] at e_1070_19
  have hr := e_1070_19
  simp only [e_1070_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9228)) := by
  have c66165 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66166 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66167 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66168 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66169 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66170 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_0 := arithEq_of_rows h (row := 1071) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_0
  simp only [← c66165, ← c66166, ← c66167] at e_1071_0
  have e_1071_1 := arithEq_of_rows h (row := 1071) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_1
  simp only [← c66168, ← c66169, k2, ← c66170] at e_1071_1
  have hr := e_1071_1
  simp only [e_1071_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9229)) := by
  have c66171 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66172 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66173 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66174 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66175 := (publicBatchWrapper86_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66176 := (publicBatchWrapper86_copies2068 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_2 := arithEq_of_rows h (row := 1071) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_2
  simp only [← c66171, ← c66172, ← c66173] at e_1071_2
  have e_1071_3 := arithEq_of_rows h (row := 1071) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_3
  simp only [← c66174, ← c66175, k2, ← c66176] at e_1071_3
  have hr := e_1071_3
  simp only [e_1071_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9230)) := by
  have c66177 := (publicBatchWrapper86_copies2068 a h).2.1
  have c66178 := (publicBatchWrapper86_copies2068 a h).2.2.1
  have c66179 := (publicBatchWrapper86_copies2068 a h).2.2.2.1
  have c66180 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.1
  have c66181 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.1
  have c66182 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_4 := arithEq_of_rows h (row := 1071) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_4
  simp only [← c66177, ← c66178, ← c66179] at e_1071_4
  have e_1071_5 := arithEq_of_rows h (row := 1071) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_5
  simp only [← c66180, ← c66181, k2, ← c66182] at e_1071_5
  have hr := e_1071_5
  simp only [e_1071_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9231)) := by
  have c66183 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.1
  have c66184 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.1
  have c66185 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.1
  have c66186 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66187 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66188 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_6 := arithEq_of_rows h (row := 1071) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_6
  simp only [← c66183, ← c66184, ← c66185] at e_1071_6
  have e_1071_7 := arithEq_of_rows h (row := 1071) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_7
  simp only [← c66186, ← c66187, k2, ← c66188] at e_1071_7
  have hr := e_1071_7
  simp only [e_1071_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9232)) := by
  have c66189 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66190 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66191 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66192 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66193 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66194 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_8 := arithEq_of_rows h (row := 1071) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_8
  simp only [← c66189, ← c66190, ← c66191] at e_1071_8
  have e_1071_9 := arithEq_of_rows h (row := 1071) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_9
  simp only [← c66192, ← c66193, k2, ← c66194] at e_1071_9
  have hr := e_1071_9
  simp only [e_1071_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9233)) := by
  have c66195 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66196 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66197 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66198 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66199 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66200 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_10 := arithEq_of_rows h (row := 1071) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_10
  simp only [← c66195, ← c66196, ← c66197] at e_1071_10
  have e_1071_11 := arithEq_of_rows h (row := 1071) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_11
  simp only [← c66198, ← c66199, k2, ← c66200] at e_1071_11
  have hr := e_1071_11
  simp only [e_1071_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9234)) := by
  have c66201 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66202 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66203 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66204 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66205 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66206 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_12 := arithEq_of_rows h (row := 1071) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_12
  simp only [← c66201, ← c66202, ← c66203] at e_1071_12
  have e_1071_13 := arithEq_of_rows h (row := 1071) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_13
  simp only [← c66204, ← c66205, k2, ← c66206] at e_1071_13
  have hr := e_1071_13
  simp only [e_1071_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9235)) := by
  have c66207 := (publicBatchWrapper86_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66208 := (publicBatchWrapper86_copies2069 a h).1
  have c66209 := (publicBatchWrapper86_copies2069 a h).2.1
  have c66210 := (publicBatchWrapper86_copies2069 a h).2.2.1
  have c66211 := (publicBatchWrapper86_copies2069 a h).2.2.2.1
  have c66212 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_14 := arithEq_of_rows h (row := 1071) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_14
  simp only [← c66207, ← c66208, ← c66209] at e_1071_14
  have e_1071_15 := arithEq_of_rows h (row := 1071) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_15
  simp only [← c66210, ← c66211, k2, ← c66212] at e_1071_15
  have hr := e_1071_15
  simp only [e_1071_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9236)) := by
  have c66213 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.1
  have c66214 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.1
  have c66215 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.1
  have c66216 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.1
  have c66217 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.1
  have c66218 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_16 := arithEq_of_rows h (row := 1071) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_16
  simp only [← c66213, ← c66214, ← c66215] at e_1071_16
  have e_1071_17 := arithEq_of_rows h (row := 1071) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_17
  simp only [← c66216, ← c66217, k2, ← c66218] at e_1071_17
  have hr := e_1071_17
  simp only [e_1071_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1071 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9237)) := by
  have c66219 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66220 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66221 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66222 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66223 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66224 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1071_18 := arithEq_of_rows h (row := 1071) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_18
  simp only [← c66219, ← c66220, ← c66221] at e_1071_18
  have e_1071_19 := arithEq_of_rows h (row := 1071) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_19
  simp only [← c66222, ← c66223, k2, ← c66224] at e_1071_19
  have hr := e_1071_19
  simp only [e_1071_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9238)) := by
  have c66225 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66226 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66227 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66228 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66229 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66230 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_0 := arithEq_of_rows h (row := 1072) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_0
  simp only [← c66225, ← c66226, ← c66227] at e_1072_0
  have e_1072_1 := arithEq_of_rows h (row := 1072) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_1
  simp only [← c66228, ← c66229, k2, ← c66230] at e_1072_1
  have hr := e_1072_1
  simp only [e_1072_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9239)) := by
  have c66231 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66232 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66233 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66234 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66235 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66236 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_2 := arithEq_of_rows h (row := 1072) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_2
  simp only [← c66231, ← c66232, ← c66233] at e_1072_2
  have e_1072_3 := arithEq_of_rows h (row := 1072) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_3
  simp only [← c66234, ← c66235, k2, ← c66236] at e_1072_3
  have hr := e_1072_3
  simp only [e_1072_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9240)) := by
  have c66237 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66238 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66239 := (publicBatchWrapper86_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66240 := (publicBatchWrapper86_copies2070 a h).1
  have c66241 := (publicBatchWrapper86_copies2070 a h).2.1
  have c66242 := (publicBatchWrapper86_copies2070 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_4 := arithEq_of_rows h (row := 1072) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_4
  simp only [← c66237, ← c66238, ← c66239] at e_1072_4
  have e_1072_5 := arithEq_of_rows h (row := 1072) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_5
  simp only [← c66240, ← c66241, k2, ← c66242] at e_1072_5
  have hr := e_1072_5
  simp only [e_1072_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9241)) := by
  have c66243 := (publicBatchWrapper86_copies2070 a h).2.2.2.1
  have c66244 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.1
  have c66245 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.1
  have c66246 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.1
  have c66247 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.1
  have c66248 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_6 := arithEq_of_rows h (row := 1072) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_6
  simp only [← c66243, ← c66244, ← c66245] at e_1072_6
  have e_1072_7 := arithEq_of_rows h (row := 1072) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_7
  simp only [← c66246, ← c66247, k2, ← c66248] at e_1072_7
  have hr := e_1072_7
  simp only [e_1072_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9242)) := by
  have c66249 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.1
  have c66250 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66251 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66252 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66253 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66254 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_8 := arithEq_of_rows h (row := 1072) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_8
  simp only [← c66249, ← c66250, ← c66251] at e_1072_8
  have e_1072_9 := arithEq_of_rows h (row := 1072) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_9
  simp only [← c66252, ← c66253, k2, ← c66254] at e_1072_9
  have hr := e_1072_9
  simp only [e_1072_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9243)) := by
  have c66255 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66256 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66257 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66258 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66259 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66260 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_10 := arithEq_of_rows h (row := 1072) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_10
  simp only [← c66255, ← c66256, ← c66257] at e_1072_10
  have e_1072_11 := arithEq_of_rows h (row := 1072) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_11
  simp only [← c66258, ← c66259, k2, ← c66260] at e_1072_11
  have hr := e_1072_11
  simp only [e_1072_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9244)) := by
  have c66261 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66262 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66263 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66264 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66265 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66266 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_12 := arithEq_of_rows h (row := 1072) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_12
  simp only [← c66261, ← c66262, ← c66263] at e_1072_12
  have e_1072_13 := arithEq_of_rows h (row := 1072) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_13
  simp only [← c66264, ← c66265, k2, ← c66266] at e_1072_13
  have hr := e_1072_13
  simp only [e_1072_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9245)) := by
  have c66267 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66268 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66269 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66270 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66271 := (publicBatchWrapper86_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66272 := (publicBatchWrapper86_copies2071 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_14 := arithEq_of_rows h (row := 1072) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_14
  simp only [← c66267, ← c66268, ← c66269] at e_1072_14
  have e_1072_15 := arithEq_of_rows h (row := 1072) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_15
  simp only [← c66270, ← c66271, k2, ← c66272] at e_1072_15
  have hr := e_1072_15
  simp only [e_1072_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9246)) := by
  have c66273 := (publicBatchWrapper86_copies2071 a h).2.1
  have c66274 := (publicBatchWrapper86_copies2071 a h).2.2.1
  have c66275 := (publicBatchWrapper86_copies2071 a h).2.2.2.1
  have c66276 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.1
  have c66277 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.1
  have c66278 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_16 := arithEq_of_rows h (row := 1072) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_16
  simp only [← c66273, ← c66274, ← c66275] at e_1072_16
  have e_1072_17 := arithEq_of_rows h (row := 1072) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_17
  simp only [← c66276, ← c66277, k2, ← c66278] at e_1072_17
  have hr := e_1072_17
  simp only [e_1072_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1072 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9247)) := by
  have c66279 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.1
  have c66280 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.1
  have c66281 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.1
  have c66282 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66283 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66284 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1072_18 := arithEq_of_rows h (row := 1072) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_18
  simp only [← c66279, ← c66280, ← c66281] at e_1072_18
  have e_1072_19 := arithEq_of_rows h (row := 1072) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_19
  simp only [← c66282, ← c66283, k2, ← c66284] at e_1072_19
  have hr := e_1072_19
  simp only [e_1072_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9248)) := by
  have c66285 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66286 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66287 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66288 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66289 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66290 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_0 := arithEq_of_rows h (row := 1073) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_0
  simp only [← c66285, ← c66286, ← c66287] at e_1073_0
  have e_1073_1 := arithEq_of_rows h (row := 1073) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_1
  simp only [← c66288, ← c66289, k2, ← c66290] at e_1073_1
  have hr := e_1073_1
  simp only [e_1073_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9249)) := by
  have c66291 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66292 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66293 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66294 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66295 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66296 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_2 := arithEq_of_rows h (row := 1073) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_2
  simp only [← c66291, ← c66292, ← c66293] at e_1073_2
  have e_1073_3 := arithEq_of_rows h (row := 1073) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_3
  simp only [← c66294, ← c66295, k2, ← c66296] at e_1073_3
  have hr := e_1073_3
  simp only [e_1073_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9250)) := by
  have c66297 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66298 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66299 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66300 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66301 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66302 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_4 := arithEq_of_rows h (row := 1073) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_4
  simp only [← c66297, ← c66298, ← c66299] at e_1073_4
  have e_1073_5 := arithEq_of_rows h (row := 1073) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_5
  simp only [← c66300, ← c66301, k2, ← c66302] at e_1073_5
  have hr := e_1073_5
  simp only [e_1073_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9251)) := by
  have c66303 := (publicBatchWrapper86_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66304 := (publicBatchWrapper86_copies2072 a h).1
  have c66305 := (publicBatchWrapper86_copies2072 a h).2.1
  have c66306 := (publicBatchWrapper86_copies2072 a h).2.2.1
  have c66307 := (publicBatchWrapper86_copies2072 a h).2.2.2.1
  have c66308 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_6 := arithEq_of_rows h (row := 1073) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_6
  simp only [← c66303, ← c66304, ← c66305] at e_1073_6
  have e_1073_7 := arithEq_of_rows h (row := 1073) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_7
  simp only [← c66306, ← c66307, k2, ← c66308] at e_1073_7
  have hr := e_1073_7
  simp only [e_1073_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9252)) := by
  have c66309 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.1
  have c66310 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.1
  have c66311 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.1
  have c66312 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.1
  have c66313 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.1
  have c66314 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_8 := arithEq_of_rows h (row := 1073) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_8
  simp only [← c66309, ← c66310, ← c66311] at e_1073_8
  have e_1073_9 := arithEq_of_rows h (row := 1073) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_9
  simp only [← c66312, ← c66313, k2, ← c66314] at e_1073_9
  have hr := e_1073_9
  simp only [e_1073_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9253)) := by
  have c66315 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66316 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66317 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66318 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66319 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66320 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_10 := arithEq_of_rows h (row := 1073) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_10
  simp only [← c66315, ← c66316, ← c66317] at e_1073_10
  have e_1073_11 := arithEq_of_rows h (row := 1073) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_11
  simp only [← c66318, ← c66319, k2, ← c66320] at e_1073_11
  have hr := e_1073_11
  simp only [e_1073_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9254)) := by
  have c66321 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66322 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66323 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66324 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66325 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66326 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_12 := arithEq_of_rows h (row := 1073) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_12
  simp only [← c66321, ← c66322, ← c66323] at e_1073_12
  have e_1073_13 := arithEq_of_rows h (row := 1073) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_13
  simp only [← c66324, ← c66325, k2, ← c66326] at e_1073_13
  have hr := e_1073_13
  simp only [e_1073_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9255)) := by
  have c66327 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66328 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66329 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66330 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66331 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66332 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_14 := arithEq_of_rows h (row := 1073) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_14
  simp only [← c66327, ← c66328, ← c66329] at e_1073_14
  have e_1073_15 := arithEq_of_rows h (row := 1073) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_15
  simp only [← c66330, ← c66331, k2, ← c66332] at e_1073_15
  have hr := e_1073_15
  simp only [e_1073_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9256)) := by
  have c66333 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66334 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66335 := (publicBatchWrapper86_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66336 := (publicBatchWrapper86_copies2073 a h).1
  have c66337 := (publicBatchWrapper86_copies2073 a h).2.1
  have c66338 := (publicBatchWrapper86_copies2073 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_16 := arithEq_of_rows h (row := 1073) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_16
  simp only [← c66333, ← c66334, ← c66335] at e_1073_16
  have e_1073_17 := arithEq_of_rows h (row := 1073) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_17
  simp only [← c66336, ← c66337, k2, ← c66338] at e_1073_17
  have hr := e_1073_17
  simp only [e_1073_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1073 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9257)) := by
  have c66339 := (publicBatchWrapper86_copies2073 a h).2.2.2.1
  have c66340 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.1
  have c66341 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.1
  have c66342 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.1
  have c66343 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.1
  have c66344 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1073_18 := arithEq_of_rows h (row := 1073) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_18
  simp only [← c66339, ← c66340, ← c66341] at e_1073_18
  have e_1073_19 := arithEq_of_rows h (row := 1073) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_19
  simp only [← c66342, ← c66343, k2, ← c66344] at e_1073_19
  have hr := e_1073_19
  simp only [e_1073_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9258)) := by
  have c66345 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.1
  have c66346 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66347 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66348 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66349 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66350 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_0 := arithEq_of_rows h (row := 1074) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_0
  simp only [← c66345, ← c66346, ← c66347] at e_1074_0
  have e_1074_1 := arithEq_of_rows h (row := 1074) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_1
  simp only [← c66348, ← c66349, k2, ← c66350] at e_1074_1
  have hr := e_1074_1
  simp only [e_1074_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9259)) := by
  have c66351 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66352 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66353 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66354 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66355 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66356 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_2 := arithEq_of_rows h (row := 1074) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_2
  simp only [← c66351, ← c66352, ← c66353] at e_1074_2
  have e_1074_3 := arithEq_of_rows h (row := 1074) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_3
  simp only [← c66354, ← c66355, k2, ← c66356] at e_1074_3
  have hr := e_1074_3
  simp only [e_1074_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9260)) := by
  have c66357 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66358 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66359 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66360 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66361 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66362 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_4 := arithEq_of_rows h (row := 1074) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_4
  simp only [← c66357, ← c66358, ← c66359] at e_1074_4
  have e_1074_5 := arithEq_of_rows h (row := 1074) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_5
  simp only [← c66360, ← c66361, k2, ← c66362] at e_1074_5
  have hr := e_1074_5
  simp only [e_1074_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9261)) := by
  have c66363 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66364 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66365 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66366 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66367 := (publicBatchWrapper86_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66368 := (publicBatchWrapper86_copies2074 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_6 := arithEq_of_rows h (row := 1074) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_6
  simp only [← c66363, ← c66364, ← c66365] at e_1074_6
  have e_1074_7 := arithEq_of_rows h (row := 1074) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_7
  simp only [← c66366, ← c66367, k2, ← c66368] at e_1074_7
  have hr := e_1074_7
  simp only [e_1074_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9262)) := by
  have c66369 := (publicBatchWrapper86_copies2074 a h).2.1
  have c66370 := (publicBatchWrapper86_copies2074 a h).2.2.1
  have c66371 := (publicBatchWrapper86_copies2074 a h).2.2.2.1
  have c66372 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.1
  have c66373 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.1
  have c66374 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_8 := arithEq_of_rows h (row := 1074) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_8
  simp only [← c66369, ← c66370, ← c66371] at e_1074_8
  have e_1074_9 := arithEq_of_rows h (row := 1074) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_9
  simp only [← c66372, ← c66373, k2, ← c66374] at e_1074_9
  have hr := e_1074_9
  simp only [e_1074_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9263)) := by
  have c66375 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.1
  have c66376 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.1
  have c66377 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.1
  have c66378 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66379 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66380 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_10 := arithEq_of_rows h (row := 1074) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_10
  simp only [← c66375, ← c66376, ← c66377] at e_1074_10
  have e_1074_11 := arithEq_of_rows h (row := 1074) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_11
  simp only [← c66378, ← c66379, k2, ← c66380] at e_1074_11
  have hr := e_1074_11
  simp only [e_1074_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9264)) := by
  have c66381 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66382 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66383 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66384 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66385 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66386 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_12 := arithEq_of_rows h (row := 1074) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_12
  simp only [← c66381, ← c66382, ← c66383] at e_1074_12
  have e_1074_13 := arithEq_of_rows h (row := 1074) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_13
  simp only [← c66384, ← c66385, k2, ← c66386] at e_1074_13
  have hr := e_1074_13
  simp only [e_1074_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9265)) := by
  have c66387 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66388 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66389 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66390 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66391 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66392 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_14 := arithEq_of_rows h (row := 1074) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_14
  simp only [← c66387, ← c66388, ← c66389] at e_1074_14
  have e_1074_15 := arithEq_of_rows h (row := 1074) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_15
  simp only [← c66390, ← c66391, k2, ← c66392] at e_1074_15
  have hr := e_1074_15
  simp only [e_1074_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9266)) := by
  have c66393 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66394 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66395 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66396 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66397 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66398 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_16 := arithEq_of_rows h (row := 1074) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_16
  simp only [← c66393, ← c66394, ← c66395] at e_1074_16
  have e_1074_17 := arithEq_of_rows h (row := 1074) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_17
  simp only [← c66396, ← c66397, k2, ← c66398] at e_1074_17
  have hr := e_1074_17
  simp only [e_1074_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1074 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9267)) := by
  have c66399 := (publicBatchWrapper86_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66400 := (publicBatchWrapper86_copies2075 a h).1
  have c66401 := (publicBatchWrapper86_copies2075 a h).2.1
  have c66402 := (publicBatchWrapper86_copies2075 a h).2.2.1
  have c66403 := (publicBatchWrapper86_copies2075 a h).2.2.2.1
  have c66404 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1074_18 := arithEq_of_rows h (row := 1074) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_18
  simp only [← c66399, ← c66400, ← c66401] at e_1074_18
  have e_1074_19 := arithEq_of_rows h (row := 1074) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_19
  simp only [← c66402, ← c66403, k2, ← c66404] at e_1074_19
  have hr := e_1074_19
  simp only [e_1074_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9268)) := by
  have c66405 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.1
  have c66406 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.1
  have c66407 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.1
  have c66408 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.1
  have c66409 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.1
  have c66410 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_0 := arithEq_of_rows h (row := 1075) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_0
  simp only [← c66405, ← c66406, ← c66407] at e_1075_0
  have e_1075_1 := arithEq_of_rows h (row := 1075) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_1
  simp only [← c66408, ← c66409, k2, ← c66410] at e_1075_1
  have hr := e_1075_1
  simp only [e_1075_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9269)) := by
  have c66411 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66412 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66413 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66414 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66415 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66416 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_2 := arithEq_of_rows h (row := 1075) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_2
  simp only [← c66411, ← c66412, ← c66413] at e_1075_2
  have e_1075_3 := arithEq_of_rows h (row := 1075) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_3
  simp only [← c66414, ← c66415, k2, ← c66416] at e_1075_3
  have hr := e_1075_3
  simp only [e_1075_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9270)) := by
  have c66417 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66418 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66419 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66420 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66421 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66422 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_4 := arithEq_of_rows h (row := 1075) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_4
  simp only [← c66417, ← c66418, ← c66419] at e_1075_4
  have e_1075_5 := arithEq_of_rows h (row := 1075) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_5
  simp only [← c66420, ← c66421, k2, ← c66422] at e_1075_5
  have hr := e_1075_5
  simp only [e_1075_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9271)) := by
  have c66423 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66424 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66425 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66426 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66427 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66428 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_6 := arithEq_of_rows h (row := 1075) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_6
  simp only [← c66423, ← c66424, ← c66425] at e_1075_6
  have e_1075_7 := arithEq_of_rows h (row := 1075) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_7
  simp only [← c66426, ← c66427, k2, ← c66428] at e_1075_7
  have hr := e_1075_7
  simp only [e_1075_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9272)) := by
  have c66429 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66430 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66431 := (publicBatchWrapper86_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66432 := (publicBatchWrapper86_copies2076 a h).1
  have c66433 := (publicBatchWrapper86_copies2076 a h).2.1
  have c66434 := (publicBatchWrapper86_copies2076 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_8 := arithEq_of_rows h (row := 1075) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_8
  simp only [← c66429, ← c66430, ← c66431] at e_1075_8
  have e_1075_9 := arithEq_of_rows h (row := 1075) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_9
  simp only [← c66432, ← c66433, k2, ← c66434] at e_1075_9
  have hr := e_1075_9
  simp only [e_1075_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9273)) := by
  have c66435 := (publicBatchWrapper86_copies2076 a h).2.2.2.1
  have c66436 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.1
  have c66437 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.1
  have c66438 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.1
  have c66439 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.1
  have c66440 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_10 := arithEq_of_rows h (row := 1075) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_10
  simp only [← c66435, ← c66436, ← c66437] at e_1075_10
  have e_1075_11 := arithEq_of_rows h (row := 1075) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_11
  simp only [← c66438, ← c66439, k2, ← c66440] at e_1075_11
  have hr := e_1075_11
  simp only [e_1075_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9274)) := by
  have c66441 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.1
  have c66442 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66443 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66444 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66445 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66446 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_12 := arithEq_of_rows h (row := 1075) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_12
  simp only [← c66441, ← c66442, ← c66443] at e_1075_12
  have e_1075_13 := arithEq_of_rows h (row := 1075) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_13
  simp only [← c66444, ← c66445, k2, ← c66446] at e_1075_13
  have hr := e_1075_13
  simp only [e_1075_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9275)) := by
  have c66447 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66448 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66449 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66450 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66451 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66452 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_14 := arithEq_of_rows h (row := 1075) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_14
  simp only [← c66447, ← c66448, ← c66449] at e_1075_14
  have e_1075_15 := arithEq_of_rows h (row := 1075) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_15
  simp only [← c66450, ← c66451, k2, ← c66452] at e_1075_15
  have hr := e_1075_15
  simp only [e_1075_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9276)) := by
  have c66453 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66454 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66455 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66456 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66457 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66458 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_16 := arithEq_of_rows h (row := 1075) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_16
  simp only [← c66453, ← c66454, ← c66455] at e_1075_16
  have e_1075_17 := arithEq_of_rows h (row := 1075) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_17
  simp only [← c66456, ← c66457, k2, ← c66458] at e_1075_17
  have hr := e_1075_17
  simp only [e_1075_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1075 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9277)) := by
  have c66459 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66460 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66461 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66462 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66463 := (publicBatchWrapper86_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66464 := (publicBatchWrapper86_copies2077 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1075_18 := arithEq_of_rows h (row := 1075) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_18
  simp only [← c66459, ← c66460, ← c66461] at e_1075_18
  have e_1075_19 := arithEq_of_rows h (row := 1075) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_19
  simp only [← c66462, ← c66463, k2, ← c66464] at e_1075_19
  have hr := e_1075_19
  simp only [e_1075_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9278)) := by
  have c66465 := (publicBatchWrapper86_copies2077 a h).2.1
  have c66466 := (publicBatchWrapper86_copies2077 a h).2.2.1
  have c66467 := (publicBatchWrapper86_copies2077 a h).2.2.2.1
  have c66468 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.1
  have c66469 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.1
  have c66470 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_0 := arithEq_of_rows h (row := 1076) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_0
  simp only [← c66465, ← c66466, ← c66467] at e_1076_0
  have e_1076_1 := arithEq_of_rows h (row := 1076) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_1
  simp only [← c66468, ← c66469, k2, ← c66470] at e_1076_1
  have hr := e_1076_1
  simp only [e_1076_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9279)) := by
  have c66471 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.1
  have c66472 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.1
  have c66473 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.1
  have c66474 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66475 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66476 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_2 := arithEq_of_rows h (row := 1076) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_2
  simp only [← c66471, ← c66472, ← c66473] at e_1076_2
  have e_1076_3 := arithEq_of_rows h (row := 1076) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_3
  simp only [← c66474, ← c66475, k2, ← c66476] at e_1076_3
  have hr := e_1076_3
  simp only [e_1076_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9280)) := by
  have c66477 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66478 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66479 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66480 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66481 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66482 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_4 := arithEq_of_rows h (row := 1076) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_4
  simp only [← c66477, ← c66478, ← c66479] at e_1076_4
  have e_1076_5 := arithEq_of_rows h (row := 1076) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_5
  simp only [← c66480, ← c66481, k2, ← c66482] at e_1076_5
  have hr := e_1076_5
  simp only [e_1076_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9281)) := by
  have c66483 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66484 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66485 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66486 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66487 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66488 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_6 := arithEq_of_rows h (row := 1076) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_6
  simp only [← c66483, ← c66484, ← c66485] at e_1076_6
  have e_1076_7 := arithEq_of_rows h (row := 1076) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_7
  simp only [← c66486, ← c66487, k2, ← c66488] at e_1076_7
  have hr := e_1076_7
  simp only [e_1076_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9282)) := by
  have c66489 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66490 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66491 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66492 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66493 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66494 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_8 := arithEq_of_rows h (row := 1076) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_8
  simp only [← c66489, ← c66490, ← c66491] at e_1076_8
  have e_1076_9 := arithEq_of_rows h (row := 1076) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_9
  simp only [← c66492, ← c66493, k2, ← c66494] at e_1076_9
  have hr := e_1076_9
  simp only [e_1076_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9283)) := by
  have c66495 := (publicBatchWrapper86_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66496 := (publicBatchWrapper86_copies2078 a h).1
  have c66497 := (publicBatchWrapper86_copies2078 a h).2.1
  have c66498 := (publicBatchWrapper86_copies2078 a h).2.2.1
  have c66499 := (publicBatchWrapper86_copies2078 a h).2.2.2.1
  have c66500 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_10 := arithEq_of_rows h (row := 1076) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_10
  simp only [← c66495, ← c66496, ← c66497] at e_1076_10
  have e_1076_11 := arithEq_of_rows h (row := 1076) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_11
  simp only [← c66498, ← c66499, k2, ← c66500] at e_1076_11
  have hr := e_1076_11
  simp only [e_1076_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9284)) := by
  have c66501 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.1
  have c66502 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.1
  have c66503 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.1
  have c66504 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.1
  have c66505 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.1
  have c66506 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_12 := arithEq_of_rows h (row := 1076) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_12
  simp only [← c66501, ← c66502, ← c66503] at e_1076_12
  have e_1076_13 := arithEq_of_rows h (row := 1076) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_13
  simp only [← c66504, ← c66505, k2, ← c66506] at e_1076_13
  have hr := e_1076_13
  simp only [e_1076_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9285)) := by
  have c66507 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66508 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66509 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66510 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66511 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66512 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_14 := arithEq_of_rows h (row := 1076) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_14
  simp only [← c66507, ← c66508, ← c66509] at e_1076_14
  have e_1076_15 := arithEq_of_rows h (row := 1076) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_15
  simp only [← c66510, ← c66511, k2, ← c66512] at e_1076_15
  have hr := e_1076_15
  simp only [e_1076_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9286)) := by
  have c66513 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66514 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66515 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66516 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66517 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66518 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_16 := arithEq_of_rows h (row := 1076) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_16
  simp only [← c66513, ← c66514, ← c66515] at e_1076_16
  have e_1076_17 := arithEq_of_rows h (row := 1076) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_17
  simp only [← c66516, ← c66517, k2, ← c66518] at e_1076_17
  have hr := e_1076_17
  simp only [e_1076_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1076 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9287)) := by
  have c66519 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66520 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66521 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66522 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66523 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66524 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1076_18 := arithEq_of_rows h (row := 1076) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_18
  simp only [← c66519, ← c66520, ← c66521] at e_1076_18
  have e_1076_19 := arithEq_of_rows h (row := 1076) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_19
  simp only [← c66522, ← c66523, k2, ← c66524] at e_1076_19
  have hr := e_1076_19
  simp only [e_1076_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9288)) := by
  have c66525 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66526 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66527 := (publicBatchWrapper86_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66528 := (publicBatchWrapper86_copies2079 a h).1
  have c66529 := (publicBatchWrapper86_copies2079 a h).2.1
  have c66530 := (publicBatchWrapper86_copies2079 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_0 := arithEq_of_rows h (row := 1077) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_0
  simp only [← c66525, ← c66526, ← c66527] at e_1077_0
  have e_1077_1 := arithEq_of_rows h (row := 1077) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_1
  simp only [← c66528, ← c66529, k2, ← c66530] at e_1077_1
  have hr := e_1077_1
  simp only [e_1077_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9289)) := by
  have c66531 := (publicBatchWrapper86_copies2079 a h).2.2.2.1
  have c66532 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.1
  have c66533 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.1
  have c66534 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.1
  have c66535 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.1
  have c66536 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_2 := arithEq_of_rows h (row := 1077) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_2
  simp only [← c66531, ← c66532, ← c66533] at e_1077_2
  have e_1077_3 := arithEq_of_rows h (row := 1077) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_3
  simp only [← c66534, ← c66535, k2, ← c66536] at e_1077_3
  have hr := e_1077_3
  simp only [e_1077_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9290)) := by
  have c66537 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.1
  have c66538 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66539 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66540 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66541 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66542 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_4 := arithEq_of_rows h (row := 1077) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_4
  simp only [← c66537, ← c66538, ← c66539] at e_1077_4
  have e_1077_5 := arithEq_of_rows h (row := 1077) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_5
  simp only [← c66540, ← c66541, k2, ← c66542] at e_1077_5
  have hr := e_1077_5
  simp only [e_1077_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9291)) := by
  have c66543 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66544 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66545 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66546 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66547 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66548 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_6 := arithEq_of_rows h (row := 1077) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_6
  simp only [← c66543, ← c66544, ← c66545] at e_1077_6
  have e_1077_7 := arithEq_of_rows h (row := 1077) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_7
  simp only [← c66546, ← c66547, k2, ← c66548] at e_1077_7
  have hr := e_1077_7
  simp only [e_1077_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9292)) := by
  have c66549 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66550 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66551 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66552 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66553 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66554 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_8 := arithEq_of_rows h (row := 1077) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_8
  simp only [← c66549, ← c66550, ← c66551] at e_1077_8
  have e_1077_9 := arithEq_of_rows h (row := 1077) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_9
  simp only [← c66552, ← c66553, k2, ← c66554] at e_1077_9
  have hr := e_1077_9
  simp only [e_1077_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9293)) := by
  have c66555 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66556 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66557 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66558 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66559 := (publicBatchWrapper86_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66560 := (publicBatchWrapper86_copies2080 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_10 := arithEq_of_rows h (row := 1077) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_10
  simp only [← c66555, ← c66556, ← c66557] at e_1077_10
  have e_1077_11 := arithEq_of_rows h (row := 1077) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_11
  simp only [← c66558, ← c66559, k2, ← c66560] at e_1077_11
  have hr := e_1077_11
  simp only [e_1077_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9294)) := by
  have c66561 := (publicBatchWrapper86_copies2080 a h).2.1
  have c66562 := (publicBatchWrapper86_copies2080 a h).2.2.1
  have c66563 := (publicBatchWrapper86_copies2080 a h).2.2.2.1
  have c66564 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.1
  have c66565 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.1
  have c66566 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_12 := arithEq_of_rows h (row := 1077) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_12
  simp only [← c66561, ← c66562, ← c66563] at e_1077_12
  have e_1077_13 := arithEq_of_rows h (row := 1077) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_13
  simp only [← c66564, ← c66565, k2, ← c66566] at e_1077_13
  have hr := e_1077_13
  simp only [e_1077_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9295)) := by
  have c66567 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.1
  have c66568 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.1
  have c66569 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.1
  have c66570 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66571 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66572 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_14 := arithEq_of_rows h (row := 1077) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_14
  simp only [← c66567, ← c66568, ← c66569] at e_1077_14
  have e_1077_15 := arithEq_of_rows h (row := 1077) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_15
  simp only [← c66570, ← c66571, k2, ← c66572] at e_1077_15
  have hr := e_1077_15
  simp only [e_1077_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9296)) := by
  have c66573 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66574 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66575 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66576 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66577 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66578 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_16 := arithEq_of_rows h (row := 1077) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_16
  simp only [← c66573, ← c66574, ← c66575] at e_1077_16
  have e_1077_17 := arithEq_of_rows h (row := 1077) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_17
  simp only [← c66576, ← c66577, k2, ← c66578] at e_1077_17
  have hr := e_1077_17
  simp only [e_1077_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1077 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9297)) := by
  have c66579 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66580 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66581 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66582 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66583 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66584 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1077_18 := arithEq_of_rows h (row := 1077) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_18
  simp only [← c66579, ← c66580, ← c66581] at e_1077_18
  have e_1077_19 := arithEq_of_rows h (row := 1077) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_19
  simp only [← c66582, ← c66583, k2, ← c66584] at e_1077_19
  have hr := e_1077_19
  simp only [e_1077_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9298)) := by
  have c66585 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66586 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66587 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66588 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66589 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66590 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_0 := arithEq_of_rows h (row := 1078) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_0
  simp only [← c66585, ← c66586, ← c66587] at e_1078_0
  have e_1078_1 := arithEq_of_rows h (row := 1078) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_1
  simp only [← c66588, ← c66589, k2, ← c66590] at e_1078_1
  have hr := e_1078_1
  simp only [e_1078_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9299)) := by
  have c66591 := (publicBatchWrapper86_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66592 := (publicBatchWrapper86_copies2081 a h).1
  have c66593 := (publicBatchWrapper86_copies2081 a h).2.1
  have c66594 := (publicBatchWrapper86_copies2081 a h).2.2.1
  have c66595 := (publicBatchWrapper86_copies2081 a h).2.2.2.1
  have c66596 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_2 := arithEq_of_rows h (row := 1078) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_2
  simp only [← c66591, ← c66592, ← c66593] at e_1078_2
  have e_1078_3 := arithEq_of_rows h (row := 1078) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_3
  simp only [← c66594, ← c66595, k2, ← c66596] at e_1078_3
  have hr := e_1078_3
  simp only [e_1078_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9300)) := by
  have c66597 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.1
  have c66598 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.1
  have c66599 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.1
  have c66600 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.1
  have c66601 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.1
  have c66602 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_4 := arithEq_of_rows h (row := 1078) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_4
  simp only [← c66597, ← c66598, ← c66599] at e_1078_4
  have e_1078_5 := arithEq_of_rows h (row := 1078) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_5
  simp only [← c66600, ← c66601, k2, ← c66602] at e_1078_5
  have hr := e_1078_5
  simp only [e_1078_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9301)) := by
  have c66603 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66604 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66605 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66606 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66607 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66608 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_6 := arithEq_of_rows h (row := 1078) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_6
  simp only [← c66603, ← c66604, ← c66605] at e_1078_6
  have e_1078_7 := arithEq_of_rows h (row := 1078) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_7
  simp only [← c66606, ← c66607, k2, ← c66608] at e_1078_7
  have hr := e_1078_7
  simp only [e_1078_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9302)) := by
  have c66609 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66610 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66611 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66612 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66613 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66614 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_8 := arithEq_of_rows h (row := 1078) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_8
  simp only [← c66609, ← c66610, ← c66611] at e_1078_8
  have e_1078_9 := arithEq_of_rows h (row := 1078) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_9
  simp only [← c66612, ← c66613, k2, ← c66614] at e_1078_9
  have hr := e_1078_9
  simp only [e_1078_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9303)) := by
  have c66615 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66616 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66617 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66618 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66619 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66620 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_10 := arithEq_of_rows h (row := 1078) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_10
  simp only [← c66615, ← c66616, ← c66617] at e_1078_10
  have e_1078_11 := arithEq_of_rows h (row := 1078) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_11
  simp only [← c66618, ← c66619, k2, ← c66620] at e_1078_11
  have hr := e_1078_11
  simp only [e_1078_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9304)) := by
  have c66621 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66622 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66623 := (publicBatchWrapper86_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66624 := (publicBatchWrapper86_copies2082 a h).1
  have c66625 := (publicBatchWrapper86_copies2082 a h).2.1
  have c66626 := (publicBatchWrapper86_copies2082 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_12 := arithEq_of_rows h (row := 1078) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_12
  simp only [← c66621, ← c66622, ← c66623] at e_1078_12
  have e_1078_13 := arithEq_of_rows h (row := 1078) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_13
  simp only [← c66624, ← c66625, k2, ← c66626] at e_1078_13
  have hr := e_1078_13
  simp only [e_1078_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9305)) := by
  have c66627 := (publicBatchWrapper86_copies2082 a h).2.2.2.1
  have c66628 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.1
  have c66629 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.1
  have c66630 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.1
  have c66631 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.1
  have c66632 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_14 := arithEq_of_rows h (row := 1078) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_14
  simp only [← c66627, ← c66628, ← c66629] at e_1078_14
  have e_1078_15 := arithEq_of_rows h (row := 1078) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_15
  simp only [← c66630, ← c66631, k2, ← c66632] at e_1078_15
  have hr := e_1078_15
  simp only [e_1078_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9306)) := by
  have c66633 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.1
  have c66634 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66635 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66636 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66637 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66638 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_16 := arithEq_of_rows h (row := 1078) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_16
  simp only [← c66633, ← c66634, ← c66635] at e_1078_16
  have e_1078_17 := arithEq_of_rows h (row := 1078) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_17
  simp only [← c66636, ← c66637, k2, ← c66638] at e_1078_17
  have hr := e_1078_17
  simp only [e_1078_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1078 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9307)) := by
  have c66639 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66640 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66641 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66642 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66643 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66644 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1078_18 := arithEq_of_rows h (row := 1078) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_18
  simp only [← c66639, ← c66640, ← c66641] at e_1078_18
  have e_1078_19 := arithEq_of_rows h (row := 1078) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_19
  simp only [← c66642, ← c66643, k2, ← c66644] at e_1078_19
  have hr := e_1078_19
  simp only [e_1078_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9308)) := by
  have c66645 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66646 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66647 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66648 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66649 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66650 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_0 := arithEq_of_rows h (row := 1079) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_0
  simp only [← c66645, ← c66646, ← c66647] at e_1079_0
  have e_1079_1 := arithEq_of_rows h (row := 1079) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_1
  simp only [← c66648, ← c66649, k2, ← c66650] at e_1079_1
  have hr := e_1079_1
  simp only [e_1079_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9309)) := by
  have c66651 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66652 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66653 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66654 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66655 := (publicBatchWrapper86_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66656 := (publicBatchWrapper86_copies2083 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_2 := arithEq_of_rows h (row := 1079) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_2
  simp only [← c66651, ← c66652, ← c66653] at e_1079_2
  have e_1079_3 := arithEq_of_rows h (row := 1079) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_3
  simp only [← c66654, ← c66655, k2, ← c66656] at e_1079_3
  have hr := e_1079_3
  simp only [e_1079_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9310)) := by
  have c66657 := (publicBatchWrapper86_copies2083 a h).2.1
  have c66658 := (publicBatchWrapper86_copies2083 a h).2.2.1
  have c66659 := (publicBatchWrapper86_copies2083 a h).2.2.2.1
  have c66660 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.1
  have c66661 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.1
  have c66662 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_4 := arithEq_of_rows h (row := 1079) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_4
  simp only [← c66657, ← c66658, ← c66659] at e_1079_4
  have e_1079_5 := arithEq_of_rows h (row := 1079) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_5
  simp only [← c66660, ← c66661, k2, ← c66662] at e_1079_5
  have hr := e_1079_5
  simp only [e_1079_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9311)) := by
  have c66663 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.1
  have c66664 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.1
  have c66665 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.1
  have c66666 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66667 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66668 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_6 := arithEq_of_rows h (row := 1079) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_6
  simp only [← c66663, ← c66664, ← c66665] at e_1079_6
  have e_1079_7 := arithEq_of_rows h (row := 1079) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_7
  simp only [← c66666, ← c66667, k2, ← c66668] at e_1079_7
  have hr := e_1079_7
  simp only [e_1079_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9312)) := by
  have c66669 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66670 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66671 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66672 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66673 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66674 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_8 := arithEq_of_rows h (row := 1079) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_8
  simp only [← c66669, ← c66670, ← c66671] at e_1079_8
  have e_1079_9 := arithEq_of_rows h (row := 1079) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_9
  simp only [← c66672, ← c66673, k2, ← c66674] at e_1079_9
  have hr := e_1079_9
  simp only [e_1079_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9313)) := by
  have c66675 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66676 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66677 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66678 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66679 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66680 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_10 := arithEq_of_rows h (row := 1079) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_10
  simp only [← c66675, ← c66676, ← c66677] at e_1079_10
  have e_1079_11 := arithEq_of_rows h (row := 1079) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_11
  simp only [← c66678, ← c66679, k2, ← c66680] at e_1079_11
  have hr := e_1079_11
  simp only [e_1079_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9314)) := by
  have c66681 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66682 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66683 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66684 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66685 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66686 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_12 := arithEq_of_rows h (row := 1079) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_12
  simp only [← c66681, ← c66682, ← c66683] at e_1079_12
  have e_1079_13 := arithEq_of_rows h (row := 1079) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_13
  simp only [← c66684, ← c66685, k2, ← c66686] at e_1079_13
  have hr := e_1079_13
  simp only [e_1079_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9315)) := by
  have c66687 := (publicBatchWrapper86_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66688 := (publicBatchWrapper86_copies2084 a h).1
  have c66689 := (publicBatchWrapper86_copies2084 a h).2.1
  have c66690 := (publicBatchWrapper86_copies2084 a h).2.2.1
  have c66691 := (publicBatchWrapper86_copies2084 a h).2.2.2.1
  have c66692 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_14 := arithEq_of_rows h (row := 1079) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_14
  simp only [← c66687, ← c66688, ← c66689] at e_1079_14
  have e_1079_15 := arithEq_of_rows h (row := 1079) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_15
  simp only [← c66690, ← c66691, k2, ← c66692] at e_1079_15
  have hr := e_1079_15
  simp only [e_1079_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9316)) := by
  have c66693 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.1
  have c66694 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.1
  have c66695 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.1
  have c66696 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.1
  have c66697 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.1
  have c66698 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_16 := arithEq_of_rows h (row := 1079) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_16
  simp only [← c66693, ← c66694, ← c66695] at e_1079_16
  have e_1079_17 := arithEq_of_rows h (row := 1079) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_17
  simp only [← c66696, ← c66697, k2, ← c66698] at e_1079_17
  have hr := e_1079_17
  simp only [e_1079_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1079 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9317)) := by
  have c66699 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66700 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66701 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66702 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66703 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66704 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1079_18 := arithEq_of_rows h (row := 1079) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_18
  simp only [← c66699, ← c66700, ← c66701] at e_1079_18
  have e_1079_19 := arithEq_of_rows h (row := 1079) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_19
  simp only [← c66702, ← c66703, k2, ← c66704] at e_1079_19
  have hr := e_1079_19
  simp only [e_1079_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9318)) := by
  have c66705 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66706 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66707 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66708 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66709 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66710 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_0 := arithEq_of_rows h (row := 1080) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_0
  simp only [← c66705, ← c66706, ← c66707] at e_1080_0
  have e_1080_1 := arithEq_of_rows h (row := 1080) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_1
  simp only [← c66708, ← c66709, k2, ← c66710] at e_1080_1
  have hr := e_1080_1
  simp only [e_1080_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9319)) := by
  have c66711 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66712 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66713 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66714 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66715 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66716 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_2 := arithEq_of_rows h (row := 1080) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_2
  simp only [← c66711, ← c66712, ← c66713] at e_1080_2
  have e_1080_3 := arithEq_of_rows h (row := 1080) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_3
  simp only [← c66714, ← c66715, k2, ← c66716] at e_1080_3
  have hr := e_1080_3
  simp only [e_1080_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9320)) := by
  have c66717 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66718 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66719 := (publicBatchWrapper86_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66720 := (publicBatchWrapper86_copies2085 a h).1
  have c66721 := (publicBatchWrapper86_copies2085 a h).2.1
  have c66722 := (publicBatchWrapper86_copies2085 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_4 := arithEq_of_rows h (row := 1080) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_4
  simp only [← c66717, ← c66718, ← c66719] at e_1080_4
  have e_1080_5 := arithEq_of_rows h (row := 1080) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_5
  simp only [← c66720, ← c66721, k2, ← c66722] at e_1080_5
  have hr := e_1080_5
  simp only [e_1080_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9321)) := by
  have c66723 := (publicBatchWrapper86_copies2085 a h).2.2.2.1
  have c66724 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.1
  have c66725 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.1
  have c66726 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.1
  have c66727 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.1
  have c66728 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_6 := arithEq_of_rows h (row := 1080) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_6
  simp only [← c66723, ← c66724, ← c66725] at e_1080_6
  have e_1080_7 := arithEq_of_rows h (row := 1080) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_7
  simp only [← c66726, ← c66727, k2, ← c66728] at e_1080_7
  have hr := e_1080_7
  simp only [e_1080_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9322)) := by
  have c66729 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.1
  have c66730 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66731 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66732 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66733 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66734 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_8 := arithEq_of_rows h (row := 1080) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_8
  simp only [← c66729, ← c66730, ← c66731] at e_1080_8
  have e_1080_9 := arithEq_of_rows h (row := 1080) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_9
  simp only [← c66732, ← c66733, k2, ← c66734] at e_1080_9
  have hr := e_1080_9
  simp only [e_1080_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9323)) := by
  have c66735 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66736 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66737 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66738 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66739 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66740 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_10 := arithEq_of_rows h (row := 1080) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_10
  simp only [← c66735, ← c66736, ← c66737] at e_1080_10
  have e_1080_11 := arithEq_of_rows h (row := 1080) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_11
  simp only [← c66738, ← c66739, k2, ← c66740] at e_1080_11
  have hr := e_1080_11
  simp only [e_1080_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9324)) := by
  have c66741 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66742 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66743 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66744 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66745 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66746 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_12 := arithEq_of_rows h (row := 1080) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_12
  simp only [← c66741, ← c66742, ← c66743] at e_1080_12
  have e_1080_13 := arithEq_of_rows h (row := 1080) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_13
  simp only [← c66744, ← c66745, k2, ← c66746] at e_1080_13
  have hr := e_1080_13
  simp only [e_1080_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9325)) := by
  have c66747 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66748 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66749 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66750 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66751 := (publicBatchWrapper86_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66752 := (publicBatchWrapper86_copies2086 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_14 := arithEq_of_rows h (row := 1080) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_14
  simp only [← c66747, ← c66748, ← c66749] at e_1080_14
  have e_1080_15 := arithEq_of_rows h (row := 1080) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_15
  simp only [← c66750, ← c66751, k2, ← c66752] at e_1080_15
  have hr := e_1080_15
  simp only [e_1080_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9326)) := by
  have c66753 := (publicBatchWrapper86_copies2086 a h).2.1
  have c66754 := (publicBatchWrapper86_copies2086 a h).2.2.1
  have c66755 := (publicBatchWrapper86_copies2086 a h).2.2.2.1
  have c66756 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.1
  have c66757 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.1
  have c66758 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_16 := arithEq_of_rows h (row := 1080) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_16
  simp only [← c66753, ← c66754, ← c66755] at e_1080_16
  have e_1080_17 := arithEq_of_rows h (row := 1080) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_17
  simp only [← c66756, ← c66757, k2, ← c66758] at e_1080_17
  have hr := e_1080_17
  simp only [e_1080_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1080 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9327)) := by
  have c66759 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.1
  have c66760 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.1
  have c66761 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.1
  have c66762 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66763 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66764 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1080_18 := arithEq_of_rows h (row := 1080) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_18
  simp only [← c66759, ← c66760, ← c66761] at e_1080_18
  have e_1080_19 := arithEq_of_rows h (row := 1080) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_19
  simp only [← c66762, ← c66763, k2, ← c66764] at e_1080_19
  have hr := e_1080_19
  simp only [e_1080_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9328)) := by
  have c66765 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66766 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66767 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66768 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66769 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66770 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_0 := arithEq_of_rows h (row := 1081) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_0
  simp only [← c66765, ← c66766, ← c66767] at e_1081_0
  have e_1081_1 := arithEq_of_rows h (row := 1081) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_1
  simp only [← c66768, ← c66769, k2, ← c66770] at e_1081_1
  have hr := e_1081_1
  simp only [e_1081_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9329)) := by
  have c66771 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66772 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66773 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66774 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66775 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66776 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_2 := arithEq_of_rows h (row := 1081) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_2
  simp only [← c66771, ← c66772, ← c66773] at e_1081_2
  have e_1081_3 := arithEq_of_rows h (row := 1081) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_3
  simp only [← c66774, ← c66775, k2, ← c66776] at e_1081_3
  have hr := e_1081_3
  simp only [e_1081_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9330)) := by
  have c66777 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66778 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66779 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66780 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66781 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66782 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_4 := arithEq_of_rows h (row := 1081) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_4
  simp only [← c66777, ← c66778, ← c66779] at e_1081_4
  have e_1081_5 := arithEq_of_rows h (row := 1081) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_5
  simp only [← c66780, ← c66781, k2, ← c66782] at e_1081_5
  have hr := e_1081_5
  simp only [e_1081_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9331)) := by
  have c66783 := (publicBatchWrapper86_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66784 := (publicBatchWrapper86_copies2087 a h).1
  have c66785 := (publicBatchWrapper86_copies2087 a h).2.1
  have c66786 := (publicBatchWrapper86_copies2087 a h).2.2.1
  have c66787 := (publicBatchWrapper86_copies2087 a h).2.2.2.1
  have c66788 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_6 := arithEq_of_rows h (row := 1081) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_6
  simp only [← c66783, ← c66784, ← c66785] at e_1081_6
  have e_1081_7 := arithEq_of_rows h (row := 1081) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_7
  simp only [← c66786, ← c66787, k2, ← c66788] at e_1081_7
  have hr := e_1081_7
  simp only [e_1081_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9332)) := by
  have c66789 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.1
  have c66790 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.1
  have c66791 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.1
  have c66792 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.1
  have c66793 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.1
  have c66794 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_8 := arithEq_of_rows h (row := 1081) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_8
  simp only [← c66789, ← c66790, ← c66791] at e_1081_8
  have e_1081_9 := arithEq_of_rows h (row := 1081) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_9
  simp only [← c66792, ← c66793, k2, ← c66794] at e_1081_9
  have hr := e_1081_9
  simp only [e_1081_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9333)) := by
  have c66795 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66796 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66797 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66798 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66799 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66800 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_10 := arithEq_of_rows h (row := 1081) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_10
  simp only [← c66795, ← c66796, ← c66797] at e_1081_10
  have e_1081_11 := arithEq_of_rows h (row := 1081) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_11
  simp only [← c66798, ← c66799, k2, ← c66800] at e_1081_11
  have hr := e_1081_11
  simp only [e_1081_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9334)) := by
  have c66801 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66802 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66803 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66804 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66805 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66806 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_12 := arithEq_of_rows h (row := 1081) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_12
  simp only [← c66801, ← c66802, ← c66803] at e_1081_12
  have e_1081_13 := arithEq_of_rows h (row := 1081) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_13
  simp only [← c66804, ← c66805, k2, ← c66806] at e_1081_13
  have hr := e_1081_13
  simp only [e_1081_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9335)) := by
  have c66807 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66808 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66809 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66810 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66811 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66812 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_14 := arithEq_of_rows h (row := 1081) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_14
  simp only [← c66807, ← c66808, ← c66809] at e_1081_14
  have e_1081_15 := arithEq_of_rows h (row := 1081) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_15
  simp only [← c66810, ← c66811, k2, ← c66812] at e_1081_15
  have hr := e_1081_15
  simp only [e_1081_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9336)) := by
  have c66813 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66814 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66815 := (publicBatchWrapper86_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66816 := (publicBatchWrapper86_copies2088 a h).1
  have c66817 := (publicBatchWrapper86_copies2088 a h).2.1
  have c66818 := (publicBatchWrapper86_copies2088 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_16 := arithEq_of_rows h (row := 1081) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_16
  simp only [← c66813, ← c66814, ← c66815] at e_1081_16
  have e_1081_17 := arithEq_of_rows h (row := 1081) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_17
  simp only [← c66816, ← c66817, k2, ← c66818] at e_1081_17
  have hr := e_1081_17
  simp only [e_1081_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1081 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9337)) := by
  have c66819 := (publicBatchWrapper86_copies2088 a h).2.2.2.1
  have c66820 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.1
  have c66821 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.1
  have c66822 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.1
  have c66823 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.1
  have c66824 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1081_18 := arithEq_of_rows h (row := 1081) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_18
  simp only [← c66819, ← c66820, ← c66821] at e_1081_18
  have e_1081_19 := arithEq_of_rows h (row := 1081) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_19
  simp only [← c66822, ← c66823, k2, ← c66824] at e_1081_19
  have hr := e_1081_19
  simp only [e_1081_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9338)) := by
  have c66825 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.1
  have c66826 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66827 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66828 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66829 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66830 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_0 := arithEq_of_rows h (row := 1082) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_0
  simp only [← c66825, ← c66826, ← c66827] at e_1082_0
  have e_1082_1 := arithEq_of_rows h (row := 1082) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_1
  simp only [← c66828, ← c66829, k2, ← c66830] at e_1082_1
  have hr := e_1082_1
  simp only [e_1082_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9339)) := by
  have c66831 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66832 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66833 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66834 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66835 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66836 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_2 := arithEq_of_rows h (row := 1082) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_2
  simp only [← c66831, ← c66832, ← c66833] at e_1082_2
  have e_1082_3 := arithEq_of_rows h (row := 1082) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_3
  simp only [← c66834, ← c66835, k2, ← c66836] at e_1082_3
  have hr := e_1082_3
  simp only [e_1082_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9340)) := by
  have c66837 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66838 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66839 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66840 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66841 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66842 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_4 := arithEq_of_rows h (row := 1082) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_4
  simp only [← c66837, ← c66838, ← c66839] at e_1082_4
  have e_1082_5 := arithEq_of_rows h (row := 1082) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_5
  simp only [← c66840, ← c66841, k2, ← c66842] at e_1082_5
  have hr := e_1082_5
  simp only [e_1082_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9341)) := by
  have c66843 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66844 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66845 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66846 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66847 := (publicBatchWrapper86_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66848 := (publicBatchWrapper86_copies2089 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_6 := arithEq_of_rows h (row := 1082) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_6
  simp only [← c66843, ← c66844, ← c66845] at e_1082_6
  have e_1082_7 := arithEq_of_rows h (row := 1082) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_7
  simp only [← c66846, ← c66847, k2, ← c66848] at e_1082_7
  have hr := e_1082_7
  simp only [e_1082_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9342)) := by
  have c66849 := (publicBatchWrapper86_copies2089 a h).2.1
  have c66850 := (publicBatchWrapper86_copies2089 a h).2.2.1
  have c66851 := (publicBatchWrapper86_copies2089 a h).2.2.2.1
  have c66852 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.1
  have c66853 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.1
  have c66854 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_8 := arithEq_of_rows h (row := 1082) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_8
  simp only [← c66849, ← c66850, ← c66851] at e_1082_8
  have e_1082_9 := arithEq_of_rows h (row := 1082) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_9
  simp only [← c66852, ← c66853, k2, ← c66854] at e_1082_9
  have hr := e_1082_9
  simp only [e_1082_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9343)) := by
  have c66855 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.1
  have c66856 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.1
  have c66857 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.1
  have c66858 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66859 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66860 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_10 := arithEq_of_rows h (row := 1082) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_10
  simp only [← c66855, ← c66856, ← c66857] at e_1082_10
  have e_1082_11 := arithEq_of_rows h (row := 1082) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_11
  simp only [← c66858, ← c66859, k2, ← c66860] at e_1082_11
  have hr := e_1082_11
  simp only [e_1082_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9344)) := by
  have c66861 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66862 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66863 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66864 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66865 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66866 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_12 := arithEq_of_rows h (row := 1082) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_12
  simp only [← c66861, ← c66862, ← c66863] at e_1082_12
  have e_1082_13 := arithEq_of_rows h (row := 1082) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_13
  simp only [← c66864, ← c66865, k2, ← c66866] at e_1082_13
  have hr := e_1082_13
  simp only [e_1082_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9345)) := by
  have c66867 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66868 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66869 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66870 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66871 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66872 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_14 := arithEq_of_rows h (row := 1082) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_14
  simp only [← c66867, ← c66868, ← c66869] at e_1082_14
  have e_1082_15 := arithEq_of_rows h (row := 1082) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_15
  simp only [← c66870, ← c66871, k2, ← c66872] at e_1082_15
  have hr := e_1082_15
  simp only [e_1082_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9346)) := by
  have c66873 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66874 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66875 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66876 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66877 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66878 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_16 := arithEq_of_rows h (row := 1082) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_16
  simp only [← c66873, ← c66874, ← c66875] at e_1082_16
  have e_1082_17 := arithEq_of_rows h (row := 1082) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_17
  simp only [← c66876, ← c66877, k2, ← c66878] at e_1082_17
  have hr := e_1082_17
  simp only [e_1082_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1082 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9347)) := by
  have c66879 := (publicBatchWrapper86_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66880 := (publicBatchWrapper86_copies2090 a h).1
  have c66881 := (publicBatchWrapper86_copies2090 a h).2.1
  have c66882 := (publicBatchWrapper86_copies2090 a h).2.2.1
  have c66883 := (publicBatchWrapper86_copies2090 a h).2.2.2.1
  have c66884 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1082_18 := arithEq_of_rows h (row := 1082) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_18
  simp only [← c66879, ← c66880, ← c66881] at e_1082_18
  have e_1082_19 := arithEq_of_rows h (row := 1082) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_19
  simp only [← c66882, ← c66883, k2, ← c66884] at e_1082_19
  have hr := e_1082_19
  simp only [e_1082_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9348)) := by
  have c66885 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.1
  have c66886 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.1
  have c66887 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.1
  have c66888 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.1
  have c66889 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.1
  have c66890 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_0 := arithEq_of_rows h (row := 1083) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_0
  simp only [← c66885, ← c66886, ← c66887] at e_1083_0
  have e_1083_1 := arithEq_of_rows h (row := 1083) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_1
  simp only [← c66888, ← c66889, k2, ← c66890] at e_1083_1
  have hr := e_1083_1
  simp only [e_1083_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9349)) := by
  have c66891 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66892 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66893 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66894 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66895 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66896 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_2 := arithEq_of_rows h (row := 1083) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_2
  simp only [← c66891, ← c66892, ← c66893] at e_1083_2
  have e_1083_3 := arithEq_of_rows h (row := 1083) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_3
  simp only [← c66894, ← c66895, k2, ← c66896] at e_1083_3
  have hr := e_1083_3
  simp only [e_1083_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9350)) := by
  have c66897 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66898 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66899 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66900 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66901 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66902 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_4 := arithEq_of_rows h (row := 1083) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_4
  simp only [← c66897, ← c66898, ← c66899] at e_1083_4
  have e_1083_5 := arithEq_of_rows h (row := 1083) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_5
  simp only [← c66900, ← c66901, k2, ← c66902] at e_1083_5
  have hr := e_1083_5
  simp only [e_1083_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9351)) := by
  have c66903 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66904 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66905 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66906 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66907 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66908 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_6 := arithEq_of_rows h (row := 1083) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_6
  simp only [← c66903, ← c66904, ← c66905] at e_1083_6
  have e_1083_7 := arithEq_of_rows h (row := 1083) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_7
  simp only [← c66906, ← c66907, k2, ← c66908] at e_1083_7
  have hr := e_1083_7
  simp only [e_1083_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9352)) := by
  have c66909 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66910 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66911 := (publicBatchWrapper86_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66912 := (publicBatchWrapper86_copies2091 a h).1
  have c66913 := (publicBatchWrapper86_copies2091 a h).2.1
  have c66914 := (publicBatchWrapper86_copies2091 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_8 := arithEq_of_rows h (row := 1083) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_8
  simp only [← c66909, ← c66910, ← c66911] at e_1083_8
  have e_1083_9 := arithEq_of_rows h (row := 1083) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_9
  simp only [← c66912, ← c66913, k2, ← c66914] at e_1083_9
  have hr := e_1083_9
  simp only [e_1083_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9353)) := by
  have c66915 := (publicBatchWrapper86_copies2091 a h).2.2.2.1
  have c66916 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.1
  have c66917 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.1
  have c66918 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.1
  have c66919 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.1
  have c66920 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_10 := arithEq_of_rows h (row := 1083) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_10
  simp only [← c66915, ← c66916, ← c66917] at e_1083_10
  have e_1083_11 := arithEq_of_rows h (row := 1083) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_11
  simp only [← c66918, ← c66919, k2, ← c66920] at e_1083_11
  have hr := e_1083_11
  simp only [e_1083_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9354)) := by
  have c66921 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.1
  have c66922 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66923 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66924 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66925 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66926 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_12 := arithEq_of_rows h (row := 1083) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_12
  simp only [← c66921, ← c66922, ← c66923] at e_1083_12
  have e_1083_13 := arithEq_of_rows h (row := 1083) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_13
  simp only [← c66924, ← c66925, k2, ← c66926] at e_1083_13
  have hr := e_1083_13
  simp only [e_1083_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9355)) := by
  have c66927 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66928 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66929 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66930 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66931 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66932 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_14 := arithEq_of_rows h (row := 1083) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_14
  simp only [← c66927, ← c66928, ← c66929] at e_1083_14
  have e_1083_15 := arithEq_of_rows h (row := 1083) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_15
  simp only [← c66930, ← c66931, k2, ← c66932] at e_1083_15
  have hr := e_1083_15
  simp only [e_1083_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9356)) := by
  have c66933 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66934 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66935 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66936 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66937 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66938 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_16 := arithEq_of_rows h (row := 1083) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_16
  simp only [← c66933, ← c66934, ← c66935] at e_1083_16
  have e_1083_17 := arithEq_of_rows h (row := 1083) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_17
  simp only [← c66936, ← c66937, k2, ← c66938] at e_1083_17
  have hr := e_1083_17
  simp only [e_1083_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1083 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9357)) := by
  have c66939 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66940 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66941 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66942 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66943 := (publicBatchWrapper86_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66944 := (publicBatchWrapper86_copies2092 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1083_18 := arithEq_of_rows h (row := 1083) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_18
  simp only [← c66939, ← c66940, ← c66941] at e_1083_18
  have e_1083_19 := arithEq_of_rows h (row := 1083) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_19
  simp only [← c66942, ← c66943, k2, ← c66944] at e_1083_19
  have hr := e_1083_19
  simp only [e_1083_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9358)) := by
  have c66945 := (publicBatchWrapper86_copies2092 a h).2.1
  have c66946 := (publicBatchWrapper86_copies2092 a h).2.2.1
  have c66947 := (publicBatchWrapper86_copies2092 a h).2.2.2.1
  have c66948 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.1
  have c66949 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.1
  have c66950 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_0 := arithEq_of_rows h (row := 1084) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_0
  simp only [← c66945, ← c66946, ← c66947] at e_1084_0
  have e_1084_1 := arithEq_of_rows h (row := 1084) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_1
  simp only [← c66948, ← c66949, k2, ← c66950] at e_1084_1
  have hr := e_1084_1
  simp only [e_1084_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9359)) := by
  have c66951 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.1
  have c66952 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.1
  have c66953 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.1
  have c66954 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66955 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66956 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_2 := arithEq_of_rows h (row := 1084) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_2
  simp only [← c66951, ← c66952, ← c66953] at e_1084_2
  have e_1084_3 := arithEq_of_rows h (row := 1084) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_3
  simp only [← c66954, ← c66955, k2, ← c66956] at e_1084_3
  have hr := e_1084_3
  simp only [e_1084_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9360)) := by
  have c66957 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66958 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66959 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66960 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66961 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66962 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_4 := arithEq_of_rows h (row := 1084) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_4
  simp only [← c66957, ← c66958, ← c66959] at e_1084_4
  have e_1084_5 := arithEq_of_rows h (row := 1084) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_5
  simp only [← c66960, ← c66961, k2, ← c66962] at e_1084_5
  have hr := e_1084_5
  simp only [e_1084_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9361)) := by
  have c66963 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66964 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66965 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66966 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66967 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66968 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_6 := arithEq_of_rows h (row := 1084) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_6
  simp only [← c66963, ← c66964, ← c66965] at e_1084_6
  have e_1084_7 := arithEq_of_rows h (row := 1084) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_7
  simp only [← c66966, ← c66967, k2, ← c66968] at e_1084_7
  have hr := e_1084_7
  simp only [e_1084_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9362)) := by
  have c66969 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66970 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66971 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66972 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66973 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66974 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_8 := arithEq_of_rows h (row := 1084) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_8
  simp only [← c66969, ← c66970, ← c66971] at e_1084_8
  have e_1084_9 := arithEq_of_rows h (row := 1084) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_9
  simp only [← c66972, ← c66973, k2, ← c66974] at e_1084_9
  have hr := e_1084_9
  simp only [e_1084_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9363)) := by
  have c66975 := (publicBatchWrapper86_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66976 := (publicBatchWrapper86_copies2093 a h).1
  have c66977 := (publicBatchWrapper86_copies2093 a h).2.1
  have c66978 := (publicBatchWrapper86_copies2093 a h).2.2.1
  have c66979 := (publicBatchWrapper86_copies2093 a h).2.2.2.1
  have c66980 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_10 := arithEq_of_rows h (row := 1084) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_10
  simp only [← c66975, ← c66976, ← c66977] at e_1084_10
  have e_1084_11 := arithEq_of_rows h (row := 1084) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_11
  simp only [← c66978, ← c66979, k2, ← c66980] at e_1084_11
  have hr := e_1084_11
  simp only [e_1084_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9364)) := by
  have c66981 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.1
  have c66982 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.1
  have c66983 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.1
  have c66984 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.1
  have c66985 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.1
  have c66986 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_12 := arithEq_of_rows h (row := 1084) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_12
  simp only [← c66981, ← c66982, ← c66983] at e_1084_12
  have e_1084_13 := arithEq_of_rows h (row := 1084) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_13
  simp only [← c66984, ← c66985, k2, ← c66986] at e_1084_13
  have hr := e_1084_13
  simp only [e_1084_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9365)) := by
  have c66987 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66988 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66989 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66990 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66991 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66992 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_14 := arithEq_of_rows h (row := 1084) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_14
  simp only [← c66987, ← c66988, ← c66989] at e_1084_14
  have e_1084_15 := arithEq_of_rows h (row := 1084) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_15
  simp only [← c66990, ← c66991, k2, ← c66992] at e_1084_15
  have hr := e_1084_15
  simp only [e_1084_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9366)) := by
  have c66993 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66994 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66995 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66996 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66997 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66998 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_16 := arithEq_of_rows h (row := 1084) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_16
  simp only [← c66993, ← c66994, ← c66995] at e_1084_16
  have e_1084_17 := arithEq_of_rows h (row := 1084) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_17
  simp only [← c66996, ← c66997, k2, ← c66998] at e_1084_17
  have hr := e_1084_17
  simp only [e_1084_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1084 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9367)) := by
  have c66999 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67000 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67001 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67002 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67003 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67004 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1084_18 := arithEq_of_rows h (row := 1084) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_18
  simp only [← c66999, ← c67000, ← c67001] at e_1084_18
  have e_1084_19 := arithEq_of_rows h (row := 1084) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_19
  simp only [← c67002, ← c67003, k2, ← c67004] at e_1084_19
  have hr := e_1084_19
  simp only [e_1084_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9368)) := by
  have c67005 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67006 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67007 := (publicBatchWrapper86_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67008 := (publicBatchWrapper86_copies2094 a h).1
  have c67009 := (publicBatchWrapper86_copies2094 a h).2.1
  have c67010 := (publicBatchWrapper86_copies2094 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_0 := arithEq_of_rows h (row := 1085) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_0
  simp only [← c67005, ← c67006, ← c67007] at e_1085_0
  have e_1085_1 := arithEq_of_rows h (row := 1085) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_1
  simp only [← c67008, ← c67009, k2, ← c67010] at e_1085_1
  have hr := e_1085_1
  simp only [e_1085_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9369)) := by
  have c67011 := (publicBatchWrapper86_copies2094 a h).2.2.2.1
  have c67012 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.1
  have c67013 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.1
  have c67014 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.1
  have c67015 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.1
  have c67016 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_2 := arithEq_of_rows h (row := 1085) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_2
  simp only [← c67011, ← c67012, ← c67013] at e_1085_2
  have e_1085_3 := arithEq_of_rows h (row := 1085) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_3
  simp only [← c67014, ← c67015, k2, ← c67016] at e_1085_3
  have hr := e_1085_3
  simp only [e_1085_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9410)) := by
  have c67017 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.1
  have c67018 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67019 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67020 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67021 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67022 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_4 := arithEq_of_rows h (row := 1085) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_4
  simp only [← c67017, ← c67018, ← c67019] at e_1085_4
  have e_1085_5 := arithEq_of_rows h (row := 1085) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_5
  simp only [← c67020, ← c67021, k2, ← c67022] at e_1085_5
  have hr := e_1085_5
  simp only [e_1085_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9411)) := by
  have c67023 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67024 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67025 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67026 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67027 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67028 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_6 := arithEq_of_rows h (row := 1085) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_6
  simp only [← c67023, ← c67024, ← c67025] at e_1085_6
  have e_1085_7 := arithEq_of_rows h (row := 1085) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_7
  simp only [← c67026, ← c67027, k2, ← c67028] at e_1085_7
  have hr := e_1085_7
  simp only [e_1085_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9412)) := by
  have c67029 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67030 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67031 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67032 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67033 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67034 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_8 := arithEq_of_rows h (row := 1085) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_8
  simp only [← c67029, ← c67030, ← c67031] at e_1085_8
  have e_1085_9 := arithEq_of_rows h (row := 1085) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_9
  simp only [← c67032, ← c67033, k2, ← c67034] at e_1085_9
  have hr := e_1085_9
  simp only [e_1085_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9413)) := by
  have c67035 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67036 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67037 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67038 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67039 := (publicBatchWrapper86_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67040 := (publicBatchWrapper86_copies2095 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_10 := arithEq_of_rows h (row := 1085) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_10
  simp only [← c67035, ← c67036, ← c67037] at e_1085_10
  have e_1085_11 := arithEq_of_rows h (row := 1085) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_11
  simp only [← c67038, ← c67039, k2, ← c67040] at e_1085_11
  have hr := e_1085_11
  simp only [e_1085_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9414)) := by
  have c67041 := (publicBatchWrapper86_copies2095 a h).2.1
  have c67042 := (publicBatchWrapper86_copies2095 a h).2.2.1
  have c67043 := (publicBatchWrapper86_copies2095 a h).2.2.2.1
  have c67044 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.1
  have c67045 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.1
  have c67046 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_12 := arithEq_of_rows h (row := 1085) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_12
  simp only [← c67041, ← c67042, ← c67043] at e_1085_12
  have e_1085_13 := arithEq_of_rows h (row := 1085) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_13
  simp only [← c67044, ← c67045, k2, ← c67046] at e_1085_13
  have hr := e_1085_13
  simp only [e_1085_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9415)) := by
  have c67047 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.1
  have c67048 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.1
  have c67049 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.1
  have c67050 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67051 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67052 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_14 := arithEq_of_rows h (row := 1085) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_14
  simp only [← c67047, ← c67048, ← c67049] at e_1085_14
  have e_1085_15 := arithEq_of_rows h (row := 1085) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_15
  simp only [← c67050, ← c67051, k2, ← c67052] at e_1085_15
  have hr := e_1085_15
  simp only [e_1085_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9416)) := by
  have c67053 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67054 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67055 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67056 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67057 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67058 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_16 := arithEq_of_rows h (row := 1085) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_16
  simp only [← c67053, ← c67054, ← c67055] at e_1085_16
  have e_1085_17 := arithEq_of_rows h (row := 1085) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_17
  simp only [← c67056, ← c67057, k2, ← c67058] at e_1085_17
  have hr := e_1085_17
  simp only [e_1085_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1085 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9417)) := by
  have c67059 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67060 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67061 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67062 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67063 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67064 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1085_18 := arithEq_of_rows h (row := 1085) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_18
  simp only [← c67059, ← c67060, ← c67061] at e_1085_18
  have e_1085_19 := arithEq_of_rows h (row := 1085) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_19
  simp only [← c67062, ← c67063, k2, ← c67064] at e_1085_19
  have hr := e_1085_19
  simp only [e_1085_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9418)) := by
  have c67065 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67066 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67067 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67068 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67069 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67070 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_0 := arithEq_of_rows h (row := 1086) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_0
  simp only [← c67065, ← c67066, ← c67067] at e_1086_0
  have e_1086_1 := arithEq_of_rows h (row := 1086) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_1
  simp only [← c67068, ← c67069, k2, ← c67070] at e_1086_1
  have hr := e_1086_1
  simp only [e_1086_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9419)) := by
  have c67071 := (publicBatchWrapper86_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67072 := (publicBatchWrapper86_copies2096 a h).1
  have c67073 := (publicBatchWrapper86_copies2096 a h).2.1
  have c67074 := (publicBatchWrapper86_copies2096 a h).2.2.1
  have c67075 := (publicBatchWrapper86_copies2096 a h).2.2.2.1
  have c67076 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_2 := arithEq_of_rows h (row := 1086) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_2
  simp only [← c67071, ← c67072, ← c67073] at e_1086_2
  have e_1086_3 := arithEq_of_rows h (row := 1086) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_3
  simp only [← c67074, ← c67075, k2, ← c67076] at e_1086_3
  have hr := e_1086_3
  simp only [e_1086_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9420)) := by
  have c67077 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.1
  have c67078 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.1
  have c67079 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.1
  have c67080 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.1
  have c67081 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.1
  have c67082 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_4 := arithEq_of_rows h (row := 1086) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_4
  simp only [← c67077, ← c67078, ← c67079] at e_1086_4
  have e_1086_5 := arithEq_of_rows h (row := 1086) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_5
  simp only [← c67080, ← c67081, k2, ← c67082] at e_1086_5
  have hr := e_1086_5
  simp only [e_1086_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9421)) := by
  have c67083 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67084 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67085 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67086 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67087 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67088 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_6 := arithEq_of_rows h (row := 1086) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_6
  simp only [← c67083, ← c67084, ← c67085] at e_1086_6
  have e_1086_7 := arithEq_of_rows h (row := 1086) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_7
  simp only [← c67086, ← c67087, k2, ← c67088] at e_1086_7
  have hr := e_1086_7
  simp only [e_1086_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9422)) := by
  have c67089 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67090 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67091 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67092 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67093 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67094 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_8 := arithEq_of_rows h (row := 1086) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_8
  simp only [← c67089, ← c67090, ← c67091] at e_1086_8
  have e_1086_9 := arithEq_of_rows h (row := 1086) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_9
  simp only [← c67092, ← c67093, k2, ← c67094] at e_1086_9
  have hr := e_1086_9
  simp only [e_1086_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9423)) := by
  have c67095 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67096 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67097 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67098 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67099 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67100 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_10 := arithEq_of_rows h (row := 1086) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_10
  simp only [← c67095, ← c67096, ← c67097] at e_1086_10
  have e_1086_11 := arithEq_of_rows h (row := 1086) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_11
  simp only [← c67098, ← c67099, k2, ← c67100] at e_1086_11
  have hr := e_1086_11
  simp only [e_1086_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9424)) := by
  have c67101 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67102 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67103 := (publicBatchWrapper86_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67104 := (publicBatchWrapper86_copies2097 a h).1
  have c67105 := (publicBatchWrapper86_copies2097 a h).2.1
  have c67106 := (publicBatchWrapper86_copies2097 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_12 := arithEq_of_rows h (row := 1086) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_12
  simp only [← c67101, ← c67102, ← c67103] at e_1086_12
  have e_1086_13 := arithEq_of_rows h (row := 1086) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_13
  simp only [← c67104, ← c67105, k2, ← c67106] at e_1086_13
  have hr := e_1086_13
  simp only [e_1086_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9425)) := by
  have c67107 := (publicBatchWrapper86_copies2097 a h).2.2.2.1
  have c67108 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.1
  have c67109 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.1
  have c67110 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.1
  have c67111 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.1
  have c67112 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_14 := arithEq_of_rows h (row := 1086) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_14
  simp only [← c67107, ← c67108, ← c67109] at e_1086_14
  have e_1086_15 := arithEq_of_rows h (row := 1086) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_15
  simp only [← c67110, ← c67111, k2, ← c67112] at e_1086_15
  have hr := e_1086_15
  simp only [e_1086_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9426)) := by
  have c67113 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.1
  have c67114 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67115 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67116 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67117 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67118 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_16 := arithEq_of_rows h (row := 1086) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_16
  simp only [← c67113, ← c67114, ← c67115] at e_1086_16
  have e_1086_17 := arithEq_of_rows h (row := 1086) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_17
  simp only [← c67116, ← c67117, k2, ← c67118] at e_1086_17
  have hr := e_1086_17
  simp only [e_1086_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1086 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9427)) := by
  have c67119 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67120 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67121 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67122 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67123 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67124 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1086_18 := arithEq_of_rows h (row := 1086) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_18
  simp only [← c67119, ← c67120, ← c67121] at e_1086_18
  have e_1086_19 := arithEq_of_rows h (row := 1086) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_19
  simp only [← c67122, ← c67123, k2, ← c67124] at e_1086_19
  have hr := e_1086_19
  simp only [e_1086_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9428)) := by
  have c67125 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67126 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67127 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67128 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67129 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67130 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_0 := arithEq_of_rows h (row := 1087) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_0
  simp only [← c67125, ← c67126, ← c67127] at e_1087_0
  have e_1087_1 := arithEq_of_rows h (row := 1087) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_1
  simp only [← c67128, ← c67129, k2, ← c67130] at e_1087_1
  have hr := e_1087_1
  simp only [e_1087_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9429)) := by
  have c67131 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67132 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67133 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67134 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67135 := (publicBatchWrapper86_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67136 := (publicBatchWrapper86_copies2098 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_2 := arithEq_of_rows h (row := 1087) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_2
  simp only [← c67131, ← c67132, ← c67133] at e_1087_2
  have e_1087_3 := arithEq_of_rows h (row := 1087) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_3
  simp only [← c67134, ← c67135, k2, ← c67136] at e_1087_3
  have hr := e_1087_3
  simp only [e_1087_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9430)) := by
  have c67137 := (publicBatchWrapper86_copies2098 a h).2.1
  have c67138 := (publicBatchWrapper86_copies2098 a h).2.2.1
  have c67139 := (publicBatchWrapper86_copies2098 a h).2.2.2.1
  have c67140 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.1
  have c67141 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.1
  have c67142 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_4 := arithEq_of_rows h (row := 1087) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_4
  simp only [← c67137, ← c67138, ← c67139] at e_1087_4
  have e_1087_5 := arithEq_of_rows h (row := 1087) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_5
  simp only [← c67140, ← c67141, k2, ← c67142] at e_1087_5
  have hr := e_1087_5
  simp only [e_1087_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9431)) := by
  have c67143 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.1
  have c67144 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.1
  have c67145 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.1
  have c67146 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67147 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67148 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_6 := arithEq_of_rows h (row := 1087) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_6
  simp only [← c67143, ← c67144, ← c67145] at e_1087_6
  have e_1087_7 := arithEq_of_rows h (row := 1087) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_7
  simp only [← c67146, ← c67147, k2, ← c67148] at e_1087_7
  have hr := e_1087_7
  simp only [e_1087_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9432)) := by
  have c67149 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67150 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67151 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67152 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67153 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67154 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_8 := arithEq_of_rows h (row := 1087) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_8
  simp only [← c67149, ← c67150, ← c67151] at e_1087_8
  have e_1087_9 := arithEq_of_rows h (row := 1087) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_9
  simp only [← c67152, ← c67153, k2, ← c67154] at e_1087_9
  have hr := e_1087_9
  simp only [e_1087_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9433)) := by
  have c67155 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67156 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67157 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67158 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67159 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67160 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_10 := arithEq_of_rows h (row := 1087) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_10
  simp only [← c67155, ← c67156, ← c67157] at e_1087_10
  have e_1087_11 := arithEq_of_rows h (row := 1087) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_11
  simp only [← c67158, ← c67159, k2, ← c67160] at e_1087_11
  have hr := e_1087_11
  simp only [e_1087_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9434)) := by
  have c67161 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67162 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67163 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67164 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67165 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67166 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_12 := arithEq_of_rows h (row := 1087) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_12
  simp only [← c67161, ← c67162, ← c67163] at e_1087_12
  have e_1087_13 := arithEq_of_rows h (row := 1087) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_13
  simp only [← c67164, ← c67165, k2, ← c67166] at e_1087_13
  have hr := e_1087_13
  simp only [e_1087_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9435)) := by
  have c67167 := (publicBatchWrapper86_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67168 := (publicBatchWrapper86_copies2099 a h).1
  have c67169 := (publicBatchWrapper86_copies2099 a h).2.1
  have c67170 := (publicBatchWrapper86_copies2099 a h).2.2.1
  have c67171 := (publicBatchWrapper86_copies2099 a h).2.2.2.1
  have c67172 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_14 := arithEq_of_rows h (row := 1087) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_14
  simp only [← c67167, ← c67168, ← c67169] at e_1087_14
  have e_1087_15 := arithEq_of_rows h (row := 1087) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_15
  simp only [← c67170, ← c67171, k2, ← c67172] at e_1087_15
  have hr := e_1087_15
  simp only [e_1087_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9436)) := by
  have c67173 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.1
  have c67174 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.1
  have c67175 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.1
  have c67176 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.1
  have c67177 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.1
  have c67178 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_16 := arithEq_of_rows h (row := 1087) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_16
  simp only [← c67173, ← c67174, ← c67175] at e_1087_16
  have e_1087_17 := arithEq_of_rows h (row := 1087) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_17
  simp only [← c67176, ← c67177, k2, ← c67178] at e_1087_17
  have hr := e_1087_17
  simp only [e_1087_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1087 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9437)) := by
  have c67179 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67180 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67181 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67182 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67183 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67184 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1087_18 := arithEq_of_rows h (row := 1087) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_18
  simp only [← c67179, ← c67180, ← c67181] at e_1087_18
  have e_1087_19 := arithEq_of_rows h (row := 1087) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_19
  simp only [← c67182, ← c67183, k2, ← c67184] at e_1087_19
  have hr := e_1087_19
  simp only [e_1087_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9438)) := by
  have c67185 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67186 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67187 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67188 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67189 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67190 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_0 := arithEq_of_rows h (row := 1088) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_0
  simp only [← c67185, ← c67186, ← c67187] at e_1088_0
  have e_1088_1 := arithEq_of_rows h (row := 1088) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_1
  simp only [← c67188, ← c67189, k2, ← c67190] at e_1088_1
  have hr := e_1088_1
  simp only [e_1088_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9439)) := by
  have c67191 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67192 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67193 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67194 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67195 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67196 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_2 := arithEq_of_rows h (row := 1088) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_2
  simp only [← c67191, ← c67192, ← c67193] at e_1088_2
  have e_1088_3 := arithEq_of_rows h (row := 1088) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_3
  simp only [← c67194, ← c67195, k2, ← c67196] at e_1088_3
  have hr := e_1088_3
  simp only [e_1088_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9440)) := by
  have c67197 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67198 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67199 := (publicBatchWrapper86_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67200 := (publicBatchWrapper86_copies2100 a h).1
  have c67201 := (publicBatchWrapper86_copies2100 a h).2.1
  have c67202 := (publicBatchWrapper86_copies2100 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_4 := arithEq_of_rows h (row := 1088) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_4
  simp only [← c67197, ← c67198, ← c67199] at e_1088_4
  have e_1088_5 := arithEq_of_rows h (row := 1088) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_5
  simp only [← c67200, ← c67201, k2, ← c67202] at e_1088_5
  have hr := e_1088_5
  simp only [e_1088_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9441)) := by
  have c67203 := (publicBatchWrapper86_copies2100 a h).2.2.2.1
  have c67204 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.1
  have c67205 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.1
  have c67206 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.1
  have c67207 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.1
  have c67208 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_6 := arithEq_of_rows h (row := 1088) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_6
  simp only [← c67203, ← c67204, ← c67205] at e_1088_6
  have e_1088_7 := arithEq_of_rows h (row := 1088) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_7
  simp only [← c67206, ← c67207, k2, ← c67208] at e_1088_7
  have hr := e_1088_7
  simp only [e_1088_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9442)) := by
  have c67209 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.1
  have c67210 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67211 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67212 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67213 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67214 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_8 := arithEq_of_rows h (row := 1088) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_8
  simp only [← c67209, ← c67210, ← c67211] at e_1088_8
  have e_1088_9 := arithEq_of_rows h (row := 1088) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_9
  simp only [← c67212, ← c67213, k2, ← c67214] at e_1088_9
  have hr := e_1088_9
  simp only [e_1088_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9443)) := by
  have c67215 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67216 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67217 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67218 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67219 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67220 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_10 := arithEq_of_rows h (row := 1088) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_10
  simp only [← c67215, ← c67216, ← c67217] at e_1088_10
  have e_1088_11 := arithEq_of_rows h (row := 1088) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_11
  simp only [← c67218, ← c67219, k2, ← c67220] at e_1088_11
  have hr := e_1088_11
  simp only [e_1088_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9444)) := by
  have c67221 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67222 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67223 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67224 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67225 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67226 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_12 := arithEq_of_rows h (row := 1088) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_12
  simp only [← c67221, ← c67222, ← c67223] at e_1088_12
  have e_1088_13 := arithEq_of_rows h (row := 1088) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_13
  simp only [← c67224, ← c67225, k2, ← c67226] at e_1088_13
  have hr := e_1088_13
  simp only [e_1088_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9445)) := by
  have c67227 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67228 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67229 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67230 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67231 := (publicBatchWrapper86_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67232 := (publicBatchWrapper86_copies2101 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_14 := arithEq_of_rows h (row := 1088) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_14
  simp only [← c67227, ← c67228, ← c67229] at e_1088_14
  have e_1088_15 := arithEq_of_rows h (row := 1088) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_15
  simp only [← c67230, ← c67231, k2, ← c67232] at e_1088_15
  have hr := e_1088_15
  simp only [e_1088_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9446)) := by
  have c67233 := (publicBatchWrapper86_copies2101 a h).2.1
  have c67234 := (publicBatchWrapper86_copies2101 a h).2.2.1
  have c67235 := (publicBatchWrapper86_copies2101 a h).2.2.2.1
  have c67236 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.1
  have c67237 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.1
  have c67238 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_16 := arithEq_of_rows h (row := 1088) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_16
  simp only [← c67233, ← c67234, ← c67235] at e_1088_16
  have e_1088_17 := arithEq_of_rows h (row := 1088) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_17
  simp only [← c67236, ← c67237, k2, ← c67238] at e_1088_17
  have hr := e_1088_17
  simp only [e_1088_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1088 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9447)) := by
  have c67239 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.1
  have c67240 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.1
  have c67241 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.1
  have c67242 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67243 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67244 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1088_18 := arithEq_of_rows h (row := 1088) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_18
  simp only [← c67239, ← c67240, ← c67241] at e_1088_18
  have e_1088_19 := arithEq_of_rows h (row := 1088) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_19
  simp only [← c67242, ← c67243, k2, ← c67244] at e_1088_19
  have hr := e_1088_19
  simp only [e_1088_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9448)) := by
  have c67245 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67246 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67247 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67248 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67249 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67250 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_0 := arithEq_of_rows h (row := 1089) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_0
  simp only [← c67245, ← c67246, ← c67247] at e_1089_0
  have e_1089_1 := arithEq_of_rows h (row := 1089) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_1
  simp only [← c67248, ← c67249, k2, ← c67250] at e_1089_1
  have hr := e_1089_1
  simp only [e_1089_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9449)) := by
  have c67251 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67252 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67253 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67254 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67255 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67256 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_2 := arithEq_of_rows h (row := 1089) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_2
  simp only [← c67251, ← c67252, ← c67253] at e_1089_2
  have e_1089_3 := arithEq_of_rows h (row := 1089) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_3
  simp only [← c67254, ← c67255, k2, ← c67256] at e_1089_3
  have hr := e_1089_3
  simp only [e_1089_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9450)) := by
  have c67257 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67258 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67259 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67260 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67261 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67262 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_4 := arithEq_of_rows h (row := 1089) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_4
  simp only [← c67257, ← c67258, ← c67259] at e_1089_4
  have e_1089_5 := arithEq_of_rows h (row := 1089) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_5
  simp only [← c67260, ← c67261, k2, ← c67262] at e_1089_5
  have hr := e_1089_5
  simp only [e_1089_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9451)) := by
  have c67263 := (publicBatchWrapper86_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67264 := (publicBatchWrapper86_copies2102 a h).1
  have c67265 := (publicBatchWrapper86_copies2102 a h).2.1
  have c67266 := (publicBatchWrapper86_copies2102 a h).2.2.1
  have c67267 := (publicBatchWrapper86_copies2102 a h).2.2.2.1
  have c67268 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_6 := arithEq_of_rows h (row := 1089) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_6
  simp only [← c67263, ← c67264, ← c67265] at e_1089_6
  have e_1089_7 := arithEq_of_rows h (row := 1089) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_7
  simp only [← c67266, ← c67267, k2, ← c67268] at e_1089_7
  have hr := e_1089_7
  simp only [e_1089_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9452)) := by
  have c67269 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.1
  have c67270 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.1
  have c67271 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.1
  have c67272 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.1
  have c67273 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.1
  have c67274 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_8 := arithEq_of_rows h (row := 1089) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_8
  simp only [← c67269, ← c67270, ← c67271] at e_1089_8
  have e_1089_9 := arithEq_of_rows h (row := 1089) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_9
  simp only [← c67272, ← c67273, k2, ← c67274] at e_1089_9
  have hr := e_1089_9
  simp only [e_1089_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9453)) := by
  have c67275 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67276 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67277 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67278 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67279 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67280 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_10 := arithEq_of_rows h (row := 1089) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_10
  simp only [← c67275, ← c67276, ← c67277] at e_1089_10
  have e_1089_11 := arithEq_of_rows h (row := 1089) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_11
  simp only [← c67278, ← c67279, k2, ← c67280] at e_1089_11
  have hr := e_1089_11
  simp only [e_1089_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9454)) := by
  have c67281 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67282 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67283 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67284 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67285 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67286 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_12 := arithEq_of_rows h (row := 1089) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_12
  simp only [← c67281, ← c67282, ← c67283] at e_1089_12
  have e_1089_13 := arithEq_of_rows h (row := 1089) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_13
  simp only [← c67284, ← c67285, k2, ← c67286] at e_1089_13
  have hr := e_1089_13
  simp only [e_1089_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9455)) := by
  have c67287 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67288 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67289 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67290 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67291 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67292 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_14 := arithEq_of_rows h (row := 1089) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_14
  simp only [← c67287, ← c67288, ← c67289] at e_1089_14
  have e_1089_15 := arithEq_of_rows h (row := 1089) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_15
  simp only [← c67290, ← c67291, k2, ← c67292] at e_1089_15
  have hr := e_1089_15
  simp only [e_1089_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9456)) := by
  have c67293 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67294 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67295 := (publicBatchWrapper86_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67296 := (publicBatchWrapper86_copies2103 a h).1
  have c67297 := (publicBatchWrapper86_copies2103 a h).2.1
  have c67298 := (publicBatchWrapper86_copies2103 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_16 := arithEq_of_rows h (row := 1089) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_16
  simp only [← c67293, ← c67294, ← c67295] at e_1089_16
  have e_1089_17 := arithEq_of_rows h (row := 1089) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_17
  simp only [← c67296, ← c67297, k2, ← c67298] at e_1089_17
  have hr := e_1089_17
  simp only [e_1089_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1089 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9457)) := by
  have c67299 := (publicBatchWrapper86_copies2103 a h).2.2.2.1
  have c67300 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.1
  have c67301 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.1
  have c67302 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.1
  have c67303 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.1
  have c67304 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1089_18 := arithEq_of_rows h (row := 1089) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_18
  simp only [← c67299, ← c67300, ← c67301] at e_1089_18
  have e_1089_19 := arithEq_of_rows h (row := 1089) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_19
  simp only [← c67302, ← c67303, k2, ← c67304] at e_1089_19
  have hr := e_1089_19
  simp only [e_1089_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9458)) := by
  have c67305 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.1
  have c67306 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67307 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67308 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67309 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67310 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_0 := arithEq_of_rows h (row := 1090) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_0
  simp only [← c67305, ← c67306, ← c67307] at e_1090_0
  have e_1090_1 := arithEq_of_rows h (row := 1090) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_1
  simp only [← c67308, ← c67309, k2, ← c67310] at e_1090_1
  have hr := e_1090_1
  simp only [e_1090_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9459)) := by
  have c67311 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67312 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67313 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67314 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67315 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67316 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_2 := arithEq_of_rows h (row := 1090) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_2
  simp only [← c67311, ← c67312, ← c67313] at e_1090_2
  have e_1090_3 := arithEq_of_rows h (row := 1090) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_3
  simp only [← c67314, ← c67315, k2, ← c67316] at e_1090_3
  have hr := e_1090_3
  simp only [e_1090_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9460)) := by
  have c67317 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67318 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67319 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67320 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67321 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67322 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_4 := arithEq_of_rows h (row := 1090) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_4
  simp only [← c67317, ← c67318, ← c67319] at e_1090_4
  have e_1090_5 := arithEq_of_rows h (row := 1090) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_5
  simp only [← c67320, ← c67321, k2, ← c67322] at e_1090_5
  have hr := e_1090_5
  simp only [e_1090_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9461)) := by
  have c67323 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67324 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67325 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67326 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67327 := (publicBatchWrapper86_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67328 := (publicBatchWrapper86_copies2104 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_6 := arithEq_of_rows h (row := 1090) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_6
  simp only [← c67323, ← c67324, ← c67325] at e_1090_6
  have e_1090_7 := arithEq_of_rows h (row := 1090) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_7
  simp only [← c67326, ← c67327, k2, ← c67328] at e_1090_7
  have hr := e_1090_7
  simp only [e_1090_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9462)) := by
  have c67329 := (publicBatchWrapper86_copies2104 a h).2.1
  have c67330 := (publicBatchWrapper86_copies2104 a h).2.2.1
  have c67331 := (publicBatchWrapper86_copies2104 a h).2.2.2.1
  have c67332 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.1
  have c67333 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.1
  have c67334 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_8 := arithEq_of_rows h (row := 1090) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_8
  simp only [← c67329, ← c67330, ← c67331] at e_1090_8
  have e_1090_9 := arithEq_of_rows h (row := 1090) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_9
  simp only [← c67332, ← c67333, k2, ← c67334] at e_1090_9
  have hr := e_1090_9
  simp only [e_1090_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9463)) := by
  have c67335 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.1
  have c67336 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.1
  have c67337 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.1
  have c67338 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67339 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67340 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_10 := arithEq_of_rows h (row := 1090) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_10
  simp only [← c67335, ← c67336, ← c67337] at e_1090_10
  have e_1090_11 := arithEq_of_rows h (row := 1090) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_11
  simp only [← c67338, ← c67339, k2, ← c67340] at e_1090_11
  have hr := e_1090_11
  simp only [e_1090_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9464)) := by
  have c67341 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67342 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67343 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67344 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67345 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67346 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_12 := arithEq_of_rows h (row := 1090) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_12
  simp only [← c67341, ← c67342, ← c67343] at e_1090_12
  have e_1090_13 := arithEq_of_rows h (row := 1090) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_13
  simp only [← c67344, ← c67345, k2, ← c67346] at e_1090_13
  have hr := e_1090_13
  simp only [e_1090_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9465)) := by
  have c67347 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67348 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67349 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67350 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67351 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67352 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_14 := arithEq_of_rows h (row := 1090) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_14
  simp only [← c67347, ← c67348, ← c67349] at e_1090_14
  have e_1090_15 := arithEq_of_rows h (row := 1090) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_15
  simp only [← c67350, ← c67351, k2, ← c67352] at e_1090_15
  have hr := e_1090_15
  simp only [e_1090_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9466)) := by
  have c67353 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67354 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67355 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67356 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67357 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67358 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_16 := arithEq_of_rows h (row := 1090) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_16
  simp only [← c67353, ← c67354, ← c67355] at e_1090_16
  have e_1090_17 := arithEq_of_rows h (row := 1090) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_17
  simp only [← c67356, ← c67357, k2, ← c67358] at e_1090_17
  have hr := e_1090_17
  simp only [e_1090_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1090 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9467)) := by
  have c67359 := (publicBatchWrapper86_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67360 := (publicBatchWrapper86_copies2105 a h).1
  have c67361 := (publicBatchWrapper86_copies2105 a h).2.1
  have c67362 := (publicBatchWrapper86_copies2105 a h).2.2.1
  have c67363 := (publicBatchWrapper86_copies2105 a h).2.2.2.1
  have c67364 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1090_18 := arithEq_of_rows h (row := 1090) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_18
  simp only [← c67359, ← c67360, ← c67361] at e_1090_18
  have e_1090_19 := arithEq_of_rows h (row := 1090) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_19
  simp only [← c67362, ← c67363, k2, ← c67364] at e_1090_19
  have hr := e_1090_19
  simp only [e_1090_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9468)) := by
  have c67365 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.1
  have c67366 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.1
  have c67367 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.1
  have c67368 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.1
  have c67369 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.1
  have c67370 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_0 := arithEq_of_rows h (row := 1091) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_0
  simp only [← c67365, ← c67366, ← c67367] at e_1091_0
  have e_1091_1 := arithEq_of_rows h (row := 1091) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_1
  simp only [← c67368, ← c67369, k2, ← c67370] at e_1091_1
  have hr := e_1091_1
  simp only [e_1091_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9469)) := by
  have c67371 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67372 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67373 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67374 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67375 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67376 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_2 := arithEq_of_rows h (row := 1091) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_2
  simp only [← c67371, ← c67372, ← c67373] at e_1091_2
  have e_1091_3 := arithEq_of_rows h (row := 1091) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_3
  simp only [← c67374, ← c67375, k2, ← c67376] at e_1091_3
  have hr := e_1091_3
  simp only [e_1091_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9470)) := by
  have c67377 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67378 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67379 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67380 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67381 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67382 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_4 := arithEq_of_rows h (row := 1091) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_4
  simp only [← c67377, ← c67378, ← c67379] at e_1091_4
  have e_1091_5 := arithEq_of_rows h (row := 1091) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_5
  simp only [← c67380, ← c67381, k2, ← c67382] at e_1091_5
  have hr := e_1091_5
  simp only [e_1091_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9471)) := by
  have c67383 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67384 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67385 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67386 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67387 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67388 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_6 := arithEq_of_rows h (row := 1091) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_6
  simp only [← c67383, ← c67384, ← c67385] at e_1091_6
  have e_1091_7 := arithEq_of_rows h (row := 1091) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_7
  simp only [← c67386, ← c67387, k2, ← c67388] at e_1091_7
  have hr := e_1091_7
  simp only [e_1091_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9472)) := by
  have c67389 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67390 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67391 := (publicBatchWrapper86_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67392 := (publicBatchWrapper86_copies2106 a h).1
  have c67393 := (publicBatchWrapper86_copies2106 a h).2.1
  have c67394 := (publicBatchWrapper86_copies2106 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_8 := arithEq_of_rows h (row := 1091) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_8
  simp only [← c67389, ← c67390, ← c67391] at e_1091_8
  have e_1091_9 := arithEq_of_rows h (row := 1091) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_9
  simp only [← c67392, ← c67393, k2, ← c67394] at e_1091_9
  have hr := e_1091_9
  simp only [e_1091_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9473)) := by
  have c67395 := (publicBatchWrapper86_copies2106 a h).2.2.2.1
  have c67396 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.1
  have c67397 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.1
  have c67398 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.1
  have c67399 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.1
  have c67400 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_10 := arithEq_of_rows h (row := 1091) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_10
  simp only [← c67395, ← c67396, ← c67397] at e_1091_10
  have e_1091_11 := arithEq_of_rows h (row := 1091) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_11
  simp only [← c67398, ← c67399, k2, ← c67400] at e_1091_11
  have hr := e_1091_11
  simp only [e_1091_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9474)) := by
  have c67401 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.1
  have c67402 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67403 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67404 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67405 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67406 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_12 := arithEq_of_rows h (row := 1091) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_12
  simp only [← c67401, ← c67402, ← c67403] at e_1091_12
  have e_1091_13 := arithEq_of_rows h (row := 1091) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_13
  simp only [← c67404, ← c67405, k2, ← c67406] at e_1091_13
  have hr := e_1091_13
  simp only [e_1091_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9475)) := by
  have c67407 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67408 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67409 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67410 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67411 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67412 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_14 := arithEq_of_rows h (row := 1091) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_14
  simp only [← c67407, ← c67408, ← c67409] at e_1091_14
  have e_1091_15 := arithEq_of_rows h (row := 1091) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_15
  simp only [← c67410, ← c67411, k2, ← c67412] at e_1091_15
  have hr := e_1091_15
  simp only [e_1091_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9476)) := by
  have c67413 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67414 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67415 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67416 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67417 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67418 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_16 := arithEq_of_rows h (row := 1091) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_16
  simp only [← c67413, ← c67414, ← c67415] at e_1091_16
  have e_1091_17 := arithEq_of_rows h (row := 1091) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_17
  simp only [← c67416, ← c67417, k2, ← c67418] at e_1091_17
  have hr := e_1091_17
  simp only [e_1091_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1091 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9477)) := by
  have c67419 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67420 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67421 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67422 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67423 := (publicBatchWrapper86_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67424 := (publicBatchWrapper86_copies2107 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1091_18 := arithEq_of_rows h (row := 1091) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_18
  simp only [← c67419, ← c67420, ← c67421] at e_1091_18
  have e_1091_19 := arithEq_of_rows h (row := 1091) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_19
  simp only [← c67422, ← c67423, k2, ← c67424] at e_1091_19
  have hr := e_1091_19
  simp only [e_1091_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9478)) := by
  have c67425 := (publicBatchWrapper86_copies2107 a h).2.1
  have c67426 := (publicBatchWrapper86_copies2107 a h).2.2.1
  have c67427 := (publicBatchWrapper86_copies2107 a h).2.2.2.1
  have c67428 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.1
  have c67429 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.1
  have c67430 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_0 := arithEq_of_rows h (row := 1092) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_0
  simp only [← c67425, ← c67426, ← c67427] at e_1092_0
  have e_1092_1 := arithEq_of_rows h (row := 1092) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_1
  simp only [← c67428, ← c67429, k2, ← c67430] at e_1092_1
  have hr := e_1092_1
  simp only [e_1092_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9479)) := by
  have c67431 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.1
  have c67432 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.1
  have c67433 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.1
  have c67434 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67435 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67436 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_2 := arithEq_of_rows h (row := 1092) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_2
  simp only [← c67431, ← c67432, ← c67433] at e_1092_2
  have e_1092_3 := arithEq_of_rows h (row := 1092) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_3
  simp only [← c67434, ← c67435, k2, ← c67436] at e_1092_3
  have hr := e_1092_3
  simp only [e_1092_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9480)) := by
  have c67437 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67438 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67439 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67440 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67441 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67442 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_4 := arithEq_of_rows h (row := 1092) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_4
  simp only [← c67437, ← c67438, ← c67439] at e_1092_4
  have e_1092_5 := arithEq_of_rows h (row := 1092) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_5
  simp only [← c67440, ← c67441, k2, ← c67442] at e_1092_5
  have hr := e_1092_5
  simp only [e_1092_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9481)) := by
  have c67443 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67444 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67445 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67446 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67447 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67448 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_6 := arithEq_of_rows h (row := 1092) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_6
  simp only [← c67443, ← c67444, ← c67445] at e_1092_6
  have e_1092_7 := arithEq_of_rows h (row := 1092) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_7
  simp only [← c67446, ← c67447, k2, ← c67448] at e_1092_7
  have hr := e_1092_7
  simp only [e_1092_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9482)) := by
  have c67449 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67450 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67451 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67452 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67453 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67454 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_8 := arithEq_of_rows h (row := 1092) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_8
  simp only [← c67449, ← c67450, ← c67451] at e_1092_8
  have e_1092_9 := arithEq_of_rows h (row := 1092) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_9
  simp only [← c67452, ← c67453, k2, ← c67454] at e_1092_9
  have hr := e_1092_9
  simp only [e_1092_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9483)) := by
  have c67455 := (publicBatchWrapper86_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67456 := (publicBatchWrapper86_copies2108 a h).1
  have c67457 := (publicBatchWrapper86_copies2108 a h).2.1
  have c67458 := (publicBatchWrapper86_copies2108 a h).2.2.1
  have c67459 := (publicBatchWrapper86_copies2108 a h).2.2.2.1
  have c67460 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_10 := arithEq_of_rows h (row := 1092) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_10
  simp only [← c67455, ← c67456, ← c67457] at e_1092_10
  have e_1092_11 := arithEq_of_rows h (row := 1092) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_11
  simp only [← c67458, ← c67459, k2, ← c67460] at e_1092_11
  have hr := e_1092_11
  simp only [e_1092_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9484)) := by
  have c67461 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.1
  have c67462 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.1
  have c67463 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.1
  have c67464 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.1
  have c67465 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.1
  have c67466 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_12 := arithEq_of_rows h (row := 1092) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_12
  simp only [← c67461, ← c67462, ← c67463] at e_1092_12
  have e_1092_13 := arithEq_of_rows h (row := 1092) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_13
  simp only [← c67464, ← c67465, k2, ← c67466] at e_1092_13
  have hr := e_1092_13
  simp only [e_1092_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9485)) := by
  have c67467 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67468 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67469 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67470 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67471 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67472 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_14 := arithEq_of_rows h (row := 1092) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_14
  simp only [← c67467, ← c67468, ← c67469] at e_1092_14
  have e_1092_15 := arithEq_of_rows h (row := 1092) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_15
  simp only [← c67470, ← c67471, k2, ← c67472] at e_1092_15
  have hr := e_1092_15
  simp only [e_1092_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9486)) := by
  have c67473 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67474 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67475 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67476 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67477 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67478 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_16 := arithEq_of_rows h (row := 1092) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_16
  simp only [← c67473, ← c67474, ← c67475] at e_1092_16
  have e_1092_17 := arithEq_of_rows h (row := 1092) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_17
  simp only [← c67476, ← c67477, k2, ← c67478] at e_1092_17
  have hr := e_1092_17
  simp only [e_1092_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1092 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9487)) := by
  have c67479 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67480 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67481 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67482 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67483 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67484 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1092_18 := arithEq_of_rows h (row := 1092) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_18
  simp only [← c67479, ← c67480, ← c67481] at e_1092_18
  have e_1092_19 := arithEq_of_rows h (row := 1092) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_19
  simp only [← c67482, ← c67483, k2, ← c67484] at e_1092_19
  have hr := e_1092_19
  simp only [e_1092_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9488)) := by
  have c67485 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67486 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67487 := (publicBatchWrapper86_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67488 := (publicBatchWrapper86_copies2109 a h).1
  have c67489 := (publicBatchWrapper86_copies2109 a h).2.1
  have c67490 := (publicBatchWrapper86_copies2109 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_0 := arithEq_of_rows h (row := 1093) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_0
  simp only [← c67485, ← c67486, ← c67487] at e_1093_0
  have e_1093_1 := arithEq_of_rows h (row := 1093) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_1
  simp only [← c67488, ← c67489, k2, ← c67490] at e_1093_1
  have hr := e_1093_1
  simp only [e_1093_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9489)) := by
  have c67491 := (publicBatchWrapper86_copies2109 a h).2.2.2.1
  have c67492 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.1
  have c67493 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.1
  have c67494 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.1
  have c67495 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.1
  have c67496 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_2 := arithEq_of_rows h (row := 1093) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_2
  simp only [← c67491, ← c67492, ← c67493] at e_1093_2
  have e_1093_3 := arithEq_of_rows h (row := 1093) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_3
  simp only [← c67494, ← c67495, k2, ← c67496] at e_1093_3
  have hr := e_1093_3
  simp only [e_1093_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9490)) := by
  have c67497 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.1
  have c67498 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67499 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67500 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67501 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67502 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_4 := arithEq_of_rows h (row := 1093) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_4
  simp only [← c67497, ← c67498, ← c67499] at e_1093_4
  have e_1093_5 := arithEq_of_rows h (row := 1093) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_5
  simp only [← c67500, ← c67501, k2, ← c67502] at e_1093_5
  have hr := e_1093_5
  simp only [e_1093_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9491)) := by
  have c67503 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67504 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67505 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67506 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67507 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67508 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_6 := arithEq_of_rows h (row := 1093) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_6
  simp only [← c67503, ← c67504, ← c67505] at e_1093_6
  have e_1093_7 := arithEq_of_rows h (row := 1093) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_7
  simp only [← c67506, ← c67507, k2, ← c67508] at e_1093_7
  have hr := e_1093_7
  simp only [e_1093_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9492)) := by
  have c67509 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67510 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67511 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67512 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67513 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67514 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_8 := arithEq_of_rows h (row := 1093) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_8
  simp only [← c67509, ← c67510, ← c67511] at e_1093_8
  have e_1093_9 := arithEq_of_rows h (row := 1093) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_9
  simp only [← c67512, ← c67513, k2, ← c67514] at e_1093_9
  have hr := e_1093_9
  simp only [e_1093_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9493)) := by
  have c67515 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67516 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67517 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67518 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67519 := (publicBatchWrapper86_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67520 := (publicBatchWrapper86_copies2110 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_10 := arithEq_of_rows h (row := 1093) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_10
  simp only [← c67515, ← c67516, ← c67517] at e_1093_10
  have e_1093_11 := arithEq_of_rows h (row := 1093) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_11
  simp only [← c67518, ← c67519, k2, ← c67520] at e_1093_11
  have hr := e_1093_11
  simp only [e_1093_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9494)) := by
  have c67521 := (publicBatchWrapper86_copies2110 a h).2.1
  have c67522 := (publicBatchWrapper86_copies2110 a h).2.2.1
  have c67523 := (publicBatchWrapper86_copies2110 a h).2.2.2.1
  have c67524 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.1
  have c67525 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.1
  have c67526 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_12 := arithEq_of_rows h (row := 1093) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_12
  simp only [← c67521, ← c67522, ← c67523] at e_1093_12
  have e_1093_13 := arithEq_of_rows h (row := 1093) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_13
  simp only [← c67524, ← c67525, k2, ← c67526] at e_1093_13
  have hr := e_1093_13
  simp only [e_1093_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9495)) := by
  have c67527 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.1
  have c67528 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.1
  have c67529 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.1
  have c67530 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67531 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67532 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_14 := arithEq_of_rows h (row := 1093) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_14
  simp only [← c67527, ← c67528, ← c67529] at e_1093_14
  have e_1093_15 := arithEq_of_rows h (row := 1093) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_15
  simp only [← c67530, ← c67531, k2, ← c67532] at e_1093_15
  have hr := e_1093_15
  simp only [e_1093_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9496)) := by
  have c67533 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67534 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67535 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67536 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67537 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67538 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_16 := arithEq_of_rows h (row := 1093) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_16
  simp only [← c67533, ← c67534, ← c67535] at e_1093_16
  have e_1093_17 := arithEq_of_rows h (row := 1093) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_17
  simp only [← c67536, ← c67537, k2, ← c67538] at e_1093_17
  have hr := e_1093_17
  simp only [e_1093_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1093 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9497)) := by
  have c67539 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67540 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67541 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67542 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67543 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67544 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1093_18 := arithEq_of_rows h (row := 1093) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_18
  simp only [← c67539, ← c67540, ← c67541] at e_1093_18
  have e_1093_19 := arithEq_of_rows h (row := 1093) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_19
  simp only [← c67542, ← c67543, k2, ← c67544] at e_1093_19
  have hr := e_1093_19
  simp only [e_1093_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9498)) := by
  have c67545 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67546 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67547 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67548 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67549 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67550 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_0 := arithEq_of_rows h (row := 1094) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_0
  simp only [← c67545, ← c67546, ← c67547] at e_1094_0
  have e_1094_1 := arithEq_of_rows h (row := 1094) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_1
  simp only [← c67548, ← c67549, k2, ← c67550] at e_1094_1
  have hr := e_1094_1
  simp only [e_1094_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9499)) := by
  have c67551 := (publicBatchWrapper86_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67552 := (publicBatchWrapper86_copies2111 a h).1
  have c67553 := (publicBatchWrapper86_copies2111 a h).2.1
  have c67554 := (publicBatchWrapper86_copies2111 a h).2.2.1
  have c67555 := (publicBatchWrapper86_copies2111 a h).2.2.2.1
  have c67556 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_2 := arithEq_of_rows h (row := 1094) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_2
  simp only [← c67551, ← c67552, ← c67553] at e_1094_2
  have e_1094_3 := arithEq_of_rows h (row := 1094) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_3
  simp only [← c67554, ← c67555, k2, ← c67556] at e_1094_3
  have hr := e_1094_3
  simp only [e_1094_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9500)) := by
  have c67557 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.1
  have c67558 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.1
  have c67559 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.1
  have c67560 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.1
  have c67561 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.1
  have c67562 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_4 := arithEq_of_rows h (row := 1094) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_4
  simp only [← c67557, ← c67558, ← c67559] at e_1094_4
  have e_1094_5 := arithEq_of_rows h (row := 1094) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_5
  simp only [← c67560, ← c67561, k2, ← c67562] at e_1094_5
  have hr := e_1094_5
  simp only [e_1094_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9501)) := by
  have c67563 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67564 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67565 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67566 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67567 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67568 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_6 := arithEq_of_rows h (row := 1094) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_6
  simp only [← c67563, ← c67564, ← c67565] at e_1094_6
  have e_1094_7 := arithEq_of_rows h (row := 1094) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_7
  simp only [← c67566, ← c67567, k2, ← c67568] at e_1094_7
  have hr := e_1094_7
  simp only [e_1094_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9502)) := by
  have c67569 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67570 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67571 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67572 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67573 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67574 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_8 := arithEq_of_rows h (row := 1094) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_8
  simp only [← c67569, ← c67570, ← c67571] at e_1094_8
  have e_1094_9 := arithEq_of_rows h (row := 1094) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_9
  simp only [← c67572, ← c67573, k2, ← c67574] at e_1094_9
  have hr := e_1094_9
  simp only [e_1094_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9503)) := by
  have c67575 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67576 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67577 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67578 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67579 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67580 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_10 := arithEq_of_rows h (row := 1094) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_10
  simp only [← c67575, ← c67576, ← c67577] at e_1094_10
  have e_1094_11 := arithEq_of_rows h (row := 1094) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_11
  simp only [← c67578, ← c67579, k2, ← c67580] at e_1094_11
  have hr := e_1094_11
  simp only [e_1094_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9504)) := by
  have c67581 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67582 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67583 := (publicBatchWrapper86_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67584 := (publicBatchWrapper86_copies2112 a h).1
  have c67585 := (publicBatchWrapper86_copies2112 a h).2.1
  have c67586 := (publicBatchWrapper86_copies2112 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_12 := arithEq_of_rows h (row := 1094) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_12
  simp only [← c67581, ← c67582, ← c67583] at e_1094_12
  have e_1094_13 := arithEq_of_rows h (row := 1094) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_13
  simp only [← c67584, ← c67585, k2, ← c67586] at e_1094_13
  have hr := e_1094_13
  simp only [e_1094_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9505)) := by
  have c67587 := (publicBatchWrapper86_copies2112 a h).2.2.2.1
  have c67588 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.1
  have c67589 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.1
  have c67590 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.1
  have c67591 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.1
  have c67592 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_14 := arithEq_of_rows h (row := 1094) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_14
  simp only [← c67587, ← c67588, ← c67589] at e_1094_14
  have e_1094_15 := arithEq_of_rows h (row := 1094) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_15
  simp only [← c67590, ← c67591, k2, ← c67592] at e_1094_15
  have hr := e_1094_15
  simp only [e_1094_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9506)) := by
  have c67593 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.1
  have c67594 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67595 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67596 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67597 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67598 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_16 := arithEq_of_rows h (row := 1094) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_16
  simp only [← c67593, ← c67594, ← c67595] at e_1094_16
  have e_1094_17 := arithEq_of_rows h (row := 1094) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_17
  simp only [← c67596, ← c67597, k2, ← c67598] at e_1094_17
  have hr := e_1094_17
  simp only [e_1094_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1094 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9507)) := by
  have c67599 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67600 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67601 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67602 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67603 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67604 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1094_18 := arithEq_of_rows h (row := 1094) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_18
  simp only [← c67599, ← c67600, ← c67601] at e_1094_18
  have e_1094_19 := arithEq_of_rows h (row := 1094) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_19
  simp only [← c67602, ← c67603, k2, ← c67604] at e_1094_19
  have hr := e_1094_19
  simp only [e_1094_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9508)) := by
  have c67605 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67606 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67607 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67608 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67609 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67610 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_0 := arithEq_of_rows h (row := 1095) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_0
  simp only [← c67605, ← c67606, ← c67607] at e_1095_0
  have e_1095_1 := arithEq_of_rows h (row := 1095) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_1
  simp only [← c67608, ← c67609, k2, ← c67610] at e_1095_1
  have hr := e_1095_1
  simp only [e_1095_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9509)) := by
  have c67611 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67612 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67613 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67614 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67615 := (publicBatchWrapper86_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67616 := (publicBatchWrapper86_copies2113 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_2 := arithEq_of_rows h (row := 1095) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_2
  simp only [← c67611, ← c67612, ← c67613] at e_1095_2
  have e_1095_3 := arithEq_of_rows h (row := 1095) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_3
  simp only [← c67614, ← c67615, k2, ← c67616] at e_1095_3
  have hr := e_1095_3
  simp only [e_1095_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9510)) := by
  have c67617 := (publicBatchWrapper86_copies2113 a h).2.1
  have c67618 := (publicBatchWrapper86_copies2113 a h).2.2.1
  have c67619 := (publicBatchWrapper86_copies2113 a h).2.2.2.1
  have c67620 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.1
  have c67621 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.1
  have c67622 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_4 := arithEq_of_rows h (row := 1095) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_4
  simp only [← c67617, ← c67618, ← c67619] at e_1095_4
  have e_1095_5 := arithEq_of_rows h (row := 1095) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_5
  simp only [← c67620, ← c67621, k2, ← c67622] at e_1095_5
  have hr := e_1095_5
  simp only [e_1095_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9511)) := by
  have c67623 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.1
  have c67624 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.1
  have c67625 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.1
  have c67626 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67627 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67628 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_6 := arithEq_of_rows h (row := 1095) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_6
  simp only [← c67623, ← c67624, ← c67625] at e_1095_6
  have e_1095_7 := arithEq_of_rows h (row := 1095) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_7
  simp only [← c67626, ← c67627, k2, ← c67628] at e_1095_7
  have hr := e_1095_7
  simp only [e_1095_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9512)) := by
  have c67629 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67630 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67631 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67632 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67633 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67634 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_8 := arithEq_of_rows h (row := 1095) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_8
  simp only [← c67629, ← c67630, ← c67631] at e_1095_8
  have e_1095_9 := arithEq_of_rows h (row := 1095) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_9
  simp only [← c67632, ← c67633, k2, ← c67634] at e_1095_9
  have hr := e_1095_9
  simp only [e_1095_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9513)) := by
  have c67635 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67636 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67637 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67638 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67639 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67640 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_10 := arithEq_of_rows h (row := 1095) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_10
  simp only [← c67635, ← c67636, ← c67637] at e_1095_10
  have e_1095_11 := arithEq_of_rows h (row := 1095) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_11
  simp only [← c67638, ← c67639, k2, ← c67640] at e_1095_11
  have hr := e_1095_11
  simp only [e_1095_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9514)) := by
  have c67641 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67642 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67643 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67644 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67645 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67646 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_12 := arithEq_of_rows h (row := 1095) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_12
  simp only [← c67641, ← c67642, ← c67643] at e_1095_12
  have e_1095_13 := arithEq_of_rows h (row := 1095) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_13
  simp only [← c67644, ← c67645, k2, ← c67646] at e_1095_13
  have hr := e_1095_13
  simp only [e_1095_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9515)) := by
  have c67647 := (publicBatchWrapper86_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67648 := (publicBatchWrapper86_copies2114 a h).1
  have c67649 := (publicBatchWrapper86_copies2114 a h).2.1
  have c67650 := (publicBatchWrapper86_copies2114 a h).2.2.1
  have c67651 := (publicBatchWrapper86_copies2114 a h).2.2.2.1
  have c67652 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_14 := arithEq_of_rows h (row := 1095) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_14
  simp only [← c67647, ← c67648, ← c67649] at e_1095_14
  have e_1095_15 := arithEq_of_rows h (row := 1095) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_15
  simp only [← c67650, ← c67651, k2, ← c67652] at e_1095_15
  have hr := e_1095_15
  simp only [e_1095_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9516)) := by
  have c67653 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.1
  have c67654 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.1
  have c67655 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.1
  have c67656 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.1
  have c67657 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.1
  have c67658 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_16 := arithEq_of_rows h (row := 1095) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_16
  simp only [← c67653, ← c67654, ← c67655] at e_1095_16
  have e_1095_17 := arithEq_of_rows h (row := 1095) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_17
  simp only [← c67656, ← c67657, k2, ← c67658] at e_1095_17
  have hr := e_1095_17
  simp only [e_1095_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1095 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9517)) := by
  have c67659 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67660 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67661 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67662 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67663 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67664 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1095_18 := arithEq_of_rows h (row := 1095) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_18
  simp only [← c67659, ← c67660, ← c67661] at e_1095_18
  have e_1095_19 := arithEq_of_rows h (row := 1095) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_19
  simp only [← c67662, ← c67663, k2, ← c67664] at e_1095_19
  have hr := e_1095_19
  simp only [e_1095_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9518)) := by
  have c67665 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67666 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67667 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67668 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67669 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67670 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_0 := arithEq_of_rows h (row := 1096) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_0
  simp only [← c67665, ← c67666, ← c67667] at e_1096_0
  have e_1096_1 := arithEq_of_rows h (row := 1096) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_1
  simp only [← c67668, ← c67669, k2, ← c67670] at e_1096_1
  have hr := e_1096_1
  simp only [e_1096_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9519)) := by
  have c67671 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67672 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67673 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67674 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67675 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67676 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_2 := arithEq_of_rows h (row := 1096) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_2
  simp only [← c67671, ← c67672, ← c67673] at e_1096_2
  have e_1096_3 := arithEq_of_rows h (row := 1096) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_3
  simp only [← c67674, ← c67675, k2, ← c67676] at e_1096_3
  have hr := e_1096_3
  simp only [e_1096_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9520)) := by
  have c67677 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67678 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67679 := (publicBatchWrapper86_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67680 := (publicBatchWrapper86_copies2115 a h).1
  have c67681 := (publicBatchWrapper86_copies2115 a h).2.1
  have c67682 := (publicBatchWrapper86_copies2115 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_4 := arithEq_of_rows h (row := 1096) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_4
  simp only [← c67677, ← c67678, ← c67679] at e_1096_4
  have e_1096_5 := arithEq_of_rows h (row := 1096) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_5
  simp only [← c67680, ← c67681, k2, ← c67682] at e_1096_5
  have hr := e_1096_5
  simp only [e_1096_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9521)) := by
  have c67683 := (publicBatchWrapper86_copies2115 a h).2.2.2.1
  have c67684 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.1
  have c67685 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.1
  have c67686 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.1
  have c67687 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.1
  have c67688 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_6 := arithEq_of_rows h (row := 1096) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_6
  simp only [← c67683, ← c67684, ← c67685] at e_1096_6
  have e_1096_7 := arithEq_of_rows h (row := 1096) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_7
  simp only [← c67686, ← c67687, k2, ← c67688] at e_1096_7
  have hr := e_1096_7
  simp only [e_1096_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9522)) := by
  have c67689 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.1
  have c67690 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67691 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67692 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67693 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67694 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_8 := arithEq_of_rows h (row := 1096) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_8
  simp only [← c67689, ← c67690, ← c67691] at e_1096_8
  have e_1096_9 := arithEq_of_rows h (row := 1096) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_9
  simp only [← c67692, ← c67693, k2, ← c67694] at e_1096_9
  have hr := e_1096_9
  simp only [e_1096_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9523)) := by
  have c67695 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67696 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67697 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67698 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67699 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67700 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_10 := arithEq_of_rows h (row := 1096) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_10
  simp only [← c67695, ← c67696, ← c67697] at e_1096_10
  have e_1096_11 := arithEq_of_rows h (row := 1096) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_11
  simp only [← c67698, ← c67699, k2, ← c67700] at e_1096_11
  have hr := e_1096_11
  simp only [e_1096_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9524)) := by
  have c67701 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67702 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67703 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67704 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67705 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67706 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_12 := arithEq_of_rows h (row := 1096) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_12
  simp only [← c67701, ← c67702, ← c67703] at e_1096_12
  have e_1096_13 := arithEq_of_rows h (row := 1096) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_13
  simp only [← c67704, ← c67705, k2, ← c67706] at e_1096_13
  have hr := e_1096_13
  simp only [e_1096_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9525)) := by
  have c67707 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67708 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67709 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67710 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67711 := (publicBatchWrapper86_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67712 := (publicBatchWrapper86_copies2116 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_14 := arithEq_of_rows h (row := 1096) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_14
  simp only [← c67707, ← c67708, ← c67709] at e_1096_14
  have e_1096_15 := arithEq_of_rows h (row := 1096) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_15
  simp only [← c67710, ← c67711, k2, ← c67712] at e_1096_15
  have hr := e_1096_15
  simp only [e_1096_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9526)) := by
  have c67713 := (publicBatchWrapper86_copies2116 a h).2.1
  have c67714 := (publicBatchWrapper86_copies2116 a h).2.2.1
  have c67715 := (publicBatchWrapper86_copies2116 a h).2.2.2.1
  have c67716 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.1
  have c67717 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.1
  have c67718 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_16 := arithEq_of_rows h (row := 1096) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_16
  simp only [← c67713, ← c67714, ← c67715] at e_1096_16
  have e_1096_17 := arithEq_of_rows h (row := 1096) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_17
  simp only [← c67716, ← c67717, k2, ← c67718] at e_1096_17
  have hr := e_1096_17
  simp only [e_1096_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1096 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9527)) := by
  have c67719 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.1
  have c67720 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.1
  have c67721 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.1
  have c67722 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67723 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67724 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1096_18 := arithEq_of_rows h (row := 1096) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_18
  simp only [← c67719, ← c67720, ← c67721] at e_1096_18
  have e_1096_19 := arithEq_of_rows h (row := 1096) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_19
  simp only [← c67722, ← c67723, k2, ← c67724] at e_1096_19
  have hr := e_1096_19
  simp only [e_1096_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9528)) := by
  have c67725 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67726 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67727 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67728 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67729 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67730 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_0 := arithEq_of_rows h (row := 1097) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_0
  simp only [← c67725, ← c67726, ← c67727] at e_1097_0
  have e_1097_1 := arithEq_of_rows h (row := 1097) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_1
  simp only [← c67728, ← c67729, k2, ← c67730] at e_1097_1
  have hr := e_1097_1
  simp only [e_1097_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9529)) := by
  have c67731 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67732 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67733 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67734 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67735 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67736 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_2 := arithEq_of_rows h (row := 1097) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_2
  simp only [← c67731, ← c67732, ← c67733] at e_1097_2
  have e_1097_3 := arithEq_of_rows h (row := 1097) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_3
  simp only [← c67734, ← c67735, k2, ← c67736] at e_1097_3
  have hr := e_1097_3
  simp only [e_1097_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9530)) := by
  have c67737 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67738 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67739 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67740 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67741 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67742 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_4 := arithEq_of_rows h (row := 1097) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_4
  simp only [← c67737, ← c67738, ← c67739] at e_1097_4
  have e_1097_5 := arithEq_of_rows h (row := 1097) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_5
  simp only [← c67740, ← c67741, k2, ← c67742] at e_1097_5
  have hr := e_1097_5
  simp only [e_1097_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9531)) := by
  have c67743 := (publicBatchWrapper86_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67744 := (publicBatchWrapper86_copies2117 a h).1
  have c67745 := (publicBatchWrapper86_copies2117 a h).2.1
  have c67746 := (publicBatchWrapper86_copies2117 a h).2.2.1
  have c67747 := (publicBatchWrapper86_copies2117 a h).2.2.2.1
  have c67748 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_6 := arithEq_of_rows h (row := 1097) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_6
  simp only [← c67743, ← c67744, ← c67745] at e_1097_6
  have e_1097_7 := arithEq_of_rows h (row := 1097) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_7
  simp only [← c67746, ← c67747, k2, ← c67748] at e_1097_7
  have hr := e_1097_7
  simp only [e_1097_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9532)) := by
  have c67749 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.1
  have c67750 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.1
  have c67751 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.1
  have c67752 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.1
  have c67753 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.1
  have c67754 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_8 := arithEq_of_rows h (row := 1097) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_8
  simp only [← c67749, ← c67750, ← c67751] at e_1097_8
  have e_1097_9 := arithEq_of_rows h (row := 1097) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_9
  simp only [← c67752, ← c67753, k2, ← c67754] at e_1097_9
  have hr := e_1097_9
  simp only [e_1097_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9533)) := by
  have c67755 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67756 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67757 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67758 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67759 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67760 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_10 := arithEq_of_rows h (row := 1097) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_10
  simp only [← c67755, ← c67756, ← c67757] at e_1097_10
  have e_1097_11 := arithEq_of_rows h (row := 1097) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_11
  simp only [← c67758, ← c67759, k2, ← c67760] at e_1097_11
  have hr := e_1097_11
  simp only [e_1097_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9534)) := by
  have c67761 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67762 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67763 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67764 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67765 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67766 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_12 := arithEq_of_rows h (row := 1097) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_12
  simp only [← c67761, ← c67762, ← c67763] at e_1097_12
  have e_1097_13 := arithEq_of_rows h (row := 1097) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_13
  simp only [← c67764, ← c67765, k2, ← c67766] at e_1097_13
  have hr := e_1097_13
  simp only [e_1097_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9535)) := by
  have c67767 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67768 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67769 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67770 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67771 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67772 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_14 := arithEq_of_rows h (row := 1097) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_14
  simp only [← c67767, ← c67768, ← c67769] at e_1097_14
  have e_1097_15 := arithEq_of_rows h (row := 1097) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_15
  simp only [← c67770, ← c67771, k2, ← c67772] at e_1097_15
  have hr := e_1097_15
  simp only [e_1097_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9536)) := by
  have c67773 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67774 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67775 := (publicBatchWrapper86_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67776 := (publicBatchWrapper86_copies2118 a h).1
  have c67777 := (publicBatchWrapper86_copies2118 a h).2.1
  have c67778 := (publicBatchWrapper86_copies2118 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_16 := arithEq_of_rows h (row := 1097) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_16
  simp only [← c67773, ← c67774, ← c67775] at e_1097_16
  have e_1097_17 := arithEq_of_rows h (row := 1097) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_17
  simp only [← c67776, ← c67777, k2, ← c67778] at e_1097_17
  have hr := e_1097_17
  simp only [e_1097_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1097 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9537)) := by
  have c67779 := (publicBatchWrapper86_copies2118 a h).2.2.2.1
  have c67780 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.1
  have c67781 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.1
  have c67782 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.1
  have c67783 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.1
  have c67784 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1097_18 := arithEq_of_rows h (row := 1097) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_18
  simp only [← c67779, ← c67780, ← c67781] at e_1097_18
  have e_1097_19 := arithEq_of_rows h (row := 1097) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_19
  simp only [← c67782, ← c67783, k2, ← c67784] at e_1097_19
  have hr := e_1097_19
  simp only [e_1097_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9538)) := by
  have c67785 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.1
  have c67786 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67787 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67788 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67789 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67790 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_0 := arithEq_of_rows h (row := 1098) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_0
  simp only [← c67785, ← c67786, ← c67787] at e_1098_0
  have e_1098_1 := arithEq_of_rows h (row := 1098) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_1
  simp only [← c67788, ← c67789, k2, ← c67790] at e_1098_1
  have hr := e_1098_1
  simp only [e_1098_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9539)) := by
  have c67791 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67792 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67793 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67794 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67795 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67796 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_2 := arithEq_of_rows h (row := 1098) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_2
  simp only [← c67791, ← c67792, ← c67793] at e_1098_2
  have e_1098_3 := arithEq_of_rows h (row := 1098) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_3
  simp only [← c67794, ← c67795, k2, ← c67796] at e_1098_3
  have hr := e_1098_3
  simp only [e_1098_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9540)) := by
  have c67797 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67798 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67799 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67800 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67801 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67802 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_4 := arithEq_of_rows h (row := 1098) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_4
  simp only [← c67797, ← c67798, ← c67799] at e_1098_4
  have e_1098_5 := arithEq_of_rows h (row := 1098) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_5
  simp only [← c67800, ← c67801, k2, ← c67802] at e_1098_5
  have hr := e_1098_5
  simp only [e_1098_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9541)) := by
  have c67803 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67804 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67805 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67806 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67807 := (publicBatchWrapper86_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67808 := (publicBatchWrapper86_copies2119 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_6 := arithEq_of_rows h (row := 1098) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_6
  simp only [← c67803, ← c67804, ← c67805] at e_1098_6
  have e_1098_7 := arithEq_of_rows h (row := 1098) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_7
  simp only [← c67806, ← c67807, k2, ← c67808] at e_1098_7
  have hr := e_1098_7
  simp only [e_1098_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9542)) := by
  have c67809 := (publicBatchWrapper86_copies2119 a h).2.1
  have c67810 := (publicBatchWrapper86_copies2119 a h).2.2.1
  have c67811 := (publicBatchWrapper86_copies2119 a h).2.2.2.1
  have c67812 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.1
  have c67813 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.1
  have c67814 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_8 := arithEq_of_rows h (row := 1098) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_8
  simp only [← c67809, ← c67810, ← c67811] at e_1098_8
  have e_1098_9 := arithEq_of_rows h (row := 1098) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_9
  simp only [← c67812, ← c67813, k2, ← c67814] at e_1098_9
  have hr := e_1098_9
  simp only [e_1098_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9543)) := by
  have c67815 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.1
  have c67816 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.1
  have c67817 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.1
  have c67818 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67819 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67820 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_10 := arithEq_of_rows h (row := 1098) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_10
  simp only [← c67815, ← c67816, ← c67817] at e_1098_10
  have e_1098_11 := arithEq_of_rows h (row := 1098) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_11
  simp only [← c67818, ← c67819, k2, ← c67820] at e_1098_11
  have hr := e_1098_11
  simp only [e_1098_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9544)) := by
  have c67821 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67822 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67823 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67824 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67825 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67826 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_12 := arithEq_of_rows h (row := 1098) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_12
  simp only [← c67821, ← c67822, ← c67823] at e_1098_12
  have e_1098_13 := arithEq_of_rows h (row := 1098) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_13
  simp only [← c67824, ← c67825, k2, ← c67826] at e_1098_13
  have hr := e_1098_13
  simp only [e_1098_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9545)) := by
  have c67827 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67828 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67829 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67830 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67831 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67832 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_14 := arithEq_of_rows h (row := 1098) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_14
  simp only [← c67827, ← c67828, ← c67829] at e_1098_14
  have e_1098_15 := arithEq_of_rows h (row := 1098) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_15
  simp only [← c67830, ← c67831, k2, ← c67832] at e_1098_15
  have hr := e_1098_15
  simp only [e_1098_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9546)) := by
  have c67833 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67834 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67835 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67836 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67837 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67838 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_16 := arithEq_of_rows h (row := 1098) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_16
  simp only [← c67833, ← c67834, ← c67835] at e_1098_16
  have e_1098_17 := arithEq_of_rows h (row := 1098) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_17
  simp only [← c67836, ← c67837, k2, ← c67838] at e_1098_17
  have hr := e_1098_17
  simp only [e_1098_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1098 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9547)) := by
  have c67839 := (publicBatchWrapper86_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67840 := (publicBatchWrapper86_copies2120 a h).1
  have c67841 := (publicBatchWrapper86_copies2120 a h).2.1
  have c67842 := (publicBatchWrapper86_copies2120 a h).2.2.1
  have c67843 := (publicBatchWrapper86_copies2120 a h).2.2.2.1
  have c67844 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1098_18 := arithEq_of_rows h (row := 1098) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_18
  simp only [← c67839, ← c67840, ← c67841] at e_1098_18
  have e_1098_19 := arithEq_of_rows h (row := 1098) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_19
  simp only [← c67842, ← c67843, k2, ← c67844] at e_1098_19
  have hr := e_1098_19
  simp only [e_1098_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9548)) := by
  have c67845 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.1
  have c67846 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.1
  have c67847 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.1
  have c67848 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.1
  have c67849 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.1
  have c67850 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_0 := arithEq_of_rows h (row := 1099) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_0
  simp only [← c67845, ← c67846, ← c67847] at e_1099_0
  have e_1099_1 := arithEq_of_rows h (row := 1099) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_1
  simp only [← c67848, ← c67849, k2, ← c67850] at e_1099_1
  have hr := e_1099_1
  simp only [e_1099_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9549)) := by
  have c67851 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67852 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67853 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67854 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67855 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67856 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_2 := arithEq_of_rows h (row := 1099) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_2
  simp only [← c67851, ← c67852, ← c67853] at e_1099_2
  have e_1099_3 := arithEq_of_rows h (row := 1099) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_3
  simp only [← c67854, ← c67855, k2, ← c67856] at e_1099_3
  have hr := e_1099_3
  simp only [e_1099_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9550)) := by
  have c67857 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67858 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67859 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67860 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67861 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67862 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_4 := arithEq_of_rows h (row := 1099) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_4
  simp only [← c67857, ← c67858, ← c67859] at e_1099_4
  have e_1099_5 := arithEq_of_rows h (row := 1099) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_5
  simp only [← c67860, ← c67861, k2, ← c67862] at e_1099_5
  have hr := e_1099_5
  simp only [e_1099_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9551)) := by
  have c67863 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67864 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67865 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67866 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67867 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67868 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_6 := arithEq_of_rows h (row := 1099) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_6
  simp only [← c67863, ← c67864, ← c67865] at e_1099_6
  have e_1099_7 := arithEq_of_rows h (row := 1099) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_7
  simp only [← c67866, ← c67867, k2, ← c67868] at e_1099_7
  have hr := e_1099_7
  simp only [e_1099_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9552)) := by
  have c67869 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67870 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67871 := (publicBatchWrapper86_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67872 := (publicBatchWrapper86_copies2121 a h).1
  have c67873 := (publicBatchWrapper86_copies2121 a h).2.1
  have c67874 := (publicBatchWrapper86_copies2121 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_8 := arithEq_of_rows h (row := 1099) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_8
  simp only [← c67869, ← c67870, ← c67871] at e_1099_8
  have e_1099_9 := arithEq_of_rows h (row := 1099) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_9
  simp only [← c67872, ← c67873, k2, ← c67874] at e_1099_9
  have hr := e_1099_9
  simp only [e_1099_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9553)) := by
  have c67875 := (publicBatchWrapper86_copies2121 a h).2.2.2.1
  have c67876 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.1
  have c67877 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.1
  have c67878 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.1
  have c67879 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.1
  have c67880 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_10 := arithEq_of_rows h (row := 1099) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_10
  simp only [← c67875, ← c67876, ← c67877] at e_1099_10
  have e_1099_11 := arithEq_of_rows h (row := 1099) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_11
  simp only [← c67878, ← c67879, k2, ← c67880] at e_1099_11
  have hr := e_1099_11
  simp only [e_1099_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9554)) := by
  have c67881 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.1
  have c67882 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67883 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67884 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67885 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67886 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_12 := arithEq_of_rows h (row := 1099) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_12
  simp only [← c67881, ← c67882, ← c67883] at e_1099_12
  have e_1099_13 := arithEq_of_rows h (row := 1099) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_13
  simp only [← c67884, ← c67885, k2, ← c67886] at e_1099_13
  have hr := e_1099_13
  simp only [e_1099_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9555)) := by
  have c67887 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67888 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67889 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67890 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67891 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67892 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_14 := arithEq_of_rows h (row := 1099) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_14
  simp only [← c67887, ← c67888, ← c67889] at e_1099_14
  have e_1099_15 := arithEq_of_rows h (row := 1099) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_15
  simp only [← c67890, ← c67891, k2, ← c67892] at e_1099_15
  have hr := e_1099_15
  simp only [e_1099_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9556)) := by
  have c67893 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67894 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67895 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67896 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67897 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67898 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_16 := arithEq_of_rows h (row := 1099) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_16
  simp only [← c67893, ← c67894, ← c67895] at e_1099_16
  have e_1099_17 := arithEq_of_rows h (row := 1099) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_17
  simp only [← c67896, ← c67897, k2, ← c67898] at e_1099_17
  have hr := e_1099_17
  simp only [e_1099_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1099 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9557)) := by
  have c67899 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67900 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67901 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67902 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67903 := (publicBatchWrapper86_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67904 := (publicBatchWrapper86_copies2122 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1099_18 := arithEq_of_rows h (row := 1099) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_18
  simp only [← c67899, ← c67900, ← c67901] at e_1099_18
  have e_1099_19 := arithEq_of_rows h (row := 1099) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_19
  simp only [← c67902, ← c67903, k2, ← c67904] at e_1099_19
  have hr := e_1099_19
  simp only [e_1099_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9558)) := by
  have c67905 := (publicBatchWrapper86_copies2122 a h).2.1
  have c67906 := (publicBatchWrapper86_copies2122 a h).2.2.1
  have c67907 := (publicBatchWrapper86_copies2122 a h).2.2.2.1
  have c67908 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.1
  have c67909 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.1
  have c67910 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_0 := arithEq_of_rows h (row := 1100) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_0
  simp only [← c67905, ← c67906, ← c67907] at e_1100_0
  have e_1100_1 := arithEq_of_rows h (row := 1100) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_1
  simp only [← c67908, ← c67909, k2, ← c67910] at e_1100_1
  have hr := e_1100_1
  simp only [e_1100_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9559)) := by
  have c67911 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.1
  have c67912 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.1
  have c67913 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.1
  have c67914 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67915 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67916 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_2 := arithEq_of_rows h (row := 1100) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_2
  simp only [← c67911, ← c67912, ← c67913] at e_1100_2
  have e_1100_3 := arithEq_of_rows h (row := 1100) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_3
  simp only [← c67914, ← c67915, k2, ← c67916] at e_1100_3
  have hr := e_1100_3
  simp only [e_1100_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9560)) := by
  have c67917 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67918 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67919 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67920 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67921 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67922 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_4 := arithEq_of_rows h (row := 1100) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_4
  simp only [← c67917, ← c67918, ← c67919] at e_1100_4
  have e_1100_5 := arithEq_of_rows h (row := 1100) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_5
  simp only [← c67920, ← c67921, k2, ← c67922] at e_1100_5
  have hr := e_1100_5
  simp only [e_1100_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9561)) := by
  have c67923 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67924 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67925 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67926 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67927 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67928 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_6 := arithEq_of_rows h (row := 1100) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_6
  simp only [← c67923, ← c67924, ← c67925] at e_1100_6
  have e_1100_7 := arithEq_of_rows h (row := 1100) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_7
  simp only [← c67926, ← c67927, k2, ← c67928] at e_1100_7
  have hr := e_1100_7
  simp only [e_1100_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9562)) := by
  have c67929 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67930 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67931 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67932 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67933 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67934 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_8 := arithEq_of_rows h (row := 1100) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_8
  simp only [← c67929, ← c67930, ← c67931] at e_1100_8
  have e_1100_9 := arithEq_of_rows h (row := 1100) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_9
  simp only [← c67932, ← c67933, k2, ← c67934] at e_1100_9
  have hr := e_1100_9
  simp only [e_1100_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9563)) := by
  have c67935 := (publicBatchWrapper86_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67936 := (publicBatchWrapper86_copies2123 a h).1
  have c67937 := (publicBatchWrapper86_copies2123 a h).2.1
  have c67938 := (publicBatchWrapper86_copies2123 a h).2.2.1
  have c67939 := (publicBatchWrapper86_copies2123 a h).2.2.2.1
  have c67940 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_10 := arithEq_of_rows h (row := 1100) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_10
  simp only [← c67935, ← c67936, ← c67937] at e_1100_10
  have e_1100_11 := arithEq_of_rows h (row := 1100) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_11
  simp only [← c67938, ← c67939, k2, ← c67940] at e_1100_11
  have hr := e_1100_11
  simp only [e_1100_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9564)) := by
  have c67941 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.1
  have c67942 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.1
  have c67943 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.1
  have c67944 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.1
  have c67945 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.1
  have c67946 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_12 := arithEq_of_rows h (row := 1100) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_12
  simp only [← c67941, ← c67942, ← c67943] at e_1100_12
  have e_1100_13 := arithEq_of_rows h (row := 1100) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_13
  simp only [← c67944, ← c67945, k2, ← c67946] at e_1100_13
  have hr := e_1100_13
  simp only [e_1100_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9565)) := by
  have c67947 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67948 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67949 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67950 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67951 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67952 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_14 := arithEq_of_rows h (row := 1100) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_14
  simp only [← c67947, ← c67948, ← c67949] at e_1100_14
  have e_1100_15 := arithEq_of_rows h (row := 1100) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_15
  simp only [← c67950, ← c67951, k2, ← c67952] at e_1100_15
  have hr := e_1100_15
  simp only [e_1100_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9566)) := by
  have c67953 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67954 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67955 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67956 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67957 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67958 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_16 := arithEq_of_rows h (row := 1100) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_16
  simp only [← c67953, ← c67954, ← c67955] at e_1100_16
  have e_1100_17 := arithEq_of_rows h (row := 1100) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_17
  simp only [← c67956, ← c67957, k2, ← c67958] at e_1100_17
  have hr := e_1100_17
  simp only [e_1100_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1100 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9567)) := by
  have c67959 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67960 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67961 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67962 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67963 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67964 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1100_18 := arithEq_of_rows h (row := 1100) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_18
  simp only [← c67959, ← c67960, ← c67961] at e_1100_18
  have e_1100_19 := arithEq_of_rows h (row := 1100) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_19
  simp only [← c67962, ← c67963, k2, ← c67964] at e_1100_19
  have hr := e_1100_19
  simp only [e_1100_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9568)) := by
  have c67965 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67966 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67967 := (publicBatchWrapper86_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67968 := (publicBatchWrapper86_copies2124 a h).1
  have c67969 := (publicBatchWrapper86_copies2124 a h).2.1
  have c67970 := (publicBatchWrapper86_copies2124 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_0 := arithEq_of_rows h (row := 1101) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_0
  simp only [← c67965, ← c67966, ← c67967] at e_1101_0
  have e_1101_1 := arithEq_of_rows h (row := 1101) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_1
  simp only [← c67968, ← c67969, k2, ← c67970] at e_1101_1
  have hr := e_1101_1
  simp only [e_1101_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9569)) := by
  have c67971 := (publicBatchWrapper86_copies2124 a h).2.2.2.1
  have c67972 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.1
  have c67973 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.1
  have c67974 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.1
  have c67975 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.1
  have c67976 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_2 := arithEq_of_rows h (row := 1101) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_2
  simp only [← c67971, ← c67972, ← c67973] at e_1101_2
  have e_1101_3 := arithEq_of_rows h (row := 1101) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_3
  simp only [← c67974, ← c67975, k2, ← c67976] at e_1101_3
  have hr := e_1101_3
  simp only [e_1101_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9610)) := by
  have c67977 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.1
  have c67978 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67979 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67980 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67981 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67982 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_4 := arithEq_of_rows h (row := 1101) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_4
  simp only [← c67977, ← c67978, ← c67979] at e_1101_4
  have e_1101_5 := arithEq_of_rows h (row := 1101) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_5
  simp only [← c67980, ← c67981, k2, ← c67982] at e_1101_5
  have hr := e_1101_5
  simp only [e_1101_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9611)) := by
  have c67983 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67984 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67985 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67986 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67987 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67988 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_6 := arithEq_of_rows h (row := 1101) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_6
  simp only [← c67983, ← c67984, ← c67985] at e_1101_6
  have e_1101_7 := arithEq_of_rows h (row := 1101) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_7
  simp only [← c67986, ← c67987, k2, ← c67988] at e_1101_7
  have hr := e_1101_7
  simp only [e_1101_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9612)) := by
  have c67989 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67990 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67991 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67992 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67993 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67994 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_8 := arithEq_of_rows h (row := 1101) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_8
  simp only [← c67989, ← c67990, ← c67991] at e_1101_8
  have e_1101_9 := arithEq_of_rows h (row := 1101) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_9
  simp only [← c67992, ← c67993, k2, ← c67994] at e_1101_9
  have hr := e_1101_9
  simp only [e_1101_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9613)) := by
  have c67995 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67996 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67997 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67998 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67999 := (publicBatchWrapper86_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68000 := (publicBatchWrapper86_copies2125 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_10 := arithEq_of_rows h (row := 1101) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_10
  simp only [← c67995, ← c67996, ← c67997] at e_1101_10
  have e_1101_11 := arithEq_of_rows h (row := 1101) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_11
  simp only [← c67998, ← c67999, k2, ← c68000] at e_1101_11
  have hr := e_1101_11
  simp only [e_1101_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9614)) := by
  have c68001 := (publicBatchWrapper86_copies2125 a h).2.1
  have c68002 := (publicBatchWrapper86_copies2125 a h).2.2.1
  have c68003 := (publicBatchWrapper86_copies2125 a h).2.2.2.1
  have c68004 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.1
  have c68005 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.1
  have c68006 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_12 := arithEq_of_rows h (row := 1101) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_12
  simp only [← c68001, ← c68002, ← c68003] at e_1101_12
  have e_1101_13 := arithEq_of_rows h (row := 1101) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_13
  simp only [← c68004, ← c68005, k2, ← c68006] at e_1101_13
  have hr := e_1101_13
  simp only [e_1101_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9615)) := by
  have c68007 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.1
  have c68008 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.1
  have c68009 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.1
  have c68010 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68011 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68012 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_14 := arithEq_of_rows h (row := 1101) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_14
  simp only [← c68007, ← c68008, ← c68009] at e_1101_14
  have e_1101_15 := arithEq_of_rows h (row := 1101) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_15
  simp only [← c68010, ← c68011, k2, ← c68012] at e_1101_15
  have hr := e_1101_15
  simp only [e_1101_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9616)) := by
  have c68013 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68014 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68015 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68016 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68017 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68018 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_16 := arithEq_of_rows h (row := 1101) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_16
  simp only [← c68013, ← c68014, ← c68015] at e_1101_16
  have e_1101_17 := arithEq_of_rows h (row := 1101) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_17
  simp only [← c68016, ← c68017, k2, ← c68018] at e_1101_17
  have hr := e_1101_17
  simp only [e_1101_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1101 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9617)) := by
  have c68019 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68020 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68021 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68022 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68023 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68024 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1101_18 := arithEq_of_rows h (row := 1101) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_18
  simp only [← c68019, ← c68020, ← c68021] at e_1101_18
  have e_1101_19 := arithEq_of_rows h (row := 1101) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_19
  simp only [← c68022, ← c68023, k2, ← c68024] at e_1101_19
  have hr := e_1101_19
  simp only [e_1101_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9618)) := by
  have c68025 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68026 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68027 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68028 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68029 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68030 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_0 := arithEq_of_rows h (row := 1102) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_0
  simp only [← c68025, ← c68026, ← c68027] at e_1102_0
  have e_1102_1 := arithEq_of_rows h (row := 1102) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_1
  simp only [← c68028, ← c68029, k2, ← c68030] at e_1102_1
  have hr := e_1102_1
  simp only [e_1102_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9619)) := by
  have c68031 := (publicBatchWrapper86_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68032 := (publicBatchWrapper86_copies2126 a h).1
  have c68033 := (publicBatchWrapper86_copies2126 a h).2.1
  have c68034 := (publicBatchWrapper86_copies2126 a h).2.2.1
  have c68035 := (publicBatchWrapper86_copies2126 a h).2.2.2.1
  have c68036 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_2 := arithEq_of_rows h (row := 1102) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_2
  simp only [← c68031, ← c68032, ← c68033] at e_1102_2
  have e_1102_3 := arithEq_of_rows h (row := 1102) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_3
  simp only [← c68034, ← c68035, k2, ← c68036] at e_1102_3
  have hr := e_1102_3
  simp only [e_1102_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9620)) := by
  have c68037 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.1
  have c68038 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.1
  have c68039 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.1
  have c68040 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.1
  have c68041 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.1
  have c68042 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_4 := arithEq_of_rows h (row := 1102) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_4
  simp only [← c68037, ← c68038, ← c68039] at e_1102_4
  have e_1102_5 := arithEq_of_rows h (row := 1102) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_5
  simp only [← c68040, ← c68041, k2, ← c68042] at e_1102_5
  have hr := e_1102_5
  simp only [e_1102_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9621)) := by
  have c68043 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68044 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68045 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68046 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68047 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68048 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_6 := arithEq_of_rows h (row := 1102) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_6
  simp only [← c68043, ← c68044, ← c68045] at e_1102_6
  have e_1102_7 := arithEq_of_rows h (row := 1102) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_7
  simp only [← c68046, ← c68047, k2, ← c68048] at e_1102_7
  have hr := e_1102_7
  simp only [e_1102_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9622)) := by
  have c68049 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68050 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68051 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68052 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68053 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68054 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_8 := arithEq_of_rows h (row := 1102) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_8
  simp only [← c68049, ← c68050, ← c68051] at e_1102_8
  have e_1102_9 := arithEq_of_rows h (row := 1102) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_9
  simp only [← c68052, ← c68053, k2, ← c68054] at e_1102_9
  have hr := e_1102_9
  simp only [e_1102_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9623)) := by
  have c68055 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68056 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68057 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68058 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68059 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68060 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_10 := arithEq_of_rows h (row := 1102) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_10
  simp only [← c68055, ← c68056, ← c68057] at e_1102_10
  have e_1102_11 := arithEq_of_rows h (row := 1102) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_11
  simp only [← c68058, ← c68059, k2, ← c68060] at e_1102_11
  have hr := e_1102_11
  simp only [e_1102_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9624)) := by
  have c68061 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68062 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68063 := (publicBatchWrapper86_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68064 := (publicBatchWrapper86_copies2127 a h).1
  have c68065 := (publicBatchWrapper86_copies2127 a h).2.1
  have c68066 := (publicBatchWrapper86_copies2127 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_12 := arithEq_of_rows h (row := 1102) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_12
  simp only [← c68061, ← c68062, ← c68063] at e_1102_12
  have e_1102_13 := arithEq_of_rows h (row := 1102) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_13
  simp only [← c68064, ← c68065, k2, ← c68066] at e_1102_13
  have hr := e_1102_13
  simp only [e_1102_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9625)) := by
  have c68067 := (publicBatchWrapper86_copies2127 a h).2.2.2.1
  have c68068 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.1
  have c68069 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.1
  have c68070 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.1
  have c68071 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.1
  have c68072 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_14 := arithEq_of_rows h (row := 1102) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_14
  simp only [← c68067, ← c68068, ← c68069] at e_1102_14
  have e_1102_15 := arithEq_of_rows h (row := 1102) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_15
  simp only [← c68070, ← c68071, k2, ← c68072] at e_1102_15
  have hr := e_1102_15
  simp only [e_1102_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9626)) := by
  have c68073 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.1
  have c68074 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68075 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68076 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68077 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68078 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_16 := arithEq_of_rows h (row := 1102) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_16
  simp only [← c68073, ← c68074, ← c68075] at e_1102_16
  have e_1102_17 := arithEq_of_rows h (row := 1102) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_17
  simp only [← c68076, ← c68077, k2, ← c68078] at e_1102_17
  have hr := e_1102_17
  simp only [e_1102_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1102 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9627)) := by
  have c68079 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68080 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68081 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68082 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68083 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68084 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1102_18 := arithEq_of_rows h (row := 1102) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_18
  simp only [← c68079, ← c68080, ← c68081] at e_1102_18
  have e_1102_19 := arithEq_of_rows h (row := 1102) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_19
  simp only [← c68082, ← c68083, k2, ← c68084] at e_1102_19
  have hr := e_1102_19
  simp only [e_1102_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9628)) := by
  have c68085 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68086 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68087 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68088 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68089 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68090 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_0 := arithEq_of_rows h (row := 1103) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_0
  simp only [← c68085, ← c68086, ← c68087] at e_1103_0
  have e_1103_1 := arithEq_of_rows h (row := 1103) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_1
  simp only [← c68088, ← c68089, k2, ← c68090] at e_1103_1
  have hr := e_1103_1
  simp only [e_1103_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9629)) := by
  have c68091 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68092 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68093 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68094 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68095 := (publicBatchWrapper86_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68096 := (publicBatchWrapper86_copies2128 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_2 := arithEq_of_rows h (row := 1103) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_2
  simp only [← c68091, ← c68092, ← c68093] at e_1103_2
  have e_1103_3 := arithEq_of_rows h (row := 1103) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_3
  simp only [← c68094, ← c68095, k2, ← c68096] at e_1103_3
  have hr := e_1103_3
  simp only [e_1103_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9630)) := by
  have c68097 := (publicBatchWrapper86_copies2128 a h).2.1
  have c68098 := (publicBatchWrapper86_copies2128 a h).2.2.1
  have c68099 := (publicBatchWrapper86_copies2128 a h).2.2.2.1
  have c68100 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.1
  have c68101 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.1
  have c68102 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_4 := arithEq_of_rows h (row := 1103) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_4
  simp only [← c68097, ← c68098, ← c68099] at e_1103_4
  have e_1103_5 := arithEq_of_rows h (row := 1103) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_5
  simp only [← c68100, ← c68101, k2, ← c68102] at e_1103_5
  have hr := e_1103_5
  simp only [e_1103_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9631)) := by
  have c68103 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.1
  have c68104 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.1
  have c68105 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.1
  have c68106 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68107 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68108 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_6 := arithEq_of_rows h (row := 1103) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_6
  simp only [← c68103, ← c68104, ← c68105] at e_1103_6
  have e_1103_7 := arithEq_of_rows h (row := 1103) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_7
  simp only [← c68106, ← c68107, k2, ← c68108] at e_1103_7
  have hr := e_1103_7
  simp only [e_1103_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9632)) := by
  have c68109 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68110 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68111 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68112 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68113 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68114 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_8 := arithEq_of_rows h (row := 1103) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_8
  simp only [← c68109, ← c68110, ← c68111] at e_1103_8
  have e_1103_9 := arithEq_of_rows h (row := 1103) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_9
  simp only [← c68112, ← c68113, k2, ← c68114] at e_1103_9
  have hr := e_1103_9
  simp only [e_1103_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9633)) := by
  have c68115 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68116 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68117 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68118 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68119 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68120 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_10 := arithEq_of_rows h (row := 1103) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_10
  simp only [← c68115, ← c68116, ← c68117] at e_1103_10
  have e_1103_11 := arithEq_of_rows h (row := 1103) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_11
  simp only [← c68118, ← c68119, k2, ← c68120] at e_1103_11
  have hr := e_1103_11
  simp only [e_1103_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9634)) := by
  have c68121 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68122 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68123 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68124 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68125 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68126 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_12 := arithEq_of_rows h (row := 1103) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_12
  simp only [← c68121, ← c68122, ← c68123] at e_1103_12
  have e_1103_13 := arithEq_of_rows h (row := 1103) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_13
  simp only [← c68124, ← c68125, k2, ← c68126] at e_1103_13
  have hr := e_1103_13
  simp only [e_1103_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9635)) := by
  have c68127 := (publicBatchWrapper86_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68128 := (publicBatchWrapper86_copies2129 a h).1
  have c68129 := (publicBatchWrapper86_copies2129 a h).2.1
  have c68130 := (publicBatchWrapper86_copies2129 a h).2.2.1
  have c68131 := (publicBatchWrapper86_copies2129 a h).2.2.2.1
  have c68132 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_14 := arithEq_of_rows h (row := 1103) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_14
  simp only [← c68127, ← c68128, ← c68129] at e_1103_14
  have e_1103_15 := arithEq_of_rows h (row := 1103) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_15
  simp only [← c68130, ← c68131, k2, ← c68132] at e_1103_15
  have hr := e_1103_15
  simp only [e_1103_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9636)) := by
  have c68133 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.1
  have c68134 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.1
  have c68135 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.1
  have c68136 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.1
  have c68137 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.1
  have c68138 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_16 := arithEq_of_rows h (row := 1103) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_16
  simp only [← c68133, ← c68134, ← c68135] at e_1103_16
  have e_1103_17 := arithEq_of_rows h (row := 1103) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_17
  simp only [← c68136, ← c68137, k2, ← c68138] at e_1103_17
  have hr := e_1103_17
  simp only [e_1103_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1103 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9637)) := by
  have c68139 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68140 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68141 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68142 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68143 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68144 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1103_18 := arithEq_of_rows h (row := 1103) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_18
  simp only [← c68139, ← c68140, ← c68141] at e_1103_18
  have e_1103_19 := arithEq_of_rows h (row := 1103) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_19
  simp only [← c68142, ← c68143, k2, ← c68144] at e_1103_19
  have hr := e_1103_19
  simp only [e_1103_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9638)) := by
  have c68145 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68146 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68147 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68148 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68149 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68150 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_0 := arithEq_of_rows h (row := 1104) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_0
  simp only [← c68145, ← c68146, ← c68147] at e_1104_0
  have e_1104_1 := arithEq_of_rows h (row := 1104) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_1
  simp only [← c68148, ← c68149, k2, ← c68150] at e_1104_1
  have hr := e_1104_1
  simp only [e_1104_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9639)) := by
  have c68151 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68152 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68153 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68154 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68155 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68156 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_2 := arithEq_of_rows h (row := 1104) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_2
  simp only [← c68151, ← c68152, ← c68153] at e_1104_2
  have e_1104_3 := arithEq_of_rows h (row := 1104) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_3
  simp only [← c68154, ← c68155, k2, ← c68156] at e_1104_3
  have hr := e_1104_3
  simp only [e_1104_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9640)) := by
  have c68157 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68158 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68159 := (publicBatchWrapper86_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68160 := (publicBatchWrapper86_copies2130 a h).1
  have c68161 := (publicBatchWrapper86_copies2130 a h).2.1
  have c68162 := (publicBatchWrapper86_copies2130 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_4 := arithEq_of_rows h (row := 1104) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_4
  simp only [← c68157, ← c68158, ← c68159] at e_1104_4
  have e_1104_5 := arithEq_of_rows h (row := 1104) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_5
  simp only [← c68160, ← c68161, k2, ← c68162] at e_1104_5
  have hr := e_1104_5
  simp only [e_1104_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9641)) := by
  have c68163 := (publicBatchWrapper86_copies2130 a h).2.2.2.1
  have c68164 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.1
  have c68165 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.1
  have c68166 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.1
  have c68167 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.1
  have c68168 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_6 := arithEq_of_rows h (row := 1104) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_6
  simp only [← c68163, ← c68164, ← c68165] at e_1104_6
  have e_1104_7 := arithEq_of_rows h (row := 1104) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_7
  simp only [← c68166, ← c68167, k2, ← c68168] at e_1104_7
  have hr := e_1104_7
  simp only [e_1104_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9642)) := by
  have c68169 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.1
  have c68170 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68171 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68172 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68173 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68174 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_8 := arithEq_of_rows h (row := 1104) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_8
  simp only [← c68169, ← c68170, ← c68171] at e_1104_8
  have e_1104_9 := arithEq_of_rows h (row := 1104) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_9
  simp only [← c68172, ← c68173, k2, ← c68174] at e_1104_9
  have hr := e_1104_9
  simp only [e_1104_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9643)) := by
  have c68175 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68176 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68177 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68178 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68179 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68180 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_10 := arithEq_of_rows h (row := 1104) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_10
  simp only [← c68175, ← c68176, ← c68177] at e_1104_10
  have e_1104_11 := arithEq_of_rows h (row := 1104) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_11
  simp only [← c68178, ← c68179, k2, ← c68180] at e_1104_11
  have hr := e_1104_11
  simp only [e_1104_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9644)) := by
  have c68181 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68182 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68183 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68184 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68185 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68186 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_12 := arithEq_of_rows h (row := 1104) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_12
  simp only [← c68181, ← c68182, ← c68183] at e_1104_12
  have e_1104_13 := arithEq_of_rows h (row := 1104) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_13
  simp only [← c68184, ← c68185, k2, ← c68186] at e_1104_13
  have hr := e_1104_13
  simp only [e_1104_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9645)) := by
  have c68187 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68188 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68189 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68190 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68191 := (publicBatchWrapper86_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68192 := (publicBatchWrapper86_copies2131 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_14 := arithEq_of_rows h (row := 1104) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_14
  simp only [← c68187, ← c68188, ← c68189] at e_1104_14
  have e_1104_15 := arithEq_of_rows h (row := 1104) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_15
  simp only [← c68190, ← c68191, k2, ← c68192] at e_1104_15
  have hr := e_1104_15
  simp only [e_1104_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9646)) := by
  have c68193 := (publicBatchWrapper86_copies2131 a h).2.1
  have c68194 := (publicBatchWrapper86_copies2131 a h).2.2.1
  have c68195 := (publicBatchWrapper86_copies2131 a h).2.2.2.1
  have c68196 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.1
  have c68197 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.1
  have c68198 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_16 := arithEq_of_rows h (row := 1104) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_16
  simp only [← c68193, ← c68194, ← c68195] at e_1104_16
  have e_1104_17 := arithEq_of_rows h (row := 1104) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_17
  simp only [← c68196, ← c68197, k2, ← c68198] at e_1104_17
  have hr := e_1104_17
  simp only [e_1104_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1104 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9647)) := by
  have c68199 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.1
  have c68200 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.1
  have c68201 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.1
  have c68202 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68203 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68204 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1104_18 := arithEq_of_rows h (row := 1104) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_18
  simp only [← c68199, ← c68200, ← c68201] at e_1104_18
  have e_1104_19 := arithEq_of_rows h (row := 1104) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_19
  simp only [← c68202, ← c68203, k2, ← c68204] at e_1104_19
  have hr := e_1104_19
  simp only [e_1104_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9648)) := by
  have c68205 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68206 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68207 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68208 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68209 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68210 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_0 := arithEq_of_rows h (row := 1105) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_0
  simp only [← c68205, ← c68206, ← c68207] at e_1105_0
  have e_1105_1 := arithEq_of_rows h (row := 1105) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_1
  simp only [← c68208, ← c68209, k2, ← c68210] at e_1105_1
  have hr := e_1105_1
  simp only [e_1105_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9649)) := by
  have c68211 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68212 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68213 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68214 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68215 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68216 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_2 := arithEq_of_rows h (row := 1105) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_2
  simp only [← c68211, ← c68212, ← c68213] at e_1105_2
  have e_1105_3 := arithEq_of_rows h (row := 1105) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_3
  simp only [← c68214, ← c68215, k2, ← c68216] at e_1105_3
  have hr := e_1105_3
  simp only [e_1105_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9650)) := by
  have c68217 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68218 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68219 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68220 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68221 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68222 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_4 := arithEq_of_rows h (row := 1105) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_4
  simp only [← c68217, ← c68218, ← c68219] at e_1105_4
  have e_1105_5 := arithEq_of_rows h (row := 1105) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_5
  simp only [← c68220, ← c68221, k2, ← c68222] at e_1105_5
  have hr := e_1105_5
  simp only [e_1105_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9651)) := by
  have c68223 := (publicBatchWrapper86_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68224 := (publicBatchWrapper86_copies2132 a h).1
  have c68225 := (publicBatchWrapper86_copies2132 a h).2.1
  have c68226 := (publicBatchWrapper86_copies2132 a h).2.2.1
  have c68227 := (publicBatchWrapper86_copies2132 a h).2.2.2.1
  have c68228 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_6 := arithEq_of_rows h (row := 1105) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_6
  simp only [← c68223, ← c68224, ← c68225] at e_1105_6
  have e_1105_7 := arithEq_of_rows h (row := 1105) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_7
  simp only [← c68226, ← c68227, k2, ← c68228] at e_1105_7
  have hr := e_1105_7
  simp only [e_1105_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9652)) := by
  have c68229 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.1
  have c68230 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.1
  have c68231 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.1
  have c68232 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.1
  have c68233 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.1
  have c68234 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_8 := arithEq_of_rows h (row := 1105) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_8
  simp only [← c68229, ← c68230, ← c68231] at e_1105_8
  have e_1105_9 := arithEq_of_rows h (row := 1105) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_9
  simp only [← c68232, ← c68233, k2, ← c68234] at e_1105_9
  have hr := e_1105_9
  simp only [e_1105_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9653)) := by
  have c68235 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68236 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68237 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68238 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68239 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68240 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_10 := arithEq_of_rows h (row := 1105) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_10
  simp only [← c68235, ← c68236, ← c68237] at e_1105_10
  have e_1105_11 := arithEq_of_rows h (row := 1105) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_11
  simp only [← c68238, ← c68239, k2, ← c68240] at e_1105_11
  have hr := e_1105_11
  simp only [e_1105_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9654)) := by
  have c68241 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68242 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68243 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68244 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68245 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68246 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_12 := arithEq_of_rows h (row := 1105) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_12
  simp only [← c68241, ← c68242, ← c68243] at e_1105_12
  have e_1105_13 := arithEq_of_rows h (row := 1105) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_13
  simp only [← c68244, ← c68245, k2, ← c68246] at e_1105_13
  have hr := e_1105_13
  simp only [e_1105_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9655)) := by
  have c68247 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68248 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68249 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68250 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68251 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68252 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_14 := arithEq_of_rows h (row := 1105) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_14
  simp only [← c68247, ← c68248, ← c68249] at e_1105_14
  have e_1105_15 := arithEq_of_rows h (row := 1105) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_15
  simp only [← c68250, ← c68251, k2, ← c68252] at e_1105_15
  have hr := e_1105_15
  simp only [e_1105_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9656)) := by
  have c68253 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68254 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68255 := (publicBatchWrapper86_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68256 := (publicBatchWrapper86_copies2133 a h).1
  have c68257 := (publicBatchWrapper86_copies2133 a h).2.1
  have c68258 := (publicBatchWrapper86_copies2133 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_16 := arithEq_of_rows h (row := 1105) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_16
  simp only [← c68253, ← c68254, ← c68255] at e_1105_16
  have e_1105_17 := arithEq_of_rows h (row := 1105) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_17
  simp only [← c68256, ← c68257, k2, ← c68258] at e_1105_17
  have hr := e_1105_17
  simp only [e_1105_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1105 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9657)) := by
  have c68259 := (publicBatchWrapper86_copies2133 a h).2.2.2.1
  have c68260 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.1
  have c68261 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.1
  have c68262 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.1
  have c68263 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.1
  have c68264 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1105_18 := arithEq_of_rows h (row := 1105) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_18
  simp only [← c68259, ← c68260, ← c68261] at e_1105_18
  have e_1105_19 := arithEq_of_rows h (row := 1105) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_19
  simp only [← c68262, ← c68263, k2, ← c68264] at e_1105_19
  have hr := e_1105_19
  simp only [e_1105_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9658)) := by
  have c68265 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.1
  have c68266 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68267 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68268 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68269 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68270 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_0 := arithEq_of_rows h (row := 1106) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_0
  simp only [← c68265, ← c68266, ← c68267] at e_1106_0
  have e_1106_1 := arithEq_of_rows h (row := 1106) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_1
  simp only [← c68268, ← c68269, k2, ← c68270] at e_1106_1
  have hr := e_1106_1
  simp only [e_1106_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9659)) := by
  have c68271 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68272 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68273 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68274 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68275 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68276 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_2 := arithEq_of_rows h (row := 1106) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_2
  simp only [← c68271, ← c68272, ← c68273] at e_1106_2
  have e_1106_3 := arithEq_of_rows h (row := 1106) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_3
  simp only [← c68274, ← c68275, k2, ← c68276] at e_1106_3
  have hr := e_1106_3
  simp only [e_1106_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9660)) := by
  have c68277 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68278 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68279 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68280 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68281 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68282 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_4 := arithEq_of_rows h (row := 1106) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_4
  simp only [← c68277, ← c68278, ← c68279] at e_1106_4
  have e_1106_5 := arithEq_of_rows h (row := 1106) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_5
  simp only [← c68280, ← c68281, k2, ← c68282] at e_1106_5
  have hr := e_1106_5
  simp only [e_1106_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9661)) := by
  have c68283 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68284 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68285 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68286 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68287 := (publicBatchWrapper86_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68288 := (publicBatchWrapper86_copies2134 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_6 := arithEq_of_rows h (row := 1106) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_6
  simp only [← c68283, ← c68284, ← c68285] at e_1106_6
  have e_1106_7 := arithEq_of_rows h (row := 1106) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_7
  simp only [← c68286, ← c68287, k2, ← c68288] at e_1106_7
  have hr := e_1106_7
  simp only [e_1106_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9662)) := by
  have c68289 := (publicBatchWrapper86_copies2134 a h).2.1
  have c68290 := (publicBatchWrapper86_copies2134 a h).2.2.1
  have c68291 := (publicBatchWrapper86_copies2134 a h).2.2.2.1
  have c68292 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.1
  have c68293 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.1
  have c68294 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_8 := arithEq_of_rows h (row := 1106) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_8
  simp only [← c68289, ← c68290, ← c68291] at e_1106_8
  have e_1106_9 := arithEq_of_rows h (row := 1106) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_9
  simp only [← c68292, ← c68293, k2, ← c68294] at e_1106_9
  have hr := e_1106_9
  simp only [e_1106_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9663)) := by
  have c68295 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.1
  have c68296 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.1
  have c68297 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.1
  have c68298 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68299 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68300 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_10 := arithEq_of_rows h (row := 1106) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_10
  simp only [← c68295, ← c68296, ← c68297] at e_1106_10
  have e_1106_11 := arithEq_of_rows h (row := 1106) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_11
  simp only [← c68298, ← c68299, k2, ← c68300] at e_1106_11
  have hr := e_1106_11
  simp only [e_1106_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9664)) := by
  have c68301 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68302 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68303 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68304 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68305 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68306 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_12 := arithEq_of_rows h (row := 1106) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_12
  simp only [← c68301, ← c68302, ← c68303] at e_1106_12
  have e_1106_13 := arithEq_of_rows h (row := 1106) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_13
  simp only [← c68304, ← c68305, k2, ← c68306] at e_1106_13
  have hr := e_1106_13
  simp only [e_1106_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9665)) := by
  have c68307 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68308 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68309 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68310 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68311 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68312 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_14 := arithEq_of_rows h (row := 1106) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_14
  simp only [← c68307, ← c68308, ← c68309] at e_1106_14
  have e_1106_15 := arithEq_of_rows h (row := 1106) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_15
  simp only [← c68310, ← c68311, k2, ← c68312] at e_1106_15
  have hr := e_1106_15
  simp only [e_1106_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9666)) := by
  have c68313 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68314 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68315 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68316 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68317 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68318 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_16 := arithEq_of_rows h (row := 1106) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_16
  simp only [← c68313, ← c68314, ← c68315] at e_1106_16
  have e_1106_17 := arithEq_of_rows h (row := 1106) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_17
  simp only [← c68316, ← c68317, k2, ← c68318] at e_1106_17
  have hr := e_1106_17
  simp only [e_1106_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1106 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9667)) := by
  have c68319 := (publicBatchWrapper86_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68320 := (publicBatchWrapper86_copies2135 a h).1
  have c68321 := (publicBatchWrapper86_copies2135 a h).2.1
  have c68322 := (publicBatchWrapper86_copies2135 a h).2.2.1
  have c68323 := (publicBatchWrapper86_copies2135 a h).2.2.2.1
  have c68324 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1106_18 := arithEq_of_rows h (row := 1106) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_18
  simp only [← c68319, ← c68320, ← c68321] at e_1106_18
  have e_1106_19 := arithEq_of_rows h (row := 1106) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_19
  simp only [← c68322, ← c68323, k2, ← c68324] at e_1106_19
  have hr := e_1106_19
  simp only [e_1106_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9668)) := by
  have c68325 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.1
  have c68326 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.1
  have c68327 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.1
  have c68328 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.1
  have c68329 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.1
  have c68330 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_0 := arithEq_of_rows h (row := 1107) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_0
  simp only [← c68325, ← c68326, ← c68327] at e_1107_0
  have e_1107_1 := arithEq_of_rows h (row := 1107) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_1
  simp only [← c68328, ← c68329, k2, ← c68330] at e_1107_1
  have hr := e_1107_1
  simp only [e_1107_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9669)) := by
  have c68331 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68332 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68333 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68334 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68335 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68336 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_2 := arithEq_of_rows h (row := 1107) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_2
  simp only [← c68331, ← c68332, ← c68333] at e_1107_2
  have e_1107_3 := arithEq_of_rows h (row := 1107) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_3
  simp only [← c68334, ← c68335, k2, ← c68336] at e_1107_3
  have hr := e_1107_3
  simp only [e_1107_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9670)) := by
  have c68337 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68338 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68339 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68340 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68341 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68342 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_4 := arithEq_of_rows h (row := 1107) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_4
  simp only [← c68337, ← c68338, ← c68339] at e_1107_4
  have e_1107_5 := arithEq_of_rows h (row := 1107) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_5
  simp only [← c68340, ← c68341, k2, ← c68342] at e_1107_5
  have hr := e_1107_5
  simp only [e_1107_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9671)) := by
  have c68343 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68344 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68345 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68346 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68347 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68348 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_6 := arithEq_of_rows h (row := 1107) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_6
  simp only [← c68343, ← c68344, ← c68345] at e_1107_6
  have e_1107_7 := arithEq_of_rows h (row := 1107) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_7
  simp only [← c68346, ← c68347, k2, ← c68348] at e_1107_7
  have hr := e_1107_7
  simp only [e_1107_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9672)) := by
  have c68349 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68350 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68351 := (publicBatchWrapper86_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68352 := (publicBatchWrapper86_copies2136 a h).1
  have c68353 := (publicBatchWrapper86_copies2136 a h).2.1
  have c68354 := (publicBatchWrapper86_copies2136 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_8 := arithEq_of_rows h (row := 1107) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_8
  simp only [← c68349, ← c68350, ← c68351] at e_1107_8
  have e_1107_9 := arithEq_of_rows h (row := 1107) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_9
  simp only [← c68352, ← c68353, k2, ← c68354] at e_1107_9
  have hr := e_1107_9
  simp only [e_1107_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9673)) := by
  have c68355 := (publicBatchWrapper86_copies2136 a h).2.2.2.1
  have c68356 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.1
  have c68357 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.1
  have c68358 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.1
  have c68359 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.1
  have c68360 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_10 := arithEq_of_rows h (row := 1107) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_10
  simp only [← c68355, ← c68356, ← c68357] at e_1107_10
  have e_1107_11 := arithEq_of_rows h (row := 1107) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_11
  simp only [← c68358, ← c68359, k2, ← c68360] at e_1107_11
  have hr := e_1107_11
  simp only [e_1107_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9674)) := by
  have c68361 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.1
  have c68362 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68363 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68364 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68365 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68366 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_12 := arithEq_of_rows h (row := 1107) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_12
  simp only [← c68361, ← c68362, ← c68363] at e_1107_12
  have e_1107_13 := arithEq_of_rows h (row := 1107) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_13
  simp only [← c68364, ← c68365, k2, ← c68366] at e_1107_13
  have hr := e_1107_13
  simp only [e_1107_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9675)) := by
  have c68367 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68368 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68369 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68370 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68371 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68372 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_14 := arithEq_of_rows h (row := 1107) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_14
  simp only [← c68367, ← c68368, ← c68369] at e_1107_14
  have e_1107_15 := arithEq_of_rows h (row := 1107) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_15
  simp only [← c68370, ← c68371, k2, ← c68372] at e_1107_15
  have hr := e_1107_15
  simp only [e_1107_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9676)) := by
  have c68373 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68374 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68375 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68376 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68377 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68378 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_16 := arithEq_of_rows h (row := 1107) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_16
  simp only [← c68373, ← c68374, ← c68375] at e_1107_16
  have e_1107_17 := arithEq_of_rows h (row := 1107) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_17
  simp only [← c68376, ← c68377, k2, ← c68378] at e_1107_17
  have hr := e_1107_17
  simp only [e_1107_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1107 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9677)) := by
  have c68379 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68380 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68381 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68382 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68383 := (publicBatchWrapper86_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68384 := (publicBatchWrapper86_copies2137 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1107_18 := arithEq_of_rows h (row := 1107) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_18
  simp only [← c68379, ← c68380, ← c68381] at e_1107_18
  have e_1107_19 := arithEq_of_rows h (row := 1107) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_19
  simp only [← c68382, ← c68383, k2, ← c68384] at e_1107_19
  have hr := e_1107_19
  simp only [e_1107_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9678)) := by
  have c68385 := (publicBatchWrapper86_copies2137 a h).2.1
  have c68386 := (publicBatchWrapper86_copies2137 a h).2.2.1
  have c68387 := (publicBatchWrapper86_copies2137 a h).2.2.2.1
  have c68388 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.1
  have c68389 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.1
  have c68390 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_0 := arithEq_of_rows h (row := 1108) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_0
  simp only [← c68385, ← c68386, ← c68387] at e_1108_0
  have e_1108_1 := arithEq_of_rows h (row := 1108) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_1
  simp only [← c68388, ← c68389, k2, ← c68390] at e_1108_1
  have hr := e_1108_1
  simp only [e_1108_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9679)) := by
  have c68391 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.1
  have c68392 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.1
  have c68393 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.1
  have c68394 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68395 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68396 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_2 := arithEq_of_rows h (row := 1108) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_2
  simp only [← c68391, ← c68392, ← c68393] at e_1108_2
  have e_1108_3 := arithEq_of_rows h (row := 1108) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_3
  simp only [← c68394, ← c68395, k2, ← c68396] at e_1108_3
  have hr := e_1108_3
  simp only [e_1108_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9680)) := by
  have c68397 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68398 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68399 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68400 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68401 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68402 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_4 := arithEq_of_rows h (row := 1108) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_4
  simp only [← c68397, ← c68398, ← c68399] at e_1108_4
  have e_1108_5 := arithEq_of_rows h (row := 1108) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_5
  simp only [← c68400, ← c68401, k2, ← c68402] at e_1108_5
  have hr := e_1108_5
  simp only [e_1108_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9681)) := by
  have c68403 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68404 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68405 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68406 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68407 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68408 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_6 := arithEq_of_rows h (row := 1108) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_6
  simp only [← c68403, ← c68404, ← c68405] at e_1108_6
  have e_1108_7 := arithEq_of_rows h (row := 1108) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_7
  simp only [← c68406, ← c68407, k2, ← c68408] at e_1108_7
  have hr := e_1108_7
  simp only [e_1108_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9682)) := by
  have c68409 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68410 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68411 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68412 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68413 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68414 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_8 := arithEq_of_rows h (row := 1108) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_8
  simp only [← c68409, ← c68410, ← c68411] at e_1108_8
  have e_1108_9 := arithEq_of_rows h (row := 1108) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_9
  simp only [← c68412, ← c68413, k2, ← c68414] at e_1108_9
  have hr := e_1108_9
  simp only [e_1108_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9683)) := by
  have c68415 := (publicBatchWrapper86_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68416 := (publicBatchWrapper86_copies2138 a h).1
  have c68417 := (publicBatchWrapper86_copies2138 a h).2.1
  have c68418 := (publicBatchWrapper86_copies2138 a h).2.2.1
  have c68419 := (publicBatchWrapper86_copies2138 a h).2.2.2.1
  have c68420 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_10 := arithEq_of_rows h (row := 1108) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_10
  simp only [← c68415, ← c68416, ← c68417] at e_1108_10
  have e_1108_11 := arithEq_of_rows h (row := 1108) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_11
  simp only [← c68418, ← c68419, k2, ← c68420] at e_1108_11
  have hr := e_1108_11
  simp only [e_1108_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9684)) := by
  have c68421 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.1
  have c68422 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.1
  have c68423 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.1
  have c68424 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.1
  have c68425 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.1
  have c68426 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_12 := arithEq_of_rows h (row := 1108) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_12
  simp only [← c68421, ← c68422, ← c68423] at e_1108_12
  have e_1108_13 := arithEq_of_rows h (row := 1108) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_13
  simp only [← c68424, ← c68425, k2, ← c68426] at e_1108_13
  have hr := e_1108_13
  simp only [e_1108_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9685)) := by
  have c68427 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68428 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68429 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68430 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68431 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68432 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_14 := arithEq_of_rows h (row := 1108) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_14
  simp only [← c68427, ← c68428, ← c68429] at e_1108_14
  have e_1108_15 := arithEq_of_rows h (row := 1108) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_15
  simp only [← c68430, ← c68431, k2, ← c68432] at e_1108_15
  have hr := e_1108_15
  simp only [e_1108_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9686)) := by
  have c68433 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68434 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68435 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68436 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68437 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68438 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_16 := arithEq_of_rows h (row := 1108) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_16
  simp only [← c68433, ← c68434, ← c68435] at e_1108_16
  have e_1108_17 := arithEq_of_rows h (row := 1108) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_17
  simp only [← c68436, ← c68437, k2, ← c68438] at e_1108_17
  have hr := e_1108_17
  simp only [e_1108_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1108 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9687)) := by
  have c68439 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68440 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68441 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68442 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68443 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68444 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1108_18 := arithEq_of_rows h (row := 1108) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_18
  simp only [← c68439, ← c68440, ← c68441] at e_1108_18
  have e_1108_19 := arithEq_of_rows h (row := 1108) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_19
  simp only [← c68442, ← c68443, k2, ← c68444] at e_1108_19
  have hr := e_1108_19
  simp only [e_1108_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9688)) := by
  have c68445 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68446 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68447 := (publicBatchWrapper86_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68448 := (publicBatchWrapper86_copies2139 a h).1
  have c68449 := (publicBatchWrapper86_copies2139 a h).2.1
  have c68450 := (publicBatchWrapper86_copies2139 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_0 := arithEq_of_rows h (row := 1109) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_0
  simp only [← c68445, ← c68446, ← c68447] at e_1109_0
  have e_1109_1 := arithEq_of_rows h (row := 1109) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_1
  simp only [← c68448, ← c68449, k2, ← c68450] at e_1109_1
  have hr := e_1109_1
  simp only [e_1109_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9689)) := by
  have c68451 := (publicBatchWrapper86_copies2139 a h).2.2.2.1
  have c68452 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.1
  have c68453 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.1
  have c68454 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.1
  have c68455 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.1
  have c68456 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_2 := arithEq_of_rows h (row := 1109) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_2
  simp only [← c68451, ← c68452, ← c68453] at e_1109_2
  have e_1109_3 := arithEq_of_rows h (row := 1109) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_3
  simp only [← c68454, ← c68455, k2, ← c68456] at e_1109_3
  have hr := e_1109_3
  simp only [e_1109_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9690)) := by
  have c68457 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.1
  have c68458 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68459 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68460 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68461 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68462 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_4 := arithEq_of_rows h (row := 1109) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_4
  simp only [← c68457, ← c68458, ← c68459] at e_1109_4
  have e_1109_5 := arithEq_of_rows h (row := 1109) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_5
  simp only [← c68460, ← c68461, k2, ← c68462] at e_1109_5
  have hr := e_1109_5
  simp only [e_1109_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9691)) := by
  have c68463 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68464 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68465 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68466 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68467 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68468 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_6 := arithEq_of_rows h (row := 1109) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_6
  simp only [← c68463, ← c68464, ← c68465] at e_1109_6
  have e_1109_7 := arithEq_of_rows h (row := 1109) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_7
  simp only [← c68466, ← c68467, k2, ← c68468] at e_1109_7
  have hr := e_1109_7
  simp only [e_1109_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9692)) := by
  have c68469 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68470 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68471 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68472 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68473 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68474 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_8 := arithEq_of_rows h (row := 1109) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_8
  simp only [← c68469, ← c68470, ← c68471] at e_1109_8
  have e_1109_9 := arithEq_of_rows h (row := 1109) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_9
  simp only [← c68472, ← c68473, k2, ← c68474] at e_1109_9
  have hr := e_1109_9
  simp only [e_1109_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9693)) := by
  have c68475 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68476 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68477 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68478 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68479 := (publicBatchWrapper86_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68480 := (publicBatchWrapper86_copies2140 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_10 := arithEq_of_rows h (row := 1109) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_10
  simp only [← c68475, ← c68476, ← c68477] at e_1109_10
  have e_1109_11 := arithEq_of_rows h (row := 1109) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_11
  simp only [← c68478, ← c68479, k2, ← c68480] at e_1109_11
  have hr := e_1109_11
  simp only [e_1109_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9694)) := by
  have c68481 := (publicBatchWrapper86_copies2140 a h).2.1
  have c68482 := (publicBatchWrapper86_copies2140 a h).2.2.1
  have c68483 := (publicBatchWrapper86_copies2140 a h).2.2.2.1
  have c68484 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.1
  have c68485 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.1
  have c68486 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_12 := arithEq_of_rows h (row := 1109) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_12
  simp only [← c68481, ← c68482, ← c68483] at e_1109_12
  have e_1109_13 := arithEq_of_rows h (row := 1109) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_13
  simp only [← c68484, ← c68485, k2, ← c68486] at e_1109_13
  have hr := e_1109_13
  simp only [e_1109_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9695)) := by
  have c68487 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.1
  have c68488 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.1
  have c68489 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.1
  have c68490 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68491 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68492 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_14 := arithEq_of_rows h (row := 1109) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_14
  simp only [← c68487, ← c68488, ← c68489] at e_1109_14
  have e_1109_15 := arithEq_of_rows h (row := 1109) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_15
  simp only [← c68490, ← c68491, k2, ← c68492] at e_1109_15
  have hr := e_1109_15
  simp only [e_1109_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9696)) := by
  have c68493 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68494 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68495 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68496 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68497 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68498 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_16 := arithEq_of_rows h (row := 1109) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_16
  simp only [← c68493, ← c68494, ← c68495] at e_1109_16
  have e_1109_17 := arithEq_of_rows h (row := 1109) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_17
  simp only [← c68496, ← c68497, k2, ← c68498] at e_1109_17
  have hr := e_1109_17
  simp only [e_1109_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1109 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9697)) := by
  have c68499 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68500 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68501 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68502 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68503 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68504 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1109_18 := arithEq_of_rows h (row := 1109) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_18
  simp only [← c68499, ← c68500, ← c68501] at e_1109_18
  have e_1109_19 := arithEq_of_rows h (row := 1109) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_19
  simp only [← c68502, ← c68503, k2, ← c68504] at e_1109_19
  have hr := e_1109_19
  simp only [e_1109_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9698)) := by
  have c68505 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68506 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68507 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68508 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68509 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68510 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_0 := arithEq_of_rows h (row := 1110) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_0
  simp only [← c68505, ← c68506, ← c68507] at e_1110_0
  have e_1110_1 := arithEq_of_rows h (row := 1110) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_1
  simp only [← c68508, ← c68509, k2, ← c68510] at e_1110_1
  have hr := e_1110_1
  simp only [e_1110_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9699)) := by
  have c68511 := (publicBatchWrapper86_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68512 := (publicBatchWrapper86_copies2141 a h).1
  have c68513 := (publicBatchWrapper86_copies2141 a h).2.1
  have c68514 := (publicBatchWrapper86_copies2141 a h).2.2.1
  have c68515 := (publicBatchWrapper86_copies2141 a h).2.2.2.1
  have c68516 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_2 := arithEq_of_rows h (row := 1110) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_2
  simp only [← c68511, ← c68512, ← c68513] at e_1110_2
  have e_1110_3 := arithEq_of_rows h (row := 1110) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_3
  simp only [← c68514, ← c68515, k2, ← c68516] at e_1110_3
  have hr := e_1110_3
  simp only [e_1110_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9700)) := by
  have c68517 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.1
  have c68518 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.1
  have c68519 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.1
  have c68520 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.1
  have c68521 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.1
  have c68522 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_4 := arithEq_of_rows h (row := 1110) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_4
  simp only [← c68517, ← c68518, ← c68519] at e_1110_4
  have e_1110_5 := arithEq_of_rows h (row := 1110) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_5
  simp only [← c68520, ← c68521, k2, ← c68522] at e_1110_5
  have hr := e_1110_5
  simp only [e_1110_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9701)) := by
  have c68523 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68524 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68525 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68526 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68527 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68528 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_6 := arithEq_of_rows h (row := 1110) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_6
  simp only [← c68523, ← c68524, ← c68525] at e_1110_6
  have e_1110_7 := arithEq_of_rows h (row := 1110) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_7
  simp only [← c68526, ← c68527, k2, ← c68528] at e_1110_7
  have hr := e_1110_7
  simp only [e_1110_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9702)) := by
  have c68529 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68530 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68531 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68532 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68533 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68534 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_8 := arithEq_of_rows h (row := 1110) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_8
  simp only [← c68529, ← c68530, ← c68531] at e_1110_8
  have e_1110_9 := arithEq_of_rows h (row := 1110) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_9
  simp only [← c68532, ← c68533, k2, ← c68534] at e_1110_9
  have hr := e_1110_9
  simp only [e_1110_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9703)) := by
  have c68535 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68536 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68537 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68538 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68539 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68540 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_10 := arithEq_of_rows h (row := 1110) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_10
  simp only [← c68535, ← c68536, ← c68537] at e_1110_10
  have e_1110_11 := arithEq_of_rows h (row := 1110) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_11
  simp only [← c68538, ← c68539, k2, ← c68540] at e_1110_11
  have hr := e_1110_11
  simp only [e_1110_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9704)) := by
  have c68541 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68542 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68543 := (publicBatchWrapper86_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68544 := (publicBatchWrapper86_copies2142 a h).1
  have c68545 := (publicBatchWrapper86_copies2142 a h).2.1
  have c68546 := (publicBatchWrapper86_copies2142 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_12 := arithEq_of_rows h (row := 1110) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_12
  simp only [← c68541, ← c68542, ← c68543] at e_1110_12
  have e_1110_13 := arithEq_of_rows h (row := 1110) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_13
  simp only [← c68544, ← c68545, k2, ← c68546] at e_1110_13
  have hr := e_1110_13
  simp only [e_1110_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9705)) := by
  have c68547 := (publicBatchWrapper86_copies2142 a h).2.2.2.1
  have c68548 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.1
  have c68549 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.1
  have c68550 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.1
  have c68551 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.1
  have c68552 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_14 := arithEq_of_rows h (row := 1110) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_14
  simp only [← c68547, ← c68548, ← c68549] at e_1110_14
  have e_1110_15 := arithEq_of_rows h (row := 1110) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_15
  simp only [← c68550, ← c68551, k2, ← c68552] at e_1110_15
  have hr := e_1110_15
  simp only [e_1110_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9706)) := by
  have c68553 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.1
  have c68554 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68555 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68556 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68557 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68558 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_16 := arithEq_of_rows h (row := 1110) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_16
  simp only [← c68553, ← c68554, ← c68555] at e_1110_16
  have e_1110_17 := arithEq_of_rows h (row := 1110) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_17
  simp only [← c68556, ← c68557, k2, ← c68558] at e_1110_17
  have hr := e_1110_17
  simp only [e_1110_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1110 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9707)) := by
  have c68559 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68560 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68561 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68562 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68563 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68564 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1110_18 := arithEq_of_rows h (row := 1110) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_18
  simp only [← c68559, ← c68560, ← c68561] at e_1110_18
  have e_1110_19 := arithEq_of_rows h (row := 1110) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_19
  simp only [← c68562, ← c68563, k2, ← c68564] at e_1110_19
  have hr := e_1110_19
  simp only [e_1110_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9708)) := by
  have c68565 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68566 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68567 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68568 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68569 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68570 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_0 := arithEq_of_rows h (row := 1111) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_0
  simp only [← c68565, ← c68566, ← c68567] at e_1111_0
  have e_1111_1 := arithEq_of_rows h (row := 1111) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_1
  simp only [← c68568, ← c68569, k2, ← c68570] at e_1111_1
  have hr := e_1111_1
  simp only [e_1111_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9709)) := by
  have c68571 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68572 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68573 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68574 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68575 := (publicBatchWrapper86_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68576 := (publicBatchWrapper86_copies2143 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_2 := arithEq_of_rows h (row := 1111) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_2
  simp only [← c68571, ← c68572, ← c68573] at e_1111_2
  have e_1111_3 := arithEq_of_rows h (row := 1111) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_3
  simp only [← c68574, ← c68575, k2, ← c68576] at e_1111_3
  have hr := e_1111_3
  simp only [e_1111_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9710)) := by
  have c68577 := (publicBatchWrapper86_copies2143 a h).2.1
  have c68578 := (publicBatchWrapper86_copies2143 a h).2.2.1
  have c68579 := (publicBatchWrapper86_copies2143 a h).2.2.2.1
  have c68580 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.1
  have c68581 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.1
  have c68582 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_4 := arithEq_of_rows h (row := 1111) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_4
  simp only [← c68577, ← c68578, ← c68579] at e_1111_4
  have e_1111_5 := arithEq_of_rows h (row := 1111) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_5
  simp only [← c68580, ← c68581, k2, ← c68582] at e_1111_5
  have hr := e_1111_5
  simp only [e_1111_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9711)) := by
  have c68583 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.1
  have c68584 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.1
  have c68585 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.1
  have c68586 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68587 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68588 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_6 := arithEq_of_rows h (row := 1111) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_6
  simp only [← c68583, ← c68584, ← c68585] at e_1111_6
  have e_1111_7 := arithEq_of_rows h (row := 1111) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_7
  simp only [← c68586, ← c68587, k2, ← c68588] at e_1111_7
  have hr := e_1111_7
  simp only [e_1111_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9712)) := by
  have c68589 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68590 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68591 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68592 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68593 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68594 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_8 := arithEq_of_rows h (row := 1111) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_8
  simp only [← c68589, ← c68590, ← c68591] at e_1111_8
  have e_1111_9 := arithEq_of_rows h (row := 1111) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_9
  simp only [← c68592, ← c68593, k2, ← c68594] at e_1111_9
  have hr := e_1111_9
  simp only [e_1111_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9713)) := by
  have c68595 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68596 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68597 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68598 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68599 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68600 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_10 := arithEq_of_rows h (row := 1111) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_10
  simp only [← c68595, ← c68596, ← c68597] at e_1111_10
  have e_1111_11 := arithEq_of_rows h (row := 1111) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_11
  simp only [← c68598, ← c68599, k2, ← c68600] at e_1111_11
  have hr := e_1111_11
  simp only [e_1111_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9714)) := by
  have c68601 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68602 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68603 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68604 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68605 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68606 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_12 := arithEq_of_rows h (row := 1111) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_12
  simp only [← c68601, ← c68602, ← c68603] at e_1111_12
  have e_1111_13 := arithEq_of_rows h (row := 1111) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_13
  simp only [← c68604, ← c68605, k2, ← c68606] at e_1111_13
  have hr := e_1111_13
  simp only [e_1111_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9715)) := by
  have c68607 := (publicBatchWrapper86_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68608 := (publicBatchWrapper86_copies2144 a h).1
  have c68609 := (publicBatchWrapper86_copies2144 a h).2.1
  have c68610 := (publicBatchWrapper86_copies2144 a h).2.2.1
  have c68611 := (publicBatchWrapper86_copies2144 a h).2.2.2.1
  have c68612 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_14 := arithEq_of_rows h (row := 1111) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_14
  simp only [← c68607, ← c68608, ← c68609] at e_1111_14
  have e_1111_15 := arithEq_of_rows h (row := 1111) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_15
  simp only [← c68610, ← c68611, k2, ← c68612] at e_1111_15
  have hr := e_1111_15
  simp only [e_1111_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9716)) := by
  have c68613 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.1
  have c68614 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.1
  have c68615 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.1
  have c68616 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.1
  have c68617 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.1
  have c68618 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_16 := arithEq_of_rows h (row := 1111) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_16
  simp only [← c68613, ← c68614, ← c68615] at e_1111_16
  have e_1111_17 := arithEq_of_rows h (row := 1111) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_17
  simp only [← c68616, ← c68617, k2, ← c68618] at e_1111_17
  have hr := e_1111_17
  simp only [e_1111_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1111 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9717)) := by
  have c68619 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68620 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68621 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68622 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68623 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68624 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1111_18 := arithEq_of_rows h (row := 1111) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_18
  simp only [← c68619, ← c68620, ← c68621] at e_1111_18
  have e_1111_19 := arithEq_of_rows h (row := 1111) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_19
  simp only [← c68622, ← c68623, k2, ← c68624] at e_1111_19
  have hr := e_1111_19
  simp only [e_1111_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9718)) := by
  have c68625 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68626 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68627 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68628 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68629 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68630 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_0 := arithEq_of_rows h (row := 1112) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_0
  simp only [← c68625, ← c68626, ← c68627] at e_1112_0
  have e_1112_1 := arithEq_of_rows h (row := 1112) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_1
  simp only [← c68628, ← c68629, k2, ← c68630] at e_1112_1
  have hr := e_1112_1
  simp only [e_1112_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9719)) := by
  have c68631 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68632 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68633 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68634 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68635 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68636 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_2 := arithEq_of_rows h (row := 1112) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_2
  simp only [← c68631, ← c68632, ← c68633] at e_1112_2
  have e_1112_3 := arithEq_of_rows h (row := 1112) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_3
  simp only [← c68634, ← c68635, k2, ← c68636] at e_1112_3
  have hr := e_1112_3
  simp only [e_1112_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9720)) := by
  have c68637 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68638 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68639 := (publicBatchWrapper86_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68640 := (publicBatchWrapper86_copies2145 a h).1
  have c68641 := (publicBatchWrapper86_copies2145 a h).2.1
  have c68642 := (publicBatchWrapper86_copies2145 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_4 := arithEq_of_rows h (row := 1112) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_4
  simp only [← c68637, ← c68638, ← c68639] at e_1112_4
  have e_1112_5 := arithEq_of_rows h (row := 1112) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_5
  simp only [← c68640, ← c68641, k2, ← c68642] at e_1112_5
  have hr := e_1112_5
  simp only [e_1112_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9721)) := by
  have c68643 := (publicBatchWrapper86_copies2145 a h).2.2.2.1
  have c68644 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.1
  have c68645 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.1
  have c68646 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.1
  have c68647 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.1
  have c68648 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_6 := arithEq_of_rows h (row := 1112) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_6
  simp only [← c68643, ← c68644, ← c68645] at e_1112_6
  have e_1112_7 := arithEq_of_rows h (row := 1112) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_7
  simp only [← c68646, ← c68647, k2, ← c68648] at e_1112_7
  have hr := e_1112_7
  simp only [e_1112_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9722)) := by
  have c68649 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.1
  have c68650 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68651 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68652 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68653 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68654 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_8 := arithEq_of_rows h (row := 1112) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_8
  simp only [← c68649, ← c68650, ← c68651] at e_1112_8
  have e_1112_9 := arithEq_of_rows h (row := 1112) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_9
  simp only [← c68652, ← c68653, k2, ← c68654] at e_1112_9
  have hr := e_1112_9
  simp only [e_1112_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9723)) := by
  have c68655 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68656 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68657 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68658 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68659 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68660 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_10 := arithEq_of_rows h (row := 1112) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_10
  simp only [← c68655, ← c68656, ← c68657] at e_1112_10
  have e_1112_11 := arithEq_of_rows h (row := 1112) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_11
  simp only [← c68658, ← c68659, k2, ← c68660] at e_1112_11
  have hr := e_1112_11
  simp only [e_1112_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9724)) := by
  have c68661 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68662 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68663 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68664 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68665 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68666 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_12 := arithEq_of_rows h (row := 1112) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_12
  simp only [← c68661, ← c68662, ← c68663] at e_1112_12
  have e_1112_13 := arithEq_of_rows h (row := 1112) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_13
  simp only [← c68664, ← c68665, k2, ← c68666] at e_1112_13
  have hr := e_1112_13
  simp only [e_1112_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9725)) := by
  have c68667 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68668 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68669 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68670 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68671 := (publicBatchWrapper86_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68672 := (publicBatchWrapper86_copies2146 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_14 := arithEq_of_rows h (row := 1112) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_14
  simp only [← c68667, ← c68668, ← c68669] at e_1112_14
  have e_1112_15 := arithEq_of_rows h (row := 1112) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_15
  simp only [← c68670, ← c68671, k2, ← c68672] at e_1112_15
  have hr := e_1112_15
  simp only [e_1112_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9726)) := by
  have c68673 := (publicBatchWrapper86_copies2146 a h).2.1
  have c68674 := (publicBatchWrapper86_copies2146 a h).2.2.1
  have c68675 := (publicBatchWrapper86_copies2146 a h).2.2.2.1
  have c68676 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.1
  have c68677 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.1
  have c68678 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_16 := arithEq_of_rows h (row := 1112) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_16
  simp only [← c68673, ← c68674, ← c68675] at e_1112_16
  have e_1112_17 := arithEq_of_rows h (row := 1112) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_17
  simp only [← c68676, ← c68677, k2, ← c68678] at e_1112_17
  have hr := e_1112_17
  simp only [e_1112_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1112 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9727)) := by
  have c68679 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.1
  have c68680 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.1
  have c68681 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.1
  have c68682 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68683 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68684 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1112_18 := arithEq_of_rows h (row := 1112) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_18
  simp only [← c68679, ← c68680, ← c68681] at e_1112_18
  have e_1112_19 := arithEq_of_rows h (row := 1112) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_19
  simp only [← c68682, ← c68683, k2, ← c68684] at e_1112_19
  have hr := e_1112_19
  simp only [e_1112_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9728)) := by
  have c68685 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68686 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68687 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68688 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68689 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68690 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_0 := arithEq_of_rows h (row := 1113) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_0
  simp only [← c68685, ← c68686, ← c68687] at e_1113_0
  have e_1113_1 := arithEq_of_rows h (row := 1113) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_1
  simp only [← c68688, ← c68689, k2, ← c68690] at e_1113_1
  have hr := e_1113_1
  simp only [e_1113_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9729)) := by
  have c68691 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68692 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68693 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68694 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68695 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68696 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_2 := arithEq_of_rows h (row := 1113) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_2
  simp only [← c68691, ← c68692, ← c68693] at e_1113_2
  have e_1113_3 := arithEq_of_rows h (row := 1113) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_3
  simp only [← c68694, ← c68695, k2, ← c68696] at e_1113_3
  have hr := e_1113_3
  simp only [e_1113_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9730)) := by
  have c68697 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68698 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68699 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68700 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68701 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68702 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_4 := arithEq_of_rows h (row := 1113) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_4
  simp only [← c68697, ← c68698, ← c68699] at e_1113_4
  have e_1113_5 := arithEq_of_rows h (row := 1113) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_5
  simp only [← c68700, ← c68701, k2, ← c68702] at e_1113_5
  have hr := e_1113_5
  simp only [e_1113_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9731)) := by
  have c68703 := (publicBatchWrapper86_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68704 := (publicBatchWrapper86_copies2147 a h).1
  have c68705 := (publicBatchWrapper86_copies2147 a h).2.1
  have c68706 := (publicBatchWrapper86_copies2147 a h).2.2.1
  have c68707 := (publicBatchWrapper86_copies2147 a h).2.2.2.1
  have c68708 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_6 := arithEq_of_rows h (row := 1113) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_6
  simp only [← c68703, ← c68704, ← c68705] at e_1113_6
  have e_1113_7 := arithEq_of_rows h (row := 1113) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_7
  simp only [← c68706, ← c68707, k2, ← c68708] at e_1113_7
  have hr := e_1113_7
  simp only [e_1113_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9732)) := by
  have c68709 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.1
  have c68710 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.1
  have c68711 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.1
  have c68712 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.1
  have c68713 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.1
  have c68714 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_8 := arithEq_of_rows h (row := 1113) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_8
  simp only [← c68709, ← c68710, ← c68711] at e_1113_8
  have e_1113_9 := arithEq_of_rows h (row := 1113) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_9
  simp only [← c68712, ← c68713, k2, ← c68714] at e_1113_9
  have hr := e_1113_9
  simp only [e_1113_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9733)) := by
  have c68715 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68716 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68717 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68718 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68719 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68720 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_10 := arithEq_of_rows h (row := 1113) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_10
  simp only [← c68715, ← c68716, ← c68717] at e_1113_10
  have e_1113_11 := arithEq_of_rows h (row := 1113) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_11
  simp only [← c68718, ← c68719, k2, ← c68720] at e_1113_11
  have hr := e_1113_11
  simp only [e_1113_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9734)) := by
  have c68721 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68722 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68723 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68724 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68725 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68726 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_12 := arithEq_of_rows h (row := 1113) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_12
  simp only [← c68721, ← c68722, ← c68723] at e_1113_12
  have e_1113_13 := arithEq_of_rows h (row := 1113) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_13
  simp only [← c68724, ← c68725, k2, ← c68726] at e_1113_13
  have hr := e_1113_13
  simp only [e_1113_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9735)) := by
  have c68727 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68728 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68729 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68730 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68731 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68732 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_14 := arithEq_of_rows h (row := 1113) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_14
  simp only [← c68727, ← c68728, ← c68729] at e_1113_14
  have e_1113_15 := arithEq_of_rows h (row := 1113) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_15
  simp only [← c68730, ← c68731, k2, ← c68732] at e_1113_15
  have hr := e_1113_15
  simp only [e_1113_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9736)) := by
  have c68733 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68734 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68735 := (publicBatchWrapper86_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68736 := (publicBatchWrapper86_copies2148 a h).1
  have c68737 := (publicBatchWrapper86_copies2148 a h).2.1
  have c68738 := (publicBatchWrapper86_copies2148 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_16 := arithEq_of_rows h (row := 1113) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_16
  simp only [← c68733, ← c68734, ← c68735] at e_1113_16
  have e_1113_17 := arithEq_of_rows h (row := 1113) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_17
  simp only [← c68736, ← c68737, k2, ← c68738] at e_1113_17
  have hr := e_1113_17
  simp only [e_1113_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1113 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9737)) := by
  have c68739 := (publicBatchWrapper86_copies2148 a h).2.2.2.1
  have c68740 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.1
  have c68741 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.1
  have c68742 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.1
  have c68743 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.1
  have c68744 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1113_18 := arithEq_of_rows h (row := 1113) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_18
  simp only [← c68739, ← c68740, ← c68741] at e_1113_18
  have e_1113_19 := arithEq_of_rows h (row := 1113) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_19
  simp only [← c68742, ← c68743, k2, ← c68744] at e_1113_19
  have hr := e_1113_19
  simp only [e_1113_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9738)) := by
  have c68745 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.1
  have c68746 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68747 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68748 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68749 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68750 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_0 := arithEq_of_rows h (row := 1114) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_0
  simp only [← c68745, ← c68746, ← c68747] at e_1114_0
  have e_1114_1 := arithEq_of_rows h (row := 1114) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_1
  simp only [← c68748, ← c68749, k2, ← c68750] at e_1114_1
  have hr := e_1114_1
  simp only [e_1114_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9739)) := by
  have c68751 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68752 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68753 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68754 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68755 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68756 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_2 := arithEq_of_rows h (row := 1114) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_2
  simp only [← c68751, ← c68752, ← c68753] at e_1114_2
  have e_1114_3 := arithEq_of_rows h (row := 1114) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_3
  simp only [← c68754, ← c68755, k2, ← c68756] at e_1114_3
  have hr := e_1114_3
  simp only [e_1114_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9740)) := by
  have c68757 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68758 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68759 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68760 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68761 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68762 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_4 := arithEq_of_rows h (row := 1114) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_4
  simp only [← c68757, ← c68758, ← c68759] at e_1114_4
  have e_1114_5 := arithEq_of_rows h (row := 1114) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_5
  simp only [← c68760, ← c68761, k2, ← c68762] at e_1114_5
  have hr := e_1114_5
  simp only [e_1114_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9741)) := by
  have c68763 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68764 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68765 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68766 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68767 := (publicBatchWrapper86_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68768 := (publicBatchWrapper86_copies2149 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_6 := arithEq_of_rows h (row := 1114) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_6
  simp only [← c68763, ← c68764, ← c68765] at e_1114_6
  have e_1114_7 := arithEq_of_rows h (row := 1114) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_7
  simp only [← c68766, ← c68767, k2, ← c68768] at e_1114_7
  have hr := e_1114_7
  simp only [e_1114_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9742)) := by
  have c68769 := (publicBatchWrapper86_copies2149 a h).2.1
  have c68770 := (publicBatchWrapper86_copies2149 a h).2.2.1
  have c68771 := (publicBatchWrapper86_copies2149 a h).2.2.2.1
  have c68772 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.1
  have c68773 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.1
  have c68774 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_8 := arithEq_of_rows h (row := 1114) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_8
  simp only [← c68769, ← c68770, ← c68771] at e_1114_8
  have e_1114_9 := arithEq_of_rows h (row := 1114) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_9
  simp only [← c68772, ← c68773, k2, ← c68774] at e_1114_9
  have hr := e_1114_9
  simp only [e_1114_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9743)) := by
  have c68775 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.1
  have c68776 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.1
  have c68777 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.1
  have c68778 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68779 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68780 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_10 := arithEq_of_rows h (row := 1114) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_10
  simp only [← c68775, ← c68776, ← c68777] at e_1114_10
  have e_1114_11 := arithEq_of_rows h (row := 1114) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_11
  simp only [← c68778, ← c68779, k2, ← c68780] at e_1114_11
  have hr := e_1114_11
  simp only [e_1114_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9744)) := by
  have c68781 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68782 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68783 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68784 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68785 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68786 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_12 := arithEq_of_rows h (row := 1114) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_12
  simp only [← c68781, ← c68782, ← c68783] at e_1114_12
  have e_1114_13 := arithEq_of_rows h (row := 1114) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_13
  simp only [← c68784, ← c68785, k2, ← c68786] at e_1114_13
  have hr := e_1114_13
  simp only [e_1114_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9745)) := by
  have c68787 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68788 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68789 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68790 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68791 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68792 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_14 := arithEq_of_rows h (row := 1114) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_14
  simp only [← c68787, ← c68788, ← c68789] at e_1114_14
  have e_1114_15 := arithEq_of_rows h (row := 1114) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_15
  simp only [← c68790, ← c68791, k2, ← c68792] at e_1114_15
  have hr := e_1114_15
  simp only [e_1114_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9746)) := by
  have c68793 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68794 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68795 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68796 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68797 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68798 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_16 := arithEq_of_rows h (row := 1114) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_16
  simp only [← c68793, ← c68794, ← c68795] at e_1114_16
  have e_1114_17 := arithEq_of_rows h (row := 1114) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_17
  simp only [← c68796, ← c68797, k2, ← c68798] at e_1114_17
  have hr := e_1114_17
  simp only [e_1114_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1114 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9747)) := by
  have c68799 := (publicBatchWrapper86_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68800 := (publicBatchWrapper86_copies2150 a h).1
  have c68801 := (publicBatchWrapper86_copies2150 a h).2.1
  have c68802 := (publicBatchWrapper86_copies2150 a h).2.2.1
  have c68803 := (publicBatchWrapper86_copies2150 a h).2.2.2.1
  have c68804 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1114_18 := arithEq_of_rows h (row := 1114) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_18
  simp only [← c68799, ← c68800, ← c68801] at e_1114_18
  have e_1114_19 := arithEq_of_rows h (row := 1114) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_19
  simp only [← c68802, ← c68803, k2, ← c68804] at e_1114_19
  have hr := e_1114_19
  simp only [e_1114_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9748)) := by
  have c68805 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.1
  have c68806 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.1
  have c68807 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.1
  have c68808 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.1
  have c68809 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.1
  have c68810 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_0 := arithEq_of_rows h (row := 1115) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_0
  simp only [← c68805, ← c68806, ← c68807] at e_1115_0
  have e_1115_1 := arithEq_of_rows h (row := 1115) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_1
  simp only [← c68808, ← c68809, k2, ← c68810] at e_1115_1
  have hr := e_1115_1
  simp only [e_1115_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9749)) := by
  have c68811 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68812 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68813 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68814 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68815 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68816 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_2 := arithEq_of_rows h (row := 1115) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_2
  simp only [← c68811, ← c68812, ← c68813] at e_1115_2
  have e_1115_3 := arithEq_of_rows h (row := 1115) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_3
  simp only [← c68814, ← c68815, k2, ← c68816] at e_1115_3
  have hr := e_1115_3
  simp only [e_1115_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9750)) := by
  have c68817 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68818 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68819 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68820 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68821 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68822 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_4 := arithEq_of_rows h (row := 1115) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_4
  simp only [← c68817, ← c68818, ← c68819] at e_1115_4
  have e_1115_5 := arithEq_of_rows h (row := 1115) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_5
  simp only [← c68820, ← c68821, k2, ← c68822] at e_1115_5
  have hr := e_1115_5
  simp only [e_1115_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9751)) := by
  have c68823 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68824 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68825 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68826 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68827 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68828 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_6 := arithEq_of_rows h (row := 1115) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_6
  simp only [← c68823, ← c68824, ← c68825] at e_1115_6
  have e_1115_7 := arithEq_of_rows h (row := 1115) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_7
  simp only [← c68826, ← c68827, k2, ← c68828] at e_1115_7
  have hr := e_1115_7
  simp only [e_1115_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9752)) := by
  have c68829 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68830 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68831 := (publicBatchWrapper86_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68832 := (publicBatchWrapper86_copies2151 a h).1
  have c68833 := (publicBatchWrapper86_copies2151 a h).2.1
  have c68834 := (publicBatchWrapper86_copies2151 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_8 := arithEq_of_rows h (row := 1115) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_8
  simp only [← c68829, ← c68830, ← c68831] at e_1115_8
  have e_1115_9 := arithEq_of_rows h (row := 1115) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_9
  simp only [← c68832, ← c68833, k2, ← c68834] at e_1115_9
  have hr := e_1115_9
  simp only [e_1115_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9753)) := by
  have c68835 := (publicBatchWrapper86_copies2151 a h).2.2.2.1
  have c68836 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.1
  have c68837 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.1
  have c68838 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.1
  have c68839 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.1
  have c68840 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_10 := arithEq_of_rows h (row := 1115) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_10
  simp only [← c68835, ← c68836, ← c68837] at e_1115_10
  have e_1115_11 := arithEq_of_rows h (row := 1115) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_11
  simp only [← c68838, ← c68839, k2, ← c68840] at e_1115_11
  have hr := e_1115_11
  simp only [e_1115_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9754)) := by
  have c68841 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.1
  have c68842 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68843 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68844 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68845 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68846 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_12 := arithEq_of_rows h (row := 1115) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_12
  simp only [← c68841, ← c68842, ← c68843] at e_1115_12
  have e_1115_13 := arithEq_of_rows h (row := 1115) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_13
  simp only [← c68844, ← c68845, k2, ← c68846] at e_1115_13
  have hr := e_1115_13
  simp only [e_1115_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9755)) := by
  have c68847 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68848 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68849 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68850 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68851 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68852 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_14 := arithEq_of_rows h (row := 1115) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_14
  simp only [← c68847, ← c68848, ← c68849] at e_1115_14
  have e_1115_15 := arithEq_of_rows h (row := 1115) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_15
  simp only [← c68850, ← c68851, k2, ← c68852] at e_1115_15
  have hr := e_1115_15
  simp only [e_1115_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9756)) := by
  have c68853 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68854 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68855 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68856 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68857 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68858 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_16 := arithEq_of_rows h (row := 1115) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_16
  simp only [← c68853, ← c68854, ← c68855] at e_1115_16
  have e_1115_17 := arithEq_of_rows h (row := 1115) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_17
  simp only [← c68856, ← c68857, k2, ← c68858] at e_1115_17
  have hr := e_1115_17
  simp only [e_1115_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1115 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9757)) := by
  have c68859 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68860 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68861 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68862 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68863 := (publicBatchWrapper86_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68864 := (publicBatchWrapper86_copies2152 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1115_18 := arithEq_of_rows h (row := 1115) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_18
  simp only [← c68859, ← c68860, ← c68861] at e_1115_18
  have e_1115_19 := arithEq_of_rows h (row := 1115) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_19
  simp only [← c68862, ← c68863, k2, ← c68864] at e_1115_19
  have hr := e_1115_19
  simp only [e_1115_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1116 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9758)) := by
  have c68865 := (publicBatchWrapper86_copies2152 a h).2.1
  have c68866 := (publicBatchWrapper86_copies2152 a h).2.2.1
  have c68867 := (publicBatchWrapper86_copies2152 a h).2.2.2.1
  have c68868 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.1
  have c68869 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.1
  have c68870 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1116_0 := arithEq_of_rows h (row := 1116) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_0
  simp only [← c68865, ← c68866, ← c68867] at e_1116_0
  have e_1116_1 := arithEq_of_rows h (row := 1116) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_1
  simp only [← c68868, ← c68869, k2, ← c68870] at e_1116_1
  have hr := e_1116_1
  simp only [e_1116_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1116 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9759)) := by
  have c68871 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.1
  have c68872 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.1
  have c68873 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.1
  have c68874 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68875 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68876 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1116_2 := arithEq_of_rows h (row := 1116) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_2
  simp only [← c68871, ← c68872, ← c68873] at e_1116_2
  have e_1116_3 := arithEq_of_rows h (row := 1116) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_3
  simp only [← c68874, ← c68875, k2, ← c68876] at e_1116_3
  have hr := e_1116_3
  simp only [e_1116_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1116 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9760)) := by
  have c68877 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68878 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68879 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68880 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68881 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68882 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1116_4 := arithEq_of_rows h (row := 1116) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_4
  simp only [← c68877, ← c68878, ← c68879] at e_1116_4
  have e_1116_5 := arithEq_of_rows h (row := 1116) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_5
  simp only [← c68880, ← c68881, k2, ← c68882] at e_1116_5
  have hr := e_1116_5
  simp only [e_1116_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1116 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9761)) := by
  have c68883 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68884 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68885 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68886 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68887 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68888 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1116_6 := arithEq_of_rows h (row := 1116) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_6
  simp only [← c68883, ← c68884, ← c68885] at e_1116_6
  have e_1116_7 := arithEq_of_rows h (row := 1116) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_7
  simp only [← c68886, ← c68887, k2, ← c68888] at e_1116_7
  have hr := e_1116_7
  simp only [e_1116_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f10756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1116 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9762)) := by
  have c68889 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68890 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68891 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68892 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68893 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68894 := (publicBatchWrapper86_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1116_8 := arithEq_of_rows h (row := 1116) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_8
  simp only [← c68889, ← c68890, ← c68891] at e_1116_8
  have e_1116_9 := arithEq_of_rows h (row := 1116) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_9
  simp only [← c68892, ← c68893, k2, ← c68894] at e_1116_9
  have hr := e_1116_9
  simp only [e_1116_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
