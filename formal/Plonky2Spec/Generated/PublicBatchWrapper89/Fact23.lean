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

theorem publicBatchWrapper89_f17466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1378)) := by
  have c109302 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109303 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109304 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109305 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109306 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109307 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_10
  simp only [← c109302, ← c109303, ← c109304] at e_1788_10
  have e_1788_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_11
  simp only [← c109305, ← c109306, k2, ← c109307] at e_1788_11
  have hr := e_1788_11
  simp only [e_1788_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1379)) := by
  have c109308 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109309 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109310 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109311 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109312 := (publicBatchWrapper89_copies3416 a h).1
  have c109313 := (publicBatchWrapper89_copies3416 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_12
  simp only [← c109308, ← c109309, ← c109310] at e_1788_12
  have e_1788_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_13
  simp only [← c109311, ← c109312, k2, ← c109313] at e_1788_13
  have hr := e_1788_13
  simp only [e_1788_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1380)) := by
  have c109314 := (publicBatchWrapper89_copies3416 a h).2.2.1
  have c109315 := (publicBatchWrapper89_copies3416 a h).2.2.2.1
  have c109316 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.1
  have c109317 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.1
  have c109318 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.1
  have c109319 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_14
  simp only [← c109314, ← c109315, ← c109316] at e_1788_14
  have e_1788_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_15
  simp only [← c109317, ← c109318, k2, ← c109319] at e_1788_15
  have hr := e_1788_15
  simp only [e_1788_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1381)) := by
  have c109320 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.1
  have c109321 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.1
  have c109322 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109323 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109324 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109325 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_16
  simp only [← c109320, ← c109321, ← c109322] at e_1788_16
  have e_1788_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_17
  simp only [← c109323, ← c109324, k2, ← c109325] at e_1788_17
  have hr := e_1788_17
  simp only [e_1788_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1382)) := by
  have c109326 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109327 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109328 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109329 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109330 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109331 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_18
  simp only [← c109326, ← c109327, ← c109328] at e_1788_18
  have e_1788_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_19
  simp only [← c109329, ← c109330, k2, ← c109331] at e_1788_19
  have hr := e_1788_19
  simp only [e_1788_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1383)) := by
  have c109332 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109333 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109334 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109335 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109336 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109337 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_0
  simp only [← c109332, ← c109333, ← c109334] at e_1789_0
  have e_1789_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_1
  simp only [← c109335, ← c109336, k2, ← c109337] at e_1789_1
  have hr := e_1789_1
  simp only [e_1789_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1384)) := by
  have c109338 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109339 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109340 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109341 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109342 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109343 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_2
  simp only [← c109338, ← c109339, ← c109340] at e_1789_2
  have e_1789_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_3
  simp only [← c109341, ← c109342, k2, ← c109343] at e_1789_3
  have hr := e_1789_3
  simp only [e_1789_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1385)) := by
  have c109344 := (publicBatchWrapper89_copies3417 a h).1
  have c109345 := (publicBatchWrapper89_copies3417 a h).2.1
  have c109346 := (publicBatchWrapper89_copies3417 a h).2.2.1
  have c109347 := (publicBatchWrapper89_copies3417 a h).2.2.2.1
  have c109348 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.1
  have c109349 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_4
  simp only [← c109344, ← c109345, ← c109346] at e_1789_4
  have e_1789_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_5
  simp only [← c109347, ← c109348, k2, ← c109349] at e_1789_5
  have hr := e_1789_5
  simp only [e_1789_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1386)) := by
  have c109350 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.1
  have c109351 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.1
  have c109352 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.1
  have c109353 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.1
  have c109354 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109355 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_6
  simp only [← c109350, ← c109351, ← c109352] at e_1789_6
  have e_1789_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_7
  simp only [← c109353, ← c109354, k2, ← c109355] at e_1789_7
  have hr := e_1789_7
  simp only [e_1789_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1387)) := by
  have c109356 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109357 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109358 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109359 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109360 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109361 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_8
  simp only [← c109356, ← c109357, ← c109358] at e_1789_8
  have e_1789_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_9
  simp only [← c109359, ← c109360, k2, ← c109361] at e_1789_9
  have hr := e_1789_9
  simp only [e_1789_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1388)) := by
  have c109362 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109363 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109364 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109365 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109366 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109367 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_10
  simp only [← c109362, ← c109363, ← c109364] at e_1789_10
  have e_1789_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_11
  simp only [← c109365, ← c109366, k2, ← c109367] at e_1789_11
  have hr := e_1789_11
  simp only [e_1789_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1389)) := by
  have c109368 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109369 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109370 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109371 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109372 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109373 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_12
  simp only [← c109368, ← c109369, ← c109370] at e_1789_12
  have e_1789_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_13
  simp only [← c109371, ← c109372, k2, ← c109373] at e_1789_13
  have hr := e_1789_13
  simp only [e_1789_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1390)) := by
  have c109374 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109375 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109376 := (publicBatchWrapper89_copies3418 a h).1
  have c109377 := (publicBatchWrapper89_copies3418 a h).2.1
  have c109378 := (publicBatchWrapper89_copies3418 a h).2.2.1
  have c109379 := (publicBatchWrapper89_copies3418 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_14
  simp only [← c109374, ← c109375, ← c109376] at e_1789_14
  have e_1789_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_15
  simp only [← c109377, ← c109378, k2, ← c109379] at e_1789_15
  have hr := e_1789_15
  simp only [e_1789_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1391)) := by
  have c109380 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.1
  have c109381 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.1
  have c109382 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.1
  have c109383 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.1
  have c109384 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.1
  have c109385 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_16
  simp only [← c109380, ← c109381, ← c109382] at e_1789_16
  have e_1789_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_17
  simp only [← c109383, ← c109384, k2, ← c109385] at e_1789_17
  have hr := e_1789_17
  simp only [e_1789_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1392)) := by
  have c109386 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109387 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109388 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109389 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109390 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109391 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_18
  simp only [← c109386, ← c109387, ← c109388] at e_1789_18
  have e_1789_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_19
  simp only [← c109389, ← c109390, k2, ← c109391] at e_1789_19
  have hr := e_1789_19
  simp only [e_1789_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1393)) := by
  have c109392 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109393 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109394 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109395 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109396 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109397 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_0
  simp only [← c109392, ← c109393, ← c109394] at e_1790_0
  have e_1790_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_1
  simp only [← c109395, ← c109396, k2, ← c109397] at e_1790_1
  have hr := e_1790_1
  simp only [e_1790_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1394)) := by
  have c109398 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109399 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109400 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109401 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109402 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109403 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_2
  simp only [← c109398, ← c109399, ← c109400] at e_1790_2
  have e_1790_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_3
  simp only [← c109401, ← c109402, k2, ← c109403] at e_1790_3
  have hr := e_1790_3
  simp only [e_1790_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1395)) := by
  have c109404 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109405 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109406 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109407 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109408 := (publicBatchWrapper89_copies3419 a h).1
  have c109409 := (publicBatchWrapper89_copies3419 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_4
  simp only [← c109404, ← c109405, ← c109406] at e_1790_4
  have e_1790_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_5
  simp only [← c109407, ← c109408, k2, ← c109409] at e_1790_5
  have hr := e_1790_5
  simp only [e_1790_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1396)) := by
  have c109410 := (publicBatchWrapper89_copies3419 a h).2.2.1
  have c109411 := (publicBatchWrapper89_copies3419 a h).2.2.2.1
  have c109412 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.1
  have c109413 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.1
  have c109414 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.1
  have c109415 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_6
  simp only [← c109410, ← c109411, ← c109412] at e_1790_6
  have e_1790_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_7
  simp only [← c109413, ← c109414, k2, ← c109415] at e_1790_7
  have hr := e_1790_7
  simp only [e_1790_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1397)) := by
  have c109416 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.1
  have c109417 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.1
  have c109418 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109419 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109420 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109421 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_8
  simp only [← c109416, ← c109417, ← c109418] at e_1790_8
  have e_1790_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_9
  simp only [← c109419, ← c109420, k2, ← c109421] at e_1790_9
  have hr := e_1790_9
  simp only [e_1790_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1398)) := by
  have c109422 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109423 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109424 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109425 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109426 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109427 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_10
  simp only [← c109422, ← c109423, ← c109424] at e_1790_10
  have e_1790_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_11
  simp only [← c109425, ← c109426, k2, ← c109427] at e_1790_11
  have hr := e_1790_11
  simp only [e_1790_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1399)) := by
  have c109428 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109429 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109430 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109431 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109432 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109433 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_12
  simp only [← c109428, ← c109429, ← c109430] at e_1790_12
  have e_1790_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_13
  simp only [← c109431, ← c109432, k2, ← c109433] at e_1790_13
  have hr := e_1790_13
  simp only [e_1790_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1400)) := by
  have c109434 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109435 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109436 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109437 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109438 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109439 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_14
  simp only [← c109434, ← c109435, ← c109436] at e_1790_14
  have e_1790_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_15
  simp only [← c109437, ← c109438, k2, ← c109439] at e_1790_15
  have hr := e_1790_15
  simp only [e_1790_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1401)) := by
  have c109440 := (publicBatchWrapper89_copies3420 a h).1
  have c109441 := (publicBatchWrapper89_copies3420 a h).2.1
  have c109442 := (publicBatchWrapper89_copies3420 a h).2.2.1
  have c109443 := (publicBatchWrapper89_copies3420 a h).2.2.2.1
  have c109444 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.1
  have c109445 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_16
  simp only [← c109440, ← c109441, ← c109442] at e_1790_16
  have e_1790_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_17
  simp only [← c109443, ← c109444, k2, ← c109445] at e_1790_17
  have hr := e_1790_17
  simp only [e_1790_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1570)) := by
  have c109446 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.1
  have c109447 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.1
  have c109448 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.1
  have c109449 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.1
  have c109450 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109451 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_18
  simp only [← c109446, ← c109447, ← c109448] at e_1790_18
  have e_1790_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_19
  simp only [← c109449, ← c109450, k2, ← c109451] at e_1790_19
  have hr := e_1790_19
  simp only [e_1790_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1571)) := by
  have c109452 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109453 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109454 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109455 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109456 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109457 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_0
  simp only [← c109452, ← c109453, ← c109454] at e_1791_0
  have e_1791_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_1
  simp only [← c109455, ← c109456, k2, ← c109457] at e_1791_1
  have hr := e_1791_1
  simp only [e_1791_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1572)) := by
  have c109458 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109459 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109460 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109461 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109462 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109463 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_2
  simp only [← c109458, ← c109459, ← c109460] at e_1791_2
  have e_1791_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_3
  simp only [← c109461, ← c109462, k2, ← c109463] at e_1791_3
  have hr := e_1791_3
  simp only [e_1791_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1573)) := by
  have c109464 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109465 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109466 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109467 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109468 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109469 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_4
  simp only [← c109464, ← c109465, ← c109466] at e_1791_4
  have e_1791_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_5
  simp only [← c109467, ← c109468, k2, ← c109469] at e_1791_5
  have hr := e_1791_5
  simp only [e_1791_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1574)) := by
  have c109470 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109471 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109472 := (publicBatchWrapper89_copies3421 a h).1
  have c109473 := (publicBatchWrapper89_copies3421 a h).2.1
  have c109474 := (publicBatchWrapper89_copies3421 a h).2.2.1
  have c109475 := (publicBatchWrapper89_copies3421 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_6
  simp only [← c109470, ← c109471, ← c109472] at e_1791_6
  have e_1791_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_7
  simp only [← c109473, ← c109474, k2, ← c109475] at e_1791_7
  have hr := e_1791_7
  simp only [e_1791_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1575)) := by
  have c109476 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.1
  have c109477 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.1
  have c109478 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.1
  have c109479 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.1
  have c109480 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.1
  have c109481 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_8
  simp only [← c109476, ← c109477, ← c109478] at e_1791_8
  have e_1791_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_9
  simp only [← c109479, ← c109480, k2, ← c109481] at e_1791_9
  have hr := e_1791_9
  simp only [e_1791_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1576)) := by
  have c109482 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109483 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109484 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109485 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109486 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109487 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_10
  simp only [← c109482, ← c109483, ← c109484] at e_1791_10
  have e_1791_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_11
  simp only [← c109485, ← c109486, k2, ← c109487] at e_1791_11
  have hr := e_1791_11
  simp only [e_1791_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1577)) := by
  have c109488 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109489 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109490 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109491 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109492 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109493 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_12
  simp only [← c109488, ← c109489, ← c109490] at e_1791_12
  have e_1791_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_13
  simp only [← c109491, ← c109492, k2, ← c109493] at e_1791_13
  have hr := e_1791_13
  simp only [e_1791_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1578)) := by
  have c109494 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109495 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109496 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109497 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109498 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109499 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_14
  simp only [← c109494, ← c109495, ← c109496] at e_1791_14
  have e_1791_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_15
  simp only [← c109497, ← c109498, k2, ← c109499] at e_1791_15
  have hr := e_1791_15
  simp only [e_1791_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1579)) := by
  have c109500 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109501 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109502 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109503 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109504 := (publicBatchWrapper89_copies3422 a h).1
  have c109505 := (publicBatchWrapper89_copies3422 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_16
  simp only [← c109500, ← c109501, ← c109502] at e_1791_16
  have e_1791_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_17
  simp only [← c109503, ← c109504, k2, ← c109505] at e_1791_17
  have hr := e_1791_17
  simp only [e_1791_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1580)) := by
  have c109506 := (publicBatchWrapper89_copies3422 a h).2.2.1
  have c109507 := (publicBatchWrapper89_copies3422 a h).2.2.2.1
  have c109508 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.1
  have c109509 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.1
  have c109510 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.1
  have c109511 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_18
  simp only [← c109506, ← c109507, ← c109508] at e_1791_18
  have e_1791_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_19
  simp only [← c109509, ← c109510, k2, ← c109511] at e_1791_19
  have hr := e_1791_19
  simp only [e_1791_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1581)) := by
  have c109512 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.1
  have c109513 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.1
  have c109514 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109515 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109516 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109517 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_0
  simp only [← c109512, ← c109513, ← c109514] at e_1792_0
  have e_1792_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_1
  simp only [← c109515, ← c109516, k2, ← c109517] at e_1792_1
  have hr := e_1792_1
  simp only [e_1792_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1582)) := by
  have c109518 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109519 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109520 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109521 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109522 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109523 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_2
  simp only [← c109518, ← c109519, ← c109520] at e_1792_2
  have e_1792_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_3
  simp only [← c109521, ← c109522, k2, ← c109523] at e_1792_3
  have hr := e_1792_3
  simp only [e_1792_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1583)) := by
  have c109524 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109525 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109526 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109527 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109528 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109529 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_4
  simp only [← c109524, ← c109525, ← c109526] at e_1792_4
  have e_1792_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_5
  simp only [← c109527, ← c109528, k2, ← c109529] at e_1792_5
  have hr := e_1792_5
  simp only [e_1792_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1584)) := by
  have c109530 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109531 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109532 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109533 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109534 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109535 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_6
  simp only [← c109530, ← c109531, ← c109532] at e_1792_6
  have e_1792_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_7
  simp only [← c109533, ← c109534, k2, ← c109535] at e_1792_7
  have hr := e_1792_7
  simp only [e_1792_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1585)) := by
  have c109536 := (publicBatchWrapper89_copies3423 a h).1
  have c109537 := (publicBatchWrapper89_copies3423 a h).2.1
  have c109538 := (publicBatchWrapper89_copies3423 a h).2.2.1
  have c109539 := (publicBatchWrapper89_copies3423 a h).2.2.2.1
  have c109540 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.1
  have c109541 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_8
  simp only [← c109536, ← c109537, ← c109538] at e_1792_8
  have e_1792_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_9
  simp only [← c109539, ← c109540, k2, ← c109541] at e_1792_9
  have hr := e_1792_9
  simp only [e_1792_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1586)) := by
  have c109542 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.1
  have c109543 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.1
  have c109544 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.1
  have c109545 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.1
  have c109546 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109547 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_10
  simp only [← c109542, ← c109543, ← c109544] at e_1792_10
  have e_1792_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_11
  simp only [← c109545, ← c109546, k2, ← c109547] at e_1792_11
  have hr := e_1792_11
  simp only [e_1792_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1587)) := by
  have c109548 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109549 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109550 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109551 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109552 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109553 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_12
  simp only [← c109548, ← c109549, ← c109550] at e_1792_12
  have e_1792_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_13
  simp only [← c109551, ← c109552, k2, ← c109553] at e_1792_13
  have hr := e_1792_13
  simp only [e_1792_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1588)) := by
  have c109554 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109555 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109556 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109557 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109558 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109559 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_14
  simp only [← c109554, ← c109555, ← c109556] at e_1792_14
  have e_1792_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_15
  simp only [← c109557, ← c109558, k2, ← c109559] at e_1792_15
  have hr := e_1792_15
  simp only [e_1792_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1589)) := by
  have c109560 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109561 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109562 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109563 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109564 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109565 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_16
  simp only [← c109560, ← c109561, ← c109562] at e_1792_16
  have e_1792_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_17
  simp only [← c109563, ← c109564, k2, ← c109565] at e_1792_17
  have hr := e_1792_17
  simp only [e_1792_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1590)) := by
  have c109566 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109567 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109568 := (publicBatchWrapper89_copies3424 a h).1
  have c109569 := (publicBatchWrapper89_copies3424 a h).2.1
  have c109570 := (publicBatchWrapper89_copies3424 a h).2.2.1
  have c109571 := (publicBatchWrapper89_copies3424 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_18
  simp only [← c109566, ← c109567, ← c109568] at e_1792_18
  have e_1792_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_19
  simp only [← c109569, ← c109570, k2, ← c109571] at e_1792_19
  have hr := e_1792_19
  simp only [e_1792_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1591)) := by
  have c109572 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.1
  have c109573 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.1
  have c109574 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.1
  have c109575 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.1
  have c109576 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.1
  have c109577 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_0
  simp only [← c109572, ← c109573, ← c109574] at e_1793_0
  have e_1793_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_1
  simp only [← c109575, ← c109576, k2, ← c109577] at e_1793_1
  have hr := e_1793_1
  simp only [e_1793_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1592)) := by
  have c109578 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109579 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109580 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109581 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109582 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109583 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_2
  simp only [← c109578, ← c109579, ← c109580] at e_1793_2
  have e_1793_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_3
  simp only [← c109581, ← c109582, k2, ← c109583] at e_1793_3
  have hr := e_1793_3
  simp only [e_1793_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1593)) := by
  have c109584 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109585 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109586 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109587 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109588 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109589 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_4
  simp only [← c109584, ← c109585, ← c109586] at e_1793_4
  have e_1793_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_5
  simp only [← c109587, ← c109588, k2, ← c109589] at e_1793_5
  have hr := e_1793_5
  simp only [e_1793_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1594)) := by
  have c109590 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109591 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109592 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109593 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109594 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109595 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_6
  simp only [← c109590, ← c109591, ← c109592] at e_1793_6
  have e_1793_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_7
  simp only [← c109593, ← c109594, k2, ← c109595] at e_1793_7
  have hr := e_1793_7
  simp only [e_1793_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1595)) := by
  have c109596 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109597 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109598 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109599 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109600 := (publicBatchWrapper89_copies3425 a h).1
  have c109601 := (publicBatchWrapper89_copies3425 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_8
  simp only [← c109596, ← c109597, ← c109598] at e_1793_8
  have e_1793_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_9
  simp only [← c109599, ← c109600, k2, ← c109601] at e_1793_9
  have hr := e_1793_9
  simp only [e_1793_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1596)) := by
  have c109602 := (publicBatchWrapper89_copies3425 a h).2.2.1
  have c109603 := (publicBatchWrapper89_copies3425 a h).2.2.2.1
  have c109604 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.1
  have c109605 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.1
  have c109606 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.1
  have c109607 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_10
  simp only [← c109602, ← c109603, ← c109604] at e_1793_10
  have e_1793_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_11
  simp only [← c109605, ← c109606, k2, ← c109607] at e_1793_11
  have hr := e_1793_11
  simp only [e_1793_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1597)) := by
  have c109608 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.1
  have c109609 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.1
  have c109610 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109611 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109612 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109613 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_12
  simp only [← c109608, ← c109609, ← c109610] at e_1793_12
  have e_1793_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_13
  simp only [← c109611, ← c109612, k2, ← c109613] at e_1793_13
  have hr := e_1793_13
  simp only [e_1793_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1598)) := by
  have c109614 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109615 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109616 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109617 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109618 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109619 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_14
  simp only [← c109614, ← c109615, ← c109616] at e_1793_14
  have e_1793_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_15
  simp only [← c109617, ← c109618, k2, ← c109619] at e_1793_15
  have hr := e_1793_15
  simp only [e_1793_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1599)) := by
  have c109620 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109621 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109622 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109623 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109624 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109625 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_16
  simp only [← c109620, ← c109621, ← c109622] at e_1793_16
  have e_1793_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_17
  simp only [← c109623, ← c109624, k2, ← c109625] at e_1793_17
  have hr := e_1793_17
  simp only [e_1793_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1600)) := by
  have c109626 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109627 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109628 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109629 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109630 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109631 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_18
  simp only [← c109626, ← c109627, ← c109628] at e_1793_18
  have e_1793_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_19
  simp only [← c109629, ← c109630, k2, ← c109631] at e_1793_19
  have hr := e_1793_19
  simp only [e_1793_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1601)) := by
  have c109632 := (publicBatchWrapper89_copies3426 a h).1
  have c109633 := (publicBatchWrapper89_copies3426 a h).2.1
  have c109634 := (publicBatchWrapper89_copies3426 a h).2.2.1
  have c109635 := (publicBatchWrapper89_copies3426 a h).2.2.2.1
  have c109636 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.1
  have c109637 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_0
  simp only [← c109632, ← c109633, ← c109634] at e_1794_0
  have e_1794_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_1
  simp only [← c109635, ← c109636, k2, ← c109637] at e_1794_1
  have hr := e_1794_1
  simp only [e_1794_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1770)) := by
  have c109638 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.1
  have c109639 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.1
  have c109640 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.1
  have c109641 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.1
  have c109642 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109643 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_2
  simp only [← c109638, ← c109639, ← c109640] at e_1794_2
  have e_1794_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_3
  simp only [← c109641, ← c109642, k2, ← c109643] at e_1794_3
  have hr := e_1794_3
  simp only [e_1794_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1771)) := by
  have c109644 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109645 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109646 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109647 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109648 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109649 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_4
  simp only [← c109644, ← c109645, ← c109646] at e_1794_4
  have e_1794_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_5
  simp only [← c109647, ← c109648, k2, ← c109649] at e_1794_5
  have hr := e_1794_5
  simp only [e_1794_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1772)) := by
  have c109650 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109651 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109652 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109653 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109654 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109655 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_6
  simp only [← c109650, ← c109651, ← c109652] at e_1794_6
  have e_1794_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_7
  simp only [← c109653, ← c109654, k2, ← c109655] at e_1794_7
  have hr := e_1794_7
  simp only [e_1794_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1773)) := by
  have c109656 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109657 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109658 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109659 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109660 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109661 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_8
  simp only [← c109656, ← c109657, ← c109658] at e_1794_8
  have e_1794_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_9
  simp only [← c109659, ← c109660, k2, ← c109661] at e_1794_9
  have hr := e_1794_9
  simp only [e_1794_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1774)) := by
  have c109662 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109663 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109664 := (publicBatchWrapper89_copies3427 a h).1
  have c109665 := (publicBatchWrapper89_copies3427 a h).2.1
  have c109666 := (publicBatchWrapper89_copies3427 a h).2.2.1
  have c109667 := (publicBatchWrapper89_copies3427 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_10
  simp only [← c109662, ← c109663, ← c109664] at e_1794_10
  have e_1794_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_11
  simp only [← c109665, ← c109666, k2, ← c109667] at e_1794_11
  have hr := e_1794_11
  simp only [e_1794_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1775)) := by
  have c109668 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.1
  have c109669 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.1
  have c109670 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.1
  have c109671 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.1
  have c109672 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.1
  have c109673 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_12
  simp only [← c109668, ← c109669, ← c109670] at e_1794_12
  have e_1794_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_13
  simp only [← c109671, ← c109672, k2, ← c109673] at e_1794_13
  have hr := e_1794_13
  simp only [e_1794_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1776)) := by
  have c109674 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109675 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109676 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109677 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109678 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109679 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_14
  simp only [← c109674, ← c109675, ← c109676] at e_1794_14
  have e_1794_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_15
  simp only [← c109677, ← c109678, k2, ← c109679] at e_1794_15
  have hr := e_1794_15
  simp only [e_1794_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1777)) := by
  have c109680 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109681 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109682 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109683 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109684 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109685 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_16
  simp only [← c109680, ← c109681, ← c109682] at e_1794_16
  have e_1794_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_17
  simp only [← c109683, ← c109684, k2, ← c109685] at e_1794_17
  have hr := e_1794_17
  simp only [e_1794_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1778)) := by
  have c109686 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109687 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109688 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109689 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109690 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109691 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_18
  simp only [← c109686, ← c109687, ← c109688] at e_1794_18
  have e_1794_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_19
  simp only [← c109689, ← c109690, k2, ← c109691] at e_1794_19
  have hr := e_1794_19
  simp only [e_1794_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1779)) := by
  have c109692 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109693 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109694 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109695 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109696 := (publicBatchWrapper89_copies3428 a h).1
  have c109697 := (publicBatchWrapper89_copies3428 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_0
  simp only [← c109692, ← c109693, ← c109694] at e_1795_0
  have e_1795_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_1
  simp only [← c109695, ← c109696, k2, ← c109697] at e_1795_1
  have hr := e_1795_1
  simp only [e_1795_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1780)) := by
  have c109698 := (publicBatchWrapper89_copies3428 a h).2.2.1
  have c109699 := (publicBatchWrapper89_copies3428 a h).2.2.2.1
  have c109700 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.1
  have c109701 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.1
  have c109702 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.1
  have c109703 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_2
  simp only [← c109698, ← c109699, ← c109700] at e_1795_2
  have e_1795_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_3
  simp only [← c109701, ← c109702, k2, ← c109703] at e_1795_3
  have hr := e_1795_3
  simp only [e_1795_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1781)) := by
  have c109704 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.1
  have c109705 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.1
  have c109706 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109707 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109708 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109709 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_4
  simp only [← c109704, ← c109705, ← c109706] at e_1795_4
  have e_1795_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_5
  simp only [← c109707, ← c109708, k2, ← c109709] at e_1795_5
  have hr := e_1795_5
  simp only [e_1795_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1782)) := by
  have c109710 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109711 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109712 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109713 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109714 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109715 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_6
  simp only [← c109710, ← c109711, ← c109712] at e_1795_6
  have e_1795_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_7
  simp only [← c109713, ← c109714, k2, ← c109715] at e_1795_7
  have hr := e_1795_7
  simp only [e_1795_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1783)) := by
  have c109716 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109717 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109718 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109719 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109720 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109721 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_8
  simp only [← c109716, ← c109717, ← c109718] at e_1795_8
  have e_1795_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_9
  simp only [← c109719, ← c109720, k2, ← c109721] at e_1795_9
  have hr := e_1795_9
  simp only [e_1795_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1784)) := by
  have c109722 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109723 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109724 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109725 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109726 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109727 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_10
  simp only [← c109722, ← c109723, ← c109724] at e_1795_10
  have e_1795_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_11
  simp only [← c109725, ← c109726, k2, ← c109727] at e_1795_11
  have hr := e_1795_11
  simp only [e_1795_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1785)) := by
  have c109728 := (publicBatchWrapper89_copies3429 a h).1
  have c109729 := (publicBatchWrapper89_copies3429 a h).2.1
  have c109730 := (publicBatchWrapper89_copies3429 a h).2.2.1
  have c109731 := (publicBatchWrapper89_copies3429 a h).2.2.2.1
  have c109732 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.1
  have c109733 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_12
  simp only [← c109728, ← c109729, ← c109730] at e_1795_12
  have e_1795_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_13
  simp only [← c109731, ← c109732, k2, ← c109733] at e_1795_13
  have hr := e_1795_13
  simp only [e_1795_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1786)) := by
  have c109734 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.1
  have c109735 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.1
  have c109736 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.1
  have c109737 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.1
  have c109738 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109739 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_14
  simp only [← c109734, ← c109735, ← c109736] at e_1795_14
  have e_1795_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_15
  simp only [← c109737, ← c109738, k2, ← c109739] at e_1795_15
  have hr := e_1795_15
  simp only [e_1795_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1787)) := by
  have c109740 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109741 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109742 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109743 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109744 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109745 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_16
  simp only [← c109740, ← c109741, ← c109742] at e_1795_16
  have e_1795_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_17
  simp only [← c109743, ← c109744, k2, ← c109745] at e_1795_17
  have hr := e_1795_17
  simp only [e_1795_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1788)) := by
  have c109746 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109747 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109748 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109749 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109750 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109751 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_18
  simp only [← c109746, ← c109747, ← c109748] at e_1795_18
  have e_1795_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_19
  simp only [← c109749, ← c109750, k2, ← c109751] at e_1795_19
  have hr := e_1795_19
  simp only [e_1795_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1789)) := by
  have c109752 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109753 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109754 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109755 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109756 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109757 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_0
  simp only [← c109752, ← c109753, ← c109754] at e_1796_0
  have e_1796_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_1
  simp only [← c109755, ← c109756, k2, ← c109757] at e_1796_1
  have hr := e_1796_1
  simp only [e_1796_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1790)) := by
  have c109758 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109759 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109760 := (publicBatchWrapper89_copies3430 a h).1
  have c109761 := (publicBatchWrapper89_copies3430 a h).2.1
  have c109762 := (publicBatchWrapper89_copies3430 a h).2.2.1
  have c109763 := (publicBatchWrapper89_copies3430 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_2
  simp only [← c109758, ← c109759, ← c109760] at e_1796_2
  have e_1796_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_3
  simp only [← c109761, ← c109762, k2, ← c109763] at e_1796_3
  have hr := e_1796_3
  simp only [e_1796_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1791)) := by
  have c109764 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.1
  have c109765 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.1
  have c109766 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.1
  have c109767 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.1
  have c109768 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.1
  have c109769 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_4
  simp only [← c109764, ← c109765, ← c109766] at e_1796_4
  have e_1796_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_5
  simp only [← c109767, ← c109768, k2, ← c109769] at e_1796_5
  have hr := e_1796_5
  simp only [e_1796_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1792)) := by
  have c109770 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109771 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109772 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109773 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109774 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109775 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_6
  simp only [← c109770, ← c109771, ← c109772] at e_1796_6
  have e_1796_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_7
  simp only [← c109773, ← c109774, k2, ← c109775] at e_1796_7
  have hr := e_1796_7
  simp only [e_1796_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1793)) := by
  have c109776 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109777 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109778 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109779 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109780 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109781 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_8
  simp only [← c109776, ← c109777, ← c109778] at e_1796_8
  have e_1796_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_9
  simp only [← c109779, ← c109780, k2, ← c109781] at e_1796_9
  have hr := e_1796_9
  simp only [e_1796_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1794)) := by
  have c109782 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109783 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109784 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109785 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109786 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109787 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_10
  simp only [← c109782, ← c109783, ← c109784] at e_1796_10
  have e_1796_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_11
  simp only [← c109785, ← c109786, k2, ← c109787] at e_1796_11
  have hr := e_1796_11
  simp only [e_1796_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1795)) := by
  have c109788 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109789 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109790 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109791 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109792 := (publicBatchWrapper89_copies3431 a h).1
  have c109793 := (publicBatchWrapper89_copies3431 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_12
  simp only [← c109788, ← c109789, ← c109790] at e_1796_12
  have e_1796_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_13
  simp only [← c109791, ← c109792, k2, ← c109793] at e_1796_13
  have hr := e_1796_13
  simp only [e_1796_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1796)) := by
  have c109794 := (publicBatchWrapper89_copies3431 a h).2.2.1
  have c109795 := (publicBatchWrapper89_copies3431 a h).2.2.2.1
  have c109796 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.1
  have c109797 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.1
  have c109798 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.1
  have c109799 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_14
  simp only [← c109794, ← c109795, ← c109796] at e_1796_14
  have e_1796_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_15
  simp only [← c109797, ← c109798, k2, ← c109799] at e_1796_15
  have hr := e_1796_15
  simp only [e_1796_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1797)) := by
  have c109800 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.1
  have c109801 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.1
  have c109802 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109803 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109804 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109805 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_16
  simp only [← c109800, ← c109801, ← c109802] at e_1796_16
  have e_1796_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_17
  simp only [← c109803, ← c109804, k2, ← c109805] at e_1796_17
  have hr := e_1796_17
  simp only [e_1796_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1798)) := by
  have c109806 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109807 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109808 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109809 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109810 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109811 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_18
  simp only [← c109806, ← c109807, ← c109808] at e_1796_18
  have e_1796_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_19
  simp only [← c109809, ← c109810, k2, ← c109811] at e_1796_19
  have hr := e_1796_19
  simp only [e_1796_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1799)) := by
  have c109812 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109813 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109814 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109815 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109816 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109817 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_0
  simp only [← c109812, ← c109813, ← c109814] at e_1797_0
  have e_1797_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_1
  simp only [← c109815, ← c109816, k2, ← c109817] at e_1797_1
  have hr := e_1797_1
  simp only [e_1797_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1800)) := by
  have c109818 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109819 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109820 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109821 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109822 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109823 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_2
  simp only [← c109818, ← c109819, ← c109820] at e_1797_2
  have e_1797_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_3
  simp only [← c109821, ← c109822, k2, ← c109823] at e_1797_3
  have hr := e_1797_3
  simp only [e_1797_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1801)) := by
  have c109824 := (publicBatchWrapper89_copies3432 a h).1
  have c109825 := (publicBatchWrapper89_copies3432 a h).2.1
  have c109826 := (publicBatchWrapper89_copies3432 a h).2.2.1
  have c109827 := (publicBatchWrapper89_copies3432 a h).2.2.2.1
  have c109828 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.1
  have c109829 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_4
  simp only [← c109824, ← c109825, ← c109826] at e_1797_4
  have e_1797_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_5
  simp only [← c109827, ← c109828, k2, ← c109829] at e_1797_5
  have hr := e_1797_5
  simp only [e_1797_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1970)) := by
  have c109830 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.1
  have c109831 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.1
  have c109832 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.1
  have c109833 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.1
  have c109834 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109835 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_6
  simp only [← c109830, ← c109831, ← c109832] at e_1797_6
  have e_1797_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_7
  simp only [← c109833, ← c109834, k2, ← c109835] at e_1797_7
  have hr := e_1797_7
  simp only [e_1797_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1971)) := by
  have c109836 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109837 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109838 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109839 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109840 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109841 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_8
  simp only [← c109836, ← c109837, ← c109838] at e_1797_8
  have e_1797_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_9
  simp only [← c109839, ← c109840, k2, ← c109841] at e_1797_9
  have hr := e_1797_9
  simp only [e_1797_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1972)) := by
  have c109842 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109843 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109844 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109845 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109846 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109847 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_10
  simp only [← c109842, ← c109843, ← c109844] at e_1797_10
  have e_1797_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_11
  simp only [← c109845, ← c109846, k2, ← c109847] at e_1797_11
  have hr := e_1797_11
  simp only [e_1797_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1973)) := by
  have c109848 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109849 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109850 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109851 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109852 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109853 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_12
  simp only [← c109848, ← c109849, ← c109850] at e_1797_12
  have e_1797_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_13
  simp only [← c109851, ← c109852, k2, ← c109853] at e_1797_13
  have hr := e_1797_13
  simp only [e_1797_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1974)) := by
  have c109854 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109855 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109856 := (publicBatchWrapper89_copies3433 a h).1
  have c109857 := (publicBatchWrapper89_copies3433 a h).2.1
  have c109858 := (publicBatchWrapper89_copies3433 a h).2.2.1
  have c109859 := (publicBatchWrapper89_copies3433 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_14
  simp only [← c109854, ← c109855, ← c109856] at e_1797_14
  have e_1797_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_15
  simp only [← c109857, ← c109858, k2, ← c109859] at e_1797_15
  have hr := e_1797_15
  simp only [e_1797_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1975)) := by
  have c109860 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.1
  have c109861 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.1
  have c109862 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.1
  have c109863 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.1
  have c109864 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.1
  have c109865 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_16
  simp only [← c109860, ← c109861, ← c109862] at e_1797_16
  have e_1797_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_17
  simp only [← c109863, ← c109864, k2, ← c109865] at e_1797_17
  have hr := e_1797_17
  simp only [e_1797_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1976)) := by
  have c109866 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109867 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109868 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109869 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109870 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109871 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_18
  simp only [← c109866, ← c109867, ← c109868] at e_1797_18
  have e_1797_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_19
  simp only [← c109869, ← c109870, k2, ← c109871] at e_1797_19
  have hr := e_1797_19
  simp only [e_1797_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1977)) := by
  have c109872 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109873 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109874 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109875 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109876 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109877 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_0
  simp only [← c109872, ← c109873, ← c109874] at e_1798_0
  have e_1798_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_1
  simp only [← c109875, ← c109876, k2, ← c109877] at e_1798_1
  have hr := e_1798_1
  simp only [e_1798_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1978)) := by
  have c109878 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109879 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109880 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109881 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109882 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109883 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_2
  simp only [← c109878, ← c109879, ← c109880] at e_1798_2
  have e_1798_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_3
  simp only [← c109881, ← c109882, k2, ← c109883] at e_1798_3
  have hr := e_1798_3
  simp only [e_1798_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1979)) := by
  have c109884 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109885 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109886 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109887 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109888 := (publicBatchWrapper89_copies3434 a h).1
  have c109889 := (publicBatchWrapper89_copies3434 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_4
  simp only [← c109884, ← c109885, ← c109886] at e_1798_4
  have e_1798_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_5
  simp only [← c109887, ← c109888, k2, ← c109889] at e_1798_5
  have hr := e_1798_5
  simp only [e_1798_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1980)) := by
  have c109890 := (publicBatchWrapper89_copies3434 a h).2.2.1
  have c109891 := (publicBatchWrapper89_copies3434 a h).2.2.2.1
  have c109892 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.1
  have c109893 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.1
  have c109894 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.1
  have c109895 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_6
  simp only [← c109890, ← c109891, ← c109892] at e_1798_6
  have e_1798_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_7
  simp only [← c109893, ← c109894, k2, ← c109895] at e_1798_7
  have hr := e_1798_7
  simp only [e_1798_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1981)) := by
  have c109896 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.1
  have c109897 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.1
  have c109898 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109899 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109900 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109901 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_8
  simp only [← c109896, ← c109897, ← c109898] at e_1798_8
  have e_1798_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_9
  simp only [← c109899, ← c109900, k2, ← c109901] at e_1798_9
  have hr := e_1798_9
  simp only [e_1798_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1982)) := by
  have c109902 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109903 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109904 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109905 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109906 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109907 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_10
  simp only [← c109902, ← c109903, ← c109904] at e_1798_10
  have e_1798_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_11
  simp only [← c109905, ← c109906, k2, ← c109907] at e_1798_11
  have hr := e_1798_11
  simp only [e_1798_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1983)) := by
  have c109908 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109909 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109910 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109911 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109912 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109913 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_12
  simp only [← c109908, ← c109909, ← c109910] at e_1798_12
  have e_1798_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_13
  simp only [← c109911, ← c109912, k2, ← c109913] at e_1798_13
  have hr := e_1798_13
  simp only [e_1798_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1984)) := by
  have c109914 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109915 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109916 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109917 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109918 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109919 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_14
  simp only [← c109914, ← c109915, ← c109916] at e_1798_14
  have e_1798_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_15
  simp only [← c109917, ← c109918, k2, ← c109919] at e_1798_15
  have hr := e_1798_15
  simp only [e_1798_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1985)) := by
  have c109920 := (publicBatchWrapper89_copies3435 a h).1
  have c109921 := (publicBatchWrapper89_copies3435 a h).2.1
  have c109922 := (publicBatchWrapper89_copies3435 a h).2.2.1
  have c109923 := (publicBatchWrapper89_copies3435 a h).2.2.2.1
  have c109924 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.1
  have c109925 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_16
  simp only [← c109920, ← c109921, ← c109922] at e_1798_16
  have e_1798_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_17
  simp only [← c109923, ← c109924, k2, ← c109925] at e_1798_17
  have hr := e_1798_17
  simp only [e_1798_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1986)) := by
  have c109926 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.1
  have c109927 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.1
  have c109928 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.1
  have c109929 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.1
  have c109930 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109931 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_18
  simp only [← c109926, ← c109927, ← c109928] at e_1798_18
  have e_1798_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_19
  simp only [← c109929, ← c109930, k2, ← c109931] at e_1798_19
  have hr := e_1798_19
  simp only [e_1798_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1987)) := by
  have c109932 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109933 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109934 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109935 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109936 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109937 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_0
  simp only [← c109932, ← c109933, ← c109934] at e_1799_0
  have e_1799_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_1
  simp only [← c109935, ← c109936, k2, ← c109937] at e_1799_1
  have hr := e_1799_1
  simp only [e_1799_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1988)) := by
  have c109938 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109939 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109940 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109941 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109942 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109943 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_2
  simp only [← c109938, ← c109939, ← c109940] at e_1799_2
  have e_1799_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_3
  simp only [← c109941, ← c109942, k2, ← c109943] at e_1799_3
  have hr := e_1799_3
  simp only [e_1799_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1989)) := by
  have c109944 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109945 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109946 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109947 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109948 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109949 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_4
  simp only [← c109944, ← c109945, ← c109946] at e_1799_4
  have e_1799_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_5
  simp only [← c109947, ← c109948, k2, ← c109949] at e_1799_5
  have hr := e_1799_5
  simp only [e_1799_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1990)) := by
  have c109950 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109951 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109952 := (publicBatchWrapper89_copies3436 a h).1
  have c109953 := (publicBatchWrapper89_copies3436 a h).2.1
  have c109954 := (publicBatchWrapper89_copies3436 a h).2.2.1
  have c109955 := (publicBatchWrapper89_copies3436 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_6
  simp only [← c109950, ← c109951, ← c109952] at e_1799_6
  have e_1799_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_7
  simp only [← c109953, ← c109954, k2, ← c109955] at e_1799_7
  have hr := e_1799_7
  simp only [e_1799_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1991)) := by
  have c109956 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.1
  have c109957 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.1
  have c109958 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.1
  have c109959 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.1
  have c109960 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.1
  have c109961 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_8
  simp only [← c109956, ← c109957, ← c109958] at e_1799_8
  have e_1799_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_9
  simp only [← c109959, ← c109960, k2, ← c109961] at e_1799_9
  have hr := e_1799_9
  simp only [e_1799_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1992)) := by
  have c109962 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109963 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109964 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109965 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109966 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109967 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_10
  simp only [← c109962, ← c109963, ← c109964] at e_1799_10
  have e_1799_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_11
  simp only [← c109965, ← c109966, k2, ← c109967] at e_1799_11
  have hr := e_1799_11
  simp only [e_1799_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1993)) := by
  have c109968 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109969 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109970 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109971 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109972 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109973 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_12
  simp only [← c109968, ← c109969, ← c109970] at e_1799_12
  have e_1799_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_13
  simp only [← c109971, ← c109972, k2, ← c109973] at e_1799_13
  have hr := e_1799_13
  simp only [e_1799_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1994)) := by
  have c109974 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109975 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109976 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109977 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109978 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109979 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_14
  simp only [← c109974, ← c109975, ← c109976] at e_1799_14
  have e_1799_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_15
  simp only [← c109977, ← c109978, k2, ← c109979] at e_1799_15
  have hr := e_1799_15
  simp only [e_1799_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1995)) := by
  have c109980 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109981 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109982 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109983 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109984 := (publicBatchWrapper89_copies3437 a h).1
  have c109985 := (publicBatchWrapper89_copies3437 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_16
  simp only [← c109980, ← c109981, ← c109982] at e_1799_16
  have e_1799_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_17
  simp only [← c109983, ← c109984, k2, ← c109985] at e_1799_17
  have hr := e_1799_17
  simp only [e_1799_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1996)) := by
  have c109986 := (publicBatchWrapper89_copies3437 a h).2.2.1
  have c109987 := (publicBatchWrapper89_copies3437 a h).2.2.2.1
  have c109988 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.1
  have c109989 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.1
  have c109990 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.1
  have c109991 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_18
  simp only [← c109986, ← c109987, ← c109988] at e_1799_18
  have e_1799_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_19
  simp only [← c109989, ← c109990, k2, ← c109991] at e_1799_19
  have hr := e_1799_19
  simp only [e_1799_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1997)) := by
  have c109992 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.1
  have c109993 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.1
  have c109994 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109995 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109996 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109997 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_0
  simp only [← c109992, ← c109993, ← c109994] at e_1800_0
  have e_1800_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_1
  simp only [← c109995, ← c109996, k2, ← c109997] at e_1800_1
  have hr := e_1800_1
  simp only [e_1800_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1998)) := by
  have c109998 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109999 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110000 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110001 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110002 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110003 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_2
  simp only [← c109998, ← c109999, ← c110000] at e_1800_2
  have e_1800_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_3
  simp only [← c110001, ← c110002, k2, ← c110003] at e_1800_3
  have hr := e_1800_3
  simp only [e_1800_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1999)) := by
  have c110004 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110005 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110006 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110007 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110008 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110009 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_4
  simp only [← c110004, ← c110005, ← c110006] at e_1800_4
  have e_1800_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_5
  simp only [← c110007, ← c110008, k2, ← c110009] at e_1800_5
  have hr := e_1800_5
  simp only [e_1800_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 2000)) := by
  have c110010 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110011 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110012 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110013 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110014 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110015 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_6
  simp only [← c110010, ← c110011, ← c110012] at e_1800_6
  have e_1800_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_7
  simp only [← c110013, ← c110014, k2, ← c110015] at e_1800_7
  have hr := e_1800_7
  simp only [e_1800_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 2001)) := by
  have c110016 := (publicBatchWrapper89_copies3438 a h).1
  have c110017 := (publicBatchWrapper89_copies3438 a h).2.1
  have c110018 := (publicBatchWrapper89_copies3438 a h).2.2.1
  have c110019 := (publicBatchWrapper89_copies3438 a h).2.2.2.1
  have c110020 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.1
  have c110021 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_8
  simp only [← c110016, ← c110017, ← c110018] at e_1800_8
  have e_1800_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_9
  simp only [← c110019, ← c110020, k2, ← c110021] at e_1800_9
  have hr := e_1800_9
  simp only [e_1800_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2170)) := by
  have c110022 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.1
  have c110023 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.1
  have c110024 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.1
  have c110025 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.1
  have c110026 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110027 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_10
  simp only [← c110022, ← c110023, ← c110024] at e_1800_10
  have e_1800_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_11
  simp only [← c110025, ← c110026, k2, ← c110027] at e_1800_11
  have hr := e_1800_11
  simp only [e_1800_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2171)) := by
  have c110028 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110029 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110030 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110031 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110032 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110033 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_12
  simp only [← c110028, ← c110029, ← c110030] at e_1800_12
  have e_1800_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_13
  simp only [← c110031, ← c110032, k2, ← c110033] at e_1800_13
  have hr := e_1800_13
  simp only [e_1800_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2172)) := by
  have c110034 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110035 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110036 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110037 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110038 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110039 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_14
  simp only [← c110034, ← c110035, ← c110036] at e_1800_14
  have e_1800_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_15
  simp only [← c110037, ← c110038, k2, ← c110039] at e_1800_15
  have hr := e_1800_15
  simp only [e_1800_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2173)) := by
  have c110040 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110041 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110042 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110043 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110044 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110045 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_16
  simp only [← c110040, ← c110041, ← c110042] at e_1800_16
  have e_1800_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_17
  simp only [← c110043, ← c110044, k2, ← c110045] at e_1800_17
  have hr := e_1800_17
  simp only [e_1800_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2174)) := by
  have c110046 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110047 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110048 := (publicBatchWrapper89_copies3439 a h).1
  have c110049 := (publicBatchWrapper89_copies3439 a h).2.1
  have c110050 := (publicBatchWrapper89_copies3439 a h).2.2.1
  have c110051 := (publicBatchWrapper89_copies3439 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_18
  simp only [← c110046, ← c110047, ← c110048] at e_1800_18
  have e_1800_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_19
  simp only [← c110049, ← c110050, k2, ← c110051] at e_1800_19
  have hr := e_1800_19
  simp only [e_1800_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2175)) := by
  have c110052 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.1
  have c110053 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.1
  have c110054 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.1
  have c110055 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.1
  have c110056 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.1
  have c110057 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_0
  simp only [← c110052, ← c110053, ← c110054] at e_1801_0
  have e_1801_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_1
  simp only [← c110055, ← c110056, k2, ← c110057] at e_1801_1
  have hr := e_1801_1
  simp only [e_1801_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2176)) := by
  have c110058 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110059 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110060 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110061 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110062 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110063 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_2
  simp only [← c110058, ← c110059, ← c110060] at e_1801_2
  have e_1801_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_3
  simp only [← c110061, ← c110062, k2, ← c110063] at e_1801_3
  have hr := e_1801_3
  simp only [e_1801_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2177)) := by
  have c110064 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110065 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110066 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110067 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110068 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110069 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_4
  simp only [← c110064, ← c110065, ← c110066] at e_1801_4
  have e_1801_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_5
  simp only [← c110067, ← c110068, k2, ← c110069] at e_1801_5
  have hr := e_1801_5
  simp only [e_1801_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2178)) := by
  have c110070 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110071 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110072 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110073 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110074 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110075 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_6
  simp only [← c110070, ← c110071, ← c110072] at e_1801_6
  have e_1801_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_7
  simp only [← c110073, ← c110074, k2, ← c110075] at e_1801_7
  have hr := e_1801_7
  simp only [e_1801_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2179)) := by
  have c110076 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110077 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110078 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110079 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110080 := (publicBatchWrapper89_copies3440 a h).1
  have c110081 := (publicBatchWrapper89_copies3440 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_8
  simp only [← c110076, ← c110077, ← c110078] at e_1801_8
  have e_1801_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_9
  simp only [← c110079, ← c110080, k2, ← c110081] at e_1801_9
  have hr := e_1801_9
  simp only [e_1801_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2180)) := by
  have c110082 := (publicBatchWrapper89_copies3440 a h).2.2.1
  have c110083 := (publicBatchWrapper89_copies3440 a h).2.2.2.1
  have c110084 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.1
  have c110085 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.1
  have c110086 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.1
  have c110087 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_10
  simp only [← c110082, ← c110083, ← c110084] at e_1801_10
  have e_1801_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_11
  simp only [← c110085, ← c110086, k2, ← c110087] at e_1801_11
  have hr := e_1801_11
  simp only [e_1801_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2181)) := by
  have c110088 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.1
  have c110089 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.1
  have c110090 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110091 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110092 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110093 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_12
  simp only [← c110088, ← c110089, ← c110090] at e_1801_12
  have e_1801_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_13
  simp only [← c110091, ← c110092, k2, ← c110093] at e_1801_13
  have hr := e_1801_13
  simp only [e_1801_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2182)) := by
  have c110094 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110095 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110096 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110097 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110098 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110099 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_14
  simp only [← c110094, ← c110095, ← c110096] at e_1801_14
  have e_1801_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_15
  simp only [← c110097, ← c110098, k2, ← c110099] at e_1801_15
  have hr := e_1801_15
  simp only [e_1801_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2183)) := by
  have c110100 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110101 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110102 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110103 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110104 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110105 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_16
  simp only [← c110100, ← c110101, ← c110102] at e_1801_16
  have e_1801_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_17
  simp only [← c110103, ← c110104, k2, ← c110105] at e_1801_17
  have hr := e_1801_17
  simp only [e_1801_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2184)) := by
  have c110106 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110107 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110108 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110109 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110110 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110111 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_18
  simp only [← c110106, ← c110107, ← c110108] at e_1801_18
  have e_1801_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_19
  simp only [← c110109, ← c110110, k2, ← c110111] at e_1801_19
  have hr := e_1801_19
  simp only [e_1801_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2185)) := by
  have c110112 := (publicBatchWrapper89_copies3441 a h).1
  have c110113 := (publicBatchWrapper89_copies3441 a h).2.1
  have c110114 := (publicBatchWrapper89_copies3441 a h).2.2.1
  have c110115 := (publicBatchWrapper89_copies3441 a h).2.2.2.1
  have c110116 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.1
  have c110117 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_0
  simp only [← c110112, ← c110113, ← c110114] at e_1802_0
  have e_1802_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_1
  simp only [← c110115, ← c110116, k2, ← c110117] at e_1802_1
  have hr := e_1802_1
  simp only [e_1802_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2186)) := by
  have c110118 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.1
  have c110119 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.1
  have c110120 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.1
  have c110121 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.1
  have c110122 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110123 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_2
  simp only [← c110118, ← c110119, ← c110120] at e_1802_2
  have e_1802_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_3
  simp only [← c110121, ← c110122, k2, ← c110123] at e_1802_3
  have hr := e_1802_3
  simp only [e_1802_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2187)) := by
  have c110124 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110125 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110126 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110127 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110128 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110129 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_4
  simp only [← c110124, ← c110125, ← c110126] at e_1802_4
  have e_1802_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_5
  simp only [← c110127, ← c110128, k2, ← c110129] at e_1802_5
  have hr := e_1802_5
  simp only [e_1802_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2188)) := by
  have c110130 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110131 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110132 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110133 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110134 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110135 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_6
  simp only [← c110130, ← c110131, ← c110132] at e_1802_6
  have e_1802_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_7
  simp only [← c110133, ← c110134, k2, ← c110135] at e_1802_7
  have hr := e_1802_7
  simp only [e_1802_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2189)) := by
  have c110136 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110137 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110138 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110139 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110140 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110141 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_8
  simp only [← c110136, ← c110137, ← c110138] at e_1802_8
  have e_1802_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_9
  simp only [← c110139, ← c110140, k2, ← c110141] at e_1802_9
  have hr := e_1802_9
  simp only [e_1802_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2190)) := by
  have c110142 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110143 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110144 := (publicBatchWrapper89_copies3442 a h).1
  have c110145 := (publicBatchWrapper89_copies3442 a h).2.1
  have c110146 := (publicBatchWrapper89_copies3442 a h).2.2.1
  have c110147 := (publicBatchWrapper89_copies3442 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_10
  simp only [← c110142, ← c110143, ← c110144] at e_1802_10
  have e_1802_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_11
  simp only [← c110145, ← c110146, k2, ← c110147] at e_1802_11
  have hr := e_1802_11
  simp only [e_1802_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2191)) := by
  have c110148 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.1
  have c110149 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.1
  have c110150 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.1
  have c110151 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.1
  have c110152 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.1
  have c110153 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_12
  simp only [← c110148, ← c110149, ← c110150] at e_1802_12
  have e_1802_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_13
  simp only [← c110151, ← c110152, k2, ← c110153] at e_1802_13
  have hr := e_1802_13
  simp only [e_1802_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2192)) := by
  have c110154 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110155 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110156 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110157 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110158 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110159 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_14
  simp only [← c110154, ← c110155, ← c110156] at e_1802_14
  have e_1802_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_15
  simp only [← c110157, ← c110158, k2, ← c110159] at e_1802_15
  have hr := e_1802_15
  simp only [e_1802_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2193)) := by
  have c110160 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110161 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110162 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110163 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110164 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110165 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_16
  simp only [← c110160, ← c110161, ← c110162] at e_1802_16
  have e_1802_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_17
  simp only [← c110163, ← c110164, k2, ← c110165] at e_1802_17
  have hr := e_1802_17
  simp only [e_1802_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2194)) := by
  have c110166 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110167 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110168 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110169 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110170 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110171 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_18
  simp only [← c110166, ← c110167, ← c110168] at e_1802_18
  have e_1802_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_19
  simp only [← c110169, ← c110170, k2, ← c110171] at e_1802_19
  have hr := e_1802_19
  simp only [e_1802_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2195)) := by
  have c110172 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110173 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110174 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110175 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110176 := (publicBatchWrapper89_copies3443 a h).1
  have c110177 := (publicBatchWrapper89_copies3443 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_0
  simp only [← c110172, ← c110173, ← c110174] at e_1803_0
  have e_1803_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_1
  simp only [← c110175, ← c110176, k2, ← c110177] at e_1803_1
  have hr := e_1803_1
  simp only [e_1803_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2196)) := by
  have c110178 := (publicBatchWrapper89_copies3443 a h).2.2.1
  have c110179 := (publicBatchWrapper89_copies3443 a h).2.2.2.1
  have c110180 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.1
  have c110181 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.1
  have c110182 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.1
  have c110183 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_2
  simp only [← c110178, ← c110179, ← c110180] at e_1803_2
  have e_1803_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_3
  simp only [← c110181, ← c110182, k2, ← c110183] at e_1803_3
  have hr := e_1803_3
  simp only [e_1803_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2197)) := by
  have c110184 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.1
  have c110185 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.1
  have c110186 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110187 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110188 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110189 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_4
  simp only [← c110184, ← c110185, ← c110186] at e_1803_4
  have e_1803_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_5
  simp only [← c110187, ← c110188, k2, ← c110189] at e_1803_5
  have hr := e_1803_5
  simp only [e_1803_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2198)) := by
  have c110190 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110191 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110192 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110193 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110194 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110195 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_6
  simp only [← c110190, ← c110191, ← c110192] at e_1803_6
  have e_1803_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_7
  simp only [← c110193, ← c110194, k2, ← c110195] at e_1803_7
  have hr := e_1803_7
  simp only [e_1803_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2199)) := by
  have c110196 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110197 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110198 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110199 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110200 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110201 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_8
  simp only [← c110196, ← c110197, ← c110198] at e_1803_8
  have e_1803_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_9
  simp only [← c110199, ← c110200, k2, ← c110201] at e_1803_9
  have hr := e_1803_9
  simp only [e_1803_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2200)) := by
  have c110202 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110203 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110204 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110205 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110206 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110207 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_10
  simp only [← c110202, ← c110203, ← c110204] at e_1803_10
  have e_1803_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_11
  simp only [← c110205, ← c110206, k2, ← c110207] at e_1803_11
  have hr := e_1803_11
  simp only [e_1803_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2201)) := by
  have c110208 := (publicBatchWrapper89_copies3444 a h).1
  have c110209 := (publicBatchWrapper89_copies3444 a h).2.1
  have c110210 := (publicBatchWrapper89_copies3444 a h).2.2.1
  have c110211 := (publicBatchWrapper89_copies3444 a h).2.2.2.1
  have c110212 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.1
  have c110213 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_12
  simp only [← c110208, ← c110209, ← c110210] at e_1803_12
  have e_1803_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_13
  simp only [← c110211, ← c110212, k2, ← c110213] at e_1803_13
  have hr := e_1803_13
  simp only [e_1803_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2370)) := by
  have c110214 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.1
  have c110215 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.1
  have c110216 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.1
  have c110217 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.1
  have c110218 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110219 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_14
  simp only [← c110214, ← c110215, ← c110216] at e_1803_14
  have e_1803_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_15
  simp only [← c110217, ← c110218, k2, ← c110219] at e_1803_15
  have hr := e_1803_15
  simp only [e_1803_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2371)) := by
  have c110220 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110221 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110222 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110223 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110224 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110225 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_16
  simp only [← c110220, ← c110221, ← c110222] at e_1803_16
  have e_1803_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_17
  simp only [← c110223, ← c110224, k2, ← c110225] at e_1803_17
  have hr := e_1803_17
  simp only [e_1803_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2372)) := by
  have c110226 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110227 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110228 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110229 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110230 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110231 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_18
  simp only [← c110226, ← c110227, ← c110228] at e_1803_18
  have e_1803_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_19
  simp only [← c110229, ← c110230, k2, ← c110231] at e_1803_19
  have hr := e_1803_19
  simp only [e_1803_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2373)) := by
  have c110232 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110233 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110234 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110235 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110236 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110237 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_0
  simp only [← c110232, ← c110233, ← c110234] at e_1804_0
  have e_1804_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_1
  simp only [← c110235, ← c110236, k2, ← c110237] at e_1804_1
  have hr := e_1804_1
  simp only [e_1804_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2374)) := by
  have c110238 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110239 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110240 := (publicBatchWrapper89_copies3445 a h).1
  have c110241 := (publicBatchWrapper89_copies3445 a h).2.1
  have c110242 := (publicBatchWrapper89_copies3445 a h).2.2.1
  have c110243 := (publicBatchWrapper89_copies3445 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_2
  simp only [← c110238, ← c110239, ← c110240] at e_1804_2
  have e_1804_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_3
  simp only [← c110241, ← c110242, k2, ← c110243] at e_1804_3
  have hr := e_1804_3
  simp only [e_1804_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2375)) := by
  have c110244 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.1
  have c110245 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.1
  have c110246 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.1
  have c110247 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.1
  have c110248 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.1
  have c110249 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_4
  simp only [← c110244, ← c110245, ← c110246] at e_1804_4
  have e_1804_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_5
  simp only [← c110247, ← c110248, k2, ← c110249] at e_1804_5
  have hr := e_1804_5
  simp only [e_1804_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2376)) := by
  have c110250 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110251 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110252 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110253 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110254 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110255 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_6
  simp only [← c110250, ← c110251, ← c110252] at e_1804_6
  have e_1804_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_7
  simp only [← c110253, ← c110254, k2, ← c110255] at e_1804_7
  have hr := e_1804_7
  simp only [e_1804_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2377)) := by
  have c110256 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110257 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110258 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110259 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110260 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110261 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_8
  simp only [← c110256, ← c110257, ← c110258] at e_1804_8
  have e_1804_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_9
  simp only [← c110259, ← c110260, k2, ← c110261] at e_1804_9
  have hr := e_1804_9
  simp only [e_1804_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2378)) := by
  have c110262 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110263 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110264 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110265 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110266 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110267 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_10
  simp only [← c110262, ← c110263, ← c110264] at e_1804_10
  have e_1804_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_11
  simp only [← c110265, ← c110266, k2, ← c110267] at e_1804_11
  have hr := e_1804_11
  simp only [e_1804_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2379)) := by
  have c110268 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110269 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110270 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110271 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110272 := (publicBatchWrapper89_copies3446 a h).1
  have c110273 := (publicBatchWrapper89_copies3446 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_12
  simp only [← c110268, ← c110269, ← c110270] at e_1804_12
  have e_1804_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_13
  simp only [← c110271, ← c110272, k2, ← c110273] at e_1804_13
  have hr := e_1804_13
  simp only [e_1804_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2380)) := by
  have c110274 := (publicBatchWrapper89_copies3446 a h).2.2.1
  have c110275 := (publicBatchWrapper89_copies3446 a h).2.2.2.1
  have c110276 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.1
  have c110277 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.1
  have c110278 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.1
  have c110279 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_14
  simp only [← c110274, ← c110275, ← c110276] at e_1804_14
  have e_1804_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_15
  simp only [← c110277, ← c110278, k2, ← c110279] at e_1804_15
  have hr := e_1804_15
  simp only [e_1804_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2381)) := by
  have c110280 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.1
  have c110281 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.1
  have c110282 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110283 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110284 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110285 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_16
  simp only [← c110280, ← c110281, ← c110282] at e_1804_16
  have e_1804_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_17
  simp only [← c110283, ← c110284, k2, ← c110285] at e_1804_17
  have hr := e_1804_17
  simp only [e_1804_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2382)) := by
  have c110286 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110287 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110288 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110289 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110290 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110291 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_18
  simp only [← c110286, ← c110287, ← c110288] at e_1804_18
  have e_1804_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_19
  simp only [← c110289, ← c110290, k2, ← c110291] at e_1804_19
  have hr := e_1804_19
  simp only [e_1804_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2383)) := by
  have c110292 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110293 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110294 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110295 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110296 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110297 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_0
  simp only [← c110292, ← c110293, ← c110294] at e_1805_0
  have e_1805_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_1
  simp only [← c110295, ← c110296, k2, ← c110297] at e_1805_1
  have hr := e_1805_1
  simp only [e_1805_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2384)) := by
  have c110298 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110299 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110300 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110301 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110302 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110303 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_2
  simp only [← c110298, ← c110299, ← c110300] at e_1805_2
  have e_1805_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_3
  simp only [← c110301, ← c110302, k2, ← c110303] at e_1805_3
  have hr := e_1805_3
  simp only [e_1805_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2385)) := by
  have c110304 := (publicBatchWrapper89_copies3447 a h).1
  have c110305 := (publicBatchWrapper89_copies3447 a h).2.1
  have c110306 := (publicBatchWrapper89_copies3447 a h).2.2.1
  have c110307 := (publicBatchWrapper89_copies3447 a h).2.2.2.1
  have c110308 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.1
  have c110309 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_4
  simp only [← c110304, ← c110305, ← c110306] at e_1805_4
  have e_1805_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_5
  simp only [← c110307, ← c110308, k2, ← c110309] at e_1805_5
  have hr := e_1805_5
  simp only [e_1805_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2386)) := by
  have c110310 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.1
  have c110311 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.1
  have c110312 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.1
  have c110313 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.1
  have c110314 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110315 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_6
  simp only [← c110310, ← c110311, ← c110312] at e_1805_6
  have e_1805_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_7
  simp only [← c110313, ← c110314, k2, ← c110315] at e_1805_7
  have hr := e_1805_7
  simp only [e_1805_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2387)) := by
  have c110316 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110317 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110318 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110319 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110320 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110321 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_8
  simp only [← c110316, ← c110317, ← c110318] at e_1805_8
  have e_1805_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_9
  simp only [← c110319, ← c110320, k2, ← c110321] at e_1805_9
  have hr := e_1805_9
  simp only [e_1805_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2388)) := by
  have c110322 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110323 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110324 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110325 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110326 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110327 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_10
  simp only [← c110322, ← c110323, ← c110324] at e_1805_10
  have e_1805_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_11
  simp only [← c110325, ← c110326, k2, ← c110327] at e_1805_11
  have hr := e_1805_11
  simp only [e_1805_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2389)) := by
  have c110328 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110329 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110330 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110331 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110332 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110333 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_12
  simp only [← c110328, ← c110329, ← c110330] at e_1805_12
  have e_1805_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_13
  simp only [← c110331, ← c110332, k2, ← c110333] at e_1805_13
  have hr := e_1805_13
  simp only [e_1805_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2390)) := by
  have c110334 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110335 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110336 := (publicBatchWrapper89_copies3448 a h).1
  have c110337 := (publicBatchWrapper89_copies3448 a h).2.1
  have c110338 := (publicBatchWrapper89_copies3448 a h).2.2.1
  have c110339 := (publicBatchWrapper89_copies3448 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_14
  simp only [← c110334, ← c110335, ← c110336] at e_1805_14
  have e_1805_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_15
  simp only [← c110337, ← c110338, k2, ← c110339] at e_1805_15
  have hr := e_1805_15
  simp only [e_1805_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2391)) := by
  have c110340 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.1
  have c110341 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.1
  have c110342 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.1
  have c110343 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.1
  have c110344 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.1
  have c110345 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_16
  simp only [← c110340, ← c110341, ← c110342] at e_1805_16
  have e_1805_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_17
  simp only [← c110343, ← c110344, k2, ← c110345] at e_1805_17
  have hr := e_1805_17
  simp only [e_1805_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2392)) := by
  have c110346 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110347 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110348 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110349 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110350 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110351 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_18
  simp only [← c110346, ← c110347, ← c110348] at e_1805_18
  have e_1805_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_19
  simp only [← c110349, ← c110350, k2, ← c110351] at e_1805_19
  have hr := e_1805_19
  simp only [e_1805_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2393)) := by
  have c110352 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110353 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110354 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110355 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110356 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110357 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_0
  simp only [← c110352, ← c110353, ← c110354] at e_1806_0
  have e_1806_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_1
  simp only [← c110355, ← c110356, k2, ← c110357] at e_1806_1
  have hr := e_1806_1
  simp only [e_1806_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2394)) := by
  have c110358 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110359 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110360 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110361 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110362 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110363 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_2
  simp only [← c110358, ← c110359, ← c110360] at e_1806_2
  have e_1806_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_3
  simp only [← c110361, ← c110362, k2, ← c110363] at e_1806_3
  have hr := e_1806_3
  simp only [e_1806_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2395)) := by
  have c110364 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110365 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110366 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110367 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110368 := (publicBatchWrapper89_copies3449 a h).1
  have c110369 := (publicBatchWrapper89_copies3449 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_4
  simp only [← c110364, ← c110365, ← c110366] at e_1806_4
  have e_1806_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_5
  simp only [← c110367, ← c110368, k2, ← c110369] at e_1806_5
  have hr := e_1806_5
  simp only [e_1806_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2396)) := by
  have c110370 := (publicBatchWrapper89_copies3449 a h).2.2.1
  have c110371 := (publicBatchWrapper89_copies3449 a h).2.2.2.1
  have c110372 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.1
  have c110373 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.1
  have c110374 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.1
  have c110375 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_6
  simp only [← c110370, ← c110371, ← c110372] at e_1806_6
  have e_1806_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_7
  simp only [← c110373, ← c110374, k2, ← c110375] at e_1806_7
  have hr := e_1806_7
  simp only [e_1806_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2397)) := by
  have c110376 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.1
  have c110377 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.1
  have c110378 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110379 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110380 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110381 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_8
  simp only [← c110376, ← c110377, ← c110378] at e_1806_8
  have e_1806_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_9
  simp only [← c110379, ← c110380, k2, ← c110381] at e_1806_9
  have hr := e_1806_9
  simp only [e_1806_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2398)) := by
  have c110382 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110383 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110384 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110385 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110386 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110387 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_10
  simp only [← c110382, ← c110383, ← c110384] at e_1806_10
  have e_1806_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_11
  simp only [← c110385, ← c110386, k2, ← c110387] at e_1806_11
  have hr := e_1806_11
  simp only [e_1806_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2399)) := by
  have c110388 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110389 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110390 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110391 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110392 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110393 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_12
  simp only [← c110388, ← c110389, ← c110390] at e_1806_12
  have e_1806_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_13
  simp only [← c110391, ← c110392, k2, ← c110393] at e_1806_13
  have hr := e_1806_13
  simp only [e_1806_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2400)) := by
  have c110394 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110395 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110396 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110397 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110398 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110399 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_14
  simp only [← c110394, ← c110395, ← c110396] at e_1806_14
  have e_1806_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_15
  simp only [← c110397, ← c110398, k2, ← c110399] at e_1806_15
  have hr := e_1806_15
  simp only [e_1806_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2401)) := by
  have c110400 := (publicBatchWrapper89_copies3450 a h).1
  have c110401 := (publicBatchWrapper89_copies3450 a h).2.1
  have c110402 := (publicBatchWrapper89_copies3450 a h).2.2.1
  have c110403 := (publicBatchWrapper89_copies3450 a h).2.2.2.1
  have c110404 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.1
  have c110405 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_16
  simp only [← c110400, ← c110401, ← c110402] at e_1806_16
  have e_1806_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_17
  simp only [← c110403, ← c110404, k2, ← c110405] at e_1806_17
  have hr := e_1806_17
  simp only [e_1806_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2570)) := by
  have c110406 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.1
  have c110407 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.1
  have c110408 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.1
  have c110409 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.1
  have c110410 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110411 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_18
  simp only [← c110406, ← c110407, ← c110408] at e_1806_18
  have e_1806_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_19
  simp only [← c110409, ← c110410, k2, ← c110411] at e_1806_19
  have hr := e_1806_19
  simp only [e_1806_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2571)) := by
  have c110412 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110413 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110414 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110415 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110416 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110417 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_0
  simp only [← c110412, ← c110413, ← c110414] at e_1807_0
  have e_1807_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_1
  simp only [← c110415, ← c110416, k2, ← c110417] at e_1807_1
  have hr := e_1807_1
  simp only [e_1807_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2572)) := by
  have c110418 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110419 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110420 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110421 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110422 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110423 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_2
  simp only [← c110418, ← c110419, ← c110420] at e_1807_2
  have e_1807_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_3
  simp only [← c110421, ← c110422, k2, ← c110423] at e_1807_3
  have hr := e_1807_3
  simp only [e_1807_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2573)) := by
  have c110424 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110425 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110426 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110427 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110428 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110429 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_4
  simp only [← c110424, ← c110425, ← c110426] at e_1807_4
  have e_1807_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_5
  simp only [← c110427, ← c110428, k2, ← c110429] at e_1807_5
  have hr := e_1807_5
  simp only [e_1807_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2574)) := by
  have c110430 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110431 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110432 := (publicBatchWrapper89_copies3451 a h).1
  have c110433 := (publicBatchWrapper89_copies3451 a h).2.1
  have c110434 := (publicBatchWrapper89_copies3451 a h).2.2.1
  have c110435 := (publicBatchWrapper89_copies3451 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_6
  simp only [← c110430, ← c110431, ← c110432] at e_1807_6
  have e_1807_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_7
  simp only [← c110433, ← c110434, k2, ← c110435] at e_1807_7
  have hr := e_1807_7
  simp only [e_1807_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2575)) := by
  have c110436 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.1
  have c110437 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.1
  have c110438 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.1
  have c110439 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.1
  have c110440 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.1
  have c110441 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_8
  simp only [← c110436, ← c110437, ← c110438] at e_1807_8
  have e_1807_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_9
  simp only [← c110439, ← c110440, k2, ← c110441] at e_1807_9
  have hr := e_1807_9
  simp only [e_1807_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2576)) := by
  have c110442 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110443 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110444 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110445 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110446 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110447 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_10
  simp only [← c110442, ← c110443, ← c110444] at e_1807_10
  have e_1807_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_11
  simp only [← c110445, ← c110446, k2, ← c110447] at e_1807_11
  have hr := e_1807_11
  simp only [e_1807_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2577)) := by
  have c110448 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110449 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110450 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110451 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110452 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110453 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_12
  simp only [← c110448, ← c110449, ← c110450] at e_1807_12
  have e_1807_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_13
  simp only [← c110451, ← c110452, k2, ← c110453] at e_1807_13
  have hr := e_1807_13
  simp only [e_1807_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2578)) := by
  have c110454 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110455 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110456 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110457 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110458 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110459 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_14
  simp only [← c110454, ← c110455, ← c110456] at e_1807_14
  have e_1807_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_15
  simp only [← c110457, ← c110458, k2, ← c110459] at e_1807_15
  have hr := e_1807_15
  simp only [e_1807_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2579)) := by
  have c110460 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110461 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110462 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110463 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110464 := (publicBatchWrapper89_copies3452 a h).1
  have c110465 := (publicBatchWrapper89_copies3452 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_16
  simp only [← c110460, ← c110461, ← c110462] at e_1807_16
  have e_1807_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_17
  simp only [← c110463, ← c110464, k2, ← c110465] at e_1807_17
  have hr := e_1807_17
  simp only [e_1807_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2580)) := by
  have c110466 := (publicBatchWrapper89_copies3452 a h).2.2.1
  have c110467 := (publicBatchWrapper89_copies3452 a h).2.2.2.1
  have c110468 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.1
  have c110469 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.1
  have c110470 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.1
  have c110471 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_18
  simp only [← c110466, ← c110467, ← c110468] at e_1807_18
  have e_1807_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_19
  simp only [← c110469, ← c110470, k2, ← c110471] at e_1807_19
  have hr := e_1807_19
  simp only [e_1807_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2581)) := by
  have c110472 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.1
  have c110473 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.1
  have c110474 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110475 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110476 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110477 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_0
  simp only [← c110472, ← c110473, ← c110474] at e_1808_0
  have e_1808_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_1
  simp only [← c110475, ← c110476, k2, ← c110477] at e_1808_1
  have hr := e_1808_1
  simp only [e_1808_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2582)) := by
  have c110478 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110479 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110480 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110481 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110482 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110483 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_2
  simp only [← c110478, ← c110479, ← c110480] at e_1808_2
  have e_1808_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_3
  simp only [← c110481, ← c110482, k2, ← c110483] at e_1808_3
  have hr := e_1808_3
  simp only [e_1808_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2583)) := by
  have c110484 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110485 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110486 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110487 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110488 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110489 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_4
  simp only [← c110484, ← c110485, ← c110486] at e_1808_4
  have e_1808_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_5
  simp only [← c110487, ← c110488, k2, ← c110489] at e_1808_5
  have hr := e_1808_5
  simp only [e_1808_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2584)) := by
  have c110490 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110491 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110492 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110493 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110494 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110495 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_6
  simp only [← c110490, ← c110491, ← c110492] at e_1808_6
  have e_1808_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_7
  simp only [← c110493, ← c110494, k2, ← c110495] at e_1808_7
  have hr := e_1808_7
  simp only [e_1808_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2585)) := by
  have c110496 := (publicBatchWrapper89_copies3453 a h).1
  have c110497 := (publicBatchWrapper89_copies3453 a h).2.1
  have c110498 := (publicBatchWrapper89_copies3453 a h).2.2.1
  have c110499 := (publicBatchWrapper89_copies3453 a h).2.2.2.1
  have c110500 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.1
  have c110501 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_8
  simp only [← c110496, ← c110497, ← c110498] at e_1808_8
  have e_1808_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_9
  simp only [← c110499, ← c110500, k2, ← c110501] at e_1808_9
  have hr := e_1808_9
  simp only [e_1808_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2586)) := by
  have c110502 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.1
  have c110503 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.1
  have c110504 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.1
  have c110505 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.1
  have c110506 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110507 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_10
  simp only [← c110502, ← c110503, ← c110504] at e_1808_10
  have e_1808_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_11
  simp only [← c110505, ← c110506, k2, ← c110507] at e_1808_11
  have hr := e_1808_11
  simp only [e_1808_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2587)) := by
  have c110508 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110509 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110510 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110511 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110512 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110513 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_12
  simp only [← c110508, ← c110509, ← c110510] at e_1808_12
  have e_1808_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_13
  simp only [← c110511, ← c110512, k2, ← c110513] at e_1808_13
  have hr := e_1808_13
  simp only [e_1808_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2588)) := by
  have c110514 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110515 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110516 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110517 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110518 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110519 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_14
  simp only [← c110514, ← c110515, ← c110516] at e_1808_14
  have e_1808_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_15
  simp only [← c110517, ← c110518, k2, ← c110519] at e_1808_15
  have hr := e_1808_15
  simp only [e_1808_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2589)) := by
  have c110520 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110521 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110522 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110523 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110524 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110525 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_16
  simp only [← c110520, ← c110521, ← c110522] at e_1808_16
  have e_1808_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_17
  simp only [← c110523, ← c110524, k2, ← c110525] at e_1808_17
  have hr := e_1808_17
  simp only [e_1808_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2590)) := by
  have c110526 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110527 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110528 := (publicBatchWrapper89_copies3454 a h).1
  have c110529 := (publicBatchWrapper89_copies3454 a h).2.1
  have c110530 := (publicBatchWrapper89_copies3454 a h).2.2.1
  have c110531 := (publicBatchWrapper89_copies3454 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_18
  simp only [← c110526, ← c110527, ← c110528] at e_1808_18
  have e_1808_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_19
  simp only [← c110529, ← c110530, k2, ← c110531] at e_1808_19
  have hr := e_1808_19
  simp only [e_1808_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2591)) := by
  have c110532 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.1
  have c110533 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.1
  have c110534 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.1
  have c110535 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.1
  have c110536 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.1
  have c110537 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_0
  simp only [← c110532, ← c110533, ← c110534] at e_1809_0
  have e_1809_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_1
  simp only [← c110535, ← c110536, k2, ← c110537] at e_1809_1
  have hr := e_1809_1
  simp only [e_1809_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2592)) := by
  have c110538 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110539 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110540 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110541 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110542 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110543 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_2
  simp only [← c110538, ← c110539, ← c110540] at e_1809_2
  have e_1809_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_3
  simp only [← c110541, ← c110542, k2, ← c110543] at e_1809_3
  have hr := e_1809_3
  simp only [e_1809_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2593)) := by
  have c110544 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110545 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110546 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110547 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110548 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110549 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_4
  simp only [← c110544, ← c110545, ← c110546] at e_1809_4
  have e_1809_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_5
  simp only [← c110547, ← c110548, k2, ← c110549] at e_1809_5
  have hr := e_1809_5
  simp only [e_1809_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2594)) := by
  have c110550 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110551 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110552 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110553 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110554 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110555 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_6
  simp only [← c110550, ← c110551, ← c110552] at e_1809_6
  have e_1809_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_7
  simp only [← c110553, ← c110554, k2, ← c110555] at e_1809_7
  have hr := e_1809_7
  simp only [e_1809_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2595)) := by
  have c110556 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110557 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110558 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110559 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110560 := (publicBatchWrapper89_copies3455 a h).1
  have c110561 := (publicBatchWrapper89_copies3455 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_8
  simp only [← c110556, ← c110557, ← c110558] at e_1809_8
  have e_1809_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_9
  simp only [← c110559, ← c110560, k2, ← c110561] at e_1809_9
  have hr := e_1809_9
  simp only [e_1809_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2596)) := by
  have c110562 := (publicBatchWrapper89_copies3455 a h).2.2.1
  have c110563 := (publicBatchWrapper89_copies3455 a h).2.2.2.1
  have c110564 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.1
  have c110565 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.1
  have c110566 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.1
  have c110567 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_10
  simp only [← c110562, ← c110563, ← c110564] at e_1809_10
  have e_1809_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_11
  simp only [← c110565, ← c110566, k2, ← c110567] at e_1809_11
  have hr := e_1809_11
  simp only [e_1809_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2597)) := by
  have c110568 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.1
  have c110569 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.1
  have c110570 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110571 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110572 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110573 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_12
  simp only [← c110568, ← c110569, ← c110570] at e_1809_12
  have e_1809_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_13
  simp only [← c110571, ← c110572, k2, ← c110573] at e_1809_13
  have hr := e_1809_13
  simp only [e_1809_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2598)) := by
  have c110574 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110575 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110576 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110577 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110578 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110579 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_14
  simp only [← c110574, ← c110575, ← c110576] at e_1809_14
  have e_1809_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_15
  simp only [← c110577, ← c110578, k2, ← c110579] at e_1809_15
  have hr := e_1809_15
  simp only [e_1809_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2599)) := by
  have c110580 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110581 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110582 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110583 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110584 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110585 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_16
  simp only [← c110580, ← c110581, ← c110582] at e_1809_16
  have e_1809_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_17
  simp only [← c110583, ← c110584, k2, ← c110585] at e_1809_17
  have hr := e_1809_17
  simp only [e_1809_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2600)) := by
  have c110586 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110587 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110588 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110589 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110590 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110591 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_18
  simp only [← c110586, ← c110587, ← c110588] at e_1809_18
  have e_1809_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_19
  simp only [← c110589, ← c110590, k2, ← c110591] at e_1809_19
  have hr := e_1809_19
  simp only [e_1809_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2601)) := by
  have c110592 := (publicBatchWrapper89_copies3456 a h).1
  have c110593 := (publicBatchWrapper89_copies3456 a h).2.1
  have c110594 := (publicBatchWrapper89_copies3456 a h).2.2.1
  have c110595 := (publicBatchWrapper89_copies3456 a h).2.2.2.1
  have c110596 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.1
  have c110597 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_0
  simp only [← c110592, ← c110593, ← c110594] at e_1810_0
  have e_1810_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_1
  simp only [← c110595, ← c110596, k2, ← c110597] at e_1810_1
  have hr := e_1810_1
  simp only [e_1810_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2770)) := by
  have c110598 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.1
  have c110599 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.1
  have c110600 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.1
  have c110601 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.1
  have c110602 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110603 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_2
  simp only [← c110598, ← c110599, ← c110600] at e_1810_2
  have e_1810_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_3
  simp only [← c110601, ← c110602, k2, ← c110603] at e_1810_3
  have hr := e_1810_3
  simp only [e_1810_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2771)) := by
  have c110604 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110605 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110606 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110607 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110608 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110609 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_4
  simp only [← c110604, ← c110605, ← c110606] at e_1810_4
  have e_1810_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_5
  simp only [← c110607, ← c110608, k2, ← c110609] at e_1810_5
  have hr := e_1810_5
  simp only [e_1810_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2772)) := by
  have c110610 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110611 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110612 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110613 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110614 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110615 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_6
  simp only [← c110610, ← c110611, ← c110612] at e_1810_6
  have e_1810_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_7
  simp only [← c110613, ← c110614, k2, ← c110615] at e_1810_7
  have hr := e_1810_7
  simp only [e_1810_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2773)) := by
  have c110616 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110617 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110618 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110619 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110620 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110621 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_8
  simp only [← c110616, ← c110617, ← c110618] at e_1810_8
  have e_1810_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_9
  simp only [← c110619, ← c110620, k2, ← c110621] at e_1810_9
  have hr := e_1810_9
  simp only [e_1810_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2774)) := by
  have c110622 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110623 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110624 := (publicBatchWrapper89_copies3457 a h).1
  have c110625 := (publicBatchWrapper89_copies3457 a h).2.1
  have c110626 := (publicBatchWrapper89_copies3457 a h).2.2.1
  have c110627 := (publicBatchWrapper89_copies3457 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_10
  simp only [← c110622, ← c110623, ← c110624] at e_1810_10
  have e_1810_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_11
  simp only [← c110625, ← c110626, k2, ← c110627] at e_1810_11
  have hr := e_1810_11
  simp only [e_1810_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2775)) := by
  have c110628 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.1
  have c110629 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.1
  have c110630 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.1
  have c110631 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.1
  have c110632 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.1
  have c110633 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_12
  simp only [← c110628, ← c110629, ← c110630] at e_1810_12
  have e_1810_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_13
  simp only [← c110631, ← c110632, k2, ← c110633] at e_1810_13
  have hr := e_1810_13
  simp only [e_1810_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2776)) := by
  have c110634 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110635 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110636 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110637 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110638 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110639 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_14
  simp only [← c110634, ← c110635, ← c110636] at e_1810_14
  have e_1810_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_15
  simp only [← c110637, ← c110638, k2, ← c110639] at e_1810_15
  have hr := e_1810_15
  simp only [e_1810_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2777)) := by
  have c110640 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110641 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110642 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110643 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110644 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110645 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_16
  simp only [← c110640, ← c110641, ← c110642] at e_1810_16
  have e_1810_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_17
  simp only [← c110643, ← c110644, k2, ← c110645] at e_1810_17
  have hr := e_1810_17
  simp only [e_1810_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2778)) := by
  have c110646 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110647 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110648 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110649 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110650 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110651 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_18
  simp only [← c110646, ← c110647, ← c110648] at e_1810_18
  have e_1810_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_19
  simp only [← c110649, ← c110650, k2, ← c110651] at e_1810_19
  have hr := e_1810_19
  simp only [e_1810_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2779)) := by
  have c110652 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110653 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110654 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110655 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110656 := (publicBatchWrapper89_copies3458 a h).1
  have c110657 := (publicBatchWrapper89_copies3458 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_0
  simp only [← c110652, ← c110653, ← c110654] at e_1811_0
  have e_1811_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_1
  simp only [← c110655, ← c110656, k2, ← c110657] at e_1811_1
  have hr := e_1811_1
  simp only [e_1811_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2780)) := by
  have c110658 := (publicBatchWrapper89_copies3458 a h).2.2.1
  have c110659 := (publicBatchWrapper89_copies3458 a h).2.2.2.1
  have c110660 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.1
  have c110661 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.1
  have c110662 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.1
  have c110663 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_2
  simp only [← c110658, ← c110659, ← c110660] at e_1811_2
  have e_1811_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_3
  simp only [← c110661, ← c110662, k2, ← c110663] at e_1811_3
  have hr := e_1811_3
  simp only [e_1811_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2781)) := by
  have c110664 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.1
  have c110665 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.1
  have c110666 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110667 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110668 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110669 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_4
  simp only [← c110664, ← c110665, ← c110666] at e_1811_4
  have e_1811_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_5
  simp only [← c110667, ← c110668, k2, ← c110669] at e_1811_5
  have hr := e_1811_5
  simp only [e_1811_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2782)) := by
  have c110670 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110671 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110672 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110673 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110674 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110675 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_6
  simp only [← c110670, ← c110671, ← c110672] at e_1811_6
  have e_1811_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_7
  simp only [← c110673, ← c110674, k2, ← c110675] at e_1811_7
  have hr := e_1811_7
  simp only [e_1811_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2783)) := by
  have c110676 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110677 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110678 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110679 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110680 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110681 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_8
  simp only [← c110676, ← c110677, ← c110678] at e_1811_8
  have e_1811_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_9
  simp only [← c110679, ← c110680, k2, ← c110681] at e_1811_9
  have hr := e_1811_9
  simp only [e_1811_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2784)) := by
  have c110682 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110683 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110684 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110685 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110686 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110687 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_10
  simp only [← c110682, ← c110683, ← c110684] at e_1811_10
  have e_1811_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_11
  simp only [← c110685, ← c110686, k2, ← c110687] at e_1811_11
  have hr := e_1811_11
  simp only [e_1811_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2785)) := by
  have c110688 := (publicBatchWrapper89_copies3459 a h).1
  have c110689 := (publicBatchWrapper89_copies3459 a h).2.1
  have c110690 := (publicBatchWrapper89_copies3459 a h).2.2.1
  have c110691 := (publicBatchWrapper89_copies3459 a h).2.2.2.1
  have c110692 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.1
  have c110693 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_12
  simp only [← c110688, ← c110689, ← c110690] at e_1811_12
  have e_1811_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_13
  simp only [← c110691, ← c110692, k2, ← c110693] at e_1811_13
  have hr := e_1811_13
  simp only [e_1811_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2786)) := by
  have c110694 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.1
  have c110695 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.1
  have c110696 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.1
  have c110697 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.1
  have c110698 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110699 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_14
  simp only [← c110694, ← c110695, ← c110696] at e_1811_14
  have e_1811_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_15
  simp only [← c110697, ← c110698, k2, ← c110699] at e_1811_15
  have hr := e_1811_15
  simp only [e_1811_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2787)) := by
  have c110700 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110701 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110702 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110703 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110704 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110705 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_16
  simp only [← c110700, ← c110701, ← c110702] at e_1811_16
  have e_1811_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_17
  simp only [← c110703, ← c110704, k2, ← c110705] at e_1811_17
  have hr := e_1811_17
  simp only [e_1811_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2788)) := by
  have c110706 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110707 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110708 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110709 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110710 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110711 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_18
  simp only [← c110706, ← c110707, ← c110708] at e_1811_18
  have e_1811_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_19
  simp only [← c110709, ← c110710, k2, ← c110711] at e_1811_19
  have hr := e_1811_19
  simp only [e_1811_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2789)) := by
  have c110712 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110713 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110714 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110715 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110716 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110717 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_0
  simp only [← c110712, ← c110713, ← c110714] at e_1812_0
  have e_1812_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_1
  simp only [← c110715, ← c110716, k2, ← c110717] at e_1812_1
  have hr := e_1812_1
  simp only [e_1812_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2790)) := by
  have c110718 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110719 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110720 := (publicBatchWrapper89_copies3460 a h).1
  have c110721 := (publicBatchWrapper89_copies3460 a h).2.1
  have c110722 := (publicBatchWrapper89_copies3460 a h).2.2.1
  have c110723 := (publicBatchWrapper89_copies3460 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_2
  simp only [← c110718, ← c110719, ← c110720] at e_1812_2
  have e_1812_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_3
  simp only [← c110721, ← c110722, k2, ← c110723] at e_1812_3
  have hr := e_1812_3
  simp only [e_1812_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2791)) := by
  have c110724 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.1
  have c110725 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.1
  have c110726 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.1
  have c110727 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.1
  have c110728 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.1
  have c110729 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_4
  simp only [← c110724, ← c110725, ← c110726] at e_1812_4
  have e_1812_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_5
  simp only [← c110727, ← c110728, k2, ← c110729] at e_1812_5
  have hr := e_1812_5
  simp only [e_1812_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2792)) := by
  have c110730 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110731 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110732 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110733 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110734 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110735 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_6
  simp only [← c110730, ← c110731, ← c110732] at e_1812_6
  have e_1812_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_7
  simp only [← c110733, ← c110734, k2, ← c110735] at e_1812_7
  have hr := e_1812_7
  simp only [e_1812_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2793)) := by
  have c110736 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110737 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110738 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110739 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110740 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110741 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_8
  simp only [← c110736, ← c110737, ← c110738] at e_1812_8
  have e_1812_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_9
  simp only [← c110739, ← c110740, k2, ← c110741] at e_1812_9
  have hr := e_1812_9
  simp only [e_1812_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2794)) := by
  have c110742 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110743 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110744 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110745 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110746 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110747 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_10
  simp only [← c110742, ← c110743, ← c110744] at e_1812_10
  have e_1812_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_11
  simp only [← c110745, ← c110746, k2, ← c110747] at e_1812_11
  have hr := e_1812_11
  simp only [e_1812_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2795)) := by
  have c110748 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110749 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110750 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110751 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110752 := (publicBatchWrapper89_copies3461 a h).1
  have c110753 := (publicBatchWrapper89_copies3461 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_12
  simp only [← c110748, ← c110749, ← c110750] at e_1812_12
  have e_1812_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_13
  simp only [← c110751, ← c110752, k2, ← c110753] at e_1812_13
  have hr := e_1812_13
  simp only [e_1812_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2796)) := by
  have c110754 := (publicBatchWrapper89_copies3461 a h).2.2.1
  have c110755 := (publicBatchWrapper89_copies3461 a h).2.2.2.1
  have c110756 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.1
  have c110757 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.1
  have c110758 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.1
  have c110759 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_14
  simp only [← c110754, ← c110755, ← c110756] at e_1812_14
  have e_1812_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_15
  simp only [← c110757, ← c110758, k2, ← c110759] at e_1812_15
  have hr := e_1812_15
  simp only [e_1812_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2797)) := by
  have c110760 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.1
  have c110761 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.1
  have c110762 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110763 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110764 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110765 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_16
  simp only [← c110760, ← c110761, ← c110762] at e_1812_16
  have e_1812_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_17
  simp only [← c110763, ← c110764, k2, ← c110765] at e_1812_17
  have hr := e_1812_17
  simp only [e_1812_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2798)) := by
  have c110766 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110767 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110768 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110769 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110770 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110771 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_18
  simp only [← c110766, ← c110767, ← c110768] at e_1812_18
  have e_1812_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_19
  simp only [← c110769, ← c110770, k2, ← c110771] at e_1812_19
  have hr := e_1812_19
  simp only [e_1812_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2799)) := by
  have c110772 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110773 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110774 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110775 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110776 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110777 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_0
  simp only [← c110772, ← c110773, ← c110774] at e_1813_0
  have e_1813_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_1
  simp only [← c110775, ← c110776, k2, ← c110777] at e_1813_1
  have hr := e_1813_1
  simp only [e_1813_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2800)) := by
  have c110778 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110779 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110780 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110781 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110782 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110783 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_2
  simp only [← c110778, ← c110779, ← c110780] at e_1813_2
  have e_1813_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_3
  simp only [← c110781, ← c110782, k2, ← c110783] at e_1813_3
  have hr := e_1813_3
  simp only [e_1813_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2801)) := by
  have c110784 := (publicBatchWrapper89_copies3462 a h).1
  have c110785 := (publicBatchWrapper89_copies3462 a h).2.1
  have c110786 := (publicBatchWrapper89_copies3462 a h).2.2.1
  have c110787 := (publicBatchWrapper89_copies3462 a h).2.2.2.1
  have c110788 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.1
  have c110789 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_4
  simp only [← c110784, ← c110785, ← c110786] at e_1813_4
  have e_1813_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_5
  simp only [← c110787, ← c110788, k2, ← c110789] at e_1813_5
  have hr := e_1813_5
  simp only [e_1813_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2970)) := by
  have c110790 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.1
  have c110791 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.1
  have c110792 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.1
  have c110793 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.1
  have c110794 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110795 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_6
  simp only [← c110790, ← c110791, ← c110792] at e_1813_6
  have e_1813_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_7
  simp only [← c110793, ← c110794, k2, ← c110795] at e_1813_7
  have hr := e_1813_7
  simp only [e_1813_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2971)) := by
  have c110796 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110797 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110798 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110799 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110800 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110801 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_8
  simp only [← c110796, ← c110797, ← c110798] at e_1813_8
  have e_1813_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_9
  simp only [← c110799, ← c110800, k2, ← c110801] at e_1813_9
  have hr := e_1813_9
  simp only [e_1813_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2972)) := by
  have c110802 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110803 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110804 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110805 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110806 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110807 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_10
  simp only [← c110802, ← c110803, ← c110804] at e_1813_10
  have e_1813_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_11
  simp only [← c110805, ← c110806, k2, ← c110807] at e_1813_11
  have hr := e_1813_11
  simp only [e_1813_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2973)) := by
  have c110808 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110809 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110810 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110811 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110812 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110813 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_12
  simp only [← c110808, ← c110809, ← c110810] at e_1813_12
  have e_1813_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_13
  simp only [← c110811, ← c110812, k2, ← c110813] at e_1813_13
  have hr := e_1813_13
  simp only [e_1813_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2974)) := by
  have c110814 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110815 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110816 := (publicBatchWrapper89_copies3463 a h).1
  have c110817 := (publicBatchWrapper89_copies3463 a h).2.1
  have c110818 := (publicBatchWrapper89_copies3463 a h).2.2.1
  have c110819 := (publicBatchWrapper89_copies3463 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_14
  simp only [← c110814, ← c110815, ← c110816] at e_1813_14
  have e_1813_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_15
  simp only [← c110817, ← c110818, k2, ← c110819] at e_1813_15
  have hr := e_1813_15
  simp only [e_1813_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2975)) := by
  have c110820 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.1
  have c110821 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.1
  have c110822 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.1
  have c110823 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.1
  have c110824 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.1
  have c110825 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_16
  simp only [← c110820, ← c110821, ← c110822] at e_1813_16
  have e_1813_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_17
  simp only [← c110823, ← c110824, k2, ← c110825] at e_1813_17
  have hr := e_1813_17
  simp only [e_1813_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2976)) := by
  have c110826 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110827 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110828 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110829 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110830 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110831 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_18
  simp only [← c110826, ← c110827, ← c110828] at e_1813_18
  have e_1813_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_19
  simp only [← c110829, ← c110830, k2, ← c110831] at e_1813_19
  have hr := e_1813_19
  simp only [e_1813_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2977)) := by
  have c110832 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110833 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110834 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110835 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110836 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110837 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_0
  simp only [← c110832, ← c110833, ← c110834] at e_1814_0
  have e_1814_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_1
  simp only [← c110835, ← c110836, k2, ← c110837] at e_1814_1
  have hr := e_1814_1
  simp only [e_1814_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2978)) := by
  have c110838 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110839 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110840 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110841 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110842 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110843 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_2
  simp only [← c110838, ← c110839, ← c110840] at e_1814_2
  have e_1814_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_3
  simp only [← c110841, ← c110842, k2, ← c110843] at e_1814_3
  have hr := e_1814_3
  simp only [e_1814_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2979)) := by
  have c110844 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110845 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110846 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110847 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110848 := (publicBatchWrapper89_copies3464 a h).1
  have c110849 := (publicBatchWrapper89_copies3464 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_4
  simp only [← c110844, ← c110845, ← c110846] at e_1814_4
  have e_1814_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_5
  simp only [← c110847, ← c110848, k2, ← c110849] at e_1814_5
  have hr := e_1814_5
  simp only [e_1814_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2980)) := by
  have c110850 := (publicBatchWrapper89_copies3464 a h).2.2.1
  have c110851 := (publicBatchWrapper89_copies3464 a h).2.2.2.1
  have c110852 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.1
  have c110853 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.1
  have c110854 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.1
  have c110855 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_6
  simp only [← c110850, ← c110851, ← c110852] at e_1814_6
  have e_1814_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_7
  simp only [← c110853, ← c110854, k2, ← c110855] at e_1814_7
  have hr := e_1814_7
  simp only [e_1814_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2981)) := by
  have c110856 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.1
  have c110857 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.1
  have c110858 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110859 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110860 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110861 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_8
  simp only [← c110856, ← c110857, ← c110858] at e_1814_8
  have e_1814_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_9
  simp only [← c110859, ← c110860, k2, ← c110861] at e_1814_9
  have hr := e_1814_9
  simp only [e_1814_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2982)) := by
  have c110862 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110863 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110864 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110865 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110866 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110867 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_10
  simp only [← c110862, ← c110863, ← c110864] at e_1814_10
  have e_1814_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_11
  simp only [← c110865, ← c110866, k2, ← c110867] at e_1814_11
  have hr := e_1814_11
  simp only [e_1814_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2983)) := by
  have c110868 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110869 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110870 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110871 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110872 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110873 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_12
  simp only [← c110868, ← c110869, ← c110870] at e_1814_12
  have e_1814_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_13
  simp only [← c110871, ← c110872, k2, ← c110873] at e_1814_13
  have hr := e_1814_13
  simp only [e_1814_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2984)) := by
  have c110874 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110875 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110876 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110877 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110878 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110879 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_14
  simp only [← c110874, ← c110875, ← c110876] at e_1814_14
  have e_1814_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_15
  simp only [← c110877, ← c110878, k2, ← c110879] at e_1814_15
  have hr := e_1814_15
  simp only [e_1814_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2985)) := by
  have c110880 := (publicBatchWrapper89_copies3465 a h).1
  have c110881 := (publicBatchWrapper89_copies3465 a h).2.1
  have c110882 := (publicBatchWrapper89_copies3465 a h).2.2.1
  have c110883 := (publicBatchWrapper89_copies3465 a h).2.2.2.1
  have c110884 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.1
  have c110885 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_16
  simp only [← c110880, ← c110881, ← c110882] at e_1814_16
  have e_1814_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_17
  simp only [← c110883, ← c110884, k2, ← c110885] at e_1814_17
  have hr := e_1814_17
  simp only [e_1814_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2986)) := by
  have c110886 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.1
  have c110887 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.1
  have c110888 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.1
  have c110889 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.1
  have c110890 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110891 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_18
  simp only [← c110886, ← c110887, ← c110888] at e_1814_18
  have e_1814_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_19
  simp only [← c110889, ← c110890, k2, ← c110891] at e_1814_19
  have hr := e_1814_19
  simp only [e_1814_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2987)) := by
  have c110892 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110893 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110894 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110895 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110896 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110897 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_0
  simp only [← c110892, ← c110893, ← c110894] at e_1815_0
  have e_1815_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_1
  simp only [← c110895, ← c110896, k2, ← c110897] at e_1815_1
  have hr := e_1815_1
  simp only [e_1815_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2988)) := by
  have c110898 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110899 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110900 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110901 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110902 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110903 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_2
  simp only [← c110898, ← c110899, ← c110900] at e_1815_2
  have e_1815_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_3
  simp only [← c110901, ← c110902, k2, ← c110903] at e_1815_3
  have hr := e_1815_3
  simp only [e_1815_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2989)) := by
  have c110904 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110905 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110906 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110907 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110908 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110909 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_4
  simp only [← c110904, ← c110905, ← c110906] at e_1815_4
  have e_1815_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_5
  simp only [← c110907, ← c110908, k2, ← c110909] at e_1815_5
  have hr := e_1815_5
  simp only [e_1815_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2990)) := by
  have c110910 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110911 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110912 := (publicBatchWrapper89_copies3466 a h).1
  have c110913 := (publicBatchWrapper89_copies3466 a h).2.1
  have c110914 := (publicBatchWrapper89_copies3466 a h).2.2.1
  have c110915 := (publicBatchWrapper89_copies3466 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_6
  simp only [← c110910, ← c110911, ← c110912] at e_1815_6
  have e_1815_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_7
  simp only [← c110913, ← c110914, k2, ← c110915] at e_1815_7
  have hr := e_1815_7
  simp only [e_1815_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2991)) := by
  have c110916 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.1
  have c110917 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.1
  have c110918 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.1
  have c110919 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.1
  have c110920 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.1
  have c110921 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_8
  simp only [← c110916, ← c110917, ← c110918] at e_1815_8
  have e_1815_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_9
  simp only [← c110919, ← c110920, k2, ← c110921] at e_1815_9
  have hr := e_1815_9
  simp only [e_1815_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2992)) := by
  have c110922 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110923 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110924 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110925 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110926 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110927 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_10
  simp only [← c110922, ← c110923, ← c110924] at e_1815_10
  have e_1815_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_11
  simp only [← c110925, ← c110926, k2, ← c110927] at e_1815_11
  have hr := e_1815_11
  simp only [e_1815_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2993)) := by
  have c110928 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110929 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110930 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110931 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110932 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110933 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_12
  simp only [← c110928, ← c110929, ← c110930] at e_1815_12
  have e_1815_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_13
  simp only [← c110931, ← c110932, k2, ← c110933] at e_1815_13
  have hr := e_1815_13
  simp only [e_1815_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2994)) := by
  have c110934 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110935 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110936 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110937 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110938 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110939 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_14
  simp only [← c110934, ← c110935, ← c110936] at e_1815_14
  have e_1815_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_15
  simp only [← c110937, ← c110938, k2, ← c110939] at e_1815_15
  have hr := e_1815_15
  simp only [e_1815_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2995)) := by
  have c110940 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110941 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110942 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110943 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110944 := (publicBatchWrapper89_copies3467 a h).1
  have c110945 := (publicBatchWrapper89_copies3467 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_16
  simp only [← c110940, ← c110941, ← c110942] at e_1815_16
  have e_1815_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_17
  simp only [← c110943, ← c110944, k2, ← c110945] at e_1815_17
  have hr := e_1815_17
  simp only [e_1815_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2996)) := by
  have c110946 := (publicBatchWrapper89_copies3467 a h).2.2.1
  have c110947 := (publicBatchWrapper89_copies3467 a h).2.2.2.1
  have c110948 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.1
  have c110949 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.1
  have c110950 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.1
  have c110951 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_18
  simp only [← c110946, ← c110947, ← c110948] at e_1815_18
  have e_1815_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_19
  simp only [← c110949, ← c110950, k2, ← c110951] at e_1815_19
  have hr := e_1815_19
  simp only [e_1815_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2997)) := by
  have c110952 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.1
  have c110953 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.1
  have c110954 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110955 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110956 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110957 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_0
  simp only [← c110952, ← c110953, ← c110954] at e_1816_0
  have e_1816_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_1
  simp only [← c110955, ← c110956, k2, ← c110957] at e_1816_1
  have hr := e_1816_1
  simp only [e_1816_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2998)) := by
  have c110958 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110959 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110960 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110961 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110962 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110963 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_2
  simp only [← c110958, ← c110959, ← c110960] at e_1816_2
  have e_1816_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_3
  simp only [← c110961, ← c110962, k2, ← c110963] at e_1816_3
  have hr := e_1816_3
  simp only [e_1816_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2999)) := by
  have c110964 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110965 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110966 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110967 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110968 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110969 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_4
  simp only [← c110964, ← c110965, ← c110966] at e_1816_4
  have e_1816_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_5
  simp only [← c110967, ← c110968, k2, ← c110969] at e_1816_5
  have hr := e_1816_5
  simp only [e_1816_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 3000)) := by
  have c110970 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110971 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110972 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110973 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110974 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110975 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_6
  simp only [← c110970, ← c110971, ← c110972] at e_1816_6
  have e_1816_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_7
  simp only [← c110973, ← c110974, k2, ← c110975] at e_1816_7
  have hr := e_1816_7
  simp only [e_1816_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 3001)) := by
  have c110976 := (publicBatchWrapper89_copies3468 a h).1
  have c110977 := (publicBatchWrapper89_copies3468 a h).2.1
  have c110978 := (publicBatchWrapper89_copies3468 a h).2.2.1
  have c110979 := (publicBatchWrapper89_copies3468 a h).2.2.2.1
  have c110980 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.1
  have c110981 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_8
  simp only [← c110976, ← c110977, ← c110978] at e_1816_8
  have e_1816_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_9
  simp only [← c110979, ← c110980, k2, ← c110981] at e_1816_9
  have hr := e_1816_9
  simp only [e_1816_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3170)) := by
  have c110982 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.1
  have c110983 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.1
  have c110984 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.1
  have c110985 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.1
  have c110986 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110987 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_10
  simp only [← c110982, ← c110983, ← c110984] at e_1816_10
  have e_1816_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_11
  simp only [← c110985, ← c110986, k2, ← c110987] at e_1816_11
  have hr := e_1816_11
  simp only [e_1816_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3171)) := by
  have c110988 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110989 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110990 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110991 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110992 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110993 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_12
  simp only [← c110988, ← c110989, ← c110990] at e_1816_12
  have e_1816_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_13
  simp only [← c110991, ← c110992, k2, ← c110993] at e_1816_13
  have hr := e_1816_13
  simp only [e_1816_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3172)) := by
  have c110994 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110995 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110996 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110997 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110998 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110999 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_14
  simp only [← c110994, ← c110995, ← c110996] at e_1816_14
  have e_1816_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_15
  simp only [← c110997, ← c110998, k2, ← c110999] at e_1816_15
  have hr := e_1816_15
  simp only [e_1816_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3173)) := by
  have c111000 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111001 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111002 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111003 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111004 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111005 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_16
  simp only [← c111000, ← c111001, ← c111002] at e_1816_16
  have e_1816_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_17
  simp only [← c111003, ← c111004, k2, ← c111005] at e_1816_17
  have hr := e_1816_17
  simp only [e_1816_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3174)) := by
  have c111006 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111007 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111008 := (publicBatchWrapper89_copies3469 a h).1
  have c111009 := (publicBatchWrapper89_copies3469 a h).2.1
  have c111010 := (publicBatchWrapper89_copies3469 a h).2.2.1
  have c111011 := (publicBatchWrapper89_copies3469 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_18
  simp only [← c111006, ← c111007, ← c111008] at e_1816_18
  have e_1816_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_19
  simp only [← c111009, ← c111010, k2, ← c111011] at e_1816_19
  have hr := e_1816_19
  simp only [e_1816_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3175)) := by
  have c111012 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.1
  have c111013 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.1
  have c111014 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.1
  have c111015 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.1
  have c111016 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.1
  have c111017 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_0
  simp only [← c111012, ← c111013, ← c111014] at e_1817_0
  have e_1817_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_1
  simp only [← c111015, ← c111016, k2, ← c111017] at e_1817_1
  have hr := e_1817_1
  simp only [e_1817_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3176)) := by
  have c111018 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111019 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111020 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111021 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111022 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111023 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_2
  simp only [← c111018, ← c111019, ← c111020] at e_1817_2
  have e_1817_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_3
  simp only [← c111021, ← c111022, k2, ← c111023] at e_1817_3
  have hr := e_1817_3
  simp only [e_1817_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3177)) := by
  have c111024 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111025 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111026 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111027 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111028 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111029 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_4
  simp only [← c111024, ← c111025, ← c111026] at e_1817_4
  have e_1817_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_5
  simp only [← c111027, ← c111028, k2, ← c111029] at e_1817_5
  have hr := e_1817_5
  simp only [e_1817_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3178)) := by
  have c111030 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111031 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111032 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111033 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111034 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111035 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_6
  simp only [← c111030, ← c111031, ← c111032] at e_1817_6
  have e_1817_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_7
  simp only [← c111033, ← c111034, k2, ← c111035] at e_1817_7
  have hr := e_1817_7
  simp only [e_1817_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3179)) := by
  have c111036 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111037 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111038 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111039 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111040 := (publicBatchWrapper89_copies3470 a h).1
  have c111041 := (publicBatchWrapper89_copies3470 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_8
  simp only [← c111036, ← c111037, ← c111038] at e_1817_8
  have e_1817_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_9
  simp only [← c111039, ← c111040, k2, ← c111041] at e_1817_9
  have hr := e_1817_9
  simp only [e_1817_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3180)) := by
  have c111042 := (publicBatchWrapper89_copies3470 a h).2.2.1
  have c111043 := (publicBatchWrapper89_copies3470 a h).2.2.2.1
  have c111044 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.1
  have c111045 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.1
  have c111046 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.1
  have c111047 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_10
  simp only [← c111042, ← c111043, ← c111044] at e_1817_10
  have e_1817_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_11
  simp only [← c111045, ← c111046, k2, ← c111047] at e_1817_11
  have hr := e_1817_11
  simp only [e_1817_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3181)) := by
  have c111048 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.1
  have c111049 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.1
  have c111050 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111051 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111052 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111053 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_12
  simp only [← c111048, ← c111049, ← c111050] at e_1817_12
  have e_1817_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_13
  simp only [← c111051, ← c111052, k2, ← c111053] at e_1817_13
  have hr := e_1817_13
  simp only [e_1817_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3182)) := by
  have c111054 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111055 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111056 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111057 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111058 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111059 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_14
  simp only [← c111054, ← c111055, ← c111056] at e_1817_14
  have e_1817_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_15
  simp only [← c111057, ← c111058, k2, ← c111059] at e_1817_15
  have hr := e_1817_15
  simp only [e_1817_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3183)) := by
  have c111060 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111061 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111062 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111063 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111064 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111065 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_16
  simp only [← c111060, ← c111061, ← c111062] at e_1817_16
  have e_1817_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_17
  simp only [← c111063, ← c111064, k2, ← c111065] at e_1817_17
  have hr := e_1817_17
  simp only [e_1817_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3184)) := by
  have c111066 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111067 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111068 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111069 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111070 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111071 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_18
  simp only [← c111066, ← c111067, ← c111068] at e_1817_18
  have e_1817_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_19
  simp only [← c111069, ← c111070, k2, ← c111071] at e_1817_19
  have hr := e_1817_19
  simp only [e_1817_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3185)) := by
  have c111072 := (publicBatchWrapper89_copies3471 a h).1
  have c111073 := (publicBatchWrapper89_copies3471 a h).2.1
  have c111074 := (publicBatchWrapper89_copies3471 a h).2.2.1
  have c111075 := (publicBatchWrapper89_copies3471 a h).2.2.2.1
  have c111076 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.1
  have c111077 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_0
  simp only [← c111072, ← c111073, ← c111074] at e_1818_0
  have e_1818_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_1
  simp only [← c111075, ← c111076, k2, ← c111077] at e_1818_1
  have hr := e_1818_1
  simp only [e_1818_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3186)) := by
  have c111078 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.1
  have c111079 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.1
  have c111080 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.1
  have c111081 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.1
  have c111082 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111083 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_2
  simp only [← c111078, ← c111079, ← c111080] at e_1818_2
  have e_1818_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_3
  simp only [← c111081, ← c111082, k2, ← c111083] at e_1818_3
  have hr := e_1818_3
  simp only [e_1818_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3187)) := by
  have c111084 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111085 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111086 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111087 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111088 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111089 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_4
  simp only [← c111084, ← c111085, ← c111086] at e_1818_4
  have e_1818_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_5
  simp only [← c111087, ← c111088, k2, ← c111089] at e_1818_5
  have hr := e_1818_5
  simp only [e_1818_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3188)) := by
  have c111090 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111091 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111092 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111093 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111094 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111095 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_6
  simp only [← c111090, ← c111091, ← c111092] at e_1818_6
  have e_1818_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_7
  simp only [← c111093, ← c111094, k2, ← c111095] at e_1818_7
  have hr := e_1818_7
  simp only [e_1818_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3189)) := by
  have c111096 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111097 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111098 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111099 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111100 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111101 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_8
  simp only [← c111096, ← c111097, ← c111098] at e_1818_8
  have e_1818_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_9
  simp only [← c111099, ← c111100, k2, ← c111101] at e_1818_9
  have hr := e_1818_9
  simp only [e_1818_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3190)) := by
  have c111102 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111103 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111104 := (publicBatchWrapper89_copies3472 a h).1
  have c111105 := (publicBatchWrapper89_copies3472 a h).2.1
  have c111106 := (publicBatchWrapper89_copies3472 a h).2.2.1
  have c111107 := (publicBatchWrapper89_copies3472 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_10
  simp only [← c111102, ← c111103, ← c111104] at e_1818_10
  have e_1818_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_11
  simp only [← c111105, ← c111106, k2, ← c111107] at e_1818_11
  have hr := e_1818_11
  simp only [e_1818_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3191)) := by
  have c111108 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.1
  have c111109 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.1
  have c111110 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.1
  have c111111 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.1
  have c111112 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.1
  have c111113 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_12
  simp only [← c111108, ← c111109, ← c111110] at e_1818_12
  have e_1818_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_13
  simp only [← c111111, ← c111112, k2, ← c111113] at e_1818_13
  have hr := e_1818_13
  simp only [e_1818_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3192)) := by
  have c111114 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111115 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111116 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111117 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111118 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111119 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_14
  simp only [← c111114, ← c111115, ← c111116] at e_1818_14
  have e_1818_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_15
  simp only [← c111117, ← c111118, k2, ← c111119] at e_1818_15
  have hr := e_1818_15
  simp only [e_1818_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3193)) := by
  have c111120 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111121 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111122 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111123 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111124 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111125 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_16
  simp only [← c111120, ← c111121, ← c111122] at e_1818_16
  have e_1818_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_17
  simp only [← c111123, ← c111124, k2, ← c111125] at e_1818_17
  have hr := e_1818_17
  simp only [e_1818_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3194)) := by
  have c111126 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111127 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111128 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111129 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111130 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111131 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_18
  simp only [← c111126, ← c111127, ← c111128] at e_1818_18
  have e_1818_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_19
  simp only [← c111129, ← c111130, k2, ← c111131] at e_1818_19
  have hr := e_1818_19
  simp only [e_1818_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3195)) := by
  have c111132 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111133 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111134 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111135 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111136 := (publicBatchWrapper89_copies3473 a h).1
  have c111137 := (publicBatchWrapper89_copies3473 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_0
  simp only [← c111132, ← c111133, ← c111134] at e_1819_0
  have e_1819_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_1
  simp only [← c111135, ← c111136, k2, ← c111137] at e_1819_1
  have hr := e_1819_1
  simp only [e_1819_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3196)) := by
  have c111138 := (publicBatchWrapper89_copies3473 a h).2.2.1
  have c111139 := (publicBatchWrapper89_copies3473 a h).2.2.2.1
  have c111140 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.1
  have c111141 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.1
  have c111142 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.1
  have c111143 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_2
  simp only [← c111138, ← c111139, ← c111140] at e_1819_2
  have e_1819_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_3
  simp only [← c111141, ← c111142, k2, ← c111143] at e_1819_3
  have hr := e_1819_3
  simp only [e_1819_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3197)) := by
  have c111144 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.1
  have c111145 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.1
  have c111146 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111147 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111148 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111149 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_4
  simp only [← c111144, ← c111145, ← c111146] at e_1819_4
  have e_1819_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_5
  simp only [← c111147, ← c111148, k2, ← c111149] at e_1819_5
  have hr := e_1819_5
  simp only [e_1819_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3198)) := by
  have c111150 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111151 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111152 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111153 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111154 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111155 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_6
  simp only [← c111150, ← c111151, ← c111152] at e_1819_6
  have e_1819_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_7
  simp only [← c111153, ← c111154, k2, ← c111155] at e_1819_7
  have hr := e_1819_7
  simp only [e_1819_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3199)) := by
  have c111156 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111157 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111158 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111159 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111160 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111161 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_8
  simp only [← c111156, ← c111157, ← c111158] at e_1819_8
  have e_1819_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_9
  simp only [← c111159, ← c111160, k2, ← c111161] at e_1819_9
  have hr := e_1819_9
  simp only [e_1819_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3200)) := by
  have c111162 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111163 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111164 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111165 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111166 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111167 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_10
  simp only [← c111162, ← c111163, ← c111164] at e_1819_10
  have e_1819_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_11
  simp only [← c111165, ← c111166, k2, ← c111167] at e_1819_11
  have hr := e_1819_11
  simp only [e_1819_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3201)) := by
  have c111168 := (publicBatchWrapper89_copies3474 a h).1
  have c111169 := (publicBatchWrapper89_copies3474 a h).2.1
  have c111170 := (publicBatchWrapper89_copies3474 a h).2.2.1
  have c111171 := (publicBatchWrapper89_copies3474 a h).2.2.2.1
  have c111172 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.1
  have c111173 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_12
  simp only [← c111168, ← c111169, ← c111170] at e_1819_12
  have e_1819_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_13
  simp only [← c111171, ← c111172, k2, ← c111173] at e_1819_13
  have hr := e_1819_13
  simp only [e_1819_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3370)) := by
  have c111174 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.1
  have c111175 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.1
  have c111176 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.1
  have c111177 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.1
  have c111178 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111179 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_14
  simp only [← c111174, ← c111175, ← c111176] at e_1819_14
  have e_1819_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_15
  simp only [← c111177, ← c111178, k2, ← c111179] at e_1819_15
  have hr := e_1819_15
  simp only [e_1819_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3371)) := by
  have c111180 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111181 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111182 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111183 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111184 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111185 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_16
  simp only [← c111180, ← c111181, ← c111182] at e_1819_16
  have e_1819_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_17
  simp only [← c111183, ← c111184, k2, ← c111185] at e_1819_17
  have hr := e_1819_17
  simp only [e_1819_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3372)) := by
  have c111186 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111187 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111188 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111189 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111190 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111191 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_18
  simp only [← c111186, ← c111187, ← c111188] at e_1819_18
  have e_1819_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_19
  simp only [← c111189, ← c111190, k2, ← c111191] at e_1819_19
  have hr := e_1819_19
  simp only [e_1819_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3373)) := by
  have c111192 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111193 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111194 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111195 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111196 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111197 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_0
  simp only [← c111192, ← c111193, ← c111194] at e_1820_0
  have e_1820_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_1
  simp only [← c111195, ← c111196, k2, ← c111197] at e_1820_1
  have hr := e_1820_1
  simp only [e_1820_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3374)) := by
  have c111198 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111199 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111200 := (publicBatchWrapper89_copies3475 a h).1
  have c111201 := (publicBatchWrapper89_copies3475 a h).2.1
  have c111202 := (publicBatchWrapper89_copies3475 a h).2.2.1
  have c111203 := (publicBatchWrapper89_copies3475 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_2
  simp only [← c111198, ← c111199, ← c111200] at e_1820_2
  have e_1820_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_3
  simp only [← c111201, ← c111202, k2, ← c111203] at e_1820_3
  have hr := e_1820_3
  simp only [e_1820_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3375)) := by
  have c111204 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.1
  have c111205 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.1
  have c111206 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.1
  have c111207 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.1
  have c111208 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.1
  have c111209 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_4
  simp only [← c111204, ← c111205, ← c111206] at e_1820_4
  have e_1820_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_5
  simp only [← c111207, ← c111208, k2, ← c111209] at e_1820_5
  have hr := e_1820_5
  simp only [e_1820_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3376)) := by
  have c111210 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111211 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111212 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111213 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111214 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111215 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_6
  simp only [← c111210, ← c111211, ← c111212] at e_1820_6
  have e_1820_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_7
  simp only [← c111213, ← c111214, k2, ← c111215] at e_1820_7
  have hr := e_1820_7
  simp only [e_1820_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3377)) := by
  have c111216 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111217 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111218 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111219 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111220 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111221 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_8
  simp only [← c111216, ← c111217, ← c111218] at e_1820_8
  have e_1820_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_9
  simp only [← c111219, ← c111220, k2, ← c111221] at e_1820_9
  have hr := e_1820_9
  simp only [e_1820_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3378)) := by
  have c111222 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111223 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111224 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111225 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111226 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111227 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_10
  simp only [← c111222, ← c111223, ← c111224] at e_1820_10
  have e_1820_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_11
  simp only [← c111225, ← c111226, k2, ← c111227] at e_1820_11
  have hr := e_1820_11
  simp only [e_1820_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3379)) := by
  have c111228 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111229 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111230 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111231 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111232 := (publicBatchWrapper89_copies3476 a h).1
  have c111233 := (publicBatchWrapper89_copies3476 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_12
  simp only [← c111228, ← c111229, ← c111230] at e_1820_12
  have e_1820_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_13
  simp only [← c111231, ← c111232, k2, ← c111233] at e_1820_13
  have hr := e_1820_13
  simp only [e_1820_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3380)) := by
  have c111234 := (publicBatchWrapper89_copies3476 a h).2.2.1
  have c111235 := (publicBatchWrapper89_copies3476 a h).2.2.2.1
  have c111236 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.1
  have c111237 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.1
  have c111238 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.1
  have c111239 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_14
  simp only [← c111234, ← c111235, ← c111236] at e_1820_14
  have e_1820_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_15
  simp only [← c111237, ← c111238, k2, ← c111239] at e_1820_15
  have hr := e_1820_15
  simp only [e_1820_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3381)) := by
  have c111240 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.1
  have c111241 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.1
  have c111242 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111243 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111244 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111245 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_16
  simp only [← c111240, ← c111241, ← c111242] at e_1820_16
  have e_1820_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_17
  simp only [← c111243, ← c111244, k2, ← c111245] at e_1820_17
  have hr := e_1820_17
  simp only [e_1820_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3382)) := by
  have c111246 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111247 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111248 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111249 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111250 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111251 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_18
  simp only [← c111246, ← c111247, ← c111248] at e_1820_18
  have e_1820_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_19
  simp only [← c111249, ← c111250, k2, ← c111251] at e_1820_19
  have hr := e_1820_19
  simp only [e_1820_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3383)) := by
  have c111252 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111253 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111254 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111255 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111256 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111257 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_0
  simp only [← c111252, ← c111253, ← c111254] at e_1821_0
  have e_1821_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_1
  simp only [← c111255, ← c111256, k2, ← c111257] at e_1821_1
  have hr := e_1821_1
  simp only [e_1821_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3384)) := by
  have c111258 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111259 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111260 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111261 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111262 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111263 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_2
  simp only [← c111258, ← c111259, ← c111260] at e_1821_2
  have e_1821_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_3
  simp only [← c111261, ← c111262, k2, ← c111263] at e_1821_3
  have hr := e_1821_3
  simp only [e_1821_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3385)) := by
  have c111264 := (publicBatchWrapper89_copies3477 a h).1
  have c111265 := (publicBatchWrapper89_copies3477 a h).2.1
  have c111266 := (publicBatchWrapper89_copies3477 a h).2.2.1
  have c111267 := (publicBatchWrapper89_copies3477 a h).2.2.2.1
  have c111268 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.1
  have c111269 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_4
  simp only [← c111264, ← c111265, ← c111266] at e_1821_4
  have e_1821_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_5
  simp only [← c111267, ← c111268, k2, ← c111269] at e_1821_5
  have hr := e_1821_5
  simp only [e_1821_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3386)) := by
  have c111270 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.1
  have c111271 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.1
  have c111272 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.1
  have c111273 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.1
  have c111274 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111275 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_6
  simp only [← c111270, ← c111271, ← c111272] at e_1821_6
  have e_1821_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_7
  simp only [← c111273, ← c111274, k2, ← c111275] at e_1821_7
  have hr := e_1821_7
  simp only [e_1821_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3387)) := by
  have c111276 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111277 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111278 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111279 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111280 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111281 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_8
  simp only [← c111276, ← c111277, ← c111278] at e_1821_8
  have e_1821_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_9
  simp only [← c111279, ← c111280, k2, ← c111281] at e_1821_9
  have hr := e_1821_9
  simp only [e_1821_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3388)) := by
  have c111282 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111283 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111284 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111285 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111286 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111287 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_10
  simp only [← c111282, ← c111283, ← c111284] at e_1821_10
  have e_1821_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_11
  simp only [← c111285, ← c111286, k2, ← c111287] at e_1821_11
  have hr := e_1821_11
  simp only [e_1821_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3389)) := by
  have c111288 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111289 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111290 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111291 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111292 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111293 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_12
  simp only [← c111288, ← c111289, ← c111290] at e_1821_12
  have e_1821_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_13
  simp only [← c111291, ← c111292, k2, ← c111293] at e_1821_13
  have hr := e_1821_13
  simp only [e_1821_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3390)) := by
  have c111294 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111295 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111296 := (publicBatchWrapper89_copies3478 a h).1
  have c111297 := (publicBatchWrapper89_copies3478 a h).2.1
  have c111298 := (publicBatchWrapper89_copies3478 a h).2.2.1
  have c111299 := (publicBatchWrapper89_copies3478 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_14
  simp only [← c111294, ← c111295, ← c111296] at e_1821_14
  have e_1821_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_15
  simp only [← c111297, ← c111298, k2, ← c111299] at e_1821_15
  have hr := e_1821_15
  simp only [e_1821_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3391)) := by
  have c111300 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.1
  have c111301 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.1
  have c111302 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.1
  have c111303 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.1
  have c111304 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.1
  have c111305 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_16
  simp only [← c111300, ← c111301, ← c111302] at e_1821_16
  have e_1821_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_17
  simp only [← c111303, ← c111304, k2, ← c111305] at e_1821_17
  have hr := e_1821_17
  simp only [e_1821_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3392)) := by
  have c111306 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111307 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111308 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111309 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111310 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111311 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_18
  simp only [← c111306, ← c111307, ← c111308] at e_1821_18
  have e_1821_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_19
  simp only [← c111309, ← c111310, k2, ← c111311] at e_1821_19
  have hr := e_1821_19
  simp only [e_1821_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3393)) := by
  have c111312 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111313 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111314 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111315 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111316 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111317 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_0
  simp only [← c111312, ← c111313, ← c111314] at e_1822_0
  have e_1822_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_1
  simp only [← c111315, ← c111316, k2, ← c111317] at e_1822_1
  have hr := e_1822_1
  simp only [e_1822_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3394)) := by
  have c111318 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111319 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111320 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111321 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111322 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111323 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_2
  simp only [← c111318, ← c111319, ← c111320] at e_1822_2
  have e_1822_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_3
  simp only [← c111321, ← c111322, k2, ← c111323] at e_1822_3
  have hr := e_1822_3
  simp only [e_1822_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3395)) := by
  have c111324 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111325 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111326 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111327 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111328 := (publicBatchWrapper89_copies3479 a h).1
  have c111329 := (publicBatchWrapper89_copies3479 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_4
  simp only [← c111324, ← c111325, ← c111326] at e_1822_4
  have e_1822_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_5
  simp only [← c111327, ← c111328, k2, ← c111329] at e_1822_5
  have hr := e_1822_5
  simp only [e_1822_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3396)) := by
  have c111330 := (publicBatchWrapper89_copies3479 a h).2.2.1
  have c111331 := (publicBatchWrapper89_copies3479 a h).2.2.2.1
  have c111332 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.1
  have c111333 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.1
  have c111334 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.1
  have c111335 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_6
  simp only [← c111330, ← c111331, ← c111332] at e_1822_6
  have e_1822_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_7
  simp only [← c111333, ← c111334, k2, ← c111335] at e_1822_7
  have hr := e_1822_7
  simp only [e_1822_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3397)) := by
  have c111336 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.1
  have c111337 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.1
  have c111338 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111339 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111340 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111341 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_8
  simp only [← c111336, ← c111337, ← c111338] at e_1822_8
  have e_1822_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_9
  simp only [← c111339, ← c111340, k2, ← c111341] at e_1822_9
  have hr := e_1822_9
  simp only [e_1822_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3398)) := by
  have c111342 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111343 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111344 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111345 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111346 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111347 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_10
  simp only [← c111342, ← c111343, ← c111344] at e_1822_10
  have e_1822_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_11
  simp only [← c111345, ← c111346, k2, ← c111347] at e_1822_11
  have hr := e_1822_11
  simp only [e_1822_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3399)) := by
  have c111348 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111349 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111350 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111351 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111352 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111353 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_12
  simp only [← c111348, ← c111349, ← c111350] at e_1822_12
  have e_1822_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_13
  simp only [← c111351, ← c111352, k2, ← c111353] at e_1822_13
  have hr := e_1822_13
  simp only [e_1822_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3400)) := by
  have c111354 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111355 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111356 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111357 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111358 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111359 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_14
  simp only [← c111354, ← c111355, ← c111356] at e_1822_14
  have e_1822_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_15
  simp only [← c111357, ← c111358, k2, ← c111359] at e_1822_15
  have hr := e_1822_15
  simp only [e_1822_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3401)) := by
  have c111360 := (publicBatchWrapper89_copies3480 a h).1
  have c111361 := (publicBatchWrapper89_copies3480 a h).2.1
  have c111362 := (publicBatchWrapper89_copies3480 a h).2.2.1
  have c111363 := (publicBatchWrapper89_copies3480 a h).2.2.2.1
  have c111364 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.1
  have c111365 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_16
  simp only [← c111360, ← c111361, ← c111362] at e_1822_16
  have e_1822_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_17
  simp only [← c111363, ← c111364, k2, ← c111365] at e_1822_17
  have hr := e_1822_17
  simp only [e_1822_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3570)) := by
  have c111366 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.1
  have c111367 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.1
  have c111368 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.1
  have c111369 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.1
  have c111370 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111371 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_18
  simp only [← c111366, ← c111367, ← c111368] at e_1822_18
  have e_1822_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_19
  simp only [← c111369, ← c111370, k2, ← c111371] at e_1822_19
  have hr := e_1822_19
  simp only [e_1822_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3571)) := by
  have c111372 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111373 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111374 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111375 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111376 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111377 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_0
  simp only [← c111372, ← c111373, ← c111374] at e_1823_0
  have e_1823_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_1
  simp only [← c111375, ← c111376, k2, ← c111377] at e_1823_1
  have hr := e_1823_1
  simp only [e_1823_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3572)) := by
  have c111378 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111379 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111380 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111381 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111382 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111383 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_2
  simp only [← c111378, ← c111379, ← c111380] at e_1823_2
  have e_1823_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_3
  simp only [← c111381, ← c111382, k2, ← c111383] at e_1823_3
  have hr := e_1823_3
  simp only [e_1823_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3573)) := by
  have c111384 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111385 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111386 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111387 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111388 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111389 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_4
  simp only [← c111384, ← c111385, ← c111386] at e_1823_4
  have e_1823_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_5
  simp only [← c111387, ← c111388, k2, ← c111389] at e_1823_5
  have hr := e_1823_5
  simp only [e_1823_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3574)) := by
  have c111390 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111391 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111392 := (publicBatchWrapper89_copies3481 a h).1
  have c111393 := (publicBatchWrapper89_copies3481 a h).2.1
  have c111394 := (publicBatchWrapper89_copies3481 a h).2.2.1
  have c111395 := (publicBatchWrapper89_copies3481 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_6
  simp only [← c111390, ← c111391, ← c111392] at e_1823_6
  have e_1823_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_7
  simp only [← c111393, ← c111394, k2, ← c111395] at e_1823_7
  have hr := e_1823_7
  simp only [e_1823_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3575)) := by
  have c111396 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.1
  have c111397 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.1
  have c111398 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.1
  have c111399 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.1
  have c111400 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.1
  have c111401 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_8
  simp only [← c111396, ← c111397, ← c111398] at e_1823_8
  have e_1823_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_9
  simp only [← c111399, ← c111400, k2, ← c111401] at e_1823_9
  have hr := e_1823_9
  simp only [e_1823_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3576)) := by
  have c111402 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111403 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111404 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111405 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111406 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111407 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_10
  simp only [← c111402, ← c111403, ← c111404] at e_1823_10
  have e_1823_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_11
  simp only [← c111405, ← c111406, k2, ← c111407] at e_1823_11
  have hr := e_1823_11
  simp only [e_1823_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3577)) := by
  have c111408 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111409 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111410 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111411 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111412 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111413 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_12
  simp only [← c111408, ← c111409, ← c111410] at e_1823_12
  have e_1823_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_13
  simp only [← c111411, ← c111412, k2, ← c111413] at e_1823_13
  have hr := e_1823_13
  simp only [e_1823_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3578)) := by
  have c111414 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111415 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111416 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111417 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111418 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111419 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_14
  simp only [← c111414, ← c111415, ← c111416] at e_1823_14
  have e_1823_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_15
  simp only [← c111417, ← c111418, k2, ← c111419] at e_1823_15
  have hr := e_1823_15
  simp only [e_1823_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3579)) := by
  have c111420 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111421 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111422 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111423 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111424 := (publicBatchWrapper89_copies3482 a h).1
  have c111425 := (publicBatchWrapper89_copies3482 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_16
  simp only [← c111420, ← c111421, ← c111422] at e_1823_16
  have e_1823_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_17
  simp only [← c111423, ← c111424, k2, ← c111425] at e_1823_17
  have hr := e_1823_17
  simp only [e_1823_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3580)) := by
  have c111426 := (publicBatchWrapper89_copies3482 a h).2.2.1
  have c111427 := (publicBatchWrapper89_copies3482 a h).2.2.2.1
  have c111428 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.1
  have c111429 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.1
  have c111430 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.1
  have c111431 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_18
  simp only [← c111426, ← c111427, ← c111428] at e_1823_18
  have e_1823_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_19
  simp only [← c111429, ← c111430, k2, ← c111431] at e_1823_19
  have hr := e_1823_19
  simp only [e_1823_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3581)) := by
  have c111432 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.1
  have c111433 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.1
  have c111434 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111435 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111436 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111437 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_0
  simp only [← c111432, ← c111433, ← c111434] at e_1824_0
  have e_1824_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_1
  simp only [← c111435, ← c111436, k2, ← c111437] at e_1824_1
  have hr := e_1824_1
  simp only [e_1824_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3582)) := by
  have c111438 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111439 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111440 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111441 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111442 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111443 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_2
  simp only [← c111438, ← c111439, ← c111440] at e_1824_2
  have e_1824_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_3
  simp only [← c111441, ← c111442, k2, ← c111443] at e_1824_3
  have hr := e_1824_3
  simp only [e_1824_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3583)) := by
  have c111444 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111445 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111446 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111447 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111448 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111449 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_4
  simp only [← c111444, ← c111445, ← c111446] at e_1824_4
  have e_1824_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_5
  simp only [← c111447, ← c111448, k2, ← c111449] at e_1824_5
  have hr := e_1824_5
  simp only [e_1824_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3584)) := by
  have c111450 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111451 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111452 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111453 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111454 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111455 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_6
  simp only [← c111450, ← c111451, ← c111452] at e_1824_6
  have e_1824_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_7
  simp only [← c111453, ← c111454, k2, ← c111455] at e_1824_7
  have hr := e_1824_7
  simp only [e_1824_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3585)) := by
  have c111456 := (publicBatchWrapper89_copies3483 a h).1
  have c111457 := (publicBatchWrapper89_copies3483 a h).2.1
  have c111458 := (publicBatchWrapper89_copies3483 a h).2.2.1
  have c111459 := (publicBatchWrapper89_copies3483 a h).2.2.2.1
  have c111460 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.1
  have c111461 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_8
  simp only [← c111456, ← c111457, ← c111458] at e_1824_8
  have e_1824_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_9
  simp only [← c111459, ← c111460, k2, ← c111461] at e_1824_9
  have hr := e_1824_9
  simp only [e_1824_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3586)) := by
  have c111462 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.1
  have c111463 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.1
  have c111464 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.1
  have c111465 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.1
  have c111466 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111467 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_10
  simp only [← c111462, ← c111463, ← c111464] at e_1824_10
  have e_1824_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_11
  simp only [← c111465, ← c111466, k2, ← c111467] at e_1824_11
  have hr := e_1824_11
  simp only [e_1824_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3587)) := by
  have c111468 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111469 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111470 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111471 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111472 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111473 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_12
  simp only [← c111468, ← c111469, ← c111470] at e_1824_12
  have e_1824_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_13
  simp only [← c111471, ← c111472, k2, ← c111473] at e_1824_13
  have hr := e_1824_13
  simp only [e_1824_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3588)) := by
  have c111474 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111475 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111476 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111477 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111478 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111479 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_14
  simp only [← c111474, ← c111475, ← c111476] at e_1824_14
  have e_1824_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_15
  simp only [← c111477, ← c111478, k2, ← c111479] at e_1824_15
  have hr := e_1824_15
  simp only [e_1824_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3589)) := by
  have c111480 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111481 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111482 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111483 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111484 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111485 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_16
  simp only [← c111480, ← c111481, ← c111482] at e_1824_16
  have e_1824_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_17
  simp only [← c111483, ← c111484, k2, ← c111485] at e_1824_17
  have hr := e_1824_17
  simp only [e_1824_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3590)) := by
  have c111486 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111487 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111488 := (publicBatchWrapper89_copies3484 a h).1
  have c111489 := (publicBatchWrapper89_copies3484 a h).2.1
  have c111490 := (publicBatchWrapper89_copies3484 a h).2.2.1
  have c111491 := (publicBatchWrapper89_copies3484 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_18
  simp only [← c111486, ← c111487, ← c111488] at e_1824_18
  have e_1824_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_19
  simp only [← c111489, ← c111490, k2, ← c111491] at e_1824_19
  have hr := e_1824_19
  simp only [e_1824_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3591)) := by
  have c111492 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.1
  have c111493 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.1
  have c111494 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.1
  have c111495 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.1
  have c111496 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.1
  have c111497 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_0
  simp only [← c111492, ← c111493, ← c111494] at e_1825_0
  have e_1825_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_1
  simp only [← c111495, ← c111496, k2, ← c111497] at e_1825_1
  have hr := e_1825_1
  simp only [e_1825_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3592)) := by
  have c111498 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111499 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111500 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111501 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111502 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111503 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_2
  simp only [← c111498, ← c111499, ← c111500] at e_1825_2
  have e_1825_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_3
  simp only [← c111501, ← c111502, k2, ← c111503] at e_1825_3
  have hr := e_1825_3
  simp only [e_1825_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3593)) := by
  have c111504 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111505 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111506 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111507 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111508 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111509 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_4
  simp only [← c111504, ← c111505, ← c111506] at e_1825_4
  have e_1825_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_5
  simp only [← c111507, ← c111508, k2, ← c111509] at e_1825_5
  have hr := e_1825_5
  simp only [e_1825_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3594)) := by
  have c111510 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111511 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111512 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111513 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111514 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111515 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_6
  simp only [← c111510, ← c111511, ← c111512] at e_1825_6
  have e_1825_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_7
  simp only [← c111513, ← c111514, k2, ← c111515] at e_1825_7
  have hr := e_1825_7
  simp only [e_1825_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3595)) := by
  have c111516 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111517 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111518 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111519 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111520 := (publicBatchWrapper89_copies3485 a h).1
  have c111521 := (publicBatchWrapper89_copies3485 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_8
  simp only [← c111516, ← c111517, ← c111518] at e_1825_8
  have e_1825_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_9
  simp only [← c111519, ← c111520, k2, ← c111521] at e_1825_9
  have hr := e_1825_9
  simp only [e_1825_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3596)) := by
  have c111522 := (publicBatchWrapper89_copies3485 a h).2.2.1
  have c111523 := (publicBatchWrapper89_copies3485 a h).2.2.2.1
  have c111524 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.1
  have c111525 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.1
  have c111526 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.1
  have c111527 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_10
  simp only [← c111522, ← c111523, ← c111524] at e_1825_10
  have e_1825_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_11
  simp only [← c111525, ← c111526, k2, ← c111527] at e_1825_11
  have hr := e_1825_11
  simp only [e_1825_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3597)) := by
  have c111528 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.1
  have c111529 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.1
  have c111530 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111531 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111532 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111533 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_12
  simp only [← c111528, ← c111529, ← c111530] at e_1825_12
  have e_1825_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_13
  simp only [← c111531, ← c111532, k2, ← c111533] at e_1825_13
  have hr := e_1825_13
  simp only [e_1825_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3598)) := by
  have c111534 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111535 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111536 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111537 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111538 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111539 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_14
  simp only [← c111534, ← c111535, ← c111536] at e_1825_14
  have e_1825_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_15
  simp only [← c111537, ← c111538, k2, ← c111539] at e_1825_15
  have hr := e_1825_15
  simp only [e_1825_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3599)) := by
  have c111540 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111541 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111542 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111543 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111544 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111545 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_16
  simp only [← c111540, ← c111541, ← c111542] at e_1825_16
  have e_1825_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_17
  simp only [← c111543, ← c111544, k2, ← c111545] at e_1825_17
  have hr := e_1825_17
  simp only [e_1825_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3600)) := by
  have c111546 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111547 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111548 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111549 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111550 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111551 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_18
  simp only [← c111546, ← c111547, ← c111548] at e_1825_18
  have e_1825_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_19
  simp only [← c111549, ← c111550, k2, ← c111551] at e_1825_19
  have hr := e_1825_19
  simp only [e_1825_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3601)) := by
  have c111552 := (publicBatchWrapper89_copies3486 a h).1
  have c111553 := (publicBatchWrapper89_copies3486 a h).2.1
  have c111554 := (publicBatchWrapper89_copies3486 a h).2.2.1
  have c111555 := (publicBatchWrapper89_copies3486 a h).2.2.2.1
  have c111556 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.1
  have c111557 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_0
  simp only [← c111552, ← c111553, ← c111554] at e_1826_0
  have e_1826_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_1
  simp only [← c111555, ← c111556, k2, ← c111557] at e_1826_1
  have hr := e_1826_1
  simp only [e_1826_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3770)) := by
  have c111558 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.1
  have c111559 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.1
  have c111560 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.1
  have c111561 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.1
  have c111562 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111563 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_2
  simp only [← c111558, ← c111559, ← c111560] at e_1826_2
  have e_1826_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_3
  simp only [← c111561, ← c111562, k2, ← c111563] at e_1826_3
  have hr := e_1826_3
  simp only [e_1826_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3771)) := by
  have c111564 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111565 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111566 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111567 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111568 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111569 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_4
  simp only [← c111564, ← c111565, ← c111566] at e_1826_4
  have e_1826_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_5
  simp only [← c111567, ← c111568, k2, ← c111569] at e_1826_5
  have hr := e_1826_5
  simp only [e_1826_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3772)) := by
  have c111570 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111571 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111572 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111573 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111574 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111575 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_6
  simp only [← c111570, ← c111571, ← c111572] at e_1826_6
  have e_1826_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_7
  simp only [← c111573, ← c111574, k2, ← c111575] at e_1826_7
  have hr := e_1826_7
  simp only [e_1826_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3773)) := by
  have c111576 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111577 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111578 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111579 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111580 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111581 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_8
  simp only [← c111576, ← c111577, ← c111578] at e_1826_8
  have e_1826_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_9
  simp only [← c111579, ← c111580, k2, ← c111581] at e_1826_9
  have hr := e_1826_9
  simp only [e_1826_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3774)) := by
  have c111582 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111583 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111584 := (publicBatchWrapper89_copies3487 a h).1
  have c111585 := (publicBatchWrapper89_copies3487 a h).2.1
  have c111586 := (publicBatchWrapper89_copies3487 a h).2.2.1
  have c111587 := (publicBatchWrapper89_copies3487 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_10
  simp only [← c111582, ← c111583, ← c111584] at e_1826_10
  have e_1826_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_11
  simp only [← c111585, ← c111586, k2, ← c111587] at e_1826_11
  have hr := e_1826_11
  simp only [e_1826_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3775)) := by
  have c111588 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.1
  have c111589 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.1
  have c111590 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.1
  have c111591 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.1
  have c111592 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.1
  have c111593 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_12
  simp only [← c111588, ← c111589, ← c111590] at e_1826_12
  have e_1826_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_13
  simp only [← c111591, ← c111592, k2, ← c111593] at e_1826_13
  have hr := e_1826_13
  simp only [e_1826_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3776)) := by
  have c111594 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111595 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111596 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111597 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111598 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111599 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_14
  simp only [← c111594, ← c111595, ← c111596] at e_1826_14
  have e_1826_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_15
  simp only [← c111597, ← c111598, k2, ← c111599] at e_1826_15
  have hr := e_1826_15
  simp only [e_1826_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3777)) := by
  have c111600 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111601 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111602 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111603 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111604 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111605 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_16
  simp only [← c111600, ← c111601, ← c111602] at e_1826_16
  have e_1826_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_17
  simp only [← c111603, ← c111604, k2, ← c111605] at e_1826_17
  have hr := e_1826_17
  simp only [e_1826_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3778)) := by
  have c111606 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111607 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111608 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111609 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111610 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111611 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_18
  simp only [← c111606, ← c111607, ← c111608] at e_1826_18
  have e_1826_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_19
  simp only [← c111609, ← c111610, k2, ← c111611] at e_1826_19
  have hr := e_1826_19
  simp only [e_1826_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3779)) := by
  have c111612 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111613 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111614 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111615 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111616 := (publicBatchWrapper89_copies3488 a h).1
  have c111617 := (publicBatchWrapper89_copies3488 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_0
  simp only [← c111612, ← c111613, ← c111614] at e_1827_0
  have e_1827_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_1
  simp only [← c111615, ← c111616, k2, ← c111617] at e_1827_1
  have hr := e_1827_1
  simp only [e_1827_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3780)) := by
  have c111618 := (publicBatchWrapper89_copies3488 a h).2.2.1
  have c111619 := (publicBatchWrapper89_copies3488 a h).2.2.2.1
  have c111620 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.1
  have c111621 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.1
  have c111622 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.1
  have c111623 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_2
  simp only [← c111618, ← c111619, ← c111620] at e_1827_2
  have e_1827_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_3
  simp only [← c111621, ← c111622, k2, ← c111623] at e_1827_3
  have hr := e_1827_3
  simp only [e_1827_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3781)) := by
  have c111624 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.1
  have c111625 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.1
  have c111626 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111627 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111628 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111629 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_4
  simp only [← c111624, ← c111625, ← c111626] at e_1827_4
  have e_1827_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_5
  simp only [← c111627, ← c111628, k2, ← c111629] at e_1827_5
  have hr := e_1827_5
  simp only [e_1827_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3782)) := by
  have c111630 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111631 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111632 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111633 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111634 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111635 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_6
  simp only [← c111630, ← c111631, ← c111632] at e_1827_6
  have e_1827_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_7
  simp only [← c111633, ← c111634, k2, ← c111635] at e_1827_7
  have hr := e_1827_7
  simp only [e_1827_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3783)) := by
  have c111636 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111637 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111638 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111639 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111640 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111641 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_8
  simp only [← c111636, ← c111637, ← c111638] at e_1827_8
  have e_1827_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_9
  simp only [← c111639, ← c111640, k2, ← c111641] at e_1827_9
  have hr := e_1827_9
  simp only [e_1827_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3784)) := by
  have c111642 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111643 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111644 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111645 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111646 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111647 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_10
  simp only [← c111642, ← c111643, ← c111644] at e_1827_10
  have e_1827_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_11
  simp only [← c111645, ← c111646, k2, ← c111647] at e_1827_11
  have hr := e_1827_11
  simp only [e_1827_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3785)) := by
  have c111648 := (publicBatchWrapper89_copies3489 a h).1
  have c111649 := (publicBatchWrapper89_copies3489 a h).2.1
  have c111650 := (publicBatchWrapper89_copies3489 a h).2.2.1
  have c111651 := (publicBatchWrapper89_copies3489 a h).2.2.2.1
  have c111652 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.1
  have c111653 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_12
  simp only [← c111648, ← c111649, ← c111650] at e_1827_12
  have e_1827_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_13
  simp only [← c111651, ← c111652, k2, ← c111653] at e_1827_13
  have hr := e_1827_13
  simp only [e_1827_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3786)) := by
  have c111654 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.1
  have c111655 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.1
  have c111656 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.1
  have c111657 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.1
  have c111658 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111659 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_14
  simp only [← c111654, ← c111655, ← c111656] at e_1827_14
  have e_1827_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_15
  simp only [← c111657, ← c111658, k2, ← c111659] at e_1827_15
  have hr := e_1827_15
  simp only [e_1827_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3787)) := by
  have c111660 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111661 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111662 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111663 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111664 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111665 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_16
  simp only [← c111660, ← c111661, ← c111662] at e_1827_16
  have e_1827_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_17
  simp only [← c111663, ← c111664, k2, ← c111665] at e_1827_17
  have hr := e_1827_17
  simp only [e_1827_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3788)) := by
  have c111666 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111667 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111668 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111669 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111670 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111671 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_18
  simp only [← c111666, ← c111667, ← c111668] at e_1827_18
  have e_1827_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_19
  simp only [← c111669, ← c111670, k2, ← c111671] at e_1827_19
  have hr := e_1827_19
  simp only [e_1827_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3789)) := by
  have c111672 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111673 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111674 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111675 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111676 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111677 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_0
  simp only [← c111672, ← c111673, ← c111674] at e_1828_0
  have e_1828_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_1
  simp only [← c111675, ← c111676, k2, ← c111677] at e_1828_1
  have hr := e_1828_1
  simp only [e_1828_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3790)) := by
  have c111678 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111679 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111680 := (publicBatchWrapper89_copies3490 a h).1
  have c111681 := (publicBatchWrapper89_copies3490 a h).2.1
  have c111682 := (publicBatchWrapper89_copies3490 a h).2.2.1
  have c111683 := (publicBatchWrapper89_copies3490 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_2
  simp only [← c111678, ← c111679, ← c111680] at e_1828_2
  have e_1828_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_3
  simp only [← c111681, ← c111682, k2, ← c111683] at e_1828_3
  have hr := e_1828_3
  simp only [e_1828_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3791)) := by
  have c111684 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.1
  have c111685 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.1
  have c111686 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.1
  have c111687 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.1
  have c111688 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.1
  have c111689 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_4
  simp only [← c111684, ← c111685, ← c111686] at e_1828_4
  have e_1828_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_5
  simp only [← c111687, ← c111688, k2, ← c111689] at e_1828_5
  have hr := e_1828_5
  simp only [e_1828_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3792)) := by
  have c111690 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111691 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111692 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111693 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111694 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111695 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_6
  simp only [← c111690, ← c111691, ← c111692] at e_1828_6
  have e_1828_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_7
  simp only [← c111693, ← c111694, k2, ← c111695] at e_1828_7
  have hr := e_1828_7
  simp only [e_1828_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3793)) := by
  have c111696 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111697 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111698 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111699 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111700 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111701 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_8
  simp only [← c111696, ← c111697, ← c111698] at e_1828_8
  have e_1828_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_9
  simp only [← c111699, ← c111700, k2, ← c111701] at e_1828_9
  have hr := e_1828_9
  simp only [e_1828_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3794)) := by
  have c111702 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111703 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111704 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111705 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111706 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111707 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_10
  simp only [← c111702, ← c111703, ← c111704] at e_1828_10
  have e_1828_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_11
  simp only [← c111705, ← c111706, k2, ← c111707] at e_1828_11
  have hr := e_1828_11
  simp only [e_1828_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3795)) := by
  have c111708 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111709 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111710 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111711 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111712 := (publicBatchWrapper89_copies3491 a h).1
  have c111713 := (publicBatchWrapper89_copies3491 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_12
  simp only [← c111708, ← c111709, ← c111710] at e_1828_12
  have e_1828_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_13
  simp only [← c111711, ← c111712, k2, ← c111713] at e_1828_13
  have hr := e_1828_13
  simp only [e_1828_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3796)) := by
  have c111714 := (publicBatchWrapper89_copies3491 a h).2.2.1
  have c111715 := (publicBatchWrapper89_copies3491 a h).2.2.2.1
  have c111716 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.1
  have c111717 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.1
  have c111718 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.1
  have c111719 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_14
  simp only [← c111714, ← c111715, ← c111716] at e_1828_14
  have e_1828_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_15
  simp only [← c111717, ← c111718, k2, ← c111719] at e_1828_15
  have hr := e_1828_15
  simp only [e_1828_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3797)) := by
  have c111720 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.1
  have c111721 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.1
  have c111722 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111723 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111724 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111725 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_16
  simp only [← c111720, ← c111721, ← c111722] at e_1828_16
  have e_1828_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_17
  simp only [← c111723, ← c111724, k2, ← c111725] at e_1828_17
  have hr := e_1828_17
  simp only [e_1828_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3798)) := by
  have c111726 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111727 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111728 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111729 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111730 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111731 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_18
  simp only [← c111726, ← c111727, ← c111728] at e_1828_18
  have e_1828_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_19
  simp only [← c111729, ← c111730, k2, ← c111731] at e_1828_19
  have hr := e_1828_19
  simp only [e_1828_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3799)) := by
  have c111732 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111733 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111734 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111735 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111736 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111737 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_0
  simp only [← c111732, ← c111733, ← c111734] at e_1829_0
  have e_1829_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_1
  simp only [← c111735, ← c111736, k2, ← c111737] at e_1829_1
  have hr := e_1829_1
  simp only [e_1829_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3800)) := by
  have c111738 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111739 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111740 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111741 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111742 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111743 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_2
  simp only [← c111738, ← c111739, ← c111740] at e_1829_2
  have e_1829_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_3
  simp only [← c111741, ← c111742, k2, ← c111743] at e_1829_3
  have hr := e_1829_3
  simp only [e_1829_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3801)) := by
  have c111744 := (publicBatchWrapper89_copies3492 a h).1
  have c111745 := (publicBatchWrapper89_copies3492 a h).2.1
  have c111746 := (publicBatchWrapper89_copies3492 a h).2.2.1
  have c111747 := (publicBatchWrapper89_copies3492 a h).2.2.2.1
  have c111748 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.1
  have c111749 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_4
  simp only [← c111744, ← c111745, ← c111746] at e_1829_4
  have e_1829_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_5
  simp only [← c111747, ← c111748, k2, ← c111749] at e_1829_5
  have hr := e_1829_5
  simp only [e_1829_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3970)) := by
  have c111750 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.1
  have c111751 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.1
  have c111752 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.1
  have c111753 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.1
  have c111754 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111755 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_6
  simp only [← c111750, ← c111751, ← c111752] at e_1829_6
  have e_1829_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_7
  simp only [← c111753, ← c111754, k2, ← c111755] at e_1829_7
  have hr := e_1829_7
  simp only [e_1829_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3971)) := by
  have c111756 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111757 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111758 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111759 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111760 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111761 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_8
  simp only [← c111756, ← c111757, ← c111758] at e_1829_8
  have e_1829_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_9
  simp only [← c111759, ← c111760, k2, ← c111761] at e_1829_9
  have hr := e_1829_9
  simp only [e_1829_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3972)) := by
  have c111762 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111763 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111764 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111765 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111766 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111767 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_10
  simp only [← c111762, ← c111763, ← c111764] at e_1829_10
  have e_1829_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_11
  simp only [← c111765, ← c111766, k2, ← c111767] at e_1829_11
  have hr := e_1829_11
  simp only [e_1829_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3973)) := by
  have c111768 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111769 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111770 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111771 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111772 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111773 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_12
  simp only [← c111768, ← c111769, ← c111770] at e_1829_12
  have e_1829_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_13
  simp only [← c111771, ← c111772, k2, ← c111773] at e_1829_13
  have hr := e_1829_13
  simp only [e_1829_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3974)) := by
  have c111774 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111775 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111776 := (publicBatchWrapper89_copies3493 a h).1
  have c111777 := (publicBatchWrapper89_copies3493 a h).2.1
  have c111778 := (publicBatchWrapper89_copies3493 a h).2.2.1
  have c111779 := (publicBatchWrapper89_copies3493 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_14
  simp only [← c111774, ← c111775, ← c111776] at e_1829_14
  have e_1829_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_15
  simp only [← c111777, ← c111778, k2, ← c111779] at e_1829_15
  have hr := e_1829_15
  simp only [e_1829_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3975)) := by
  have c111780 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.1
  have c111781 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.1
  have c111782 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.1
  have c111783 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.1
  have c111784 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.1
  have c111785 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_16
  simp only [← c111780, ← c111781, ← c111782] at e_1829_16
  have e_1829_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_17
  simp only [← c111783, ← c111784, k2, ← c111785] at e_1829_17
  have hr := e_1829_17
  simp only [e_1829_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3976)) := by
  have c111786 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111787 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111788 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111789 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111790 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111791 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_18
  simp only [← c111786, ← c111787, ← c111788] at e_1829_18
  have e_1829_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_19
  simp only [← c111789, ← c111790, k2, ← c111791] at e_1829_19
  have hr := e_1829_19
  simp only [e_1829_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3977)) := by
  have c111792 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111793 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111794 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111795 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111796 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111797 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_0
  simp only [← c111792, ← c111793, ← c111794] at e_1830_0
  have e_1830_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_1
  simp only [← c111795, ← c111796, k2, ← c111797] at e_1830_1
  have hr := e_1830_1
  simp only [e_1830_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3978)) := by
  have c111798 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111799 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111800 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111801 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111802 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111803 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_2
  simp only [← c111798, ← c111799, ← c111800] at e_1830_2
  have e_1830_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_3
  simp only [← c111801, ← c111802, k2, ← c111803] at e_1830_3
  have hr := e_1830_3
  simp only [e_1830_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3979)) := by
  have c111804 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111805 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111806 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111807 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111808 := (publicBatchWrapper89_copies3494 a h).1
  have c111809 := (publicBatchWrapper89_copies3494 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_4
  simp only [← c111804, ← c111805, ← c111806] at e_1830_4
  have e_1830_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_5
  simp only [← c111807, ← c111808, k2, ← c111809] at e_1830_5
  have hr := e_1830_5
  simp only [e_1830_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3980)) := by
  have c111810 := (publicBatchWrapper89_copies3494 a h).2.2.1
  have c111811 := (publicBatchWrapper89_copies3494 a h).2.2.2.1
  have c111812 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.1
  have c111813 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.1
  have c111814 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.1
  have c111815 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_6
  simp only [← c111810, ← c111811, ← c111812] at e_1830_6
  have e_1830_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_7
  simp only [← c111813, ← c111814, k2, ← c111815] at e_1830_7
  have hr := e_1830_7
  simp only [e_1830_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3981)) := by
  have c111816 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.1
  have c111817 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.1
  have c111818 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111819 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111820 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111821 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_8
  simp only [← c111816, ← c111817, ← c111818] at e_1830_8
  have e_1830_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_9
  simp only [← c111819, ← c111820, k2, ← c111821] at e_1830_9
  have hr := e_1830_9
  simp only [e_1830_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3982)) := by
  have c111822 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111823 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111824 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111825 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111826 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111827 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_10
  simp only [← c111822, ← c111823, ← c111824] at e_1830_10
  have e_1830_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_11
  simp only [← c111825, ← c111826, k2, ← c111827] at e_1830_11
  have hr := e_1830_11
  simp only [e_1830_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3983)) := by
  have c111828 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111829 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111830 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111831 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111832 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111833 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_12
  simp only [← c111828, ← c111829, ← c111830] at e_1830_12
  have e_1830_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_13
  simp only [← c111831, ← c111832, k2, ← c111833] at e_1830_13
  have hr := e_1830_13
  simp only [e_1830_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3984)) := by
  have c111834 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111835 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111836 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111837 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111838 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111839 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_14
  simp only [← c111834, ← c111835, ← c111836] at e_1830_14
  have e_1830_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_15
  simp only [← c111837, ← c111838, k2, ← c111839] at e_1830_15
  have hr := e_1830_15
  simp only [e_1830_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3985)) := by
  have c111840 := (publicBatchWrapper89_copies3495 a h).1
  have c111841 := (publicBatchWrapper89_copies3495 a h).2.1
  have c111842 := (publicBatchWrapper89_copies3495 a h).2.2.1
  have c111843 := (publicBatchWrapper89_copies3495 a h).2.2.2.1
  have c111844 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.1
  have c111845 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_16
  simp only [← c111840, ← c111841, ← c111842] at e_1830_16
  have e_1830_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_17
  simp only [← c111843, ← c111844, k2, ← c111845] at e_1830_17
  have hr := e_1830_17
  simp only [e_1830_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3986)) := by
  have c111846 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.1
  have c111847 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.1
  have c111848 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.1
  have c111849 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.1
  have c111850 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111851 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_18
  simp only [← c111846, ← c111847, ← c111848] at e_1830_18
  have e_1830_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_19
  simp only [← c111849, ← c111850, k2, ← c111851] at e_1830_19
  have hr := e_1830_19
  simp only [e_1830_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3987)) := by
  have c111852 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111853 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111854 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111855 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111856 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111857 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_0
  simp only [← c111852, ← c111853, ← c111854] at e_1831_0
  have e_1831_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_1
  simp only [← c111855, ← c111856, k2, ← c111857] at e_1831_1
  have hr := e_1831_1
  simp only [e_1831_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3988)) := by
  have c111858 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111859 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111860 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111861 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111862 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111863 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_2
  simp only [← c111858, ← c111859, ← c111860] at e_1831_2
  have e_1831_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_3
  simp only [← c111861, ← c111862, k2, ← c111863] at e_1831_3
  have hr := e_1831_3
  simp only [e_1831_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3989)) := by
  have c111864 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111865 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111866 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111867 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111868 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111869 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_4
  simp only [← c111864, ← c111865, ← c111866] at e_1831_4
  have e_1831_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_5
  simp only [← c111867, ← c111868, k2, ← c111869] at e_1831_5
  have hr := e_1831_5
  simp only [e_1831_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3990)) := by
  have c111870 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111871 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111872 := (publicBatchWrapper89_copies3496 a h).1
  have c111873 := (publicBatchWrapper89_copies3496 a h).2.1
  have c111874 := (publicBatchWrapper89_copies3496 a h).2.2.1
  have c111875 := (publicBatchWrapper89_copies3496 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_6
  simp only [← c111870, ← c111871, ← c111872] at e_1831_6
  have e_1831_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_7
  simp only [← c111873, ← c111874, k2, ← c111875] at e_1831_7
  have hr := e_1831_7
  simp only [e_1831_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3991)) := by
  have c111876 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.1
  have c111877 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.1
  have c111878 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.1
  have c111879 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.1
  have c111880 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.1
  have c111881 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_8
  simp only [← c111876, ← c111877, ← c111878] at e_1831_8
  have e_1831_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_9
  simp only [← c111879, ← c111880, k2, ← c111881] at e_1831_9
  have hr := e_1831_9
  simp only [e_1831_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3992)) := by
  have c111882 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111883 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111884 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111885 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111886 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111887 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_10
  simp only [← c111882, ← c111883, ← c111884] at e_1831_10
  have e_1831_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_11
  simp only [← c111885, ← c111886, k2, ← c111887] at e_1831_11
  have hr := e_1831_11
  simp only [e_1831_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3993)) := by
  have c111888 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111889 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111890 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111891 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111892 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111893 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_12
  simp only [← c111888, ← c111889, ← c111890] at e_1831_12
  have e_1831_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_13
  simp only [← c111891, ← c111892, k2, ← c111893] at e_1831_13
  have hr := e_1831_13
  simp only [e_1831_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3994)) := by
  have c111894 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111895 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111896 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111897 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111898 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111899 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_14
  simp only [← c111894, ← c111895, ← c111896] at e_1831_14
  have e_1831_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_15
  simp only [← c111897, ← c111898, k2, ← c111899] at e_1831_15
  have hr := e_1831_15
  simp only [e_1831_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3995)) := by
  have c111900 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111901 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111902 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111903 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111904 := (publicBatchWrapper89_copies3497 a h).1
  have c111905 := (publicBatchWrapper89_copies3497 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_16
  simp only [← c111900, ← c111901, ← c111902] at e_1831_16
  have e_1831_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_17
  simp only [← c111903, ← c111904, k2, ← c111905] at e_1831_17
  have hr := e_1831_17
  simp only [e_1831_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3996)) := by
  have c111906 := (publicBatchWrapper89_copies3497 a h).2.2.1
  have c111907 := (publicBatchWrapper89_copies3497 a h).2.2.2.1
  have c111908 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.1
  have c111909 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.1
  have c111910 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.1
  have c111911 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_18
  simp only [← c111906, ← c111907, ← c111908] at e_1831_18
  have e_1831_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_19
  simp only [← c111909, ← c111910, k2, ← c111911] at e_1831_19
  have hr := e_1831_19
  simp only [e_1831_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3997)) := by
  have c111912 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.1
  have c111913 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.1
  have c111914 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111915 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111916 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111917 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_0
  simp only [← c111912, ← c111913, ← c111914] at e_1832_0
  have e_1832_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_1
  simp only [← c111915, ← c111916, k2, ← c111917] at e_1832_1
  have hr := e_1832_1
  simp only [e_1832_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3998)) := by
  have c111918 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111919 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111920 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111921 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111922 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111923 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_2
  simp only [← c111918, ← c111919, ← c111920] at e_1832_2
  have e_1832_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_3
  simp only [← c111921, ← c111922, k2, ← c111923] at e_1832_3
  have hr := e_1832_3
  simp only [e_1832_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3999)) := by
  have c111924 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111925 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111926 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111927 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111928 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111929 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_4
  simp only [← c111924, ← c111925, ← c111926] at e_1832_4
  have e_1832_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_5
  simp only [← c111927, ← c111928, k2, ← c111929] at e_1832_5
  have hr := e_1832_5
  simp only [e_1832_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 4000)) := by
  have c111930 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111931 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111932 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111933 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111934 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111935 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_6
  simp only [← c111930, ← c111931, ← c111932] at e_1832_6
  have e_1832_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_7
  simp only [← c111933, ← c111934, k2, ← c111935] at e_1832_7
  have hr := e_1832_7
  simp only [e_1832_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 4001)) := by
  have c111936 := (publicBatchWrapper89_copies3498 a h).1
  have c111937 := (publicBatchWrapper89_copies3498 a h).2.1
  have c111938 := (publicBatchWrapper89_copies3498 a h).2.2.1
  have c111939 := (publicBatchWrapper89_copies3498 a h).2.2.2.1
  have c111940 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.1
  have c111941 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_8
  simp only [← c111936, ← c111937, ← c111938] at e_1832_8
  have e_1832_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_9
  simp only [← c111939, ← c111940, k2, ← c111941] at e_1832_9
  have hr := e_1832_9
  simp only [e_1832_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4170)) := by
  have c111942 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.1
  have c111943 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.1
  have c111944 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.1
  have c111945 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.1
  have c111946 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111947 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_10
  simp only [← c111942, ← c111943, ← c111944] at e_1832_10
  have e_1832_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_11
  simp only [← c111945, ← c111946, k2, ← c111947] at e_1832_11
  have hr := e_1832_11
  simp only [e_1832_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4171)) := by
  have c111948 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111949 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111950 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111951 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111952 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111953 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_12
  simp only [← c111948, ← c111949, ← c111950] at e_1832_12
  have e_1832_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_13
  simp only [← c111951, ← c111952, k2, ← c111953] at e_1832_13
  have hr := e_1832_13
  simp only [e_1832_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4172)) := by
  have c111954 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111955 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111956 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111957 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111958 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111959 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_14
  simp only [← c111954, ← c111955, ← c111956] at e_1832_14
  have e_1832_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_15
  simp only [← c111957, ← c111958, k2, ← c111959] at e_1832_15
  have hr := e_1832_15
  simp only [e_1832_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4173)) := by
  have c111960 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111961 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111962 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111963 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111964 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111965 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_16
  simp only [← c111960, ← c111961, ← c111962] at e_1832_16
  have e_1832_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_17
  simp only [← c111963, ← c111964, k2, ← c111965] at e_1832_17
  have hr := e_1832_17
  simp only [e_1832_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4174)) := by
  have c111966 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111967 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111968 := (publicBatchWrapper89_copies3499 a h).1
  have c111969 := (publicBatchWrapper89_copies3499 a h).2.1
  have c111970 := (publicBatchWrapper89_copies3499 a h).2.2.1
  have c111971 := (publicBatchWrapper89_copies3499 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_18
  simp only [← c111966, ← c111967, ← c111968] at e_1832_18
  have e_1832_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_19
  simp only [← c111969, ← c111970, k2, ← c111971] at e_1832_19
  have hr := e_1832_19
  simp only [e_1832_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4175)) := by
  have c111972 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.1
  have c111973 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.1
  have c111974 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.1
  have c111975 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.1
  have c111976 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.1
  have c111977 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_0
  simp only [← c111972, ← c111973, ← c111974] at e_1833_0
  have e_1833_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_1
  simp only [← c111975, ← c111976, k2, ← c111977] at e_1833_1
  have hr := e_1833_1
  simp only [e_1833_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4176)) := by
  have c111978 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111979 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111980 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111981 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111982 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111983 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_2
  simp only [← c111978, ← c111979, ← c111980] at e_1833_2
  have e_1833_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_3
  simp only [← c111981, ← c111982, k2, ← c111983] at e_1833_3
  have hr := e_1833_3
  simp only [e_1833_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4177)) := by
  have c111984 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111985 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111986 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111987 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111988 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111989 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_4
  simp only [← c111984, ← c111985, ← c111986] at e_1833_4
  have e_1833_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_5
  simp only [← c111987, ← c111988, k2, ← c111989] at e_1833_5
  have hr := e_1833_5
  simp only [e_1833_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4178)) := by
  have c111990 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111991 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111992 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111993 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111994 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111995 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_6
  simp only [← c111990, ← c111991, ← c111992] at e_1833_6
  have e_1833_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_7
  simp only [← c111993, ← c111994, k2, ← c111995] at e_1833_7
  have hr := e_1833_7
  simp only [e_1833_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4179)) := by
  have c111996 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111997 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111998 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111999 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112000 := (publicBatchWrapper89_copies3500 a h).1
  have c112001 := (publicBatchWrapper89_copies3500 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_8
  simp only [← c111996, ← c111997, ← c111998] at e_1833_8
  have e_1833_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_9
  simp only [← c111999, ← c112000, k2, ← c112001] at e_1833_9
  have hr := e_1833_9
  simp only [e_1833_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4180)) := by
  have c112002 := (publicBatchWrapper89_copies3500 a h).2.2.1
  have c112003 := (publicBatchWrapper89_copies3500 a h).2.2.2.1
  have c112004 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.1
  have c112005 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.1
  have c112006 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.1
  have c112007 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_10
  simp only [← c112002, ← c112003, ← c112004] at e_1833_10
  have e_1833_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_11
  simp only [← c112005, ← c112006, k2, ← c112007] at e_1833_11
  have hr := e_1833_11
  simp only [e_1833_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4181)) := by
  have c112008 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.1
  have c112009 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.1
  have c112010 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112011 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112012 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112013 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_12
  simp only [← c112008, ← c112009, ← c112010] at e_1833_12
  have e_1833_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_13
  simp only [← c112011, ← c112012, k2, ← c112013] at e_1833_13
  have hr := e_1833_13
  simp only [e_1833_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4182)) := by
  have c112014 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112015 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112016 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112017 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112018 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112019 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_14
  simp only [← c112014, ← c112015, ← c112016] at e_1833_14
  have e_1833_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_15
  simp only [← c112017, ← c112018, k2, ← c112019] at e_1833_15
  have hr := e_1833_15
  simp only [e_1833_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4183)) := by
  have c112020 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112021 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112022 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112023 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112024 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112025 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_16
  simp only [← c112020, ← c112021, ← c112022] at e_1833_16
  have e_1833_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_17
  simp only [← c112023, ← c112024, k2, ← c112025] at e_1833_17
  have hr := e_1833_17
  simp only [e_1833_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4184)) := by
  have c112026 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112027 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112028 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112029 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112030 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112031 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_18
  simp only [← c112026, ← c112027, ← c112028] at e_1833_18
  have e_1833_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_19
  simp only [← c112029, ← c112030, k2, ← c112031] at e_1833_19
  have hr := e_1833_19
  simp only [e_1833_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4185)) := by
  have c112032 := (publicBatchWrapper89_copies3501 a h).1
  have c112033 := (publicBatchWrapper89_copies3501 a h).2.1
  have c112034 := (publicBatchWrapper89_copies3501 a h).2.2.1
  have c112035 := (publicBatchWrapper89_copies3501 a h).2.2.2.1
  have c112036 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.1
  have c112037 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_0
  simp only [← c112032, ← c112033, ← c112034] at e_1834_0
  have e_1834_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_1
  simp only [← c112035, ← c112036, k2, ← c112037] at e_1834_1
  have hr := e_1834_1
  simp only [e_1834_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4186)) := by
  have c112038 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.1
  have c112039 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.1
  have c112040 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.1
  have c112041 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.1
  have c112042 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112043 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_2
  simp only [← c112038, ← c112039, ← c112040] at e_1834_2
  have e_1834_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_3
  simp only [← c112041, ← c112042, k2, ← c112043] at e_1834_3
  have hr := e_1834_3
  simp only [e_1834_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4187)) := by
  have c112044 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112045 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112046 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112047 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112048 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112049 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_4
  simp only [← c112044, ← c112045, ← c112046] at e_1834_4
  have e_1834_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_5
  simp only [← c112047, ← c112048, k2, ← c112049] at e_1834_5
  have hr := e_1834_5
  simp only [e_1834_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4188)) := by
  have c112050 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112051 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112052 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112053 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112054 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112055 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_6
  simp only [← c112050, ← c112051, ← c112052] at e_1834_6
  have e_1834_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_7
  simp only [← c112053, ← c112054, k2, ← c112055] at e_1834_7
  have hr := e_1834_7
  simp only [e_1834_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4189)) := by
  have c112056 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112057 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112058 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112059 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112060 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112061 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_8
  simp only [← c112056, ← c112057, ← c112058] at e_1834_8
  have e_1834_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_9
  simp only [← c112059, ← c112060, k2, ← c112061] at e_1834_9
  have hr := e_1834_9
  simp only [e_1834_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4190)) := by
  have c112062 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112063 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112064 := (publicBatchWrapper89_copies3502 a h).1
  have c112065 := (publicBatchWrapper89_copies3502 a h).2.1
  have c112066 := (publicBatchWrapper89_copies3502 a h).2.2.1
  have c112067 := (publicBatchWrapper89_copies3502 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_10
  simp only [← c112062, ← c112063, ← c112064] at e_1834_10
  have e_1834_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_11
  simp only [← c112065, ← c112066, k2, ← c112067] at e_1834_11
  have hr := e_1834_11
  simp only [e_1834_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4191)) := by
  have c112068 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.1
  have c112069 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.1
  have c112070 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.1
  have c112071 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.1
  have c112072 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.1
  have c112073 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_12
  simp only [← c112068, ← c112069, ← c112070] at e_1834_12
  have e_1834_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_13
  simp only [← c112071, ← c112072, k2, ← c112073] at e_1834_13
  have hr := e_1834_13
  simp only [e_1834_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4192)) := by
  have c112074 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112075 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112076 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112077 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112078 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112079 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_14
  simp only [← c112074, ← c112075, ← c112076] at e_1834_14
  have e_1834_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_15
  simp only [← c112077, ← c112078, k2, ← c112079] at e_1834_15
  have hr := e_1834_15
  simp only [e_1834_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4193)) := by
  have c112080 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112081 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112082 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112083 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112084 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112085 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_16
  simp only [← c112080, ← c112081, ← c112082] at e_1834_16
  have e_1834_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_17
  simp only [← c112083, ← c112084, k2, ← c112085] at e_1834_17
  have hr := e_1834_17
  simp only [e_1834_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4194)) := by
  have c112086 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112087 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112088 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112089 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112090 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112091 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_18
  simp only [← c112086, ← c112087, ← c112088] at e_1834_18
  have e_1834_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_19
  simp only [← c112089, ← c112090, k2, ← c112091] at e_1834_19
  have hr := e_1834_19
  simp only [e_1834_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4195)) := by
  have c112092 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112093 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112094 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112095 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112096 := (publicBatchWrapper89_copies3503 a h).1
  have c112097 := (publicBatchWrapper89_copies3503 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_0
  simp only [← c112092, ← c112093, ← c112094] at e_1835_0
  have e_1835_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_1
  simp only [← c112095, ← c112096, k2, ← c112097] at e_1835_1
  have hr := e_1835_1
  simp only [e_1835_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4196)) := by
  have c112098 := (publicBatchWrapper89_copies3503 a h).2.2.1
  have c112099 := (publicBatchWrapper89_copies3503 a h).2.2.2.1
  have c112100 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.1
  have c112101 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.1
  have c112102 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.1
  have c112103 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_2
  simp only [← c112098, ← c112099, ← c112100] at e_1835_2
  have e_1835_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_3
  simp only [← c112101, ← c112102, k2, ← c112103] at e_1835_3
  have hr := e_1835_3
  simp only [e_1835_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4197)) := by
  have c112104 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.1
  have c112105 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.1
  have c112106 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112107 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112108 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112109 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_4
  simp only [← c112104, ← c112105, ← c112106] at e_1835_4
  have e_1835_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_5
  simp only [← c112107, ← c112108, k2, ← c112109] at e_1835_5
  have hr := e_1835_5
  simp only [e_1835_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4198)) := by
  have c112110 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112111 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112112 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112113 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112114 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112115 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_6
  simp only [← c112110, ← c112111, ← c112112] at e_1835_6
  have e_1835_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_7
  simp only [← c112113, ← c112114, k2, ← c112115] at e_1835_7
  have hr := e_1835_7
  simp only [e_1835_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4199)) := by
  have c112116 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112117 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112118 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112119 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112120 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112121 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_8
  simp only [← c112116, ← c112117, ← c112118] at e_1835_8
  have e_1835_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_9
  simp only [← c112119, ← c112120, k2, ← c112121] at e_1835_9
  have hr := e_1835_9
  simp only [e_1835_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4200)) := by
  have c112122 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112123 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112124 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112125 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112126 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112127 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_10
  simp only [← c112122, ← c112123, ← c112124] at e_1835_10
  have e_1835_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_11
  simp only [← c112125, ← c112126, k2, ← c112127] at e_1835_11
  have hr := e_1835_11
  simp only [e_1835_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4201)) := by
  have c112128 := (publicBatchWrapper89_copies3504 a h).1
  have c112129 := (publicBatchWrapper89_copies3504 a h).2.1
  have c112130 := (publicBatchWrapper89_copies3504 a h).2.2.1
  have c112131 := (publicBatchWrapper89_copies3504 a h).2.2.2.1
  have c112132 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.1
  have c112133 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_12
  simp only [← c112128, ← c112129, ← c112130] at e_1835_12
  have e_1835_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_13
  simp only [← c112131, ← c112132, k2, ← c112133] at e_1835_13
  have hr := e_1835_13
  simp only [e_1835_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4370)) := by
  have c112134 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.1
  have c112135 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.1
  have c112136 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.1
  have c112137 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.1
  have c112138 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112139 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_14
  simp only [← c112134, ← c112135, ← c112136] at e_1835_14
  have e_1835_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_15
  simp only [← c112137, ← c112138, k2, ← c112139] at e_1835_15
  have hr := e_1835_15
  simp only [e_1835_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4371)) := by
  have c112140 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112141 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112142 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112143 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112144 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112145 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_16
  simp only [← c112140, ← c112141, ← c112142] at e_1835_16
  have e_1835_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_17
  simp only [← c112143, ← c112144, k2, ← c112145] at e_1835_17
  have hr := e_1835_17
  simp only [e_1835_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4372)) := by
  have c112146 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112147 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112148 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112149 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112150 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112151 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_18
  simp only [← c112146, ← c112147, ← c112148] at e_1835_18
  have e_1835_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_19
  simp only [← c112149, ← c112150, k2, ← c112151] at e_1835_19
  have hr := e_1835_19
  simp only [e_1835_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4373)) := by
  have c112152 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112153 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112154 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112155 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112156 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112157 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_0
  simp only [← c112152, ← c112153, ← c112154] at e_1836_0
  have e_1836_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_1
  simp only [← c112155, ← c112156, k2, ← c112157] at e_1836_1
  have hr := e_1836_1
  simp only [e_1836_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4374)) := by
  have c112158 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112159 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112160 := (publicBatchWrapper89_copies3505 a h).1
  have c112161 := (publicBatchWrapper89_copies3505 a h).2.1
  have c112162 := (publicBatchWrapper89_copies3505 a h).2.2.1
  have c112163 := (publicBatchWrapper89_copies3505 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_2
  simp only [← c112158, ← c112159, ← c112160] at e_1836_2
  have e_1836_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_3
  simp only [← c112161, ← c112162, k2, ← c112163] at e_1836_3
  have hr := e_1836_3
  simp only [e_1836_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4375)) := by
  have c112164 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.1
  have c112165 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.1
  have c112166 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.1
  have c112167 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.1
  have c112168 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.1
  have c112169 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_4
  simp only [← c112164, ← c112165, ← c112166] at e_1836_4
  have e_1836_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_5
  simp only [← c112167, ← c112168, k2, ← c112169] at e_1836_5
  have hr := e_1836_5
  simp only [e_1836_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4376)) := by
  have c112170 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112171 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112172 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112173 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112174 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112175 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_6
  simp only [← c112170, ← c112171, ← c112172] at e_1836_6
  have e_1836_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_7
  simp only [← c112173, ← c112174, k2, ← c112175] at e_1836_7
  have hr := e_1836_7
  simp only [e_1836_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4377)) := by
  have c112176 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112177 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112178 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112179 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112180 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112181 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_8
  simp only [← c112176, ← c112177, ← c112178] at e_1836_8
  have e_1836_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_9
  simp only [← c112179, ← c112180, k2, ← c112181] at e_1836_9
  have hr := e_1836_9
  simp only [e_1836_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4378)) := by
  have c112182 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112183 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112184 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112185 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112186 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112187 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_10
  simp only [← c112182, ← c112183, ← c112184] at e_1836_10
  have e_1836_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_11
  simp only [← c112185, ← c112186, k2, ← c112187] at e_1836_11
  have hr := e_1836_11
  simp only [e_1836_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4379)) := by
  have c112188 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112189 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112190 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112191 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112192 := (publicBatchWrapper89_copies3506 a h).1
  have c112193 := (publicBatchWrapper89_copies3506 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_12
  simp only [← c112188, ← c112189, ← c112190] at e_1836_12
  have e_1836_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_13
  simp only [← c112191, ← c112192, k2, ← c112193] at e_1836_13
  have hr := e_1836_13
  simp only [e_1836_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4380)) := by
  have c112194 := (publicBatchWrapper89_copies3506 a h).2.2.1
  have c112195 := (publicBatchWrapper89_copies3506 a h).2.2.2.1
  have c112196 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.1
  have c112197 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.1
  have c112198 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.1
  have c112199 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_14
  simp only [← c112194, ← c112195, ← c112196] at e_1836_14
  have e_1836_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_15
  simp only [← c112197, ← c112198, k2, ← c112199] at e_1836_15
  have hr := e_1836_15
  simp only [e_1836_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4381)) := by
  have c112200 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.1
  have c112201 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.1
  have c112202 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112203 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112204 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112205 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_16
  simp only [← c112200, ← c112201, ← c112202] at e_1836_16
  have e_1836_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_17
  simp only [← c112203, ← c112204, k2, ← c112205] at e_1836_17
  have hr := e_1836_17
  simp only [e_1836_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4382)) := by
  have c112206 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112207 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112208 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112209 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112210 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112211 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_18
  simp only [← c112206, ← c112207, ← c112208] at e_1836_18
  have e_1836_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_19
  simp only [← c112209, ← c112210, k2, ← c112211] at e_1836_19
  have hr := e_1836_19
  simp only [e_1836_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4383)) := by
  have c112212 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112213 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112214 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112215 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112216 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112217 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_0
  simp only [← c112212, ← c112213, ← c112214] at e_1837_0
  have e_1837_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_1
  simp only [← c112215, ← c112216, k2, ← c112217] at e_1837_1
  have hr := e_1837_1
  simp only [e_1837_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4384)) := by
  have c112218 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112219 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112220 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112221 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112222 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112223 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_2
  simp only [← c112218, ← c112219, ← c112220] at e_1837_2
  have e_1837_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_3
  simp only [← c112221, ← c112222, k2, ← c112223] at e_1837_3
  have hr := e_1837_3
  simp only [e_1837_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4385)) := by
  have c112224 := (publicBatchWrapper89_copies3507 a h).1
  have c112225 := (publicBatchWrapper89_copies3507 a h).2.1
  have c112226 := (publicBatchWrapper89_copies3507 a h).2.2.1
  have c112227 := (publicBatchWrapper89_copies3507 a h).2.2.2.1
  have c112228 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.1
  have c112229 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_4
  simp only [← c112224, ← c112225, ← c112226] at e_1837_4
  have e_1837_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_5
  simp only [← c112227, ← c112228, k2, ← c112229] at e_1837_5
  have hr := e_1837_5
  simp only [e_1837_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4386)) := by
  have c112230 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.1
  have c112231 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.1
  have c112232 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.1
  have c112233 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.1
  have c112234 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112235 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_6
  simp only [← c112230, ← c112231, ← c112232] at e_1837_6
  have e_1837_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_7
  simp only [← c112233, ← c112234, k2, ← c112235] at e_1837_7
  have hr := e_1837_7
  simp only [e_1837_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4387)) := by
  have c112236 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112237 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112238 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112239 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112240 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112241 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_8
  simp only [← c112236, ← c112237, ← c112238] at e_1837_8
  have e_1837_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_9
  simp only [← c112239, ← c112240, k2, ← c112241] at e_1837_9
  have hr := e_1837_9
  simp only [e_1837_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4388)) := by
  have c112242 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112243 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112244 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112245 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112246 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112247 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_10
  simp only [← c112242, ← c112243, ← c112244] at e_1837_10
  have e_1837_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_11
  simp only [← c112245, ← c112246, k2, ← c112247] at e_1837_11
  have hr := e_1837_11
  simp only [e_1837_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4389)) := by
  have c112248 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112249 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112250 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112251 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112252 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112253 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_12
  simp only [← c112248, ← c112249, ← c112250] at e_1837_12
  have e_1837_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_13
  simp only [← c112251, ← c112252, k2, ← c112253] at e_1837_13
  have hr := e_1837_13
  simp only [e_1837_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4390)) := by
  have c112254 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112255 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112256 := (publicBatchWrapper89_copies3508 a h).1
  have c112257 := (publicBatchWrapper89_copies3508 a h).2.1
  have c112258 := (publicBatchWrapper89_copies3508 a h).2.2.1
  have c112259 := (publicBatchWrapper89_copies3508 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_14
  simp only [← c112254, ← c112255, ← c112256] at e_1837_14
  have e_1837_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_15
  simp only [← c112257, ← c112258, k2, ← c112259] at e_1837_15
  have hr := e_1837_15
  simp only [e_1837_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4391)) := by
  have c112260 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.1
  have c112261 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.1
  have c112262 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.1
  have c112263 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.1
  have c112264 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.1
  have c112265 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_16
  simp only [← c112260, ← c112261, ← c112262] at e_1837_16
  have e_1837_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_17
  simp only [← c112263, ← c112264, k2, ← c112265] at e_1837_17
  have hr := e_1837_17
  simp only [e_1837_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4392)) := by
  have c112266 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112267 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112268 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112269 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112270 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112271 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_18
  simp only [← c112266, ← c112267, ← c112268] at e_1837_18
  have e_1837_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_19
  simp only [← c112269, ← c112270, k2, ← c112271] at e_1837_19
  have hr := e_1837_19
  simp only [e_1837_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4393)) := by
  have c112272 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112273 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112274 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112275 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112276 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112277 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_0
  simp only [← c112272, ← c112273, ← c112274] at e_1838_0
  have e_1838_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_1
  simp only [← c112275, ← c112276, k2, ← c112277] at e_1838_1
  have hr := e_1838_1
  simp only [e_1838_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4394)) := by
  have c112278 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112279 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112280 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112281 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112282 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112283 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_2
  simp only [← c112278, ← c112279, ← c112280] at e_1838_2
  have e_1838_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_3
  simp only [← c112281, ← c112282, k2, ← c112283] at e_1838_3
  have hr := e_1838_3
  simp only [e_1838_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4395)) := by
  have c112284 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112285 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112286 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112287 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112288 := (publicBatchWrapper89_copies3509 a h).1
  have c112289 := (publicBatchWrapper89_copies3509 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_4
  simp only [← c112284, ← c112285, ← c112286] at e_1838_4
  have e_1838_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_5
  simp only [← c112287, ← c112288, k2, ← c112289] at e_1838_5
  have hr := e_1838_5
  simp only [e_1838_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4396)) := by
  have c112290 := (publicBatchWrapper89_copies3509 a h).2.2.1
  have c112291 := (publicBatchWrapper89_copies3509 a h).2.2.2.1
  have c112292 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.1
  have c112293 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.1
  have c112294 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.1
  have c112295 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_6
  simp only [← c112290, ← c112291, ← c112292] at e_1838_6
  have e_1838_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_7
  simp only [← c112293, ← c112294, k2, ← c112295] at e_1838_7
  have hr := e_1838_7
  simp only [e_1838_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4397)) := by
  have c112296 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.1
  have c112297 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.1
  have c112298 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112299 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112300 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112301 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_8
  simp only [← c112296, ← c112297, ← c112298] at e_1838_8
  have e_1838_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_9
  simp only [← c112299, ← c112300, k2, ← c112301] at e_1838_9
  have hr := e_1838_9
  simp only [e_1838_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4398)) := by
  have c112302 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112303 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112304 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112305 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112306 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112307 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_10
  simp only [← c112302, ← c112303, ← c112304] at e_1838_10
  have e_1838_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_11
  simp only [← c112305, ← c112306, k2, ← c112307] at e_1838_11
  have hr := e_1838_11
  simp only [e_1838_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4399)) := by
  have c112308 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112309 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112310 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112311 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112312 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112313 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_12
  simp only [← c112308, ← c112309, ← c112310] at e_1838_12
  have e_1838_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_13
  simp only [← c112311, ← c112312, k2, ← c112313] at e_1838_13
  have hr := e_1838_13
  simp only [e_1838_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4400)) := by
  have c112314 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112315 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112316 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112317 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112318 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112319 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_14
  simp only [← c112314, ← c112315, ← c112316] at e_1838_14
  have e_1838_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_15
  simp only [← c112317, ← c112318, k2, ← c112319] at e_1838_15
  have hr := e_1838_15
  simp only [e_1838_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4401)) := by
  have c112320 := (publicBatchWrapper89_copies3510 a h).1
  have c112321 := (publicBatchWrapper89_copies3510 a h).2.1
  have c112322 := (publicBatchWrapper89_copies3510 a h).2.2.1
  have c112323 := (publicBatchWrapper89_copies3510 a h).2.2.2.1
  have c112324 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.1
  have c112325 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_16
  simp only [← c112320, ← c112321, ← c112322] at e_1838_16
  have e_1838_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_17
  simp only [← c112323, ← c112324, k2, ← c112325] at e_1838_17
  have hr := e_1838_17
  simp only [e_1838_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4570)) := by
  have c112326 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.1
  have c112327 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.1
  have c112328 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.1
  have c112329 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.1
  have c112330 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112331 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_18
  simp only [← c112326, ← c112327, ← c112328] at e_1838_18
  have e_1838_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_19
  simp only [← c112329, ← c112330, k2, ← c112331] at e_1838_19
  have hr := e_1838_19
  simp only [e_1838_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4571)) := by
  have c112332 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112333 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112334 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112335 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112336 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112337 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_0
  simp only [← c112332, ← c112333, ← c112334] at e_1839_0
  have e_1839_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_1
  simp only [← c112335, ← c112336, k2, ← c112337] at e_1839_1
  have hr := e_1839_1
  simp only [e_1839_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4572)) := by
  have c112338 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112339 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112340 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112341 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112342 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112343 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_2
  simp only [← c112338, ← c112339, ← c112340] at e_1839_2
  have e_1839_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_3
  simp only [← c112341, ← c112342, k2, ← c112343] at e_1839_3
  have hr := e_1839_3
  simp only [e_1839_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4573)) := by
  have c112344 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112345 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112346 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112347 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112348 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112349 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_4
  simp only [← c112344, ← c112345, ← c112346] at e_1839_4
  have e_1839_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_5
  simp only [← c112347, ← c112348, k2, ← c112349] at e_1839_5
  have hr := e_1839_5
  simp only [e_1839_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4574)) := by
  have c112350 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112351 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112352 := (publicBatchWrapper89_copies3511 a h).1
  have c112353 := (publicBatchWrapper89_copies3511 a h).2.1
  have c112354 := (publicBatchWrapper89_copies3511 a h).2.2.1
  have c112355 := (publicBatchWrapper89_copies3511 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_6
  simp only [← c112350, ← c112351, ← c112352] at e_1839_6
  have e_1839_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_7
  simp only [← c112353, ← c112354, k2, ← c112355] at e_1839_7
  have hr := e_1839_7
  simp only [e_1839_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4575)) := by
  have c112356 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.1
  have c112357 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.1
  have c112358 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.1
  have c112359 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.1
  have c112360 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.1
  have c112361 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_8
  simp only [← c112356, ← c112357, ← c112358] at e_1839_8
  have e_1839_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_9
  simp only [← c112359, ← c112360, k2, ← c112361] at e_1839_9
  have hr := e_1839_9
  simp only [e_1839_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4576)) := by
  have c112362 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112363 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112364 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112365 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112366 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112367 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_10
  simp only [← c112362, ← c112363, ← c112364] at e_1839_10
  have e_1839_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_11
  simp only [← c112365, ← c112366, k2, ← c112367] at e_1839_11
  have hr := e_1839_11
  simp only [e_1839_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4577)) := by
  have c112368 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112369 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112370 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112371 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112372 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112373 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_12
  simp only [← c112368, ← c112369, ← c112370] at e_1839_12
  have e_1839_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_13
  simp only [← c112371, ← c112372, k2, ← c112373] at e_1839_13
  have hr := e_1839_13
  simp only [e_1839_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4578)) := by
  have c112374 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112375 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112376 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112377 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112378 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112379 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_14
  simp only [← c112374, ← c112375, ← c112376] at e_1839_14
  have e_1839_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_15
  simp only [← c112377, ← c112378, k2, ← c112379] at e_1839_15
  have hr := e_1839_15
  simp only [e_1839_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4579)) := by
  have c112380 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112381 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112382 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112383 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112384 := (publicBatchWrapper89_copies3512 a h).1
  have c112385 := (publicBatchWrapper89_copies3512 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_16
  simp only [← c112380, ← c112381, ← c112382] at e_1839_16
  have e_1839_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_17
  simp only [← c112383, ← c112384, k2, ← c112385] at e_1839_17
  have hr := e_1839_17
  simp only [e_1839_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4580)) := by
  have c112386 := (publicBatchWrapper89_copies3512 a h).2.2.1
  have c112387 := (publicBatchWrapper89_copies3512 a h).2.2.2.1
  have c112388 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.1
  have c112389 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.1
  have c112390 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.1
  have c112391 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_18
  simp only [← c112386, ← c112387, ← c112388] at e_1839_18
  have e_1839_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_19
  simp only [← c112389, ← c112390, k2, ← c112391] at e_1839_19
  have hr := e_1839_19
  simp only [e_1839_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4581)) := by
  have c112392 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.1
  have c112393 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.1
  have c112394 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112395 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112396 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112397 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_0
  simp only [← c112392, ← c112393, ← c112394] at e_1840_0
  have e_1840_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_1
  simp only [← c112395, ← c112396, k2, ← c112397] at e_1840_1
  have hr := e_1840_1
  simp only [e_1840_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4582)) := by
  have c112398 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112399 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112400 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112401 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112402 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112403 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_2
  simp only [← c112398, ← c112399, ← c112400] at e_1840_2
  have e_1840_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_3
  simp only [← c112401, ← c112402, k2, ← c112403] at e_1840_3
  have hr := e_1840_3
  simp only [e_1840_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4583)) := by
  have c112404 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112405 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112406 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112407 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112408 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112409 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_4
  simp only [← c112404, ← c112405, ← c112406] at e_1840_4
  have e_1840_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_5
  simp only [← c112407, ← c112408, k2, ← c112409] at e_1840_5
  have hr := e_1840_5
  simp only [e_1840_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4584)) := by
  have c112410 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112411 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112412 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112413 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112414 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112415 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_6
  simp only [← c112410, ← c112411, ← c112412] at e_1840_6
  have e_1840_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_7
  simp only [← c112413, ← c112414, k2, ← c112415] at e_1840_7
  have hr := e_1840_7
  simp only [e_1840_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4585)) := by
  have c112416 := (publicBatchWrapper89_copies3513 a h).1
  have c112417 := (publicBatchWrapper89_copies3513 a h).2.1
  have c112418 := (publicBatchWrapper89_copies3513 a h).2.2.1
  have c112419 := (publicBatchWrapper89_copies3513 a h).2.2.2.1
  have c112420 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.1
  have c112421 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_8
  simp only [← c112416, ← c112417, ← c112418] at e_1840_8
  have e_1840_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_9
  simp only [← c112419, ← c112420, k2, ← c112421] at e_1840_9
  have hr := e_1840_9
  simp only [e_1840_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4586)) := by
  have c112422 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.1
  have c112423 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.1
  have c112424 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.1
  have c112425 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.1
  have c112426 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112427 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_10
  simp only [← c112422, ← c112423, ← c112424] at e_1840_10
  have e_1840_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_11
  simp only [← c112425, ← c112426, k2, ← c112427] at e_1840_11
  have hr := e_1840_11
  simp only [e_1840_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4587)) := by
  have c112428 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112429 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112430 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112431 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112432 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112433 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_12
  simp only [← c112428, ← c112429, ← c112430] at e_1840_12
  have e_1840_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_13
  simp only [← c112431, ← c112432, k2, ← c112433] at e_1840_13
  have hr := e_1840_13
  simp only [e_1840_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4588)) := by
  have c112434 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112435 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112436 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112437 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112438 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112439 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_14
  simp only [← c112434, ← c112435, ← c112436] at e_1840_14
  have e_1840_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_15
  simp only [← c112437, ← c112438, k2, ← c112439] at e_1840_15
  have hr := e_1840_15
  simp only [e_1840_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4589)) := by
  have c112440 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112441 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112442 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112443 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112444 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112445 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_16
  simp only [← c112440, ← c112441, ← c112442] at e_1840_16
  have e_1840_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_17
  simp only [← c112443, ← c112444, k2, ← c112445] at e_1840_17
  have hr := e_1840_17
  simp only [e_1840_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4590)) := by
  have c112446 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112447 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112448 := (publicBatchWrapper89_copies3514 a h).1
  have c112449 := (publicBatchWrapper89_copies3514 a h).2.1
  have c112450 := (publicBatchWrapper89_copies3514 a h).2.2.1
  have c112451 := (publicBatchWrapper89_copies3514 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_18
  simp only [← c112446, ← c112447, ← c112448] at e_1840_18
  have e_1840_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_19
  simp only [← c112449, ← c112450, k2, ← c112451] at e_1840_19
  have hr := e_1840_19
  simp only [e_1840_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4591)) := by
  have c112452 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.1
  have c112453 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.1
  have c112454 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.1
  have c112455 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.1
  have c112456 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.1
  have c112457 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_0
  simp only [← c112452, ← c112453, ← c112454] at e_1841_0
  have e_1841_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_1
  simp only [← c112455, ← c112456, k2, ← c112457] at e_1841_1
  have hr := e_1841_1
  simp only [e_1841_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4592)) := by
  have c112458 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112459 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112460 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112461 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112462 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112463 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_2
  simp only [← c112458, ← c112459, ← c112460] at e_1841_2
  have e_1841_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_3
  simp only [← c112461, ← c112462, k2, ← c112463] at e_1841_3
  have hr := e_1841_3
  simp only [e_1841_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4593)) := by
  have c112464 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112465 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112466 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112467 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112468 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112469 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_4
  simp only [← c112464, ← c112465, ← c112466] at e_1841_4
  have e_1841_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_5
  simp only [← c112467, ← c112468, k2, ← c112469] at e_1841_5
  have hr := e_1841_5
  simp only [e_1841_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4594)) := by
  have c112470 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112471 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112472 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112473 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112474 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112475 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_6
  simp only [← c112470, ← c112471, ← c112472] at e_1841_6
  have e_1841_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_7
  simp only [← c112473, ← c112474, k2, ← c112475] at e_1841_7
  have hr := e_1841_7
  simp only [e_1841_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4595)) := by
  have c112476 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112477 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112478 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112479 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112480 := (publicBatchWrapper89_copies3515 a h).1
  have c112481 := (publicBatchWrapper89_copies3515 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_8
  simp only [← c112476, ← c112477, ← c112478] at e_1841_8
  have e_1841_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_9
  simp only [← c112479, ← c112480, k2, ← c112481] at e_1841_9
  have hr := e_1841_9
  simp only [e_1841_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4596)) := by
  have c112482 := (publicBatchWrapper89_copies3515 a h).2.2.1
  have c112483 := (publicBatchWrapper89_copies3515 a h).2.2.2.1
  have c112484 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.1
  have c112485 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.1
  have c112486 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.1
  have c112487 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_10
  simp only [← c112482, ← c112483, ← c112484] at e_1841_10
  have e_1841_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_11
  simp only [← c112485, ← c112486, k2, ← c112487] at e_1841_11
  have hr := e_1841_11
  simp only [e_1841_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4597)) := by
  have c112488 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.1
  have c112489 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.1
  have c112490 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112491 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112492 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112493 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_12
  simp only [← c112488, ← c112489, ← c112490] at e_1841_12
  have e_1841_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_13
  simp only [← c112491, ← c112492, k2, ← c112493] at e_1841_13
  have hr := e_1841_13
  simp only [e_1841_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4598)) := by
  have c112494 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112495 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112496 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112497 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112498 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112499 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_14
  simp only [← c112494, ← c112495, ← c112496] at e_1841_14
  have e_1841_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_15
  simp only [← c112497, ← c112498, k2, ← c112499] at e_1841_15
  have hr := e_1841_15
  simp only [e_1841_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4599)) := by
  have c112500 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112501 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112502 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112503 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112504 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112505 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_16
  simp only [← c112500, ← c112501, ← c112502] at e_1841_16
  have e_1841_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_17
  simp only [← c112503, ← c112504, k2, ← c112505] at e_1841_17
  have hr := e_1841_17
  simp only [e_1841_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4600)) := by
  have c112506 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112507 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112508 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112509 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112510 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112511 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_18
  simp only [← c112506, ← c112507, ← c112508] at e_1841_18
  have e_1841_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_19
  simp only [← c112509, ← c112510, k2, ← c112511] at e_1841_19
  have hr := e_1841_19
  simp only [e_1841_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4601)) := by
  have c112512 := (publicBatchWrapper89_copies3516 a h).1
  have c112513 := (publicBatchWrapper89_copies3516 a h).2.1
  have c112514 := (publicBatchWrapper89_copies3516 a h).2.2.1
  have c112515 := (publicBatchWrapper89_copies3516 a h).2.2.2.1
  have c112516 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.1
  have c112517 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_0
  simp only [← c112512, ← c112513, ← c112514] at e_1842_0
  have e_1842_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_1
  simp only [← c112515, ← c112516, k2, ← c112517] at e_1842_1
  have hr := e_1842_1
  simp only [e_1842_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4770)) := by
  have c112518 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.1
  have c112519 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.1
  have c112520 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.1
  have c112521 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.1
  have c112522 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112523 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_2
  simp only [← c112518, ← c112519, ← c112520] at e_1842_2
  have e_1842_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_3
  simp only [← c112521, ← c112522, k2, ← c112523] at e_1842_3
  have hr := e_1842_3
  simp only [e_1842_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4771)) := by
  have c112524 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112525 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112526 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112527 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112528 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112529 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_4
  simp only [← c112524, ← c112525, ← c112526] at e_1842_4
  have e_1842_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_5
  simp only [← c112527, ← c112528, k2, ← c112529] at e_1842_5
  have hr := e_1842_5
  simp only [e_1842_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4772)) := by
  have c112530 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112531 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112532 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112533 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112534 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112535 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_6
  simp only [← c112530, ← c112531, ← c112532] at e_1842_6
  have e_1842_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_7
  simp only [← c112533, ← c112534, k2, ← c112535] at e_1842_7
  have hr := e_1842_7
  simp only [e_1842_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4773)) := by
  have c112536 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112537 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112538 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112539 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112540 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112541 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_8
  simp only [← c112536, ← c112537, ← c112538] at e_1842_8
  have e_1842_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_9
  simp only [← c112539, ← c112540, k2, ← c112541] at e_1842_9
  have hr := e_1842_9
  simp only [e_1842_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4774)) := by
  have c112542 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112543 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112544 := (publicBatchWrapper89_copies3517 a h).1
  have c112545 := (publicBatchWrapper89_copies3517 a h).2.1
  have c112546 := (publicBatchWrapper89_copies3517 a h).2.2.1
  have c112547 := (publicBatchWrapper89_copies3517 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_10
  simp only [← c112542, ← c112543, ← c112544] at e_1842_10
  have e_1842_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_11
  simp only [← c112545, ← c112546, k2, ← c112547] at e_1842_11
  have hr := e_1842_11
  simp only [e_1842_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4775)) := by
  have c112548 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.1
  have c112549 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.1
  have c112550 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.1
  have c112551 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.1
  have c112552 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.1
  have c112553 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_12
  simp only [← c112548, ← c112549, ← c112550] at e_1842_12
  have e_1842_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_13
  simp only [← c112551, ← c112552, k2, ← c112553] at e_1842_13
  have hr := e_1842_13
  simp only [e_1842_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4776)) := by
  have c112554 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112555 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112556 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112557 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112558 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112559 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_14
  simp only [← c112554, ← c112555, ← c112556] at e_1842_14
  have e_1842_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_15
  simp only [← c112557, ← c112558, k2, ← c112559] at e_1842_15
  have hr := e_1842_15
  simp only [e_1842_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4777)) := by
  have c112560 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112561 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112562 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112563 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112564 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112565 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_16
  simp only [← c112560, ← c112561, ← c112562] at e_1842_16
  have e_1842_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_17
  simp only [← c112563, ← c112564, k2, ← c112565] at e_1842_17
  have hr := e_1842_17
  simp only [e_1842_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4778)) := by
  have c112566 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112567 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112568 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112569 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112570 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112571 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_18
  simp only [← c112566, ← c112567, ← c112568] at e_1842_18
  have e_1842_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_19
  simp only [← c112569, ← c112570, k2, ← c112571] at e_1842_19
  have hr := e_1842_19
  simp only [e_1842_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4779)) := by
  have c112572 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112573 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112574 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112575 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112576 := (publicBatchWrapper89_copies3518 a h).1
  have c112577 := (publicBatchWrapper89_copies3518 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_0
  simp only [← c112572, ← c112573, ← c112574] at e_1843_0
  have e_1843_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_1
  simp only [← c112575, ← c112576, k2, ← c112577] at e_1843_1
  have hr := e_1843_1
  simp only [e_1843_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4780)) := by
  have c112578 := (publicBatchWrapper89_copies3518 a h).2.2.1
  have c112579 := (publicBatchWrapper89_copies3518 a h).2.2.2.1
  have c112580 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.1
  have c112581 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.1
  have c112582 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.1
  have c112583 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_2
  simp only [← c112578, ← c112579, ← c112580] at e_1843_2
  have e_1843_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_3
  simp only [← c112581, ← c112582, k2, ← c112583] at e_1843_3
  have hr := e_1843_3
  simp only [e_1843_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4781)) := by
  have c112584 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.1
  have c112585 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.1
  have c112586 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112587 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112588 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112589 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_4
  simp only [← c112584, ← c112585, ← c112586] at e_1843_4
  have e_1843_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_5
  simp only [← c112587, ← c112588, k2, ← c112589] at e_1843_5
  have hr := e_1843_5
  simp only [e_1843_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4782)) := by
  have c112590 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112591 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112592 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112593 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112594 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112595 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_6
  simp only [← c112590, ← c112591, ← c112592] at e_1843_6
  have e_1843_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_7
  simp only [← c112593, ← c112594, k2, ← c112595] at e_1843_7
  have hr := e_1843_7
  simp only [e_1843_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4783)) := by
  have c112596 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112597 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112598 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112599 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112600 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112601 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_8
  simp only [← c112596, ← c112597, ← c112598] at e_1843_8
  have e_1843_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_9
  simp only [← c112599, ← c112600, k2, ← c112601] at e_1843_9
  have hr := e_1843_9
  simp only [e_1843_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4784)) := by
  have c112602 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112603 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112604 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112605 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112606 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112607 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_10
  simp only [← c112602, ← c112603, ← c112604] at e_1843_10
  have e_1843_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_11
  simp only [← c112605, ← c112606, k2, ← c112607] at e_1843_11
  have hr := e_1843_11
  simp only [e_1843_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4785)) := by
  have c112608 := (publicBatchWrapper89_copies3519 a h).1
  have c112609 := (publicBatchWrapper89_copies3519 a h).2.1
  have c112610 := (publicBatchWrapper89_copies3519 a h).2.2.1
  have c112611 := (publicBatchWrapper89_copies3519 a h).2.2.2.1
  have c112612 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.1
  have c112613 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_12
  simp only [← c112608, ← c112609, ← c112610] at e_1843_12
  have e_1843_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_13
  simp only [← c112611, ← c112612, k2, ← c112613] at e_1843_13
  have hr := e_1843_13
  simp only [e_1843_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4786)) := by
  have c112614 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.1
  have c112615 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.1
  have c112616 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.1
  have c112617 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.1
  have c112618 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112619 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_14
  simp only [← c112614, ← c112615, ← c112616] at e_1843_14
  have e_1843_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_15
  simp only [← c112617, ← c112618, k2, ← c112619] at e_1843_15
  have hr := e_1843_15
  simp only [e_1843_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4787)) := by
  have c112620 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112621 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112622 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112623 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112624 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112625 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_16
  simp only [← c112620, ← c112621, ← c112622] at e_1843_16
  have e_1843_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_17
  simp only [← c112623, ← c112624, k2, ← c112625] at e_1843_17
  have hr := e_1843_17
  simp only [e_1843_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4788)) := by
  have c112626 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112627 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112628 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112629 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112630 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112631 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_18
  simp only [← c112626, ← c112627, ← c112628] at e_1843_18
  have e_1843_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_19
  simp only [← c112629, ← c112630, k2, ← c112631] at e_1843_19
  have hr := e_1843_19
  simp only [e_1843_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4789)) := by
  have c112632 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112633 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112634 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112635 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112636 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112637 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_0
  simp only [← c112632, ← c112633, ← c112634] at e_1844_0
  have e_1844_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_1
  simp only [← c112635, ← c112636, k2, ← c112637] at e_1844_1
  have hr := e_1844_1
  simp only [e_1844_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4790)) := by
  have c112638 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112639 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112640 := (publicBatchWrapper89_copies3520 a h).1
  have c112641 := (publicBatchWrapper89_copies3520 a h).2.1
  have c112642 := (publicBatchWrapper89_copies3520 a h).2.2.1
  have c112643 := (publicBatchWrapper89_copies3520 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_2
  simp only [← c112638, ← c112639, ← c112640] at e_1844_2
  have e_1844_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_3
  simp only [← c112641, ← c112642, k2, ← c112643] at e_1844_3
  have hr := e_1844_3
  simp only [e_1844_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4791)) := by
  have c112644 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.1
  have c112645 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.1
  have c112646 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.1
  have c112647 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.1
  have c112648 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.1
  have c112649 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_4
  simp only [← c112644, ← c112645, ← c112646] at e_1844_4
  have e_1844_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_5
  simp only [← c112647, ← c112648, k2, ← c112649] at e_1844_5
  have hr := e_1844_5
  simp only [e_1844_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4792)) := by
  have c112650 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112651 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112652 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112653 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112654 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112655 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_6
  simp only [← c112650, ← c112651, ← c112652] at e_1844_6
  have e_1844_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_7
  simp only [← c112653, ← c112654, k2, ← c112655] at e_1844_7
  have hr := e_1844_7
  simp only [e_1844_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4793)) := by
  have c112656 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112657 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112658 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112659 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112660 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112661 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_8
  simp only [← c112656, ← c112657, ← c112658] at e_1844_8
  have e_1844_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_9
  simp only [← c112659, ← c112660, k2, ← c112661] at e_1844_9
  have hr := e_1844_9
  simp only [e_1844_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4794)) := by
  have c112662 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112663 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112664 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112665 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112666 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112667 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_10
  simp only [← c112662, ← c112663, ← c112664] at e_1844_10
  have e_1844_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_11
  simp only [← c112665, ← c112666, k2, ← c112667] at e_1844_11
  have hr := e_1844_11
  simp only [e_1844_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4795)) := by
  have c112668 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112669 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112670 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112671 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112672 := (publicBatchWrapper89_copies3521 a h).1
  have c112673 := (publicBatchWrapper89_copies3521 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_12
  simp only [← c112668, ← c112669, ← c112670] at e_1844_12
  have e_1844_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_13
  simp only [← c112671, ← c112672, k2, ← c112673] at e_1844_13
  have hr := e_1844_13
  simp only [e_1844_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4796)) := by
  have c112674 := (publicBatchWrapper89_copies3521 a h).2.2.1
  have c112675 := (publicBatchWrapper89_copies3521 a h).2.2.2.1
  have c112676 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.1
  have c112677 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.1
  have c112678 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.1
  have c112679 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_14
  simp only [← c112674, ← c112675, ← c112676] at e_1844_14
  have e_1844_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_15
  simp only [← c112677, ← c112678, k2, ← c112679] at e_1844_15
  have hr := e_1844_15
  simp only [e_1844_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4797)) := by
  have c112680 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.1
  have c112681 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.1
  have c112682 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112683 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112684 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112685 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_16
  simp only [← c112680, ← c112681, ← c112682] at e_1844_16
  have e_1844_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_17
  simp only [← c112683, ← c112684, k2, ← c112685] at e_1844_17
  have hr := e_1844_17
  simp only [e_1844_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4798)) := by
  have c112686 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112687 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112688 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112689 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112690 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112691 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_18
  simp only [← c112686, ← c112687, ← c112688] at e_1844_18
  have e_1844_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_19
  simp only [← c112689, ← c112690, k2, ← c112691] at e_1844_19
  have hr := e_1844_19
  simp only [e_1844_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4799)) := by
  have c112692 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112693 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112694 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112695 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112696 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112697 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_0
  simp only [← c112692, ← c112693, ← c112694] at e_1845_0
  have e_1845_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_1
  simp only [← c112695, ← c112696, k2, ← c112697] at e_1845_1
  have hr := e_1845_1
  simp only [e_1845_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4800)) := by
  have c112698 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112699 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112700 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112701 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112702 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112703 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_2
  simp only [← c112698, ← c112699, ← c112700] at e_1845_2
  have e_1845_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_3
  simp only [← c112701, ← c112702, k2, ← c112703] at e_1845_3
  have hr := e_1845_3
  simp only [e_1845_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4801)) := by
  have c112704 := (publicBatchWrapper89_copies3522 a h).1
  have c112705 := (publicBatchWrapper89_copies3522 a h).2.1
  have c112706 := (publicBatchWrapper89_copies3522 a h).2.2.1
  have c112707 := (publicBatchWrapper89_copies3522 a h).2.2.2.1
  have c112708 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.1
  have c112709 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_4
  simp only [← c112704, ← c112705, ← c112706] at e_1845_4
  have e_1845_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_5
  simp only [← c112707, ← c112708, k2, ← c112709] at e_1845_5
  have hr := e_1845_5
  simp only [e_1845_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4970)) := by
  have c112710 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.1
  have c112711 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.1
  have c112712 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.1
  have c112713 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.1
  have c112714 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112715 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_6
  simp only [← c112710, ← c112711, ← c112712] at e_1845_6
  have e_1845_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_7
  simp only [← c112713, ← c112714, k2, ← c112715] at e_1845_7
  have hr := e_1845_7
  simp only [e_1845_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4971)) := by
  have c112716 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112717 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112718 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112719 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112720 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112721 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_8
  simp only [← c112716, ← c112717, ← c112718] at e_1845_8
  have e_1845_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_9
  simp only [← c112719, ← c112720, k2, ← c112721] at e_1845_9
  have hr := e_1845_9
  simp only [e_1845_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4972)) := by
  have c112722 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112723 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112724 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112725 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112726 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112727 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_10
  simp only [← c112722, ← c112723, ← c112724] at e_1845_10
  have e_1845_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_11
  simp only [← c112725, ← c112726, k2, ← c112727] at e_1845_11
  have hr := e_1845_11
  simp only [e_1845_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4973)) := by
  have c112728 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112729 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112730 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112731 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112732 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112733 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_12
  simp only [← c112728, ← c112729, ← c112730] at e_1845_12
  have e_1845_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_13
  simp only [← c112731, ← c112732, k2, ← c112733] at e_1845_13
  have hr := e_1845_13
  simp only [e_1845_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4974)) := by
  have c112734 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112735 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112736 := (publicBatchWrapper89_copies3523 a h).1
  have c112737 := (publicBatchWrapper89_copies3523 a h).2.1
  have c112738 := (publicBatchWrapper89_copies3523 a h).2.2.1
  have c112739 := (publicBatchWrapper89_copies3523 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_14
  simp only [← c112734, ← c112735, ← c112736] at e_1845_14
  have e_1845_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_15
  simp only [← c112737, ← c112738, k2, ← c112739] at e_1845_15
  have hr := e_1845_15
  simp only [e_1845_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4975)) := by
  have c112740 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.1
  have c112741 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.1
  have c112742 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.1
  have c112743 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.1
  have c112744 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.1
  have c112745 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_16
  simp only [← c112740, ← c112741, ← c112742] at e_1845_16
  have e_1845_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_17
  simp only [← c112743, ← c112744, k2, ← c112745] at e_1845_17
  have hr := e_1845_17
  simp only [e_1845_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4976)) := by
  have c112746 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112747 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112748 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112749 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112750 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112751 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_18
  simp only [← c112746, ← c112747, ← c112748] at e_1845_18
  have e_1845_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_19
  simp only [← c112749, ← c112750, k2, ← c112751] at e_1845_19
  have hr := e_1845_19
  simp only [e_1845_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4977)) := by
  have c112752 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112753 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112754 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112755 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112756 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112757 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_0
  simp only [← c112752, ← c112753, ← c112754] at e_1846_0
  have e_1846_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_1
  simp only [← c112755, ← c112756, k2, ← c112757] at e_1846_1
  have hr := e_1846_1
  simp only [e_1846_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4978)) := by
  have c112758 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112759 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112760 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112761 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112762 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112763 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_2
  simp only [← c112758, ← c112759, ← c112760] at e_1846_2
  have e_1846_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_3
  simp only [← c112761, ← c112762, k2, ← c112763] at e_1846_3
  have hr := e_1846_3
  simp only [e_1846_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4979)) := by
  have c112764 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112765 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112766 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112767 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112768 := (publicBatchWrapper89_copies3524 a h).1
  have c112769 := (publicBatchWrapper89_copies3524 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_4
  simp only [← c112764, ← c112765, ← c112766] at e_1846_4
  have e_1846_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_5
  simp only [← c112767, ← c112768, k2, ← c112769] at e_1846_5
  have hr := e_1846_5
  simp only [e_1846_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4980)) := by
  have c112770 := (publicBatchWrapper89_copies3524 a h).2.2.1
  have c112771 := (publicBatchWrapper89_copies3524 a h).2.2.2.1
  have c112772 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.1
  have c112773 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.1
  have c112774 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.1
  have c112775 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_6
  simp only [← c112770, ← c112771, ← c112772] at e_1846_6
  have e_1846_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_7
  simp only [← c112773, ← c112774, k2, ← c112775] at e_1846_7
  have hr := e_1846_7
  simp only [e_1846_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4981)) := by
  have c112776 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.1
  have c112777 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.1
  have c112778 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112779 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112780 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112781 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_8
  simp only [← c112776, ← c112777, ← c112778] at e_1846_8
  have e_1846_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_9
  simp only [← c112779, ← c112780, k2, ← c112781] at e_1846_9
  have hr := e_1846_9
  simp only [e_1846_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4982)) := by
  have c112782 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112783 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112784 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112785 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112786 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112787 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_10
  simp only [← c112782, ← c112783, ← c112784] at e_1846_10
  have e_1846_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_11
  simp only [← c112785, ← c112786, k2, ← c112787] at e_1846_11
  have hr := e_1846_11
  simp only [e_1846_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4983)) := by
  have c112788 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112789 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112790 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112791 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112792 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112793 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_12
  simp only [← c112788, ← c112789, ← c112790] at e_1846_12
  have e_1846_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_13
  simp only [← c112791, ← c112792, k2, ← c112793] at e_1846_13
  have hr := e_1846_13
  simp only [e_1846_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4984)) := by
  have c112794 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112795 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112796 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112797 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112798 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112799 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_14
  simp only [← c112794, ← c112795, ← c112796] at e_1846_14
  have e_1846_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_15
  simp only [← c112797, ← c112798, k2, ← c112799] at e_1846_15
  have hr := e_1846_15
  simp only [e_1846_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4985)) := by
  have c112800 := (publicBatchWrapper89_copies3525 a h).1
  have c112801 := (publicBatchWrapper89_copies3525 a h).2.1
  have c112802 := (publicBatchWrapper89_copies3525 a h).2.2.1
  have c112803 := (publicBatchWrapper89_copies3525 a h).2.2.2.1
  have c112804 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.1
  have c112805 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_16
  simp only [← c112800, ← c112801, ← c112802] at e_1846_16
  have e_1846_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_17
  simp only [← c112803, ← c112804, k2, ← c112805] at e_1846_17
  have hr := e_1846_17
  simp only [e_1846_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4986)) := by
  have c112806 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.1
  have c112807 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.1
  have c112808 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.1
  have c112809 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.1
  have c112810 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112811 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_18
  simp only [← c112806, ← c112807, ← c112808] at e_1846_18
  have e_1846_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_19
  simp only [← c112809, ← c112810, k2, ← c112811] at e_1846_19
  have hr := e_1846_19
  simp only [e_1846_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4987)) := by
  have c112812 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112813 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112814 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112815 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112816 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112817 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_0
  simp only [← c112812, ← c112813, ← c112814] at e_1847_0
  have e_1847_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_1
  simp only [← c112815, ← c112816, k2, ← c112817] at e_1847_1
  have hr := e_1847_1
  simp only [e_1847_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4988)) := by
  have c112818 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112819 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112820 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112821 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112822 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112823 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_2
  simp only [← c112818, ← c112819, ← c112820] at e_1847_2
  have e_1847_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_3
  simp only [← c112821, ← c112822, k2, ← c112823] at e_1847_3
  have hr := e_1847_3
  simp only [e_1847_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4989)) := by
  have c112824 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112825 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112826 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112827 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112828 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112829 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_4
  simp only [← c112824, ← c112825, ← c112826] at e_1847_4
  have e_1847_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_5
  simp only [← c112827, ← c112828, k2, ← c112829] at e_1847_5
  have hr := e_1847_5
  simp only [e_1847_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4990)) := by
  have c112830 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112831 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112832 := (publicBatchWrapper89_copies3526 a h).1
  have c112833 := (publicBatchWrapper89_copies3526 a h).2.1
  have c112834 := (publicBatchWrapper89_copies3526 a h).2.2.1
  have c112835 := (publicBatchWrapper89_copies3526 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_6
  simp only [← c112830, ← c112831, ← c112832] at e_1847_6
  have e_1847_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_7
  simp only [← c112833, ← c112834, k2, ← c112835] at e_1847_7
  have hr := e_1847_7
  simp only [e_1847_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4991)) := by
  have c112836 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.1
  have c112837 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.1
  have c112838 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.1
  have c112839 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.1
  have c112840 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.1
  have c112841 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_8
  simp only [← c112836, ← c112837, ← c112838] at e_1847_8
  have e_1847_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_9
  simp only [← c112839, ← c112840, k2, ← c112841] at e_1847_9
  have hr := e_1847_9
  simp only [e_1847_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4992)) := by
  have c112842 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112843 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112844 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112845 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112846 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112847 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_10
  simp only [← c112842, ← c112843, ← c112844] at e_1847_10
  have e_1847_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_11
  simp only [← c112845, ← c112846, k2, ← c112847] at e_1847_11
  have hr := e_1847_11
  simp only [e_1847_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4993)) := by
  have c112848 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112849 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112850 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112851 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112852 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112853 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_12
  simp only [← c112848, ← c112849, ← c112850] at e_1847_12
  have e_1847_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_13
  simp only [← c112851, ← c112852, k2, ← c112853] at e_1847_13
  have hr := e_1847_13
  simp only [e_1847_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4994)) := by
  have c112854 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112855 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112856 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112857 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112858 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112859 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_14
  simp only [← c112854, ← c112855, ← c112856] at e_1847_14
  have e_1847_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_15
  simp only [← c112857, ← c112858, k2, ← c112859] at e_1847_15
  have hr := e_1847_15
  simp only [e_1847_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4995)) := by
  have c112860 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112861 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112862 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112863 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112864 := (publicBatchWrapper89_copies3527 a h).1
  have c112865 := (publicBatchWrapper89_copies3527 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_16
  simp only [← c112860, ← c112861, ← c112862] at e_1847_16
  have e_1847_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_17
  simp only [← c112863, ← c112864, k2, ← c112865] at e_1847_17
  have hr := e_1847_17
  simp only [e_1847_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4996)) := by
  have c112866 := (publicBatchWrapper89_copies3527 a h).2.2.1
  have c112867 := (publicBatchWrapper89_copies3527 a h).2.2.2.1
  have c112868 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.1
  have c112869 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.1
  have c112870 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.1
  have c112871 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_18
  simp only [← c112866, ← c112867, ← c112868] at e_1847_18
  have e_1847_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_19
  simp only [← c112869, ← c112870, k2, ← c112871] at e_1847_19
  have hr := e_1847_19
  simp only [e_1847_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4997)) := by
  have c112872 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.1
  have c112873 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.1
  have c112874 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112875 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112876 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112877 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_0
  simp only [← c112872, ← c112873, ← c112874] at e_1848_0
  have e_1848_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_1
  simp only [← c112875, ← c112876, k2, ← c112877] at e_1848_1
  have hr := e_1848_1
  simp only [e_1848_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4998)) := by
  have c112878 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112879 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112880 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112881 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112882 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112883 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_2
  simp only [← c112878, ← c112879, ← c112880] at e_1848_2
  have e_1848_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_3
  simp only [← c112881, ← c112882, k2, ← c112883] at e_1848_3
  have hr := e_1848_3
  simp only [e_1848_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4999)) := by
  have c112884 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112885 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112886 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112887 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112888 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112889 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_4
  simp only [← c112884, ← c112885, ← c112886] at e_1848_4
  have e_1848_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_5
  simp only [← c112887, ← c112888, k2, ← c112889] at e_1848_5
  have hr := e_1848_5
  simp only [e_1848_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 5000)) := by
  have c112890 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112891 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112892 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112893 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112894 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112895 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_6
  simp only [← c112890, ← c112891, ← c112892] at e_1848_6
  have e_1848_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_7
  simp only [← c112893, ← c112894, k2, ← c112895] at e_1848_7
  have hr := e_1848_7
  simp only [e_1848_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 5001)) := by
  have c112896 := (publicBatchWrapper89_copies3528 a h).1
  have c112897 := (publicBatchWrapper89_copies3528 a h).2.1
  have c112898 := (publicBatchWrapper89_copies3528 a h).2.2.1
  have c112899 := (publicBatchWrapper89_copies3528 a h).2.2.2.1
  have c112900 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.1
  have c112901 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_8
  simp only [← c112896, ← c112897, ← c112898] at e_1848_8
  have e_1848_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_9
  simp only [← c112899, ← c112900, k2, ← c112901] at e_1848_9
  have hr := e_1848_9
  simp only [e_1848_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5170)) := by
  have c112902 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.1
  have c112903 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.1
  have c112904 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.1
  have c112905 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.1
  have c112906 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112907 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_10
  simp only [← c112902, ← c112903, ← c112904] at e_1848_10
  have e_1848_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_11
  simp only [← c112905, ← c112906, k2, ← c112907] at e_1848_11
  have hr := e_1848_11
  simp only [e_1848_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5171)) := by
  have c112908 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112909 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112910 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112911 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112912 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112913 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_12
  simp only [← c112908, ← c112909, ← c112910] at e_1848_12
  have e_1848_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_13
  simp only [← c112911, ← c112912, k2, ← c112913] at e_1848_13
  have hr := e_1848_13
  simp only [e_1848_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5172)) := by
  have c112914 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112915 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112916 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112917 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112918 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112919 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_14
  simp only [← c112914, ← c112915, ← c112916] at e_1848_14
  have e_1848_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_15
  simp only [← c112917, ← c112918, k2, ← c112919] at e_1848_15
  have hr := e_1848_15
  simp only [e_1848_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5173)) := by
  have c112920 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112921 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112922 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112923 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112924 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112925 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_16
  simp only [← c112920, ← c112921, ← c112922] at e_1848_16
  have e_1848_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_17
  simp only [← c112923, ← c112924, k2, ← c112925] at e_1848_17
  have hr := e_1848_17
  simp only [e_1848_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5174)) := by
  have c112926 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112927 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112928 := (publicBatchWrapper89_copies3529 a h).1
  have c112929 := (publicBatchWrapper89_copies3529 a h).2.1
  have c112930 := (publicBatchWrapper89_copies3529 a h).2.2.1
  have c112931 := (publicBatchWrapper89_copies3529 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_18
  simp only [← c112926, ← c112927, ← c112928] at e_1848_18
  have e_1848_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_19
  simp only [← c112929, ← c112930, k2, ← c112931] at e_1848_19
  have hr := e_1848_19
  simp only [e_1848_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5175)) := by
  have c112932 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.1
  have c112933 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.1
  have c112934 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.1
  have c112935 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.1
  have c112936 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.1
  have c112937 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_0
  simp only [← c112932, ← c112933, ← c112934] at e_1849_0
  have e_1849_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_1
  simp only [← c112935, ← c112936, k2, ← c112937] at e_1849_1
  have hr := e_1849_1
  simp only [e_1849_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5176)) := by
  have c112938 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112939 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112940 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112941 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112942 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112943 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_2
  simp only [← c112938, ← c112939, ← c112940] at e_1849_2
  have e_1849_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_3
  simp only [← c112941, ← c112942, k2, ← c112943] at e_1849_3
  have hr := e_1849_3
  simp only [e_1849_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5177)) := by
  have c112944 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112945 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112946 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112947 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112948 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112949 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_4
  simp only [← c112944, ← c112945, ← c112946] at e_1849_4
  have e_1849_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_5
  simp only [← c112947, ← c112948, k2, ← c112949] at e_1849_5
  have hr := e_1849_5
  simp only [e_1849_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5178)) := by
  have c112950 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112951 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112952 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112953 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112954 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112955 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_6
  simp only [← c112950, ← c112951, ← c112952] at e_1849_6
  have e_1849_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_7
  simp only [← c112953, ← c112954, k2, ← c112955] at e_1849_7
  have hr := e_1849_7
  simp only [e_1849_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5179)) := by
  have c112956 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112957 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112958 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112959 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112960 := (publicBatchWrapper89_copies3530 a h).1
  have c112961 := (publicBatchWrapper89_copies3530 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_8
  simp only [← c112956, ← c112957, ← c112958] at e_1849_8
  have e_1849_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_9
  simp only [← c112959, ← c112960, k2, ← c112961] at e_1849_9
  have hr := e_1849_9
  simp only [e_1849_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5180)) := by
  have c112962 := (publicBatchWrapper89_copies3530 a h).2.2.1
  have c112963 := (publicBatchWrapper89_copies3530 a h).2.2.2.1
  have c112964 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.1
  have c112965 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.1
  have c112966 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.1
  have c112967 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_10
  simp only [← c112962, ← c112963, ← c112964] at e_1849_10
  have e_1849_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_11
  simp only [← c112965, ← c112966, k2, ← c112967] at e_1849_11
  have hr := e_1849_11
  simp only [e_1849_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5181)) := by
  have c112968 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.1
  have c112969 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.1
  have c112970 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112971 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112972 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112973 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_12
  simp only [← c112968, ← c112969, ← c112970] at e_1849_12
  have e_1849_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_13
  simp only [← c112971, ← c112972, k2, ← c112973] at e_1849_13
  have hr := e_1849_13
  simp only [e_1849_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5182)) := by
  have c112974 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112975 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112976 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112977 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112978 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112979 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_14
  simp only [← c112974, ← c112975, ← c112976] at e_1849_14
  have e_1849_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_15
  simp only [← c112977, ← c112978, k2, ← c112979] at e_1849_15
  have hr := e_1849_15
  simp only [e_1849_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5183)) := by
  have c112980 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112981 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112982 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112983 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112984 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112985 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_16
  simp only [← c112980, ← c112981, ← c112982] at e_1849_16
  have e_1849_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_17
  simp only [← c112983, ← c112984, k2, ← c112985] at e_1849_17
  have hr := e_1849_17
  simp only [e_1849_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5184)) := by
  have c112986 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112987 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112988 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112989 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112990 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112991 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_18
  simp only [← c112986, ← c112987, ← c112988] at e_1849_18
  have e_1849_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_19
  simp only [← c112989, ← c112990, k2, ← c112991] at e_1849_19
  have hr := e_1849_19
  simp only [e_1849_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5185)) := by
  have c112992 := (publicBatchWrapper89_copies3531 a h).1
  have c112993 := (publicBatchWrapper89_copies3531 a h).2.1
  have c112994 := (publicBatchWrapper89_copies3531 a h).2.2.1
  have c112995 := (publicBatchWrapper89_copies3531 a h).2.2.2.1
  have c112996 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.1
  have c112997 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_0
  simp only [← c112992, ← c112993, ← c112994] at e_1850_0
  have e_1850_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_1
  simp only [← c112995, ← c112996, k2, ← c112997] at e_1850_1
  have hr := e_1850_1
  simp only [e_1850_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5186)) := by
  have c112998 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.1
  have c112999 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.1
  have c113000 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.1
  have c113001 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.1
  have c113002 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113003 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_2
  simp only [← c112998, ← c112999, ← c113000] at e_1850_2
  have e_1850_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_3
  simp only [← c113001, ← c113002, k2, ← c113003] at e_1850_3
  have hr := e_1850_3
  simp only [e_1850_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5187)) := by
  have c113004 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113005 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113006 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113007 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113008 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113009 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_4
  simp only [← c113004, ← c113005, ← c113006] at e_1850_4
  have e_1850_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_5
  simp only [← c113007, ← c113008, k2, ← c113009] at e_1850_5
  have hr := e_1850_5
  simp only [e_1850_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5188)) := by
  have c113010 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113011 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113012 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113013 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113014 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113015 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_6
  simp only [← c113010, ← c113011, ← c113012] at e_1850_6
  have e_1850_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_7
  simp only [← c113013, ← c113014, k2, ← c113015] at e_1850_7
  have hr := e_1850_7
  simp only [e_1850_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5189)) := by
  have c113016 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113017 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113018 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113019 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113020 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113021 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_8
  simp only [← c113016, ← c113017, ← c113018] at e_1850_8
  have e_1850_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_9
  simp only [← c113019, ← c113020, k2, ← c113021] at e_1850_9
  have hr := e_1850_9
  simp only [e_1850_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5190)) := by
  have c113022 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113023 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113024 := (publicBatchWrapper89_copies3532 a h).1
  have c113025 := (publicBatchWrapper89_copies3532 a h).2.1
  have c113026 := (publicBatchWrapper89_copies3532 a h).2.2.1
  have c113027 := (publicBatchWrapper89_copies3532 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_10
  simp only [← c113022, ← c113023, ← c113024] at e_1850_10
  have e_1850_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_11
  simp only [← c113025, ← c113026, k2, ← c113027] at e_1850_11
  have hr := e_1850_11
  simp only [e_1850_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5191)) := by
  have c113028 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.1
  have c113029 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.1
  have c113030 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.1
  have c113031 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.1
  have c113032 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.1
  have c113033 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_12
  simp only [← c113028, ← c113029, ← c113030] at e_1850_12
  have e_1850_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_13
  simp only [← c113031, ← c113032, k2, ← c113033] at e_1850_13
  have hr := e_1850_13
  simp only [e_1850_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5192)) := by
  have c113034 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113035 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113036 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113037 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113038 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113039 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_14
  simp only [← c113034, ← c113035, ← c113036] at e_1850_14
  have e_1850_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_15
  simp only [← c113037, ← c113038, k2, ← c113039] at e_1850_15
  have hr := e_1850_15
  simp only [e_1850_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5193)) := by
  have c113040 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113041 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113042 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113043 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113044 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113045 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_16
  simp only [← c113040, ← c113041, ← c113042] at e_1850_16
  have e_1850_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_17
  simp only [← c113043, ← c113044, k2, ← c113045] at e_1850_17
  have hr := e_1850_17
  simp only [e_1850_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5194)) := by
  have c113046 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113047 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113048 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113049 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113050 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113051 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_18
  simp only [← c113046, ← c113047, ← c113048] at e_1850_18
  have e_1850_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_19
  simp only [← c113049, ← c113050, k2, ← c113051] at e_1850_19
  have hr := e_1850_19
  simp only [e_1850_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5195)) := by
  have c113052 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113053 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113054 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113055 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113056 := (publicBatchWrapper89_copies3533 a h).1
  have c113057 := (publicBatchWrapper89_copies3533 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_0
  simp only [← c113052, ← c113053, ← c113054] at e_1851_0
  have e_1851_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_1
  simp only [← c113055, ← c113056, k2, ← c113057] at e_1851_1
  have hr := e_1851_1
  simp only [e_1851_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5196)) := by
  have c113058 := (publicBatchWrapper89_copies3533 a h).2.2.1
  have c113059 := (publicBatchWrapper89_copies3533 a h).2.2.2.1
  have c113060 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.1
  have c113061 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.1
  have c113062 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.1
  have c113063 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_2
  simp only [← c113058, ← c113059, ← c113060] at e_1851_2
  have e_1851_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_3
  simp only [← c113061, ← c113062, k2, ← c113063] at e_1851_3
  have hr := e_1851_3
  simp only [e_1851_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5197)) := by
  have c113064 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.1
  have c113065 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.1
  have c113066 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113067 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113068 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113069 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_4
  simp only [← c113064, ← c113065, ← c113066] at e_1851_4
  have e_1851_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_5
  simp only [← c113067, ← c113068, k2, ← c113069] at e_1851_5
  have hr := e_1851_5
  simp only [e_1851_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5198)) := by
  have c113070 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113071 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113072 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113073 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113074 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113075 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_6
  simp only [← c113070, ← c113071, ← c113072] at e_1851_6
  have e_1851_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_7
  simp only [← c113073, ← c113074, k2, ← c113075] at e_1851_7
  have hr := e_1851_7
  simp only [e_1851_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5199)) := by
  have c113076 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113077 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113078 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113079 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113080 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113081 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_8
  simp only [← c113076, ← c113077, ← c113078] at e_1851_8
  have e_1851_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_9
  simp only [← c113079, ← c113080, k2, ← c113081] at e_1851_9
  have hr := e_1851_9
  simp only [e_1851_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5200)) := by
  have c113082 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113083 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113084 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113085 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113086 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113087 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_10
  simp only [← c113082, ← c113083, ← c113084] at e_1851_10
  have e_1851_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_11
  simp only [← c113085, ← c113086, k2, ← c113087] at e_1851_11
  have hr := e_1851_11
  simp only [e_1851_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5201)) := by
  have c113088 := (publicBatchWrapper89_copies3534 a h).1
  have c113089 := (publicBatchWrapper89_copies3534 a h).2.1
  have c113090 := (publicBatchWrapper89_copies3534 a h).2.2.1
  have c113091 := (publicBatchWrapper89_copies3534 a h).2.2.2.1
  have c113092 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.1
  have c113093 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_12
  simp only [← c113088, ← c113089, ← c113090] at e_1851_12
  have e_1851_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_13
  simp only [← c113091, ← c113092, k2, ← c113093] at e_1851_13
  have hr := e_1851_13
  simp only [e_1851_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5370)) := by
  have c113094 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.1
  have c113095 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.1
  have c113096 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.1
  have c113097 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.1
  have c113098 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113099 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_14
  simp only [← c113094, ← c113095, ← c113096] at e_1851_14
  have e_1851_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_15
  simp only [← c113097, ← c113098, k2, ← c113099] at e_1851_15
  have hr := e_1851_15
  simp only [e_1851_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5371)) := by
  have c113100 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113101 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113102 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113103 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113104 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113105 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_16
  simp only [← c113100, ← c113101, ← c113102] at e_1851_16
  have e_1851_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_17
  simp only [← c113103, ← c113104, k2, ← c113105] at e_1851_17
  have hr := e_1851_17
  simp only [e_1851_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5372)) := by
  have c113106 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113107 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113108 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113109 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113110 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113111 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_18
  simp only [← c113106, ← c113107, ← c113108] at e_1851_18
  have e_1851_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_19
  simp only [← c113109, ← c113110, k2, ← c113111] at e_1851_19
  have hr := e_1851_19
  simp only [e_1851_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5373)) := by
  have c113112 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113113 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113114 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113115 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113116 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113117 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_0
  simp only [← c113112, ← c113113, ← c113114] at e_1852_0
  have e_1852_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_1
  simp only [← c113115, ← c113116, k2, ← c113117] at e_1852_1
  have hr := e_1852_1
  simp only [e_1852_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5374)) := by
  have c113118 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113119 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113120 := (publicBatchWrapper89_copies3535 a h).1
  have c113121 := (publicBatchWrapper89_copies3535 a h).2.1
  have c113122 := (publicBatchWrapper89_copies3535 a h).2.2.1
  have c113123 := (publicBatchWrapper89_copies3535 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_2
  simp only [← c113118, ← c113119, ← c113120] at e_1852_2
  have e_1852_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_3
  simp only [← c113121, ← c113122, k2, ← c113123] at e_1852_3
  have hr := e_1852_3
  simp only [e_1852_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5375)) := by
  have c113124 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.1
  have c113125 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.1
  have c113126 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.1
  have c113127 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.1
  have c113128 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.1
  have c113129 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_4
  simp only [← c113124, ← c113125, ← c113126] at e_1852_4
  have e_1852_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_5
  simp only [← c113127, ← c113128, k2, ← c113129] at e_1852_5
  have hr := e_1852_5
  simp only [e_1852_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5376)) := by
  have c113130 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113131 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113132 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113133 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113134 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113135 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_6
  simp only [← c113130, ← c113131, ← c113132] at e_1852_6
  have e_1852_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_7
  simp only [← c113133, ← c113134, k2, ← c113135] at e_1852_7
  have hr := e_1852_7
  simp only [e_1852_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5377)) := by
  have c113136 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113137 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113138 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113139 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113140 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113141 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_8
  simp only [← c113136, ← c113137, ← c113138] at e_1852_8
  have e_1852_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_9
  simp only [← c113139, ← c113140, k2, ← c113141] at e_1852_9
  have hr := e_1852_9
  simp only [e_1852_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5378)) := by
  have c113142 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113143 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113144 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113145 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113146 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113147 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_10
  simp only [← c113142, ← c113143, ← c113144] at e_1852_10
  have e_1852_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_11
  simp only [← c113145, ← c113146, k2, ← c113147] at e_1852_11
  have hr := e_1852_11
  simp only [e_1852_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5379)) := by
  have c113148 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113149 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113150 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113151 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113152 := (publicBatchWrapper89_copies3536 a h).1
  have c113153 := (publicBatchWrapper89_copies3536 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_12
  simp only [← c113148, ← c113149, ← c113150] at e_1852_12
  have e_1852_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_13
  simp only [← c113151, ← c113152, k2, ← c113153] at e_1852_13
  have hr := e_1852_13
  simp only [e_1852_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5380)) := by
  have c113154 := (publicBatchWrapper89_copies3536 a h).2.2.1
  have c113155 := (publicBatchWrapper89_copies3536 a h).2.2.2.1
  have c113156 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.1
  have c113157 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.1
  have c113158 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.1
  have c113159 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_14
  simp only [← c113154, ← c113155, ← c113156] at e_1852_14
  have e_1852_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_15
  simp only [← c113157, ← c113158, k2, ← c113159] at e_1852_15
  have hr := e_1852_15
  simp only [e_1852_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5381)) := by
  have c113160 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.1
  have c113161 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.1
  have c113162 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113163 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113164 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113165 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_16
  simp only [← c113160, ← c113161, ← c113162] at e_1852_16
  have e_1852_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_17
  simp only [← c113163, ← c113164, k2, ← c113165] at e_1852_17
  have hr := e_1852_17
  simp only [e_1852_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5382)) := by
  have c113166 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113167 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113168 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113169 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113170 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113171 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_18
  simp only [← c113166, ← c113167, ← c113168] at e_1852_18
  have e_1852_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_19
  simp only [← c113169, ← c113170, k2, ← c113171] at e_1852_19
  have hr := e_1852_19
  simp only [e_1852_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5383)) := by
  have c113172 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113173 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113174 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113175 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113176 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113177 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_0
  simp only [← c113172, ← c113173, ← c113174] at e_1853_0
  have e_1853_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_1
  simp only [← c113175, ← c113176, k2, ← c113177] at e_1853_1
  have hr := e_1853_1
  simp only [e_1853_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5384)) := by
  have c113178 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113179 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113180 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113181 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113182 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113183 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_2
  simp only [← c113178, ← c113179, ← c113180] at e_1853_2
  have e_1853_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_3
  simp only [← c113181, ← c113182, k2, ← c113183] at e_1853_3
  have hr := e_1853_3
  simp only [e_1853_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5385)) := by
  have c113184 := (publicBatchWrapper89_copies3537 a h).1
  have c113185 := (publicBatchWrapper89_copies3537 a h).2.1
  have c113186 := (publicBatchWrapper89_copies3537 a h).2.2.1
  have c113187 := (publicBatchWrapper89_copies3537 a h).2.2.2.1
  have c113188 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.1
  have c113189 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_4
  simp only [← c113184, ← c113185, ← c113186] at e_1853_4
  have e_1853_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_5
  simp only [← c113187, ← c113188, k2, ← c113189] at e_1853_5
  have hr := e_1853_5
  simp only [e_1853_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5386)) := by
  have c113190 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.1
  have c113191 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.1
  have c113192 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.1
  have c113193 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.1
  have c113194 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113195 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_6
  simp only [← c113190, ← c113191, ← c113192] at e_1853_6
  have e_1853_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_7
  simp only [← c113193, ← c113194, k2, ← c113195] at e_1853_7
  have hr := e_1853_7
  simp only [e_1853_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5387)) := by
  have c113196 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113197 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113198 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113199 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113200 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113201 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_8
  simp only [← c113196, ← c113197, ← c113198] at e_1853_8
  have e_1853_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_9
  simp only [← c113199, ← c113200, k2, ← c113201] at e_1853_9
  have hr := e_1853_9
  simp only [e_1853_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5388)) := by
  have c113202 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113203 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113204 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113205 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113206 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113207 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_10
  simp only [← c113202, ← c113203, ← c113204] at e_1853_10
  have e_1853_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_11
  simp only [← c113205, ← c113206, k2, ← c113207] at e_1853_11
  have hr := e_1853_11
  simp only [e_1853_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5389)) := by
  have c113208 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113209 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113210 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113211 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113212 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113213 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_12
  simp only [← c113208, ← c113209, ← c113210] at e_1853_12
  have e_1853_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_13
  simp only [← c113211, ← c113212, k2, ← c113213] at e_1853_13
  have hr := e_1853_13
  simp only [e_1853_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5390)) := by
  have c113214 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113215 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113216 := (publicBatchWrapper89_copies3538 a h).1
  have c113217 := (publicBatchWrapper89_copies3538 a h).2.1
  have c113218 := (publicBatchWrapper89_copies3538 a h).2.2.1
  have c113219 := (publicBatchWrapper89_copies3538 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_14
  simp only [← c113214, ← c113215, ← c113216] at e_1853_14
  have e_1853_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_15
  simp only [← c113217, ← c113218, k2, ← c113219] at e_1853_15
  have hr := e_1853_15
  simp only [e_1853_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5391)) := by
  have c113220 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.1
  have c113221 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.1
  have c113222 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.1
  have c113223 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.1
  have c113224 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.1
  have c113225 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_16
  simp only [← c113220, ← c113221, ← c113222] at e_1853_16
  have e_1853_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_17
  simp only [← c113223, ← c113224, k2, ← c113225] at e_1853_17
  have hr := e_1853_17
  simp only [e_1853_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5392)) := by
  have c113226 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113227 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113228 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113229 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113230 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113231 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_18
  simp only [← c113226, ← c113227, ← c113228] at e_1853_18
  have e_1853_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_19
  simp only [← c113229, ← c113230, k2, ← c113231] at e_1853_19
  have hr := e_1853_19
  simp only [e_1853_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5393)) := by
  have c113232 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113233 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113234 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113235 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113236 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113237 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_0
  simp only [← c113232, ← c113233, ← c113234] at e_1854_0
  have e_1854_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_1
  simp only [← c113235, ← c113236, k2, ← c113237] at e_1854_1
  have hr := e_1854_1
  simp only [e_1854_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5394)) := by
  have c113238 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113239 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113240 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113241 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113242 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113243 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_2
  simp only [← c113238, ← c113239, ← c113240] at e_1854_2
  have e_1854_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_3
  simp only [← c113241, ← c113242, k2, ← c113243] at e_1854_3
  have hr := e_1854_3
  simp only [e_1854_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5395)) := by
  have c113244 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113245 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113246 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113247 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113248 := (publicBatchWrapper89_copies3539 a h).1
  have c113249 := (publicBatchWrapper89_copies3539 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_4
  simp only [← c113244, ← c113245, ← c113246] at e_1854_4
  have e_1854_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_5
  simp only [← c113247, ← c113248, k2, ← c113249] at e_1854_5
  have hr := e_1854_5
  simp only [e_1854_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5396)) := by
  have c113250 := (publicBatchWrapper89_copies3539 a h).2.2.1
  have c113251 := (publicBatchWrapper89_copies3539 a h).2.2.2.1
  have c113252 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.1
  have c113253 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.1
  have c113254 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.1
  have c113255 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_6
  simp only [← c113250, ← c113251, ← c113252] at e_1854_6
  have e_1854_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_7
  simp only [← c113253, ← c113254, k2, ← c113255] at e_1854_7
  have hr := e_1854_7
  simp only [e_1854_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5397)) := by
  have c113256 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.1
  have c113257 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.1
  have c113258 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113259 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113260 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113261 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_8
  simp only [← c113256, ← c113257, ← c113258] at e_1854_8
  have e_1854_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_9
  simp only [← c113259, ← c113260, k2, ← c113261] at e_1854_9
  have hr := e_1854_9
  simp only [e_1854_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5398)) := by
  have c113262 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113263 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113264 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113265 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113266 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113267 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_10
  simp only [← c113262, ← c113263, ← c113264] at e_1854_10
  have e_1854_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_11
  simp only [← c113265, ← c113266, k2, ← c113267] at e_1854_11
  have hr := e_1854_11
  simp only [e_1854_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5399)) := by
  have c113268 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113269 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113270 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113271 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113272 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113273 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_12
  simp only [← c113268, ← c113269, ← c113270] at e_1854_12
  have e_1854_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_13
  simp only [← c113271, ← c113272, k2, ← c113273] at e_1854_13
  have hr := e_1854_13
  simp only [e_1854_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5400)) := by
  have c113274 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113275 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113276 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113277 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113278 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113279 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_14
  simp only [← c113274, ← c113275, ← c113276] at e_1854_14
  have e_1854_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_15
  simp only [← c113277, ← c113278, k2, ← c113279] at e_1854_15
  have hr := e_1854_15
  simp only [e_1854_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5401)) := by
  have c113280 := (publicBatchWrapper89_copies3540 a h).1
  have c113281 := (publicBatchWrapper89_copies3540 a h).2.1
  have c113282 := (publicBatchWrapper89_copies3540 a h).2.2.1
  have c113283 := (publicBatchWrapper89_copies3540 a h).2.2.2.1
  have c113284 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.1
  have c113285 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_16
  simp only [← c113280, ← c113281, ← c113282] at e_1854_16
  have e_1854_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_17
  simp only [← c113283, ← c113284, k2, ← c113285] at e_1854_17
  have hr := e_1854_17
  simp only [e_1854_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5570)) := by
  have c113286 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.1
  have c113287 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.1
  have c113288 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.1
  have c113289 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.1
  have c113290 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113291 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_18
  simp only [← c113286, ← c113287, ← c113288] at e_1854_18
  have e_1854_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_19
  simp only [← c113289, ← c113290, k2, ← c113291] at e_1854_19
  have hr := e_1854_19
  simp only [e_1854_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5571)) := by
  have c113292 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113293 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113294 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113295 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113296 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113297 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_0
  simp only [← c113292, ← c113293, ← c113294] at e_1855_0
  have e_1855_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_1
  simp only [← c113295, ← c113296, k2, ← c113297] at e_1855_1
  have hr := e_1855_1
  simp only [e_1855_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5572)) := by
  have c113298 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113299 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113300 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113301 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113302 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113303 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_2
  simp only [← c113298, ← c113299, ← c113300] at e_1855_2
  have e_1855_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_3
  simp only [← c113301, ← c113302, k2, ← c113303] at e_1855_3
  have hr := e_1855_3
  simp only [e_1855_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5573)) := by
  have c113304 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113305 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113306 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113307 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113308 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113309 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_4
  simp only [← c113304, ← c113305, ← c113306] at e_1855_4
  have e_1855_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_5
  simp only [← c113307, ← c113308, k2, ← c113309] at e_1855_5
  have hr := e_1855_5
  simp only [e_1855_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5574)) := by
  have c113310 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113311 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113312 := (publicBatchWrapper89_copies3541 a h).1
  have c113313 := (publicBatchWrapper89_copies3541 a h).2.1
  have c113314 := (publicBatchWrapper89_copies3541 a h).2.2.1
  have c113315 := (publicBatchWrapper89_copies3541 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_6
  simp only [← c113310, ← c113311, ← c113312] at e_1855_6
  have e_1855_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_7
  simp only [← c113313, ← c113314, k2, ← c113315] at e_1855_7
  have hr := e_1855_7
  simp only [e_1855_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5575)) := by
  have c113316 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.1
  have c113317 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.1
  have c113318 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.1
  have c113319 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.1
  have c113320 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.1
  have c113321 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_8
  simp only [← c113316, ← c113317, ← c113318] at e_1855_8
  have e_1855_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_9
  simp only [← c113319, ← c113320, k2, ← c113321] at e_1855_9
  have hr := e_1855_9
  simp only [e_1855_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5576)) := by
  have c113322 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113323 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113324 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113325 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113326 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113327 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_10
  simp only [← c113322, ← c113323, ← c113324] at e_1855_10
  have e_1855_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_11
  simp only [← c113325, ← c113326, k2, ← c113327] at e_1855_11
  have hr := e_1855_11
  simp only [e_1855_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5577)) := by
  have c113328 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113329 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113330 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113331 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113332 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113333 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_12
  simp only [← c113328, ← c113329, ← c113330] at e_1855_12
  have e_1855_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_13
  simp only [← c113331, ← c113332, k2, ← c113333] at e_1855_13
  have hr := e_1855_13
  simp only [e_1855_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5578)) := by
  have c113334 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113335 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113336 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113337 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113338 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113339 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_14
  simp only [← c113334, ← c113335, ← c113336] at e_1855_14
  have e_1855_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_15
  simp only [← c113337, ← c113338, k2, ← c113339] at e_1855_15
  have hr := e_1855_15
  simp only [e_1855_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5579)) := by
  have c113340 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113341 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113342 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113343 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113344 := (publicBatchWrapper89_copies3542 a h).1
  have c113345 := (publicBatchWrapper89_copies3542 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_16
  simp only [← c113340, ← c113341, ← c113342] at e_1855_16
  have e_1855_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_17
  simp only [← c113343, ← c113344, k2, ← c113345] at e_1855_17
  have hr := e_1855_17
  simp only [e_1855_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5580)) := by
  have c113346 := (publicBatchWrapper89_copies3542 a h).2.2.1
  have c113347 := (publicBatchWrapper89_copies3542 a h).2.2.2.1
  have c113348 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.1
  have c113349 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.1
  have c113350 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.1
  have c113351 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_18
  simp only [← c113346, ← c113347, ← c113348] at e_1855_18
  have e_1855_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_19
  simp only [← c113349, ← c113350, k2, ← c113351] at e_1855_19
  have hr := e_1855_19
  simp only [e_1855_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5581)) := by
  have c113352 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.1
  have c113353 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.1
  have c113354 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113355 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113356 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113357 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_0
  simp only [← c113352, ← c113353, ← c113354] at e_1856_0
  have e_1856_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_1
  simp only [← c113355, ← c113356, k2, ← c113357] at e_1856_1
  have hr := e_1856_1
  simp only [e_1856_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5582)) := by
  have c113358 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113359 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113360 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113361 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113362 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113363 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_2
  simp only [← c113358, ← c113359, ← c113360] at e_1856_2
  have e_1856_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_3
  simp only [← c113361, ← c113362, k2, ← c113363] at e_1856_3
  have hr := e_1856_3
  simp only [e_1856_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5583)) := by
  have c113364 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113365 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113366 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113367 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113368 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113369 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_4
  simp only [← c113364, ← c113365, ← c113366] at e_1856_4
  have e_1856_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_5
  simp only [← c113367, ← c113368, k2, ← c113369] at e_1856_5
  have hr := e_1856_5
  simp only [e_1856_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5584)) := by
  have c113370 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113371 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113372 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113373 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113374 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113375 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_6
  simp only [← c113370, ← c113371, ← c113372] at e_1856_6
  have e_1856_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_7
  simp only [← c113373, ← c113374, k2, ← c113375] at e_1856_7
  have hr := e_1856_7
  simp only [e_1856_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5585)) := by
  have c113376 := (publicBatchWrapper89_copies3543 a h).1
  have c113377 := (publicBatchWrapper89_copies3543 a h).2.1
  have c113378 := (publicBatchWrapper89_copies3543 a h).2.2.1
  have c113379 := (publicBatchWrapper89_copies3543 a h).2.2.2.1
  have c113380 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.1
  have c113381 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_8
  simp only [← c113376, ← c113377, ← c113378] at e_1856_8
  have e_1856_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_9
  simp only [← c113379, ← c113380, k2, ← c113381] at e_1856_9
  have hr := e_1856_9
  simp only [e_1856_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5586)) := by
  have c113382 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.1
  have c113383 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.1
  have c113384 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.1
  have c113385 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.1
  have c113386 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113387 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_10
  simp only [← c113382, ← c113383, ← c113384] at e_1856_10
  have e_1856_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_11
  simp only [← c113385, ← c113386, k2, ← c113387] at e_1856_11
  have hr := e_1856_11
  simp only [e_1856_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5587)) := by
  have c113388 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113389 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113390 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113391 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113392 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113393 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_12
  simp only [← c113388, ← c113389, ← c113390] at e_1856_12
  have e_1856_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_13
  simp only [← c113391, ← c113392, k2, ← c113393] at e_1856_13
  have hr := e_1856_13
  simp only [e_1856_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5588)) := by
  have c113394 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113395 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113396 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113397 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113398 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113399 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_14
  simp only [← c113394, ← c113395, ← c113396] at e_1856_14
  have e_1856_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_15
  simp only [← c113397, ← c113398, k2, ← c113399] at e_1856_15
  have hr := e_1856_15
  simp only [e_1856_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5589)) := by
  have c113400 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113401 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113402 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113403 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113404 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113405 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_16
  simp only [← c113400, ← c113401, ← c113402] at e_1856_16
  have e_1856_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_17
  simp only [← c113403, ← c113404, k2, ← c113405] at e_1856_17
  have hr := e_1856_17
  simp only [e_1856_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5590)) := by
  have c113406 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113407 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113408 := (publicBatchWrapper89_copies3544 a h).1
  have c113409 := (publicBatchWrapper89_copies3544 a h).2.1
  have c113410 := (publicBatchWrapper89_copies3544 a h).2.2.1
  have c113411 := (publicBatchWrapper89_copies3544 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_18
  simp only [← c113406, ← c113407, ← c113408] at e_1856_18
  have e_1856_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_19
  simp only [← c113409, ← c113410, k2, ← c113411] at e_1856_19
  have hr := e_1856_19
  simp only [e_1856_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5591)) := by
  have c113412 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.1
  have c113413 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.1
  have c113414 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.1
  have c113415 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.1
  have c113416 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.1
  have c113417 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_0
  simp only [← c113412, ← c113413, ← c113414] at e_1857_0
  have e_1857_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_1
  simp only [← c113415, ← c113416, k2, ← c113417] at e_1857_1
  have hr := e_1857_1
  simp only [e_1857_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5592)) := by
  have c113418 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113419 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113420 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113421 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113422 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113423 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_2
  simp only [← c113418, ← c113419, ← c113420] at e_1857_2
  have e_1857_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_3
  simp only [← c113421, ← c113422, k2, ← c113423] at e_1857_3
  have hr := e_1857_3
  simp only [e_1857_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5593)) := by
  have c113424 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113425 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113426 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113427 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113428 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113429 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_4
  simp only [← c113424, ← c113425, ← c113426] at e_1857_4
  have e_1857_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_5
  simp only [← c113427, ← c113428, k2, ← c113429] at e_1857_5
  have hr := e_1857_5
  simp only [e_1857_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5594)) := by
  have c113430 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113431 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113432 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113433 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113434 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113435 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_6
  simp only [← c113430, ← c113431, ← c113432] at e_1857_6
  have e_1857_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_7
  simp only [← c113433, ← c113434, k2, ← c113435] at e_1857_7
  have hr := e_1857_7
  simp only [e_1857_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5595)) := by
  have c113436 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113437 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113438 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113439 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113440 := (publicBatchWrapper89_copies3545 a h).1
  have c113441 := (publicBatchWrapper89_copies3545 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_8
  simp only [← c113436, ← c113437, ← c113438] at e_1857_8
  have e_1857_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_9
  simp only [← c113439, ← c113440, k2, ← c113441] at e_1857_9
  have hr := e_1857_9
  simp only [e_1857_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5596)) := by
  have c113442 := (publicBatchWrapper89_copies3545 a h).2.2.1
  have c113443 := (publicBatchWrapper89_copies3545 a h).2.2.2.1
  have c113444 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.1
  have c113445 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.1
  have c113446 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.1
  have c113447 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_10
  simp only [← c113442, ← c113443, ← c113444] at e_1857_10
  have e_1857_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_11
  simp only [← c113445, ← c113446, k2, ← c113447] at e_1857_11
  have hr := e_1857_11
  simp only [e_1857_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5597)) := by
  have c113448 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.1
  have c113449 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.1
  have c113450 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113451 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113452 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113453 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_12
  simp only [← c113448, ← c113449, ← c113450] at e_1857_12
  have e_1857_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_13
  simp only [← c113451, ← c113452, k2, ← c113453] at e_1857_13
  have hr := e_1857_13
  simp only [e_1857_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5598)) := by
  have c113454 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113455 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113456 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113457 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113458 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113459 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_14
  simp only [← c113454, ← c113455, ← c113456] at e_1857_14
  have e_1857_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_15
  simp only [← c113457, ← c113458, k2, ← c113459] at e_1857_15
  have hr := e_1857_15
  simp only [e_1857_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5599)) := by
  have c113460 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113461 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113462 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113463 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113464 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113465 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_16
  simp only [← c113460, ← c113461, ← c113462] at e_1857_16
  have e_1857_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_17
  simp only [← c113463, ← c113464, k2, ← c113465] at e_1857_17
  have hr := e_1857_17
  simp only [e_1857_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5600)) := by
  have c113466 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113467 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113468 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113469 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113470 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113471 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_18
  simp only [← c113466, ← c113467, ← c113468] at e_1857_18
  have e_1857_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_19
  simp only [← c113469, ← c113470, k2, ← c113471] at e_1857_19
  have hr := e_1857_19
  simp only [e_1857_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5601)) := by
  have c113472 := (publicBatchWrapper89_copies3546 a h).1
  have c113473 := (publicBatchWrapper89_copies3546 a h).2.1
  have c113474 := (publicBatchWrapper89_copies3546 a h).2.2.1
  have c113475 := (publicBatchWrapper89_copies3546 a h).2.2.2.1
  have c113476 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.1
  have c113477 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_0
  simp only [← c113472, ← c113473, ← c113474] at e_1858_0
  have e_1858_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_1
  simp only [← c113475, ← c113476, k2, ← c113477] at e_1858_1
  have hr := e_1858_1
  simp only [e_1858_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5770)) := by
  have c113478 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.1
  have c113479 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.1
  have c113480 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.1
  have c113481 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.1
  have c113482 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113483 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_2
  simp only [← c113478, ← c113479, ← c113480] at e_1858_2
  have e_1858_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_3
  simp only [← c113481, ← c113482, k2, ← c113483] at e_1858_3
  have hr := e_1858_3
  simp only [e_1858_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5771)) := by
  have c113484 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113485 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113486 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113487 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113488 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113489 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_4
  simp only [← c113484, ← c113485, ← c113486] at e_1858_4
  have e_1858_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_5
  simp only [← c113487, ← c113488, k2, ← c113489] at e_1858_5
  have hr := e_1858_5
  simp only [e_1858_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5772)) := by
  have c113490 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113491 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113492 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113493 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113494 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113495 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_6
  simp only [← c113490, ← c113491, ← c113492] at e_1858_6
  have e_1858_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_7
  simp only [← c113493, ← c113494, k2, ← c113495] at e_1858_7
  have hr := e_1858_7
  simp only [e_1858_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5773)) := by
  have c113496 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113497 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113498 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113499 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113500 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113501 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_8
  simp only [← c113496, ← c113497, ← c113498] at e_1858_8
  have e_1858_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_9
  simp only [← c113499, ← c113500, k2, ← c113501] at e_1858_9
  have hr := e_1858_9
  simp only [e_1858_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5774)) := by
  have c113502 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113503 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113504 := (publicBatchWrapper89_copies3547 a h).1
  have c113505 := (publicBatchWrapper89_copies3547 a h).2.1
  have c113506 := (publicBatchWrapper89_copies3547 a h).2.2.1
  have c113507 := (publicBatchWrapper89_copies3547 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_10
  simp only [← c113502, ← c113503, ← c113504] at e_1858_10
  have e_1858_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_11
  simp only [← c113505, ← c113506, k2, ← c113507] at e_1858_11
  have hr := e_1858_11
  simp only [e_1858_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5775)) := by
  have c113508 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.1
  have c113509 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.1
  have c113510 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.1
  have c113511 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.1
  have c113512 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.1
  have c113513 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_12
  simp only [← c113508, ← c113509, ← c113510] at e_1858_12
  have e_1858_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_13
  simp only [← c113511, ← c113512, k2, ← c113513] at e_1858_13
  have hr := e_1858_13
  simp only [e_1858_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5776)) := by
  have c113514 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113515 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113516 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113517 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113518 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113519 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_14
  simp only [← c113514, ← c113515, ← c113516] at e_1858_14
  have e_1858_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_15
  simp only [← c113517, ← c113518, k2, ← c113519] at e_1858_15
  have hr := e_1858_15
  simp only [e_1858_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5777)) := by
  have c113520 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113521 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113522 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113523 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113524 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113525 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_16
  simp only [← c113520, ← c113521, ← c113522] at e_1858_16
  have e_1858_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_17
  simp only [← c113523, ← c113524, k2, ← c113525] at e_1858_17
  have hr := e_1858_17
  simp only [e_1858_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5778)) := by
  have c113526 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113527 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113528 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113529 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113530 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113531 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_18
  simp only [← c113526, ← c113527, ← c113528] at e_1858_18
  have e_1858_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_19
  simp only [← c113529, ← c113530, k2, ← c113531] at e_1858_19
  have hr := e_1858_19
  simp only [e_1858_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5779)) := by
  have c113532 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113533 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113534 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113535 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113536 := (publicBatchWrapper89_copies3548 a h).1
  have c113537 := (publicBatchWrapper89_copies3548 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_0
  simp only [← c113532, ← c113533, ← c113534] at e_1859_0
  have e_1859_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_1
  simp only [← c113535, ← c113536, k2, ← c113537] at e_1859_1
  have hr := e_1859_1
  simp only [e_1859_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5780)) := by
  have c113538 := (publicBatchWrapper89_copies3548 a h).2.2.1
  have c113539 := (publicBatchWrapper89_copies3548 a h).2.2.2.1
  have c113540 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.1
  have c113541 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.1
  have c113542 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.1
  have c113543 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_2
  simp only [← c113538, ← c113539, ← c113540] at e_1859_2
  have e_1859_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_3
  simp only [← c113541, ← c113542, k2, ← c113543] at e_1859_3
  have hr := e_1859_3
  simp only [e_1859_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5781)) := by
  have c113544 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.1
  have c113545 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.1
  have c113546 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113547 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113548 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113549 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_4
  simp only [← c113544, ← c113545, ← c113546] at e_1859_4
  have e_1859_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_5
  simp only [← c113547, ← c113548, k2, ← c113549] at e_1859_5
  have hr := e_1859_5
  simp only [e_1859_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5782)) := by
  have c113550 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113551 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113552 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113553 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113554 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113555 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_6
  simp only [← c113550, ← c113551, ← c113552] at e_1859_6
  have e_1859_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_7
  simp only [← c113553, ← c113554, k2, ← c113555] at e_1859_7
  have hr := e_1859_7
  simp only [e_1859_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5783)) := by
  have c113556 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113557 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113558 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113559 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113560 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113561 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_8
  simp only [← c113556, ← c113557, ← c113558] at e_1859_8
  have e_1859_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_9
  simp only [← c113559, ← c113560, k2, ← c113561] at e_1859_9
  have hr := e_1859_9
  simp only [e_1859_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5784)) := by
  have c113562 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113563 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113564 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113565 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113566 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113567 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_10
  simp only [← c113562, ← c113563, ← c113564] at e_1859_10
  have e_1859_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_11
  simp only [← c113565, ← c113566, k2, ← c113567] at e_1859_11
  have hr := e_1859_11
  simp only [e_1859_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5785)) := by
  have c113568 := (publicBatchWrapper89_copies3549 a h).1
  have c113569 := (publicBatchWrapper89_copies3549 a h).2.1
  have c113570 := (publicBatchWrapper89_copies3549 a h).2.2.1
  have c113571 := (publicBatchWrapper89_copies3549 a h).2.2.2.1
  have c113572 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.1
  have c113573 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_12
  simp only [← c113568, ← c113569, ← c113570] at e_1859_12
  have e_1859_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_13
  simp only [← c113571, ← c113572, k2, ← c113573] at e_1859_13
  have hr := e_1859_13
  simp only [e_1859_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5786)) := by
  have c113574 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.1
  have c113575 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.1
  have c113576 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.1
  have c113577 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.1
  have c113578 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113579 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_14
  simp only [← c113574, ← c113575, ← c113576] at e_1859_14
  have e_1859_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_15
  simp only [← c113577, ← c113578, k2, ← c113579] at e_1859_15
  have hr := e_1859_15
  simp only [e_1859_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5787)) := by
  have c113580 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113581 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113582 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113583 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113584 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113585 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_16
  simp only [← c113580, ← c113581, ← c113582] at e_1859_16
  have e_1859_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_17
  simp only [← c113583, ← c113584, k2, ← c113585] at e_1859_17
  have hr := e_1859_17
  simp only [e_1859_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5788)) := by
  have c113586 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113587 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113588 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113589 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113590 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113591 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_18
  simp only [← c113586, ← c113587, ← c113588] at e_1859_18
  have e_1859_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_19
  simp only [← c113589, ← c113590, k2, ← c113591] at e_1859_19
  have hr := e_1859_19
  simp only [e_1859_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5789)) := by
  have c113592 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113593 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113594 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113595 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113596 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113597 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_0
  simp only [← c113592, ← c113593, ← c113594] at e_1860_0
  have e_1860_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_1
  simp only [← c113595, ← c113596, k2, ← c113597] at e_1860_1
  have hr := e_1860_1
  simp only [e_1860_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5790)) := by
  have c113598 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113599 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113600 := (publicBatchWrapper89_copies3550 a h).1
  have c113601 := (publicBatchWrapper89_copies3550 a h).2.1
  have c113602 := (publicBatchWrapper89_copies3550 a h).2.2.1
  have c113603 := (publicBatchWrapper89_copies3550 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_2
  simp only [← c113598, ← c113599, ← c113600] at e_1860_2
  have e_1860_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_3
  simp only [← c113601, ← c113602, k2, ← c113603] at e_1860_3
  have hr := e_1860_3
  simp only [e_1860_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5791)) := by
  have c113604 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.1
  have c113605 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.1
  have c113606 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.1
  have c113607 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.1
  have c113608 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.1
  have c113609 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_4
  simp only [← c113604, ← c113605, ← c113606] at e_1860_4
  have e_1860_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_5
  simp only [← c113607, ← c113608, k2, ← c113609] at e_1860_5
  have hr := e_1860_5
  simp only [e_1860_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5792)) := by
  have c113610 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113611 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113612 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113613 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113614 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113615 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_6
  simp only [← c113610, ← c113611, ← c113612] at e_1860_6
  have e_1860_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_7
  simp only [← c113613, ← c113614, k2, ← c113615] at e_1860_7
  have hr := e_1860_7
  simp only [e_1860_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5793)) := by
  have c113616 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113617 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113618 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113619 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113620 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113621 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_8
  simp only [← c113616, ← c113617, ← c113618] at e_1860_8
  have e_1860_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_9
  simp only [← c113619, ← c113620, k2, ← c113621] at e_1860_9
  have hr := e_1860_9
  simp only [e_1860_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5794)) := by
  have c113622 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113623 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113624 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113625 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113626 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113627 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_10
  simp only [← c113622, ← c113623, ← c113624] at e_1860_10
  have e_1860_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_11
  simp only [← c113625, ← c113626, k2, ← c113627] at e_1860_11
  have hr := e_1860_11
  simp only [e_1860_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5795)) := by
  have c113628 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113629 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113630 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113631 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113632 := (publicBatchWrapper89_copies3551 a h).1
  have c113633 := (publicBatchWrapper89_copies3551 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_12
  simp only [← c113628, ← c113629, ← c113630] at e_1860_12
  have e_1860_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_13
  simp only [← c113631, ← c113632, k2, ← c113633] at e_1860_13
  have hr := e_1860_13
  simp only [e_1860_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5796)) := by
  have c113634 := (publicBatchWrapper89_copies3551 a h).2.2.1
  have c113635 := (publicBatchWrapper89_copies3551 a h).2.2.2.1
  have c113636 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.1
  have c113637 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.1
  have c113638 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.1
  have c113639 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_14
  simp only [← c113634, ← c113635, ← c113636] at e_1860_14
  have e_1860_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_15
  simp only [← c113637, ← c113638, k2, ← c113639] at e_1860_15
  have hr := e_1860_15
  simp only [e_1860_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5797)) := by
  have c113640 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.1
  have c113641 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.1
  have c113642 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113643 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113644 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113645 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_16
  simp only [← c113640, ← c113641, ← c113642] at e_1860_16
  have e_1860_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_17
  simp only [← c113643, ← c113644, k2, ← c113645] at e_1860_17
  have hr := e_1860_17
  simp only [e_1860_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5798)) := by
  have c113646 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113647 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113648 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113649 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113650 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113651 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_18
  simp only [← c113646, ← c113647, ← c113648] at e_1860_18
  have e_1860_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_19
  simp only [← c113649, ← c113650, k2, ← c113651] at e_1860_19
  have hr := e_1860_19
  simp only [e_1860_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5799)) := by
  have c113652 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113653 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113654 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113655 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113656 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113657 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_0
  simp only [← c113652, ← c113653, ← c113654] at e_1861_0
  have e_1861_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_1
  simp only [← c113655, ← c113656, k2, ← c113657] at e_1861_1
  have hr := e_1861_1
  simp only [e_1861_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5800)) := by
  have c113658 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113659 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113660 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113661 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113662 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113663 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_2
  simp only [← c113658, ← c113659, ← c113660] at e_1861_2
  have e_1861_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_3
  simp only [← c113661, ← c113662, k2, ← c113663] at e_1861_3
  have hr := e_1861_3
  simp only [e_1861_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5801)) := by
  have c113664 := (publicBatchWrapper89_copies3552 a h).1
  have c113665 := (publicBatchWrapper89_copies3552 a h).2.1
  have c113666 := (publicBatchWrapper89_copies3552 a h).2.2.1
  have c113667 := (publicBatchWrapper89_copies3552 a h).2.2.2.1
  have c113668 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.1
  have c113669 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_4
  simp only [← c113664, ← c113665, ← c113666] at e_1861_4
  have e_1861_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_5
  simp only [← c113667, ← c113668, k2, ← c113669] at e_1861_5
  have hr := e_1861_5
  simp only [e_1861_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5970)) := by
  have c113670 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.1
  have c113671 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.1
  have c113672 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.1
  have c113673 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.1
  have c113674 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113675 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_6
  simp only [← c113670, ← c113671, ← c113672] at e_1861_6
  have e_1861_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_7
  simp only [← c113673, ← c113674, k2, ← c113675] at e_1861_7
  have hr := e_1861_7
  simp only [e_1861_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5971)) := by
  have c113676 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113677 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113678 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113679 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113680 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113681 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_8
  simp only [← c113676, ← c113677, ← c113678] at e_1861_8
  have e_1861_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_9
  simp only [← c113679, ← c113680, k2, ← c113681] at e_1861_9
  have hr := e_1861_9
  simp only [e_1861_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5972)) := by
  have c113682 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113683 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113684 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113685 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113686 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113687 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_10
  simp only [← c113682, ← c113683, ← c113684] at e_1861_10
  have e_1861_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_11
  simp only [← c113685, ← c113686, k2, ← c113687] at e_1861_11
  have hr := e_1861_11
  simp only [e_1861_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5973)) := by
  have c113688 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113689 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113690 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113691 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113692 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113693 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_12
  simp only [← c113688, ← c113689, ← c113690] at e_1861_12
  have e_1861_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_13
  simp only [← c113691, ← c113692, k2, ← c113693] at e_1861_13
  have hr := e_1861_13
  simp only [e_1861_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5974)) := by
  have c113694 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113695 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113696 := (publicBatchWrapper89_copies3553 a h).1
  have c113697 := (publicBatchWrapper89_copies3553 a h).2.1
  have c113698 := (publicBatchWrapper89_copies3553 a h).2.2.1
  have c113699 := (publicBatchWrapper89_copies3553 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_14
  simp only [← c113694, ← c113695, ← c113696] at e_1861_14
  have e_1861_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_15
  simp only [← c113697, ← c113698, k2, ← c113699] at e_1861_15
  have hr := e_1861_15
  simp only [e_1861_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5975)) := by
  have c113700 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.1
  have c113701 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.1
  have c113702 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.1
  have c113703 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.1
  have c113704 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.1
  have c113705 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_16
  simp only [← c113700, ← c113701, ← c113702] at e_1861_16
  have e_1861_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_17
  simp only [← c113703, ← c113704, k2, ← c113705] at e_1861_17
  have hr := e_1861_17
  simp only [e_1861_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5976)) := by
  have c113706 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113707 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113708 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113709 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113710 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113711 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_18
  simp only [← c113706, ← c113707, ← c113708] at e_1861_18
  have e_1861_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_19
  simp only [← c113709, ← c113710, k2, ← c113711] at e_1861_19
  have hr := e_1861_19
  simp only [e_1861_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5977)) := by
  have c113712 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113713 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113714 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113715 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113716 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113717 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_0
  simp only [← c113712, ← c113713, ← c113714] at e_1862_0
  have e_1862_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_1
  simp only [← c113715, ← c113716, k2, ← c113717] at e_1862_1
  have hr := e_1862_1
  simp only [e_1862_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5978)) := by
  have c113718 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113719 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113720 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113721 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113722 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113723 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_2
  simp only [← c113718, ← c113719, ← c113720] at e_1862_2
  have e_1862_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_3
  simp only [← c113721, ← c113722, k2, ← c113723] at e_1862_3
  have hr := e_1862_3
  simp only [e_1862_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5979)) := by
  have c113724 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113725 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113726 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113727 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113728 := (publicBatchWrapper89_copies3554 a h).1
  have c113729 := (publicBatchWrapper89_copies3554 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_4
  simp only [← c113724, ← c113725, ← c113726] at e_1862_4
  have e_1862_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_5
  simp only [← c113727, ← c113728, k2, ← c113729] at e_1862_5
  have hr := e_1862_5
  simp only [e_1862_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5980)) := by
  have c113730 := (publicBatchWrapper89_copies3554 a h).2.2.1
  have c113731 := (publicBatchWrapper89_copies3554 a h).2.2.2.1
  have c113732 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.1
  have c113733 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.1
  have c113734 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.1
  have c113735 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_6
  simp only [← c113730, ← c113731, ← c113732] at e_1862_6
  have e_1862_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_7
  simp only [← c113733, ← c113734, k2, ← c113735] at e_1862_7
  have hr := e_1862_7
  simp only [e_1862_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5981)) := by
  have c113736 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.1
  have c113737 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.1
  have c113738 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113739 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113740 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113741 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_8
  simp only [← c113736, ← c113737, ← c113738] at e_1862_8
  have e_1862_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_9
  simp only [← c113739, ← c113740, k2, ← c113741] at e_1862_9
  have hr := e_1862_9
  simp only [e_1862_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5982)) := by
  have c113742 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113743 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113744 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113745 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113746 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113747 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_10
  simp only [← c113742, ← c113743, ← c113744] at e_1862_10
  have e_1862_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_11
  simp only [← c113745, ← c113746, k2, ← c113747] at e_1862_11
  have hr := e_1862_11
  simp only [e_1862_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5983)) := by
  have c113748 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113749 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113750 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113751 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113752 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113753 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_12
  simp only [← c113748, ← c113749, ← c113750] at e_1862_12
  have e_1862_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_13
  simp only [← c113751, ← c113752, k2, ← c113753] at e_1862_13
  have hr := e_1862_13
  simp only [e_1862_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5984)) := by
  have c113754 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113755 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113756 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113757 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113758 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113759 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_14
  simp only [← c113754, ← c113755, ← c113756] at e_1862_14
  have e_1862_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_15
  simp only [← c113757, ← c113758, k2, ← c113759] at e_1862_15
  have hr := e_1862_15
  simp only [e_1862_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5985)) := by
  have c113760 := (publicBatchWrapper89_copies3555 a h).1
  have c113761 := (publicBatchWrapper89_copies3555 a h).2.1
  have c113762 := (publicBatchWrapper89_copies3555 a h).2.2.1
  have c113763 := (publicBatchWrapper89_copies3555 a h).2.2.2.1
  have c113764 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.1
  have c113765 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_16
  simp only [← c113760, ← c113761, ← c113762] at e_1862_16
  have e_1862_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_17
  simp only [← c113763, ← c113764, k2, ← c113765] at e_1862_17
  have hr := e_1862_17
  simp only [e_1862_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5986)) := by
  have c113766 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.1
  have c113767 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.1
  have c113768 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.1
  have c113769 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.1
  have c113770 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113771 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_18
  simp only [← c113766, ← c113767, ← c113768] at e_1862_18
  have e_1862_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_19
  simp only [← c113769, ← c113770, k2, ← c113771] at e_1862_19
  have hr := e_1862_19
  simp only [e_1862_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5987)) := by
  have c113772 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113773 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113774 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113775 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113776 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113777 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_0
  simp only [← c113772, ← c113773, ← c113774] at e_1863_0
  have e_1863_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_1
  simp only [← c113775, ← c113776, k2, ← c113777] at e_1863_1
  have hr := e_1863_1
  simp only [e_1863_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5988)) := by
  have c113778 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113779 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113780 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113781 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113782 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113783 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_2
  simp only [← c113778, ← c113779, ← c113780] at e_1863_2
  have e_1863_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_3
  simp only [← c113781, ← c113782, k2, ← c113783] at e_1863_3
  have hr := e_1863_3
  simp only [e_1863_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5989)) := by
  have c113784 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113785 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113786 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113787 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113788 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113789 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_4
  simp only [← c113784, ← c113785, ← c113786] at e_1863_4
  have e_1863_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_5
  simp only [← c113787, ← c113788, k2, ← c113789] at e_1863_5
  have hr := e_1863_5
  simp only [e_1863_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5990)) := by
  have c113790 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113791 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113792 := (publicBatchWrapper89_copies3556 a h).1
  have c113793 := (publicBatchWrapper89_copies3556 a h).2.1
  have c113794 := (publicBatchWrapper89_copies3556 a h).2.2.1
  have c113795 := (publicBatchWrapper89_copies3556 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_6
  simp only [← c113790, ← c113791, ← c113792] at e_1863_6
  have e_1863_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_7
  simp only [← c113793, ← c113794, k2, ← c113795] at e_1863_7
  have hr := e_1863_7
  simp only [e_1863_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5991)) := by
  have c113796 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.1
  have c113797 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.1
  have c113798 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.1
  have c113799 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.1
  have c113800 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.1
  have c113801 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_8
  simp only [← c113796, ← c113797, ← c113798] at e_1863_8
  have e_1863_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_9
  simp only [← c113799, ← c113800, k2, ← c113801] at e_1863_9
  have hr := e_1863_9
  simp only [e_1863_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5992)) := by
  have c113802 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113803 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113804 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113805 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113806 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113807 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_10
  simp only [← c113802, ← c113803, ← c113804] at e_1863_10
  have e_1863_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_11
  simp only [← c113805, ← c113806, k2, ← c113807] at e_1863_11
  have hr := e_1863_11
  simp only [e_1863_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5993)) := by
  have c113808 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113809 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113810 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113811 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113812 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113813 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_12
  simp only [← c113808, ← c113809, ← c113810] at e_1863_12
  have e_1863_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_13
  simp only [← c113811, ← c113812, k2, ← c113813] at e_1863_13
  have hr := e_1863_13
  simp only [e_1863_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5994)) := by
  have c113814 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113815 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113816 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113817 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113818 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113819 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_14
  simp only [← c113814, ← c113815, ← c113816] at e_1863_14
  have e_1863_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_15
  simp only [← c113817, ← c113818, k2, ← c113819] at e_1863_15
  have hr := e_1863_15
  simp only [e_1863_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5995)) := by
  have c113820 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113821 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113822 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113823 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113824 := (publicBatchWrapper89_copies3557 a h).1
  have c113825 := (publicBatchWrapper89_copies3557 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_16
  simp only [← c113820, ← c113821, ← c113822] at e_1863_16
  have e_1863_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_17
  simp only [← c113823, ← c113824, k2, ← c113825] at e_1863_17
  have hr := e_1863_17
  simp only [e_1863_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5996)) := by
  have c113826 := (publicBatchWrapper89_copies3557 a h).2.2.1
  have c113827 := (publicBatchWrapper89_copies3557 a h).2.2.2.1
  have c113828 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.1
  have c113829 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.1
  have c113830 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.1
  have c113831 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_18
  simp only [← c113826, ← c113827, ← c113828] at e_1863_18
  have e_1863_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_19
  simp only [← c113829, ← c113830, k2, ← c113831] at e_1863_19
  have hr := e_1863_19
  simp only [e_1863_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5997)) := by
  have c113832 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.1
  have c113833 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.1
  have c113834 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113835 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113836 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113837 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_0
  simp only [← c113832, ← c113833, ← c113834] at e_1864_0
  have e_1864_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_1
  simp only [← c113835, ← c113836, k2, ← c113837] at e_1864_1
  have hr := e_1864_1
  simp only [e_1864_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5998)) := by
  have c113838 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113839 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113840 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113841 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113842 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113843 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_2
  simp only [← c113838, ← c113839, ← c113840] at e_1864_2
  have e_1864_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_3
  simp only [← c113841, ← c113842, k2, ← c113843] at e_1864_3
  have hr := e_1864_3
  simp only [e_1864_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5999)) := by
  have c113844 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113845 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113846 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113847 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113848 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113849 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_4
  simp only [← c113844, ← c113845, ← c113846] at e_1864_4
  have e_1864_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_5
  simp only [← c113847, ← c113848, k2, ← c113849] at e_1864_5
  have hr := e_1864_5
  simp only [e_1864_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 6000)) := by
  have c113850 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113851 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113852 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113853 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113854 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113855 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_6
  simp only [← c113850, ← c113851, ← c113852] at e_1864_6
  have e_1864_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_7
  simp only [← c113853, ← c113854, k2, ← c113855] at e_1864_7
  have hr := e_1864_7
  simp only [e_1864_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 6001)) := by
  have c113856 := (publicBatchWrapper89_copies3558 a h).1
  have c113857 := (publicBatchWrapper89_copies3558 a h).2.1
  have c113858 := (publicBatchWrapper89_copies3558 a h).2.2.1
  have c113859 := (publicBatchWrapper89_copies3558 a h).2.2.2.1
  have c113860 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.1
  have c113861 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_8
  simp only [← c113856, ← c113857, ← c113858] at e_1864_8
  have e_1864_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_9
  simp only [← c113859, ← c113860, k2, ← c113861] at e_1864_9
  have hr := e_1864_9
  simp only [e_1864_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6170)) := by
  have c113862 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.1
  have c113863 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.1
  have c113864 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.1
  have c113865 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.1
  have c113866 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113867 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_10
  simp only [← c113862, ← c113863, ← c113864] at e_1864_10
  have e_1864_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_11
  simp only [← c113865, ← c113866, k2, ← c113867] at e_1864_11
  have hr := e_1864_11
  simp only [e_1864_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6171)) := by
  have c113868 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113869 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113870 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113871 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113872 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113873 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_12
  simp only [← c113868, ← c113869, ← c113870] at e_1864_12
  have e_1864_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_13
  simp only [← c113871, ← c113872, k2, ← c113873] at e_1864_13
  have hr := e_1864_13
  simp only [e_1864_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6172)) := by
  have c113874 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113875 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113876 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113877 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113878 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113879 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_14
  simp only [← c113874, ← c113875, ← c113876] at e_1864_14
  have e_1864_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_15
  simp only [← c113877, ← c113878, k2, ← c113879] at e_1864_15
  have hr := e_1864_15
  simp only [e_1864_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
